from typing import Optional
from modules import processing

from lib_cnpro import external_code

# --- host boundary: resolved through cnpro_host/adapter.py (never import the host directly) ---
from cnpro_host.adapter import image_utils

HWC3, _to_torch = image_utils()

from PIL import Image, ImageFilter, ImageOps
from lib_cnpro.lvminthin import lvmin_thin, nake_nms

import torch
import os
import numpy as np
import safetensors.torch
import cv2

from lib_cnpro.logging import logger

def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    logger.info(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict

def get_state_dict(d):
    return d.get("state_dict", d)

def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]

def align_dim_latent(x: int) -> int:
    """ Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel, i.e.,
    1 latent unit == 8 pixel unit."""
    return (x // 8) * 8

def predict_hires_dimensions(width, height, hr_scale, hr_resize_x, hr_resize_y,
                             res_step):
    """The hires-fix target size the HOST will sample at, computed pre-init.

    Replicates StableDiffusionProcessingTxt2Img.calculate_target_resolution
    (modules/processing.py) exactly, because the real thing cannot be read:
    scripts run BEFORE p.init(), so p.hr_upscale_to_x/y do not exist yet and
    the formula has to be reproduced. It has drifted once already - the host
    moved from int-truncation to sRound (round-half-up to opts.res_step,
    default 64), so 832x1216 at scale 1.5 samples at 1280x1856 while the old
    prediction said 1248x1824, and every hires hint was prepared at the wrong
    size and silently nearest-resized + center-cropped onto the real latent
    at sampling time. No error anywhere; the hires pass just came out softer
    and a sliver off its masks. tests/test_hires_dims.py holds this function
    equal to the host's own calculate_target_resolution, case by case, so the
    NEXT drift raises there instead of shipping.

    The one-sided hr_resize case (exactly one of x/y set, the other 0) is the
    host's aspect-preserving branch and must be reproduced too: the old
    prediction passed the 0 through, and a zero-height resize is an assertion
    inside cv2 at Generate time.

    Returns (hr_h, hr_w), unaligned - the caller aligns to the latent grid,
    which is also what the host's hires pass effectively samples at
    (hires latent = target // 8).
    """
    step = max(int(res_step), 1)

    def s_round(value):
        # modules.ui.sRound: round-half-up to the resolution step. int() is
        # floor for the positive values a resolution can be, matching the
        # host's math.floor without costing this module an import.
        return int(value / step + 0.5) * step

    if hr_resize_x == 0 and hr_resize_y == 0:
        return s_round(height * hr_scale), s_round(width * hr_scale)
    if hr_resize_y == 0:
        hr_x, hr_y = hr_resize_x, hr_resize_x * (height / width)
    elif hr_resize_x == 0:
        hr_x, hr_y = hr_resize_y * (width / height), hr_resize_y
    else:
        hr_x, hr_y = hr_resize_x, hr_resize_y
    return s_round(hr_y), s_round(hr_x)

def prepare_mask(
    mask: Image.Image, p: processing.StableDiffusionProcessing
) -> Image.Image:
    """
    Prepare an image mask for the inpainting process.

    This function takes as input a PIL Image object and an instance of the
    StableDiffusionProcessing class, and performs the following steps to prepare the mask:

    1. Convert the mask to grayscale (mode "L").
    2. If the 'inpainting_mask_invert' attribute of the processing instance is True,
       invert the mask colors.
    3. If the 'mask_blur' attribute of the processing instance is greater than 0,
       apply a Gaussian blur to the mask with a radius equal to 'mask_blur'.

    Args:
        mask (Image.Image): The input mask as a PIL Image object.
        p (processing.StableDiffusionProcessing): An instance of the StableDiffusionProcessing class
                                                   containing the processing parameters.

    Returns:
        mask (Image.Image): The prepared mask as a PIL Image object.
    """
    mask = mask.convert("L")
    if getattr(p, "inpainting_mask_invert", False):
        mask = ImageOps.invert(mask)

    if hasattr(p, 'mask_blur_x'):
        if getattr(p, "mask_blur_x", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
            mask = Image.fromarray(np_mask)
        if getattr(p, "mask_blur_y", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
            mask = Image.fromarray(np_mask)
    else:
        if getattr(p, "mask_blur", 0) > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    return mask

def set_numpy_seed(p: processing.StableDiffusionProcessing) -> Optional[int]:
    """
    Set the random seed for NumPy based on the provided parameters.

    Args:
        p (processing.StableDiffusionProcessing): The instance of the StableDiffusionProcessing class.

    Returns:
        Optional[int]: The computed random seed if successful, or None if an exception occurs.

    This function sets the random seed for NumPy using the seed and subseed values from the given instance of
    StableDiffusionProcessing. If either seed or subseed is -1, it uses the first value from `all_seeds`.
    Otherwise, it takes the maximum of the provided seed value and 0.

    The final random seed is computed by adding the seed and subseed values, applying a bitwise AND operation
    with 0xFFFFFFFF to ensure it fits within a 32-bit integer.
    """
    try:
        tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
        tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
        np.random.seed(seed)
        return seed
    except Exception as e:
        logger.warning(e)
        logger.warning('Warning: Failed to use consistent random seed.')
        return None

def safe_numpy(x):
    # A very safe method to make sure that Apple/Mac works
    y = x

    # below is very boring but do not change these. If you change these Apple or Mac may fail.
    y = y.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y

def high_quality_resize(x, size):
    # Written by lvmin
    # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

    if x.shape[0] != size[1] or x.shape[1] != size[0]:
        new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
        new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
        unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
        is_one_pixel_edge = False
        is_binary = False
        if unique_color_count == 2:
            is_binary = np.min(x) < 16 and np.max(x) > 240
            if is_binary:
                xc = x
                xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                all_edge_count = np.where(x > 127)[0].shape[0]
                is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

        if 2 < unique_color_count < 200:
            interpolation = cv2.INTER_NEAREST
        elif new_size_is_smaller:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

        y = cv2.resize(x, size, interpolation=interpolation)

        if is_binary:
            y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
            if is_one_pixel_edge:
                y = nake_nms(y)
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                y = lvmin_thin(y, prunings=new_size_is_bigger)
            else:
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            y = np.stack([y] * 3, axis=2)
    else:
        y = x

    return y

def mask_resize(x, size):
    """Resampler for VALUE masks (painted weights, feathered ramps).

    high_quality_resize is written for DETECTED MAPS and reads a mask's
    statistics as if it were one: a uniform paint at weight >= ~0.945 has two
    gray levels (0 and >= 241) and was OTSU-snapped to exactly 1.0 after the
    resize, and a light feather ramp (3..199 unique levels) was resized with
    INTER_NEAREST, stair-stepping the very edge the feather exists to smooth.
    A mask's gray IS the value: plain AREA on downscale (each output pixel is
    its true coverage average), LINEAR on upscale, nothing thresholded.
    tests/test_mask_resize.py pins both properties.
    """
    if x.shape[0] == size[1] and x.shape[1] == size[0]:
        return x
    smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
    return cv2.resize(x, size, interpolation=cv2.INTER_AREA if smaller else cv2.INTER_LINEAR)

def crop_and_resize_image(detected_map, resize_mode, h, w, fill_border_with_255=False,
                          resample=None):
    # `resample` swaps the RESAMPLER only; every geometric decision (fit, crop,
    # pad, border fill) below is shared by all callers, which is what keeps a
    # weight mask aligned with the hint it was painted over under every resize
    # mode. Pass mask_resize for value masks; the default is the detected-map
    # resampler.
    if resample is None:
        resample = high_quality_resize
    if resize_mode == external_code.ResizeMode.RESIZE:
        detected_map = resample(detected_map, (w, h))
        detected_map = safe_numpy(detected_map)
        return detected_map

    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    safeint = lambda x: int(np.round(x))

    if resize_mode == external_code.ResizeMode.OUTER_FIT:
        k = min(k0, k1)
        borders = np.concatenate([detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0)
        high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
        if fill_border_with_255:
            high_quality_border_color = np.zeros_like(high_quality_border_color) + 255
        high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
        detected_map = resample(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = detected_map
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
        return detected_map
    else:
        k = max(k0, k1)
        detected_map = resample(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        detected_map = detected_map[pad_h:pad_h+h, pad_w:pad_w+w]
        detected_map = safe_numpy(detected_map)
        return detected_map

def crop_and_resize_mask(mask_map, resize_mode, h, w):
    """crop_and_resize_image with the mask resampler - IDENTICAL geometry."""
    return crop_and_resize_image(mask_map, resize_mode, h, w, resample=mask_resize)

def judge_image_type(img):
    return isinstance(img, np.ndarray) and img.ndim == 3 and int(img.shape[2]) in [3, 4]
