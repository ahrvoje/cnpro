import os
import math
from typing import Optional, Tuple

import cv2
import torch

import modules.scripts as scripts
from modules import shared, script_callbacks, masking, images
from modules.ui_components import InputAccordion
import gradio as gr

from lib_cnpro import global_state, external_code
from lib_cnpro.external_code import ControlNetUnit
from lib_cnpro.utils import align_dim_latent, set_numpy_seed, crop_and_resize_image, \
    prepare_mask, judge_image_type
from lib_cnpro.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
from lib_cnpro.controlnet_ui.photopea import Photopea
from lib_cnpro.logging import logger
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessing
from lib_cnpro.infotext import Infotext
# --- host boundary: resolved through cnpro_host/adapter.py (never import the host directly) ---
from cnpro_host.adapter import image_utils

HWC3, numpy_to_pytorch = image_utils()
from lib_cnpro.enums import HiResFixOption
from lib_cnpro.api import controlnet_api

import numpy as np
import collections
import copy
import functools
import hashlib

from PIL import Image
from cnpro_host.adapter import load_control_model
# CNPro's OWN patchers - built by CNPro's registry, never the host's list.
from cnpro_host.patchers.base import CNProModelPatcher as ControlModelPatcher
# NOTE: no concrete patcher is imported here, and none should be. The UI reads
# capability flags off whatever patcher the registry built; importing one would
# both re-introduce the temptation to branch on its type and drag the whole
# torch/backend chain into the UI module's import time.

# Gradio 3.32 bug fix
import tempfile

gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_tempfile_path, exist_ok=True)

global_state.update_controlnet_filenames()


@functools.lru_cache(maxsize=shared.opts.data.get("control_net_model_cache_size", 5))
def cached_controlnet_loader(filename):
    return load_control_model(filename)


# Detected-map cache across Generate clicks: iterating on prompt/seed re-runs
# every (unchanged) preprocessor on every (unchanged) input each click - with
# heavy preprocessors (depth, normal) x several inputs that is seconds per
# iteration for identical results. Keyed by module + resolution + sliders +
# image/mask content hash; only plain ndarray outputs are cached (embedding
# preprocessors return dicts holding model references - never retain those).
PREPROCESSOR_CACHE_SIZE = 8
preprocessor_result_cache = collections.OrderedDict()


def cached_preprocessor_call(preprocessor, module_name, input_image, input_mask,
                             resolution, slider_1, slider_2):
    cacheable = isinstance(input_image, np.ndarray) and 'shuffle' not in module_name.lower()
    key = None
    if cacheable:
        digest = hashlib.sha1()
        digest.update(np.ascontiguousarray(input_image).tobytes())
        digest.update(str(input_image.shape).encode())
        if isinstance(input_mask, np.ndarray):
            digest.update(np.ascontiguousarray(input_mask).tobytes())
            digest.update(str(input_mask.shape).encode())
        key = (module_name, int(resolution), float(slider_1), float(slider_2),
               digest.hexdigest())
        cached = preprocessor_result_cache.get(key)
        if cached is not None:
            preprocessor_result_cache.move_to_end(key)
            logger.info(f"Preprocessor result reused from cache: {module_name} @ {resolution}.")
            return cached.copy()
    output = preprocessor(
        input_image=input_image,
        input_mask=input_mask,
        resolution=resolution,
        slider_1=slider_1,
        slider_2=slider_2,
    )
    if key is not None and isinstance(output, np.ndarray):
        preprocessor_result_cache[key] = output.copy()
        while len(preprocessor_result_cache) > PREPROCESSOR_CACHE_SIZE:
            preprocessor_result_cache.popitem(last=False)
    return output


WEIGHT_MASK_HUE_SPAN = 270.0


def decode_weight_mask(weight_mask):
    """Decode a painted weight mask into per-pixel weights in [0, 1].

    The painter ships masks as GRAYSCALE on the wire (pixel value = weight,
    alpha = paint coverage: 255 inside strokes, a partial ramp where the
    FEATHER slider blurred the stroke edge - canvas blur runs on premultiplied
    alpha, so the entire falloff lives in the alpha channel while the value
    channel keeps the painted weight). Alpha therefore multiplies the weight:
    binarizing it (the original decode) threw the feather ramp away and every
    mask edge came out hard. Unpainted pixels (alpha 0) stay weight 0. The
    rainbow hue is display-only on the JS side; a chromatic mask (the legacy
    rainbow wire format, still produced by old sessions and possible from API
    callers) falls back to the hue decode: (1 - hue / 270), red = 1 down to
    violet = 0.
    Returns a float32 HxW array, or None when nothing is painted.
    """
    if weight_mask is None:
        return None
    if weight_mask.ndim != 3 or weight_mask.shape[2] < 4:
        # API callers must send RGBA; a mask without alpha cannot mark painted
        # pixels and silently dropping it would be confusing
        logger.warning("ControlNet weight mask ignored: expected an RGBA image "
                       f"(alpha marks painted pixels), got shape {getattr(weight_mask, 'shape', None)}.")
        return None
    alpha = weight_mask[:, :, 3].astype(np.float32) / 255.0
    if not (alpha > 0.0).any():
        return None
    rgb = weight_mask[:, :, :3]
    # classify by the solidly painted core (a feather ramp dilutes chroma)
    core = weight_mask[:, :, 3] >= 128
    sample = core if core.any() else (weight_mask[:, :, 3] > 0)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    if int(chroma[sample].max()) <= 2:
        # achromatic = the grayscale wire format (legacy rainbow paint is
        # always fully saturated, so the two cannot be confused)
        values = rgb[:, :, 0].astype(np.float32) / 255.0
    else:
        hue = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)[:, :, 0].astype(np.float32) * 2.0
        values = np.clip(1.0 - hue / WEIGHT_MASK_HUE_SPAN, 0.0, 1.0)
    return (values * alpha).astype(np.float32)


class ControlNetCachedParameters:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.control_cond = None
        self.control_cond_for_hr_fix = None
        self.control_mask = None
        self.control_mask_for_hr_fix = None
        self.output_mask = None
        self.output_mask_for_hr_fix = None
        self.weight_mask = None
        self.weight_mask_for_hr_fix = None
        self.weight_mask_bands = {}
        self.weight_mask_bands_for_hr_fix = {}
        # parallel to control_cond: which Input slot each cond came from
        self.slot_order = []


class ControlNetForForgeOfficial(scripts.Script):
    sorting_priority = 10

    def title(self):
        return "CNPro"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        infotext = Infotext()
        ui_groups = []
        controls = []
        max_models = shared.opts.data.get("control_net_unit_count", 3)
        gen_type = "img2img" if is_img2img else "txt2img"
        elem_id_tabname = gen_type + "_controlnet"
        default_unit = ControlNetUnit(
            enabled=False,
            module="None",
            model="None",
            resize_mode=external_code.ResizeMode.OUTER_FIT,  # Resize and Fill
        )
        with gr.Group(elem_id=elem_id_tabname):
            # elem_id/elem_classes stay "controlnet": the editor, painter and
            # tab-strip JS select on them, and so do the ui-config.json keys.
            # Only the visible label is CNPro's.
            with gr.Accordion("ControlNet Pro", open=False, elem_id="controlnet",
                              elem_classes=["controlnet"]):
                photopea = (
                    Photopea()
                    if not shared.opts.data.get("controlnet_disable_photopea_edit", False)
                    else None
                )
                with gr.Row(elem_id=elem_id_tabname + "_accordions", elem_classes="accordions"):
                    for i in range(max_models):
                        with InputAccordion(
                            value=False,
                            label=f"ControlNet Unit {i}",
                            elem_classes=["cnet-unit-enabled-accordion"],  # Class on accordion
                        ):
                            group = ControlNetUiGroup(is_img2img, default_unit, photopea)
                            ui_groups.append(group)
                            controls.append(group.render(f"ControlNet-{i}", elem_id_tabname))

        for i, ui_group in enumerate(ui_groups):
            infotext.register_unit(i, ui_group)
        if shared.opts.data.get("control_net_sync_field_args", True):
            self.infotext_fields = infotext.infotext_fields
            self.paste_field_names = infotext.paste_field_names
        return tuple(controls)

    def get_enabled_units(self, units):
        # Parse dict from API calls.
        units = [
            ControlNetUnit.from_dict(unit) if isinstance(unit, dict) else unit
            for unit in units
        ]
        assert all(isinstance(unit, ControlNetUnit) for unit in units)
        # SNAPSHOT. The UI's field writers mutate the gr.State unit in place
        # (controlnet_ui_group.field_updater - that is what makes concurrent
        # per-field writes stop reverting each other), so the object handed in
        # here keeps changing under a run: touching a slider or painting a mask
        # mid-generation would otherwise be read by whatever step reaches the
        # field next. A shallow copy per unit restores the stable-per-run
        # semantics the old rebuild-the-dataclass writer gave for free. Shallow
        # is the right depth: the image/mask arrays are only ever read here,
        # and copying tens of MB per unit per run is exactly what the State
        # wiring exists to avoid.
        enabled_units = [copy.copy(x) for x in units if x.enabled]
        return enabled_units

    @staticmethod
    def try_crop_image_with_a1111_mask(
            p: StableDiffusionProcessing,
            unit: ControlNetUnit,
            input_image: np.ndarray,
            resize_mode: external_code.ResizeMode,
            preprocessor
    ) -> np.ndarray:
        a1111_mask_image: Optional[Image.Image] = getattr(p, "image_mask", None)
        is_only_masked_inpaint = (
                issubclass(type(p), StableDiffusionProcessingImg2Img) and
                p.inpaint_full_res and
                a1111_mask_image is not None
        )
        if (
                preprocessor.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab
                and is_only_masked_inpaint
        ):
            logger.info("Crop input image based on A1111 mask.")
            input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
            input_image = [Image.fromarray(x) for x in input_image]

            mask = prepare_mask(a1111_mask_image, p)

            crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

            input_image = [
                images.resize_image(resize_mode.int_value(), i, mask.width, mask.height)
                for i in input_image
            ]

            input_image = [x.crop(crop_region) for x in input_image]
            input_image = [
                images.resize_image(external_code.ResizeMode.OUTER_FIT.int_value(), x, p.width, p.height)
                for x in input_image
            ]

            input_image = [np.asarray(x)[:, :, 0] for x in input_image]
            input_image = np.stack(input_image, axis=2)
        return input_image

    def get_input_data(self, p, unit, preprocessor, h, w):
        resize_mode = external_code.resize_mode_from_value(unit.resize_mode)

        a1111_i2i_image = getattr(p, "init_images", [None])[0]
        a1111_i2i_mask = getattr(p, "image_mask", None)

        using_a1111_data = False

        # Every enabled Input tab holding an image. Slots the user never
        # opened - or closed again, which clears the canvas - are None and drop
        # out here, so the backend never has to mirror the UI's tab state.
        # Muted slots (per-tab "use this input" checkbox off) drop out the same
        # way: image and masks stay, contribution goes. Per-input WEIGHTING is
        # the weight masks' job (a flat global mask at value v scales that
        # input), not a separate scalar.
        triples = unit.input_images()
        slots = [(index, image, foreground[:, :, 3] if foreground is not None else None)
                 for index, (image, foreground, enabled) in enumerate(triples)
                 if enabled and image is not None]
        # The tab strip's visual order (move-left button) is the generation
        # order: multi-phase profiles assign input k the phase k*2pi/n, so the
        # leftmost tab is input 1. Everything downstream follows this list -
        # the weight masks travel with their slot_index regardless.
        position = {slot: pos for pos, slot
                    in enumerate(external_code.ControlNetUnit.input_order_permutation(unit.input_order))}
        slots.sort(key=lambda entry: position.get(entry[0], entry[0]))
        muted = sum(1 for image, _, enabled in triples
                    if image is not None and not enabled)
        if muted:
            logger.info(f"ControlNet: {muted} muted input(s) skipped.")

        if unit.use_preview_as_input and unit.generated_image is not None:
            # the preview replaces the image of the slot it was made from, so
            # it must be gated by THAT slot's weight masks (run_annotator
            # previews whichever tab is open, not always the first)
            preview_slot = getattr(unit, 'preview_slot', 0)
            if not isinstance(preview_slot, int) or not 0 <= preview_slot < len(triples):
                preview_slot = 0
            slots = [(preview_slot, unit.generated_image, None)]
        elif not slots:
            if any(image is not None for image, _, _ in triples):
                # every populated input is muted: the unit is deliberately
                # silenced, not empty - falling through to the img2img source
                # would silently re-target it against the user's intent
                logger.info("ControlNet: all inputs of the unit are muted - unit skipped.")
                return [], resize_mode
            if a1111_i2i_image is None:
                # txt2img with an enabled-but-empty unit used to die on a bare
                # uint8 assert inside HWC3; fail with the message instead
                raise ValueError("controlnet is enabled but no input image is given "
                                 "(and there is no img2img source image to fall back to)")
            resize_mode = external_code.resize_mode_from_value(p.resize_mode)
            slots = [(0, HWC3(np.asarray(a1111_i2i_image)), None)]
            using_a1111_data = True

        # the Use-Mask canvas (or the a1111 inpaint mask) is unit level: one
        # mask, applied to every input
        unit_mask_image = unit.mask_image
        unit_mask_image_fg = unit.mask_image_fg[:, :, 3] if unit.mask_image_fg is not None else None

        image_list = []
        for slot_index, unit_image, unit_image_fg in slots:
            if unit_image is not None and unit_image_fg is not None \
                    and (unit_image < 5).all() and (unit_image_fg > 5).any():
                # drawn on an all-black canvas: the scribble is the input
                image = unit_image_fg
            else:
                image = unit_image

            if not isinstance(image, np.ndarray):
                raise ValueError("controlnet is enabled but no input image is given")

            image = HWC3(image)

            if using_a1111_data:
                mask = HWC3(np.asarray(a1111_i2i_mask)) if a1111_i2i_mask is not None else None
            elif unit_mask_image_fg is not None and (unit_mask_image_fg > 5).any():
                mask = unit_mask_image_fg
            elif unit_mask_image is not None and (unit_mask_image > 5).any():
                mask = unit_mask_image
            elif unit_image_fg is not None and (unit_image_fg > 5).any():
                mask = unit_image_fg
            else:
                mask = None

            image = self.try_crop_image_with_a1111_mask(p, unit, image, resize_mode, preprocessor)

            if mask is not None:
                mask = cv2.resize(HWC3(mask), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = self.try_crop_image_with_a1111_mask(p, unit, mask, resize_mode, preprocessor)

            # the slot travels with the input: its weight masks are its own
            image_list.append([image, mask, slot_index])

        if resize_mode == external_code.ResizeMode.OUTER_FIT and preprocessor.expand_mask_when_resize_and_fill:
            new_image_list = []
            for input_image, input_mask, slot_index in image_list:
                if input_mask is None:
                    input_mask = np.zeros_like(input_image)
                input_mask = crop_and_resize_image(
                    input_mask,
                    external_code.ResizeMode.OUTER_FIT, h, w,
                    fill_border_with_255=True,
                )
                input_image = crop_and_resize_image(
                    input_image,
                    external_code.ResizeMode.OUTER_FIT, h, w,
                    fill_border_with_255=False,
                )
                new_image_list.append((input_image, input_mask, slot_index))
            image_list = new_image_list

        return image_list, resize_mode

    @staticmethod
    def get_target_dimensions(p: StableDiffusionProcessing) -> Tuple[int, int, int, int]:
        """Returns (h, w, hr_h, hr_w)."""
        h = align_dim_latent(p.height)
        w = align_dim_latent(p.width)

        high_res_fix = (
                isinstance(p, StableDiffusionProcessingTxt2Img)
                and getattr(p, 'enable_hr', False)
        )
        if high_res_fix:
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                hr_y = int(p.height * p.hr_scale)
                hr_x = int(p.width * p.hr_scale)
            else:
                hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
            hr_y = align_dim_latent(hr_y)
            hr_x = align_dim_latent(hr_x)
        else:
            hr_y = h
            hr_x = w

        return h, w, hr_y, hr_x

    @torch.no_grad()
    def process_unit_after_click_generate(self,
                                          p: StableDiffusionProcessing,
                                          unit: ControlNetUnit,
                                          params: ControlNetCachedParameters,
                                          *args, **kwargs):

        h, w, hr_y, hr_x = self.get_target_dimensions(p)

        has_high_res_fix = (
                isinstance(p, StableDiffusionProcessingTxt2Img)
                and getattr(p, 'enable_hr', False)
        )

        if unit.use_preview_as_input:
            unit.module = 'None'

        preprocessor = global_state.get_preprocessor(unit.module)

        input_list, resize_mode = self.get_input_data(p, unit, preprocessor, h, w)
        if not input_list:
            # all inputs muted: params.model stays None, so
            # process_unit_before_every_sampling skips the unit entirely
            return
        preprocessor_outputs = []
        control_masks = []
        preprocessor_output_is_image = False
        preprocessor_output = None

        decoded_slot_cache = {}

        # WHICH MASK SLOTS ARE LIVE IS THE PROFILE SELECTOR'S DECISION, and it
        # is read once, here, so every consumer below sees the same answer.
        # Main (and depth, which multiplies main) runs on the G mask; a band
        # selection runs on C/M/F. See external_code.masks_in_force.
        band_selected = external_code.band_mode_active(unit.weight_profile)

        def decode_slot_masks(slot_index):
            """(global, {band: values}) of one input slot, decoded.

            Only the masks the ACTIVE PROFILE uses come back - the others are
            dropped here, at the single point where the raw channels are read,
            so no consumer downstream has to remember the rule (or invent a
            second one, which is what it used to do in two places).

            Memoized: the same slot is decoded for the input gate and again
            for the mask tensors - cv2 work on full-res masks, once is enough.
            """
            if slot_index not in decoded_slot_cache:
                raw_global, raw_bands = unit.input_weight_masks(slot_index)
                bands = {band: values for band, values in
                         ((band, decode_weight_mask(raw)) for band, raw in raw_bands.items())
                         if values is not None}
                decoded_slot_cache[slot_index] = external_code.masks_in_force(
                    decode_weight_mask(raw_global), bands, band_selected)
            return decoded_slot_cache[slot_index]

        # Embedding preprocessors (CLIP vision / InsightFace) read the image
        # globally, so the weight-mask knowledge gate must blank weight-0
        # content BEFORE they run: their output is a non-spatial embedding the
        # later cond-level gate cannot act on. Spatial preprocessors are gated
        # at cond level instead (no fake edges along the blanked border).
        # Masks are per input slot, so the gate is computed per input too.
        def input_gate_for(slot_index):
            if not getattr(preprocessor, 'gate_input_by_weight_mask', False):
                return None
            gate_values, bands = decode_slot_masks(slot_index)
            # Whichever of the two the active profile uses (decode_slot_masks
            # has already dropped the other). Bands: restrict-to-painted means
            # an absent band is zero, so the gate is the union of the painted
            # ones - everywhere this unit is allowed to act at all.
            if gate_values is None and bands:
                gate_values = np.maximum.reduce(list(bands.values()))
            return None if gate_values is None else gate_values > 0

        def report_masks_not_in_force():
            """Say when paint exists that the SELECTED profile does not use.

            Dropping it silently is the worst option available: the button
            still carries its painted-mask border, the overlay still shows the
            strokes, and the generation simply ignores them. One line naming
            the mode and the slots is the difference between "the mask does
            nothing" and "the mask does nothing BECAUSE the main profile is
            selected".
            """
            # PRESENCE, not a decode. The channel only carries anything when the
            # painter exported a stroke, so `is not None` answers the question -
            # and this runs on every generation, over every slot the mode is NOT
            # using, where a full-resolution decode would be pure waste.
            ignored = set()
            for slot_index in slot_order:
                raw_global, raw_bands = unit.input_weight_masks(slot_index)
                if band_selected:
                    if raw_global is not None:
                        ignored.add('G')
                else:
                    for band, raw in raw_bands.items():
                        if raw is not None:
                            ignored.add(band[0].upper())
            if ignored:
                logger.warning(
                    f"ControlNet: {', '.join(sorted(ignored))} weight mask(s) are painted "
                    f"but NOT applied - the "
                    f"{'band' if band_selected else 'main'} profile is selected, and the "
                    f"mask slots follow that selector "
                    f"({'C/M/F' if band_selected else 'G'} is what runs). Press the "
                    f"matching profile selector to use them.")

        def optional_tqdm(iterable, use_tqdm):
            from tqdm import tqdm
            return tqdm(iterable) if use_tqdm else iterable

        # the slot each preprocessed input came from, parallel to
        # preprocessor_outputs / params.control_cond
        slot_order = [entry[2] for entry in input_list]
        params.slot_order = slot_order

        for input_image, input_mask, slot_index in optional_tqdm(input_list, len(input_list) > 1):
            input_gate = input_gate_for(slot_index)
            if input_gate is not None and isinstance(input_image, np.ndarray) \
                    and input_image.ndim == 3 and input_image.shape[:2] == input_gate.shape:
                input_image = input_image * input_gate[..., None].astype(input_image.dtype)
                logger.info("ControlNet weight mask gated the preprocessor input (embedding preprocessor).")

            if unit.pixel_perfect:
                unit.processor_res = external_code.pixel_perfect_resolution(
                    input_image,
                    target_H=h,
                    target_W=w,
                    resize_mode=resize_mode,
                )

            seed = set_numpy_seed(p)
            logger.debug(f"Use numpy seed {seed}.")
            logger.info(f"Using preprocessor: {unit.module}")
            logger.info(f'preprocessor resolution = {unit.processor_res}')

            preprocessor_output = cached_preprocessor_call(
                preprocessor, unit.module,
                input_image=input_image,
                input_mask=input_mask,
                resolution=unit.processor_res,
                slider_1=unit.threshold_a,
                slider_2=unit.threshold_b,
            )

            preprocessor_outputs.append(preprocessor_output)

            preprocessor_output_is_image = judge_image_type(preprocessor_output)

            # appended unconditionally, None included: this list is indexed in
            # parallel with the control conds, so skipping the empty ones would
            # shift every later input onto the wrong mask
            control_masks.append(input_mask)

        if len(input_list) > 1:
            logger.info(f'ControlNet unit has {len(input_list)} inputs: each one is preprocessed '
                        f'separately and their controls are summed.')

        if has_high_res_fix:
            hr_option = HiResFixOption.from_value(unit.hr_option)
        else:
            hr_option = HiResFixOption.BOTH

        def attach_extra_result_image(img: np.ndarray, is_high_res: bool = False):
            if (
                (is_high_res and hr_option.high_res_enabled) or
                (not is_high_res and hr_option.low_res_enabled)
            ) and unit.save_detected_map:
                p.extra_result_images.append(img)

        # One control cond per input image, kept as a LIST: these are separate
        # controls to be summed, not frames of one generation batch, so they
        # must not be folded onto the batch axis. Each entry stays at batch 1
        # and the patcher broadcasts it over the latent batch.
        if preprocessor_output_is_image:
            params.control_cond = []
            params.control_cond_for_hr_fix = []

            for preprocessor_output in preprocessor_outputs:
                control_cond = crop_and_resize_image(preprocessor_output, resize_mode, h, w)
                attach_extra_result_image(external_code.visualize_inpaint_mask(control_cond))
                params.control_cond.append(numpy_to_pytorch(control_cond).movedim(-1, 1).contiguous())

            if has_high_res_fix:
                for preprocessor_output in preprocessor_outputs:
                    control_cond_for_hr_fix = crop_and_resize_image(preprocessor_output, resize_mode, hr_y, hr_x)
                    attach_extra_result_image(external_code.visualize_inpaint_mask(control_cond_for_hr_fix), is_high_res=True)
                    params.control_cond_for_hr_fix.append(numpy_to_pytorch(control_cond_for_hr_fix).movedim(-1, 1).contiguous())
            else:
                params.control_cond_for_hr_fix = params.control_cond
        else:
            params.control_cond = list(preprocessor_outputs)
            params.control_cond_for_hr_fix = params.control_cond
            attach_extra_result_image(input_image)

        if any(m is not None for m in control_masks):
            params.control_mask = []
            params.control_mask_for_hr_fix = []

            for input_mask in control_masks:
                if input_mask is None:
                    # keeps the list index-aligned with control_cond
                    params.control_mask.append(None)
                    params.control_mask_for_hr_fix.append(None)
                    continue
                fill_border = preprocessor.fill_mask_with_one_when_resize_and_fill
                control_mask = crop_and_resize_image(input_mask, resize_mode, h, w, fill_border)
                attach_extra_result_image(control_mask)
                control_mask = numpy_to_pytorch(control_mask).movedim(-1, 1)[:, :1]
                params.control_mask.append(control_mask)

                if has_high_res_fix:
                    control_mask_for_hr_fix = crop_and_resize_image(input_mask, resize_mode, hr_y, hr_x, fill_border)
                    attach_extra_result_image(control_mask_for_hr_fix, is_high_res=True)
                    control_mask_for_hr_fix = numpy_to_pytorch(control_mask_for_hr_fix).movedim(-1, 1)[:, :1]
                    params.control_mask_for_hr_fix.append(control_mask_for_hr_fix)

            # parallel to control_cond: one entry per input image
            if not has_high_res_fix:
                params.control_mask_for_hr_fix = params.control_mask

        # Mask previews multiplied the result gallery (global + 3 bands +
        # output, base + hires, every run) while save_detected_map had no
        # reachable off switch - so they are opt-in now; the detected map
        # itself stays.
        mask_previews = shared.opts.data.get("controlnet_mask_preview_in_results", False)

        def prepare_weight_mask(values):
            # Same geometric mapping as the control image, so the paint stays
            # aligned with the source under every resize mode; values travel as
            # grayscale, hue decoding already happened at source resolution.
            gray = HWC3((values * 255.0).round().astype(np.uint8))
            resized = crop_and_resize_image(gray, resize_mode, h, w)
            if mask_previews:
                attach_extra_result_image(resized)
            tensor = numpy_to_pytorch(resized).movedim(-1, 1)[:, :1]
            if has_high_res_fix:
                resized_hr = crop_and_resize_image(gray, resize_mode, hr_y, hr_x)
                tensor_hr = numpy_to_pytorch(resized_hr).movedim(-1, 1)[:, :1]
            else:
                tensor_hr = tensor
            return tensor, tensor_hr

        # One set of masks per input, in the same order as the control conds:
        # every input is its own control, so its masks restrict only its own
        # contribution. Per input, EXACTLY ONE of the two is populated, and the
        # profile selector says which (external_code.masks_in_force, applied in
        # decode_slot_masks): main/depth -> the global mask, a band selection ->
        # the C/M/F masks.
        report_masks_not_in_force()
        params.weight_mask = []
        params.weight_mask_for_hr_fix = []
        params.weight_mask_bands = []
        params.weight_mask_bands_for_hr_fix = []
        for slot_index in slot_order:
            global_values, band_values = decode_slot_masks(slot_index)
            if global_values is not None:
                tensor, tensor_hr = prepare_weight_mask(global_values)
            else:
                tensor = tensor_hr = None
            params.weight_mask.append(tensor)
            params.weight_mask_for_hr_fix.append(tensor_hr)

            bands, bands_hr = {}, {}
            for band, values in band_values.items():
                band_tensor, band_tensor_hr = prepare_weight_mask(values)
                bands[band] = band_tensor
                bands_hr[band] = band_tensor_hr
            params.weight_mask_bands.append(bands)
            params.weight_mask_bands_for_hr_fix.append(bands_hr)

        # Output mask: painted over a throwaway backdrop on its own tab, so it
        # is registered with the GENERATED image, not with the control input -
        # the unit's resize mode (which maps SOURCE geometry onto the output)
        # must not be applied to it. The painted rectangle simply maps onto the
        # output rectangle, hence a plain resize.
        output_mask_values = decode_weight_mask(unit.output_mask)
        if output_mask_values is not None:
            gray = HWC3((output_mask_values * 255.0).round().astype(np.uint8))
            resized = crop_and_resize_image(gray, external_code.ResizeMode.RESIZE, h, w)
            if mask_previews:
                attach_extra_result_image(resized)
            params.output_mask = numpy_to_pytorch(resized).movedim(-1, 1)[:, :1]
            if has_high_res_fix:
                resized_hr = crop_and_resize_image(gray, external_code.ResizeMode.RESIZE, hr_y, hr_x)
                if mask_previews:
                    attach_extra_result_image(resized_hr, is_high_res=True)
                params.output_mask_for_hr_fix = numpy_to_pytorch(resized_hr).movedim(-1, 1)[:, :1]
            else:
                params.output_mask_for_hr_fix = params.output_mask
        else:
            params.output_mask = None
            params.output_mask_for_hr_fix = None

        if preprocessor.do_not_need_model:
            model_filename = 'Not Needed'
            params.model = ControlModelPatcher()
        else:
            assert unit.model != 'None', 'You have not selected any control model!'
            model_filename = global_state.get_controlnet_filename(unit.model)
            # load_control_model RAISES with a specific message now (which
            # loaders were tried, and whether one of them broke rather than
            # declined). The old `assert ... is not None` collapsed every cause
            # into "Recognizing Control Model failed", which is how a typo in the
            # ControlNet loader passed for "unsupported file". Kept as a
            # belt-and-braces check for a loader that returns None without
            # raising - that would be a contract violation, so say so.
            params.model = cached_controlnet_loader(model_filename)
            assert params.model is not None, (
                f"CNPro: the control-model registry returned None for {model_filename} "
                f"instead of raising. That is a registry contract violation - see "
                f"cnpro_host/registry.py::load_control_model.")

        params.preprocessor = preprocessor

        params.preprocessor.process_after_running_preprocessors(process=p, params=params, **kwargs)
        params.model.process_after_running_preprocessors(process=p, params=params, **kwargs)

        logger.info(f"Current ControlNet {type(params.model).__name__}: {model_filename}")
        return

    @torch.no_grad()
    def process_unit_before_every_sampling(self,
                                           p: StableDiffusionProcessing,
                                           unit: ControlNetUnit,
                                           params: ControlNetCachedParameters,
                                           *args, **kwargs):

        if params.model is None or params.preprocessor is None:
            # unit went inert at click time (e.g. all inputs muted)
            return

        # Patcher instances are cached (and shared by two units on the same
        # model file), so every per-run attribute starts from its neutral
        # default here instead of relying on each assignment below to be
        # unconditional - one place enforces "never stale" for all of them.
        params.model.reset_run_state()

        is_hr_pass = getattr(p, 'is_hr_pass', False)

        has_high_res_fix = (
                isinstance(p, StableDiffusionProcessingTxt2Img)
                and getattr(p, 'enable_hr', False)
        )

        if has_high_res_fix:
            hr_option = HiResFixOption.from_value(unit.hr_option)
        else:
            hr_option = HiResFixOption.BOTH

        if has_high_res_fix and is_hr_pass and (not hr_option.high_res_enabled):
            logger.info(f"ControlNet Skipped High-res pass.")
            return

        if has_high_res_fix and (not is_hr_pass) and (not hr_option.low_res_enabled):
            logger.info(f"ControlNet Skipped Low-res pass.")
            return

        if is_hr_pass:
            conds = params.control_cond_for_hr_fix
            masks = params.control_mask_for_hr_fix
        else:
            conds = params.control_cond
            masks = params.control_mask

        # One entry per Input tab that held an image. The mask list is unit
        # level in practice (one Use-Mask canvas), so it is padded rather than
        # required to be the same length.
        if not isinstance(conds, list):
            conds = [conds]
        if not isinstance(masks, list):
            masks = [masks] * len(conds)
        masks = (masks + [masks[-1] if masks else None] * len(conds))[:len(conds)]

        weight_profile = external_code.parse_weight_profile(unit.weight_profile)

        # Per-band (coarse/mid/fine) step profiles: per-step strength of that
        # band's injection layers, packed into the same profile string
        # ('#C/#M/#F'). Main and bands are EXCLUSIVE and the editor's band
        # SELECTOR is the switch (marker '#B<band>', band_mode_active): the
        # profile the user has selected - and is looking at on the plot - is
        # the one that runs. In band mode the unit strength is a neutral 1 and
        # the bands alone shape it (a band left flat at 1 does nothing to its
        # layers); in main mode the band curves are kept in the string but not
        # forwarded. Band values are NOT scaled by the multi-input share below
        # - that stays on the unit-level strength.
        band_mode = external_code.band_mode_active(unit.weight_profile)
        band_profiles = external_code.parse_band_profiles(unit.weight_profile) if band_mode else None
        per_depth_capable = getattr(params.model, 'supports_band_profiles', False)
        if band_profiles and not per_depth_capable:
            # Fall back to the MAIN profile rather than to nothing: band mode
            # otherwise left the unit with no profile at all and a constant
            # strength 1.0 - stronger than anything the user drew, and neither
            # what the plot shows nor a neutral no-op.
            logger.warning(f"Band weight profiles ignored: {type(params.model).__name__} "
                           f"injects through a single whole-UNet hook, so there are no "
                           f"per-depth sites to scale - falling back to the main profile.")
            band_profiles = None
            band_mode = False
        params.model.band_weight_profiles = band_profiles

        # Depth profile: a step-invariant per-layer multiplier that MULTIPLIES
        # the main profile (unlike the bands, which replace it). Bands and depth
        # are exclusive by editor contract, so in band mode the '#D' segment
        # rides along unread.
        depth_profile = None if band_mode else external_code.parse_depth_profile(unit.weight_profile)
        if depth_profile and not per_depth_capable:
            logger.warning(f"Depth profile ignored: {type(params.model).__name__} "
                           f"injects through a single whole-UNet hook, so there are no "
                           f"per-depth sites to scale.")
            depth_profile = None
        params.model.depth_profile = depth_profile
        if depth_profile:
            logger.info(f"ControlNet depth profile on {type(params.model).__name__} "
                        f"(per-layer multiplier on the per-step strength)")

        if band_mode:
            active = ', '.join(band_profiles or {}) or 'all neutral'
            logger.info(f"ControlNet band profiles ({active}) "
                        f"on {type(params.model).__name__} (band mode: the main "
                        f"profile is not applied)")
            weight_profile = None
            params.model.strength = 1.0
            params.model.start_percent = 0.0
            params.model.end_percent = 1.0
        elif weight_profile is not None:
            # The profile drives per-step strength directly. Derived scalars
            # keep patchers that only understand constant weight + timestep
            # range (e.g. IP-Adapter, ControlLLLite) behaving sensibly; the
            # derived strength is clamped at 0 because those patchers cannot
            # express negative (repulsive) control.
            guidance_start, guidance_end = external_code.weight_profile_support(weight_profile)
            params.model.strength = float(max(max(y for _, y in weight_profile), 0.0))
            params.model.start_percent = float(guidance_start)
            params.model.end_percent = float(guidance_end)
        else:
            params.model.strength = float(unit.weight)
            params.model.start_percent = float(unit.guidance_start)
            params.model.end_percent = float(unit.guidance_end)
            logger.info(f"ControlNet constant weight {params.model.strength} "
                        f"(no weight profile, raw value {unit.weight_profile!r}) on {type(params.model).__name__}")
        params.model.weight_profile = weight_profile

        # Mean, not sum. The patcher chain adds the N residuals with no
        # normalization of its own (control_merge / the IP-Adapter attention
        # patch both just accumulate), so N inputs at full weight would inject
        # N times the pull of one. Each input is scaled by 1/N up front instead,
        # which keeps the unit weight meaning "how hard this unit pulls"
        # regardless of how many Input tabs are open - the summing is an
        # implementation detail of combining them, not a strength multiplier.
        # UNEVEN per-input weighting is the weight masks' job: a flat-painted
        # global mask at value v scales exactly that input's contribution.
        if len(conds) > 1:
            share = 1.0 / len(conds)
            params.model.strength *= share
            if weight_profile is not None:
                params.model.weight_profile = [(x, y * share) for x, y in weight_profile]
            logger.info(f"ControlNet {len(conds)} inputs: each control scaled by 1/{len(conds)} "
                        f"so their sum matches a single input at the unit weight.")

        # Multi-phase cosine ('P' marker from the editor's multi preset): every
        # Input runs the SAME envelope with the cosine shifted by 2*pi/n
        # relative to the previous input, so the inputs take turns steering
        # across the steps (oscillatory amalgamation). The variants are scaled
        # by 2/n, NOT the plain 1/n share: the shifted waves sum to the
        # constant n/2, so the extra 2 cancels the wave's 0.5 mean and the
        # unit's SUMMED per-step pull equals the drawn envelope itself. With
        # 1/n it was silently half of the non-multi-phase unit - observed as
        # "half the steps missed" on an IP-Adapter face unit with 5 inputs.
        # Unit weight keeps meaning "how hard this unit pulls", the same
        # principle as the 1/N share above.
        multiphase_profiles = None
        if (weight_profile is not None and len(conds) > 1
                and external_code.weight_profile_is_multiphase(unit.weight_profile)):
            share = 2.0 / len(conds)
            multiphase_profiles = [
                [(x, y * share) for x, y in external_code.parse_weight_profile(
                    unit.weight_profile, phase_offset=index * 2.0 * math.pi / len(conds))]
                for index in range(len(conds))
            ]
            # the unit-level range gate was derived from input 1's wave
            # support; the sibling phases are active elsewhere in the range
            # and every patcher gates per step through its profile lookup
            # anyway, so the coarse gate must not chop them
            params.model.start_percent = 0.0
            params.model.end_percent = 1.0
            logger.info(f"ControlNet multi-phase profile: {len(conds)} inputs, the cosine "
                        f"phase-shifted by 2pi/{len(conds)} per input (summed pull = envelope).")

        params.model.cond_layer_weights = None
        params.model.uncond_layer_weights = None
        params.model.frame_weights = None
        params.model.sigma_weight_fn = None

        soft_weighting = {
            'input': [0.09941396206337118, 0.12050177219802567, 0.14606275417942507, 0.17704576264172736,
                      0.214600924414215,
                      0.26012233262329093, 0.3152997971191405, 0.3821815722656249, 0.4632503906249999, 0.561515625,
                      0.6806249999999999, 0.825],
            'middle': [0.561515625] if p.sd_model.is_sdxl else [1.0],
            'output': [0.09941396206337118, 0.12050177219802567, 0.14606275417942507, 0.17704576264172736,
                       0.214600924414215,
                       0.26012233262329093, 0.3152997971191405, 0.3821815722656249, 0.4632503906249999, 0.561515625,
                       0.6806249999999999, 0.825]
        }

        # The legacy Control Mode chooser (Balanced / Prompt / ControlNet) is
        # replaced by the per-step balance profile: y = 0.5 keeps control on
        # cond and uncond equally (balanced), y -> 1 removes it from uncond
        # (control matters most), y -> 0 removes it from cond (prompt matters
        # most). A flat 0.5 profile is a no-op and is not forwarded.
        balance_profile = external_code.parse_weight_profile(unit.balance_profile)
        if not external_code.balance_points_are_neutral(balance_profile):
            params.model.balance_profile = balance_profile
            if getattr(params.model, 'supports_balance_profile', False):
                logger.info(f"ControlNet balance profile on {type(params.model).__name__}")
            else:
                logger.warning(f"Balance profile ignored: {type(params.model).__name__} "
                               f"does not implement per-step cond/uncond balance.")
        else:
            params.model.balance_profile = None

        # Per-unit prompts: encoded once with the model's own text encoder
        # (attention syntax and embeddings work as in the main prompt, since it
        # is the same pipeline) and handed to the control branch as its
        # cross-attention context - the positive on the cond rows, the negative
        # ("push this control's semantics away from X") on the uncond rows.
        # Cached on params so the hires pass reuses the tensors instead of
        # re-encoding; the transport attrs are set unconditionally because
        # params.model instances are cached across runs and must never carry a
        # stale prompt.
        unit_prompt = (getattr(unit, 'unit_prompt', '') or '').strip()
        unit_negative_prompt = (getattr(unit, 'unit_negative_prompt', '') or '').strip()
        params.model.unit_prompt_cond = None
        params.model.unit_negative_prompt_cond = None
        # strength pair per side (see external_code dataclass comments); set
        # unconditionally for the same reason the conds are - patcher
        # instances are cached across runs
        params.model.unit_prompt_emb_strength = float(getattr(unit, 'unit_prompt_emb_strength', 1.0))
        params.model.unit_prompt_delta_scale = float(getattr(unit, 'unit_prompt_delta_scale', 1.0))
        params.model.unit_negative_prompt_emb_strength = float(getattr(unit, 'unit_negative_prompt_emb_strength', 1.0))
        params.model.unit_negative_prompt_delta_scale = float(getattr(unit, 'unit_negative_prompt_delta_scale', 1.0))
        params.model.unit_prompt_retention = float(getattr(unit, 'unit_prompt_retention', 0.0))
        if unit_prompt or unit_negative_prompt:
            if getattr(params.model, 'supports_unit_prompt', False):
                def encode_unit_prompt(text):
                    if not text:
                        return None
                    encoded = p.sd_model.get_learned_conditioning([text])
                    # XL/flux engines return {'crossattn', 'vector'}, sd1x a
                    # bare tensor; only the crossattn part has a control-side
                    # counterpart (pooled conds keep coming from the main
                    # prompt).
                    if isinstance(encoded, dict):
                        encoded = encoded['crossattn']
                    return encoded

                cache = getattr(params, 'unit_prompt_cond_cache', None)
                cache_key = (unit_prompt, unit_negative_prompt)
                if cache is None or cache[0] != cache_key:
                    # Encoding must not disturb the sampler's randomness. It
                    # runs between set_numpy_seed() and the sampling loop, and
                    # whether it runs at all depends on this cache - so any RNG
                    # the text encoder touches made the SAME seed and settings
                    # produce a different image depending on what the previous
                    # run did (measured: with a unit prompt set, two identical
                    # runs diverged, and clearing the prompt again did not
                    # restore the original result). Snapshot and restore.
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    numpy_rng = np.random.get_state()
                    try:
                        cache = (cache_key,
                                 encode_unit_prompt(unit_prompt),
                                 encode_unit_prompt(unit_negative_prompt))
                    finally:
                        torch.set_rng_state(cpu_rng)
                        if cuda_rng is not None:
                            torch.cuda.set_rng_state_all(cuda_rng)
                        np.random.set_state(numpy_rng)
                    params.unit_prompt_cond_cache = cache
                params.model.unit_prompt_cond = cache[1]
                params.model.unit_negative_prompt_cond = cache[2]
                def strengths_note(emb, delta):
                    parts = []
                    if abs(emb - 1.0) > 1e-4:
                        parts.append(f"emb {emb:g}")
                    if abs(delta - 1.0) > 1e-4:
                        parts.append(f"delta {delta:g}")
                    return f" [{', '.join(parts)}]" if parts else ""

                retention = params.model.unit_prompt_retention
                described = ' / '.join(filter(None, [
                    (f"prompt {unit_prompt!r}"
                     + strengths_note(params.model.unit_prompt_emb_strength,
                                      params.model.unit_prompt_delta_scale)) if unit_prompt else None,
                    (f"negative {unit_negative_prompt!r}"
                     + strengths_note(params.model.unit_negative_prompt_emb_strength,
                                      params.model.unit_negative_prompt_delta_scale)) if unit_negative_prompt else None,
                    f"retention {retention:g}" if abs(retention) > 1e-4 else None]))
                logger.info(f"ControlNet unit {described} drives the control branch "
                            f"(the main prompts still drive the unet).")
                # delta scale != 0/1 doubles the control-model cost per step
                # (a base pass with the sampled context isolates the text's
                # exact effect); retention > 0 ramps the scale off 1 on every
                # step after the first, so it does the same - say so, or the
                # slowdown looks like a bug
                double_pass = any(
                    abs(d) > 1e-4 and (abs(d - 1.0) > 1e-4 or abs(retention) > 1e-4)
                    for d, text in ((params.model.unit_prompt_delta_scale, unit_prompt),
                                    (params.model.unit_negative_prompt_delta_scale, unit_negative_prompt))
                    if text)
                if double_pass:
                    logger.info("ControlNet unit prompt effect scale is not 0/1: "
                                "the control model runs twice per step to isolate the text's effect.")
            else:
                logger.warning(f"ControlNet unit prompt ignored: {type(params.model).__name__} "
                               f"has no text cross-attention (only true ControlNet models consume a prompt).")

        if is_hr_pass and params.preprocessor.use_soft_projection_in_hr_fix:
            params.model.cond_layer_weights = soft_weighting.copy()
            params.model.uncond_layer_weights = soft_weighting.copy()

        # per-input lists, parallel to conds
        weight_masks = params.weight_mask_for_hr_fix if is_hr_pass else params.weight_mask
        band_mask_sets = params.weight_mask_bands_for_hr_fix if is_hr_pass else params.weight_mask_bands
        output_mask = params.output_mask_for_hr_fix if is_hr_pass else params.output_mask

        def per_input(values, index):
            if not isinstance(values, list):
                return values
            return values[index] if index < len(values) else None

        def fold_masks(base_mask, extra_mask):
            # multiply extra into base, resized to base's geometry; a missing
            # base simply adopts extra
            if not isinstance(base_mask, torch.Tensor):
                return extra_mask
            resized = torch.nn.functional.interpolate(
                extra_mask.to(base_mask), size=base_mask.shape[2:], mode='bilinear')
            return base_mask * resized

        def fold_output_mask(mask_tensor):
            return fold_masks(mask_tensor, output_mask)

        # Each input image is patched in as its own control on the same unet.
        # The patcher chains them (set_previous_controlnet) and control_merge
        # sums their residuals, so N inputs act in parallel and arrive as a
        # single contribution from this unit. Everything configured above is
        # unit level and shared by every pass; only the cond and its mask differ.
        for index, (cond, mask) in enumerate(zip(conds, masks)):
            first = index == 0
            weight_mask = per_input(weight_masks, index)
            band_masks = per_input(band_mask_sets, index) or {}

            # multi-phase: this input's phase-shifted variant of the profile
            # (each pass reads params.model.weight_profile at patch time)
            if multiphase_profiles is not None:
                params.model.weight_profile = multiphase_profiles[index]

            kwargs.update(dict(
                unit=unit,
                params=params,
                cond_original=cond.clone() if isinstance(cond, torch.Tensor) else cond,
                mask_original=mask.clone() if isinstance(mask, torch.Tensor) else mask,
            ))

            cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask, *args, **kwargs)

            params.model.region_masks = mask
            current_unit_mask = params.model.region_masks

            def combine_with_unit_mask(mask_tensor):
                # Fold the unit's own Use-Mask (if any) into the painted mask.
                if isinstance(current_unit_mask, torch.Tensor):
                    resized = torch.nn.functional.interpolate(
                        mask_tensor.to(current_unit_mask), size=current_unit_mask.shape[2:], mode='bilinear')
                    return current_unit_mask * resized
                return mask_tensor

            def apply_knowledge_gate(gate_mask):
                # The injection mask alone is not knowledge-proof: the control
                # encoder has an image-wide receptive field (deep conv stacks and
                # attention blocks), so hint content under weight-0 paint would
                # still leak into the residuals injected at non-zero regions.
                # Blank the hint itself wherever the weight is exactly zero, so
                # the control model never sees that content at all. In-between
                # weights keep the full hint and only scale the injection.
                if isinstance(cond, torch.Tensor) and cond.dim() == 4 and cond.shape[-2:] == gate_mask.shape[-2:]:
                    return cond * (gate_mask > 0).to(cond)
                return cond

            # Restrict-to-painted semantics. WHICH slots are live was decided
            # once, by the profile selector, before the masks were ever decoded
            # (external_code.masks_in_force): main/depth -> G, a band selection
            # -> C/M/F. So at most one of these two is populated, and the
            # branch below is not a precedence rule any more - it is just the
            # two shapes a live mask set can have:
            # 1. nothing painted in the live slots -> full image, ordinary control;
            # 2. the global mask        -> it governs ALL layers;
            # 3. the band masks         -> each painted band weights its own
            #                              layers, every ABSENT band counts as
            #                              ZERO (its layers inject nothing).
            # In cases 2 and 3 control never escapes the painted regions.
            # masks are per input, so they are reported per input too
            which = f" (input {index + 1})" if len(conds) > 1 else ""

            # everywhere this input's painted masks allow control at all; also
            # routed into the attention mask of patchers that restrict via the
            # mask ARGUMENT (below)
            painted_restrict = None
            # per-band variant of the same, for patchers whose injection sites
            # each carry a UNet depth (IP-Adapter): they take one mask PER BAND
            # instead of the union, so painting coarse differently from fine
            # restricts different depths - what the band buttons already say
            band_restrict = None

            if weight_mask is not None:
                params.model.region_masks = combine_with_unit_mask(weight_mask)
                cond = apply_knowledge_gate(weight_mask)
                painted_restrict = weight_mask
                logger.info(f"ControlNet weight mask{which} applied on {type(params.model).__name__}")
            elif band_masks:
                # union of the painted bands = everywhere any control acts at all
                # (absent bands are zero, so they cannot widen it)
                union = None
                for tensor in band_masks.values():
                    union = tensor if union is None else torch.maximum(union, tensor.to(union))
                banded = {band: combine_with_unit_mask(tensor)
                          for band, tensor in band_masks.items()}
                if getattr(params.model, 'masks_via_advanced_weighting', False):
                    # Per-band masks: coarse gates the deepest injection layers
                    # (+ middle), mid the middle band, fine the shallowest; layers
                    # of bands WITHOUT a mask are zeroed per layer in
                    # compute_controlnet_weighting.
                    #
                    # Routed by CAPABILITY, not by class. This used to read
                    # `isinstance(params.model, ControlNetPatcher)`, which meant a
                    # new residual patcher (Z-Image) silently fell into the
                    # attention branch below and got a union mask where it should
                    # have had per-band ones - the exact failure the no-model-type-
                    # branching invariant exists to prevent.
                    params.model.region_masks = banded
                else:
                    # Patchers without per-layer injection get the union here;
                    # those whose ATTENTION sites carry a depth take the
                    # per-band dict through the mask argument instead (below).
                    params.model.region_masks = combine_with_unit_mask(union)
                    if getattr(params.model, 'supports_band_profiles', False):
                        band_restrict = banded
                # content must stay unknown to the model outside every painted
                # region, so the hint is blanked beyond the union
                cond = apply_knowledge_gate(union)
                painted_restrict = union
                logger.info(f"ControlNet layer weight masks ({', '.join(sorted(band_masks))}){which} "
                            f"applied on {type(params.model).__name__}, absent bands zeroed")

            # Painted weight masks for patchers that restrict spatially via the
            # mask ARGUMENT rather than region_masks (IP-Adapter
            # takes it as attn_mask): fold the paint into it, so the painted
            # masks shape those patchers' OUTPUT region too instead of only
            # gating their input. Residual patchers ignore the argument (their
            # advanced path already carries the paint), so nothing is applied
            # twice; patchers with neither route (ControlLLLite) keep their
            # existing unsupported warning via the output-mask path.
            if painted_restrict is not None \
                    and not getattr(params.model, 'masks_via_advanced_weighting', False) \
                    and getattr(params.model, 'supports_output_mask', False):
                if band_restrict:
                    # one attn_mask per band: each patched site takes the mask
                    # of its own depth, and a band with no mask means its sites
                    # inject nothing (restrict-to-painted, per band)
                    mask = {band: fold_masks(mask, tensor)
                            for band, tensor in band_restrict.items()}
                    if first:
                        logger.info(f"ControlNet per-band weight masks routed into the "
                                    f"attention masks of {type(params.model).__name__} "
                                    f"({', '.join(sorted(band_restrict))}; absent bands inject nothing).")
                else:
                    mask = fold_masks(mask, painted_restrict)
                    if first:
                        logger.info(f"ControlNet weight mask routed into the attention mask "
                                    f"of {type(params.model).__name__}.")

            # Output mask (its own tab): output-side only, so it never feeds the
            # knowledge gate - the control model keeps seeing the whole hint and
            # only the injection is restricted. It multiplies into whatever
            # injection mask the unit already carries; nothing painted leaves the
            # unit untouched, i.e. control applies to the whole output.
            if output_mask is not None:
                current = params.model.region_masks
                if isinstance(current, dict):
                    params.model.region_masks = {
                        band: fold_output_mask(tensor) for band, tensor in current.items()
                    }
                else:
                    params.model.region_masks = fold_output_mask(current)
                # Patchers that read the mask ARGUMENT rather than the transport
                # attribute (IP-Adapter takes it as attn_mask) are served here;
                # residual patchers ignore this argument, so nothing is applied
                # twice. Folded after the preprocessor call above on purpose - the
                # inpaint preprocessors read `mask` as their hole definition.
                # The argument may now be a per-band dict (see above), and the
                # output mask is output-side: it applies to every band alike.
                if isinstance(mask, dict):
                    mask = {band: fold_output_mask(tensor) for band, tensor in mask.items()}
                else:
                    mask = fold_output_mask(mask)
                if first:
                    if getattr(params.model, 'supports_output_mask', False):
                        logger.info(f"ControlNet output mask applied on {type(params.model).__name__}")
                    else:
                        logger.warning(f"Output mask ignored: {type(params.model).__name__} "
                                       f"does not restrict its injection spatially.")

            params.model.process_before_every_sampling(p, cond, mask, *args, **kwargs)

        logger.info(f"ControlNet Method {params.preprocessor.name} patched"
                    f"{f' ({len(conds)} inputs summed)' if len(conds) > 1 else ''}.")
        return

    @staticmethod
    def bound_check_params(unit: ControlNetUnit) -> None:
        """
        Checks and corrects negative parameters in ControlNetUnit 'unit'.
        Parameters 'processor_res', 'threshold_a', 'threshold_b' are reset to
        their default values if negative.

        Args:
            unit (ControlNetUnit): The ControlNetUnit instance to check.
        """
        preprocessor = global_state.get_preprocessor(unit.module)

        if unit.processor_res < 0:
            # CNPro default (1024), not the preprocessor's own slider value: the
            # host's Preprocessor base declares 512, which predates every model
            # family CNPro now targets. Clamped to the preprocessor's declared
            # range by global_state.default_processor_res.
            unit.processor_res = global_state.default_processor_res(preprocessor)

        if unit.threshold_a < 0:
            unit.threshold_a = int(preprocessor.slider_1.gradio_update_kwargs.get('value', 1.0))

        if unit.threshold_b < 0:
            unit.threshold_b = int(preprocessor.slider_2.gradio_update_kwargs.get('value', 1.0))

        return

    @torch.no_grad()
    def process_unit_after_every_sampling(self,
                                          p: StableDiffusionProcessing,
                                          unit: ControlNetUnit,
                                          params: ControlNetCachedParameters,
                                          *args, **kwargs):

        if params.model is None or params.preprocessor is None:
            # unit went inert at click time (e.g. all inputs muted)
            return
        params.preprocessor.process_after_every_sampling(p, params, *args, **kwargs)
        params.model.process_after_every_sampling(p, params, *args, **kwargs)
        return

    @torch.no_grad()
    def process(self, p, *args, **kwargs):
        self.current_params = {}
        enabled_units = self.get_enabled_units(args)
        Infotext.write_infotext(enabled_units, p)
        for i, unit in enumerate(enabled_units):
            self.bound_check_params(unit)
            params = ControlNetCachedParameters()
            self.process_unit_after_click_generate(p, unit, params, *args, **kwargs)
            self.current_params[i] = params
        return

    @torch.no_grad()
    def process_before_every_sampling(self, p, *args, **kwargs):
        for i, unit in enumerate(self.get_enabled_units(args)):
            self.process_unit_before_every_sampling(p, unit, self.current_params[i], *args, **kwargs)
        return

    @torch.no_grad()
    def postprocess_batch_list(self, p, pp, *args, **kwargs):
        for i, unit in enumerate(self.get_enabled_units(args)):
            self.process_unit_after_every_sampling(p, unit, self.current_params[i], pp, *args, **kwargs)
        return

    def postprocess(self, p, processed, *args):
        self.current_params = {}
        return


def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_detectedmap_dir", shared.OptionInfo(
        "detected_maps", "Directory for detected maps auto saving", section=section))
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))
    shared.opts.add_option("control_net_modules_path", shared.OptionInfo(
        "",
        "Path to directory containing annotator model directories (requires restart, overrides corresponding command line flag)",
        section=section))
    shared.opts.add_option("control_net_unit_count", shared.OptionInfo(
        3, "Multi-ControlNet: ControlNet unit number (requires restart)", gr.Slider,
        {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_model_cache_size", shared.OptionInfo(
        5, "Model cache size (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_detectmap_autosaving", shared.OptionInfo(
        False, "Allow detectmap auto saving", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_sync_field_args", shared.OptionInfo(
        True, "Paste ControlNet parameters in infotext", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_mask_preview_in_results", shared.OptionInfo(
        False, "Append weight/output mask previews to generation results", gr.Checkbox,
        {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_openpose_edit", shared.OptionInfo(
        False, "Disable openpose edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_photopea_edit", shared.OptionInfo(
        False, "Disable photopea edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_photopea_warning", shared.OptionInfo(
        True, "Photopea popup warning", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_input_thumbnail", shared.OptionInfo(
        True, "Input image thumbnail on unit header", gr.Checkbox, {"interactive": True}, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)
script_callbacks.on_after_component(ControlNetUiGroup.on_after_component)
script_callbacks.on_before_reload(ControlNetUiGroup.reset)
# Routes added from on_app_started land behind gradio's catch-all and are
# silently shadowed (see cnpro_host.optional.serving_routes). CNPro's own
# API needs the same promotion as the optional features.
def _cnpro_api(demo, app):
    from cnpro_host.optional import serving_routes
    serving_routes(app, lambda a: controlnet_api(demo, a), 'cnpro api')


script_callbacks.on_app_started(_cnpro_api)

# Optional, environment-dependent extras (see cnpro_host/optional/__init__.py).
# Registered LAST and fully guarded: absence is the expected case and must cost
# nothing. The route hookup has to live in a script rather than at package
# import time - load_scripts() calls clear_callbacks(), which would wipe an
# earlier on_app_started registration.
from cnpro_host import optional as _cnpro_optional

script_callbacks.on_app_started(_cnpro_optional.register_all)
script_callbacks.on_ui_settings(lambda: _cnpro_optional.register_settings(shared))
