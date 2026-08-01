"""Forge-family host implementation of the CNPro adapter contract.

Covers ForgeNeo (Forge Classic `neo`) and, by construction, lllyasviel Forge --
the two differ by ~150 lines in the modules this file touches, none of them in
the names used here.

Everything in this file is a thin lookup. If a function here grows logic, that
logic probably belongs in cnpro_core (host-agnostic) or in lib_cnpro (UI), not
in the adapter: the adapter's job is to answer "where does the host keep X",
never "what should CNPro do with X".
"""

from __future__ import annotations


def canvas_widget():
    from modules_forge.forge_canvas.canvas import ForgeCanvas, LogicalImage
    return ForgeCanvas, LogicalImage


def image_utils():
    from modules_forge.utils import HWC3, numpy_to_pytorch
    return HWC3, numpy_to_pytorch


def model_dir() -> str:
    from modules_forge.shared import controlnet_dir
    return controlnet_dir


def preprocessors() -> dict:
    # Populated by the host's own builtin preprocessor extensions
    # (forge_legacy_preprocessors, forge_preprocessor_inpaint/reference/tile).
    # CNPro READS this registry and never mutates it -- that is what keeps the
    # host's own ControlNet working unchanged alongside CNPro.
    from modules_forge.shared import supported_preprocessors
    return supported_preprocessors


def patcher_base():
    from modules_forge.supported_controlnet import ControlModelPatcher
    return ControlModelPatcher


def load_control_model(ckpt_path):
    # CNPro's OWN registry, not the host's `try_load_supported_control_model`.
    # This is the single most important structural choice in the extension: it
    # means CNPro never inserts itself into `modules_forge.shared.
    # supported_control_models`, so a user running both CNPro and the host's
    # builtin ControlNet gets each behaving exactly as designed.
    from .registry import load_control_model as _load
    return _load(ckpt_path)


def gates_input_by_mask(preprocessor) -> bool:
    # Resolved by TYPE rather than by a flag written onto the host's class.
    # PreprocessorClipVision covers IP-Adapter/Revision; InstantID's insightface
    # preprocessor is not a subclass of it, so it is matched by name.
    try:
        from modules_forge.supported_preprocessor import PreprocessorClipVision
        if isinstance(preprocessor, PreprocessorClipVision):
            return True
    except Exception:
        pass
    name = (getattr(preprocessor, "name", "") or "").lower()
    return "instant" in name or "insightface" in name


def sampling_steps(process) -> int | None:
    """Model forwards the CURRENT pass will run - the LLLite step axis.

    NOT the UI step count: an img2img pass at denoising strength d runs only
    the tail of the schedule, and the hires pass runs its own count. LLLite
    counts forwards (its modules never see a sigma), so handing it p.steps made
    the counter traverse only the first fraction of every drawn curve on
    partial passes - an end-weighted profile silently never engaged on img2img.
    The steps->forwards mapping is the HOST'S OWN (setup_img2img_steps, with
    the same `steps` argument the host's sampler call passes), never
    replicated, so it cannot drift.
    """
    steps = getattr(process, "steps", None)
    if not steps:
        return steps
    denoise = getattr(process, "denoising_strength", None)
    is_hr = bool(getattr(process, "is_hr_pass", False))
    has_init = bool(getattr(process, "init_images", None))
    if denoise is None or not (is_hr or has_init):
        return steps
    explicit = (getattr(process, "hr_second_pass_steps", 0) or steps) if is_hr else None
    try:
        from modules.sd_samplers_common import setup_img2img_steps
        _, t_enc = setup_img2img_steps(process, explicit)
        return t_enc + 1
    except Exception:
        return int(explicit or steps)


# --- model family ----------------------------------------------------------
# Resolved from the ESTIMATED CONFIG the loader already matched, not from the
# engine class name and not from a file name. `huggingface_guess` sniffed the
# checkpoint's own tensors to pick that class, so this answer is as reliable as
# the host's own model loading; a renamed or merged checkpoint cannot fool it.
#
# The values are CNPro's vocabulary, not the host's. They exist so a patcher can
# say "I inject into a Z-Image DiT" and be refused politely on an SDXL UNet
# instead of being handed residuals of the wrong rank at sampling time. Keep them
# stable: they appear in log lines and in patcher `families` sets.
#
# Ordered most-specific first -- ZImage subclasses Lumina2 and Chroma subclasses
# FluxSchnell subclasses Flux, so an isinstance/issubclass walk up the MRO would
# answer "flux" for a Chroma model. Exact class identity is what is wanted here.
_FAMILY_BY_GUESS = {
    "SD15": "sd15",
    "SDXL": "sdxl",
    "SDXLRefiner": "sdxl",
    "Mugen": "sdxl",
    "Flux": "flux",
    "FluxSchnell": "flux",
    "Flux2K4B": "flux2",
    "Flux2K9B": "flux2",
    "Chroma": "chroma",
    "Lumina2": "lumina2",
    "ZImage": "zimage",
    "Anima": "anima",
    "WAN21_T2V": "wan",
    "WAN21_I2V": "wan",
    "QwenImage": "qwen",
    "Krea2": "krea2",
    "ErnieImage": "ernie",
    "PiD": "pid",
}


def model_family(process=None) -> str:
    """CNPro's name for the model architecture currently loaded, or 'unknown'.

    `process` is the StableDiffusionProcessing when one is in flight; without it
    the currently loaded checkpoint is used. Never raises -- an unrecognised or
    absent model answers 'unknown', and callers treat that as "do not gate",
    because refusing to run on a model CNPro merely failed to NAME would be worse
    than running on it.
    """
    sd_model = getattr(process, "sd_model", None)
    if sd_model is None:
        try:
            from modules import shared
            sd_model = shared.sd_model
        except Exception:
            return "unknown"
    if sd_model is None:
        return "unknown"

    guess = getattr(getattr(sd_model, "forge_objects", None), "unet", None)
    guess = getattr(getattr(guess, "model", None), "config", None)
    name = type(guess).__name__ if guess is not None else None
    family = _FAMILY_BY_GUESS.get(name)
    if family is not None:
        return family

    # Fallback for engines whose config class CNPro has not been taught yet: the
    # legacy booleans the host has always exposed. Deliberately last, since they
    # only distinguish the two UNet families.
    if getattr(sd_model, "is_sdxl", False):
        return "sdxl"
    if getattr(sd_model, "is_sd1", False):
        return "sd15"
    return "unknown"


def computation_dtype(process=None):
    """The dtype the loaded UNet/DiT actually COMPUTES in, or None.

    Distinct from ``memory_management.unet_dtype()``, which is the dtype weights
    are STORED in. A control model has to be built with manual casting enabled
    exactly when the two differ, so both are needed and confusing them produces
    either silent precision loss or a dtype mismatch mid-forward.

    Read from the loaded model (``KModel.computation_dtype``) because that is
    where the host itself reads it -- see modules_forge/supported_controlnet.py.
    """
    sd_model = getattr(process, "sd_model", None)
    if sd_model is None:
        try:
            from modules import shared
            sd_model = shared.sd_model
        except Exception:
            return None
    unet = getattr(getattr(sd_model, "forge_objects", None), "unet", None)
    return getattr(getattr(unet, "model", None), "computation_dtype", None)


def diffusion_model(process=None):
    """The raw ``torch.nn.Module`` doing the denoising, or None.

    The one place that knows how deep the host buries it
    (``sd_model.forge_objects.unet.model.diffusion_model``). Patchers that inject
    through module hooks rather than through a ``control`` argument need it; going
    through here keeps that path out of the patchers themselves.
    """
    sd_model = getattr(process, "sd_model", None)
    if sd_model is None:
        try:
            from modules import shared
            sd_model = shared.sd_model
        except Exception:
            return None
    unet = getattr(getattr(sd_model, "forge_objects", None), "unet", None)
    return getattr(getattr(unet, "model", None), "diffusion_model", None)
