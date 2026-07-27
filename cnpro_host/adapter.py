"""The host boundary.

This is the ONE module that knows what application CNPro is running inside.
Everything above it -- ``lib_cnpro/`` (UI) and ``cnpro_core/`` (profile maths) --
must reach the host only through here.

Why this exists
---------------
CNPro's value is the profile editor and the weighting engine behind it, not the
application hosting them. Hosts churn: the widget was built on lllyasviel's
Forge, which has had no commit since 2025-06; it now runs on ForgeNeo, whose
ControlNet backend and canvas API had both moved. Each of those moves cost a
handful of lines here and nothing above.

Adding a host
-------------
Write a module beside ``host_forge_neo.py`` exposing the same names, add one
branch to ``_detect()``. Do NOT add host branches anywhere else; if you find
yourself writing ``if HOST ==`` outside this package, the surface below is
missing something and should grow instead.

The contract (deliberately small -- ten names)
------------------------------------------------
``canvas_widget()``      -> (ImageCanvasClass, HiddenImageChannelClass)
``image_utils()``        -> (hwc3, to_torch)
``model_dir()``          -> str, where control models live
``preprocessors()``      -> dict[name, preprocessor]
``load_control_model()`` -> a CNPro patcher, or None
``patcher_base()``       -> the class CNPro patchers must subclass
``gates_input_by_mask()``-> bool, per preprocessor (knowledge-gate policy)
``sampling_steps()``     -> int for the current run, or None
``model_family()``       -> str, which architecture is loaded right now
``diffusion_model()``    -> the raw denoising nn.Module, or None

The last two were added for DiT support. They are the minimum needed to make a
patcher that cannot use the host's ``control`` argument work without editing the
host: one to know WHICH architecture is loaded, one to reach the module the
residuals have to be injected into. Both are lookups; neither carries policy.

PORTING NOTE (ComfyUI)
----------------------
Six of the eight are trivially satisfiable outside a Gradio app. The two that
are not are ``canvas_widget`` and ``preprocessors``: ComfyUI has no ForgeCanvas
and resolves preprocessors as nodes. A ComfyUI adapter therefore returns
``None`` from ``canvas_widget()`` and the UI layer is simply not built -- the
engine (cnpro_core + cnpro_host/patchers) runs headless, driven by node inputs.
That is the intended split and the reason the painter/editor JS never imports
anything from this package: see ARCHITECTURE.md, "Porting to ComfyUI".
"""

from __future__ import annotations

_HOST = None
_impl = None


def _detect():
    """Identify the host. Cheap, import-based, no version strings parsed.

    Version strings lie across forks (ForgeNeo reports "neo 2.27", Forge
    Classic's other branch reports nothing comparable). What actually matters
    is which modules exist, so that is what we test.
    """
    try:
        import modules_forge.supported_controlnet  # noqa: F401
        import modules_forge.forge_canvas.canvas  # noqa: F401
    except Exception:
        return None
    return "forge"


def _load():
    global _HOST, _impl
    if _impl is not None:
        return _impl
    _HOST = _detect()
    if _HOST == "forge":
        from . import host_forge_neo as impl
    else:
        raise RuntimeError(
            "CNPro: no supported host detected. CNPro currently targets the "
            "Forge family (ForgeNeo / Forge Classic neo, and lllyasviel Forge). "
            "To add a host, see cnpro_host/adapter.py."
        )
    _impl = impl
    return _impl


def host_name() -> str:
    _load()
    return _HOST or "unknown"


# --- the contract ----------------------------------------------------------
# Thin pass-throughs. Kept as functions rather than re-exported symbols so the
# host module is imported lazily: importing CNPro's core must not drag in the
# host, which is what lets cnpro_core be unit-tested standalone (and what lets
# tests/test_profile_parity.py run under a bare python).

def canvas_widget():
    """(canvas_class, hidden_image_channel_class) or None if the host has none."""
    return _load().canvas_widget()


def image_utils():
    """(hwc3, to_torch) -- numpy/tensor helpers with host-specific fast paths."""
    return _load().image_utils()


def model_dir() -> str:
    return _load().model_dir()


def preprocessors() -> dict:
    return _load().preprocessors()


def load_control_model(ckpt_path):
    """Sniff a control model file and return a CNPro patcher.

    RAISES rather than returning None when nothing can load the file, and the
    message distinguishes "CNPro does not support this file" from "a CNPro loader
    is broken". Those are different problems for the user and used to look
    identical -- see registry.load_control_model.
    """
    return _load().load_control_model(ckpt_path)


def patcher_base():
    """The base class CNPro's patchers subclass.

    On Forge this is ``ControlModelPatcher``, so CNPro's patchers slot into the
    host's own lifecycle (process_before_every_sampling etc.) without the host
    knowing CNPro exists. On a host with no such concept, return ``object`` and
    drive the patchers directly.
    """
    return _load().patcher_base()


def gates_input_by_mask(preprocessor) -> bool:
    """Should weight-0 regions be blanked from this preprocessor's INPUT?

    True for embedding preprocessors (CLIP-vision, InstantID): they read the
    image globally, so a cond-level gate has nothing spatial to act on.
    False for spatial preprocessors (canny, depth) -- gating their input would
    manufacture edges along the blanked border. See MAINTENANCE.md, weight-mask
    knowledge gate.

    Deliberately answered HERE rather than by a flag on the host's Preprocessor
    base class: that flag was a core-file edit in the original fork, and the
    decision is CNPro's, not the host's.
    """
    return _load().gates_input_by_mask(preprocessor)


def sampling_steps(process) -> int | None:
    return _load().sampling_steps(process)


def model_family(process=None) -> str:
    """CNPro's name for the model architecture loaded right now.

    'sd15' | 'sdxl' | 'flux' | 'zimage' | 'krea2' | ... | 'unknown'.

    The host answers this by asking its own loader what it matched, so it is a
    fact about the checkpoint's tensors rather than about its filename. Patchers
    that only make sense on one architecture declare a `families` set and are
    refused (with a readable message) on anything else; 'unknown' never gates,
    because failing to NAME a model is not a reason to refuse to run on it.

    On a host with no equivalent notion, return 'unknown' and nothing gates.
    """
    return _load().model_family(process)


def computation_dtype(process=None):
    """The dtype the loaded model computes in (not the dtype it is stored in).

    Control models must be built with manual casting enabled exactly when this
    differs from ``memory_management.unet_dtype()``.
    """
    return _load().computation_dtype(process)


def diffusion_model(process=None):
    """The raw denoising ``torch.nn.Module``, or None if the host hides it.

    Needed only by patchers that inject through module hooks because the host's
    forward ignores a ``control`` argument (see patchers/zimage.py). Everything
    that can go through the ordinary control plumbing should.
    """
    return _load().diffusion_model(process)
