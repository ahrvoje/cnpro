"""CNPro's own control-model registry.

WHY THIS IS NOT THE HOST'S REGISTRY
-----------------------------------
Forge exposes ``modules_forge.shared.supported_control_models`` (a plain list)
and ``try_load_supported_control_model`` (first match wins). CNPro could insert
its patchers at the head of that list -- and that would silently change which
classes the HOST's own ControlNet extension builds, for every user, whether or
not they use CNPro.

So CNPro keeps its own list and its own loader instead. The registry below is
private, ordered, and the only thing CNPro dispatches through. Consequences
worth knowing:

  * enabling CNPro cannot alter the host's behaviour by even one class;
  * both can coexist in one process (the UI still requires the host's builtin
    ControlNet to be disabled, but that is an elem_id collision, not a model
    conflict -- see README);
  * a new control-model family is added HERE, in one list, and needs no host
    cooperation at all.

ORDER MATTERS: first match wins, same as the host. LLLite is sniffed before
ControlNet because some LLLite files also carry keys a loose ControlNet check
would accept.

DiT NOTE
--------
Z-Image is in (``patchers/zimage.py``). Adding Flux / Qwen means the same two
steps: a patcher module beside the four below, and one entry in ``_types()``.

It does NOT mean touching the profile engine. That was the bet ARCHITECTURE.md
made and it held: ``cnpro_core`` gained two functions (a depth mapping for
block-indexed injection) and lost nothing, while the weighting engine itself
moved to ``patchers/weighting.py`` unchanged except that the three genuinely
family-specific decisions - tensor rank, mask projection, depth of site i - now
come from a ``ResidualLayout``. Flux and Qwen are easier than Z-Image was: the
host's ``backend/nn/flux.py`` and ``qwen.py`` already CONSUME a ``control``
argument, so they need no injection hooks at all, only a layout and a loader.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("CNPro")


def _types():
    # Imported lazily: the patcher modules pull in torch and the host backend,
    # and the UI layer must be importable without paying for that.
    from .patchers.zimage import ZImageControlNetPatcher
    from .patchers.lllite import ControlLLLitePatcher
    from .patchers.ipadapter import IPAdapterPatcher
    from .patchers.controlnet import ControlNetPatcher

    # Z-Image first: its discriminator (`control_layers.0.after_proj.weight`)
    # appears in no other family, so the check is exact AND cheap - it fails on
    # the first dict lookup for every UNet-era file, which is what most users
    # have most of. Ordering it early costs nothing and keeps the expensive
    # UNet-config sniff off the path for DiT models.
    return [ZImageControlNetPatcher, ControlLLLitePatcher, IPAdapterPatcher, ControlNetPatcher]


def load_control_model(ckpt_path):
    """Sniff `ckpt_path` and return a CNPro patcher.

    Each candidate gets a SHALLOW COPY of the state dict: the host's loaders
    ``pop()`` keys while converting diffusers layouts, so a failed candidate
    would otherwise hand the next one a gutted dict.

    THREE OUTCOMES, KEPT DISTINCT
    -----------------------------
    This loop used to have two: a patcher either returned something, or the file
    was "unrecognised". Every exception was logged and swallowed, and the caller
    turned a ``None`` into "Recognizing Control Model failed" no matter why.

    That is how a plain typo survived: ``patchers/controlnet.py`` called
    ``memory_management.get_computation_dtype()``, which does not exist. Every
    single SD1.5/SDXL ControlNet load raised ``AttributeError`` here, was logged
    at a level nobody reads, and was reported to the user as "this file is not a
    ControlNet". A broken loader and an unsupported file are completely different
    problems and they looked identical.

    So:

    * **claimed** -- a patcher returned a model. Done.
    * **declined** -- a patcher returned ``None``. It looked and said "not mine";
      try the next one. This is the only silent path, and it is silent because
      it is the normal case.
    * **failed** -- a patcher RAISED. It is not fine. Either it recognised the
      file and refuses it deliberately (``cnpro_recognised`` on the exception, as
      ``UnsupportedZImageControlNet`` sets), in which case that refusal IS the
      answer and is raised immediately with its own message; or it broke, in
      which case the error is remembered and re-raised at the end rather than
      being laundered into "unrecognised".

    Raises RuntimeError when nothing claimed the file. Callers that want the old
    "None means no" can catch it, but they should not: the message says which
    patchers were tried and what went wrong, and that is the information that was
    missing for however long the typo above was live.
    """
    from backend import utils

    state_dict = utils.load_torch_file(ckpt_path, safe_load=True)

    tried = []
    broke = []

    for supported_type in _types():
        name = supported_type.__name__
        tried.append(name)
        try:
            candidate = {k: v for k, v in state_dict.items()}
            model = supported_type.try_build_from_state_dict(candidate, ckpt_path)
        except Exception as exc:
            if getattr(exc, "cnpro_recognised", False):
                # A deliberate refusal: this patcher KNOWS what the file is and
                # will not run it. No later patcher can do better, and falling
                # through would replace a precise explanation with a vague one.
                raise
            logger.exception("CNPro: %s raised while sniffing %s", name, ckpt_path)
            broke.append("%s: %s: %s" % (name, type(exc).__name__, exc))
            continue
        if model is not None:
            return model

    if broke:
        raise RuntimeError(
            "CNPro: could not load %s. %d of %d loaders raised while sniffing it, "
            "so this is a BUG IN CNPRO, not an unsupported file:\n  %s"
            % (ckpt_path, len(broke), len(tried), "\n  ".join(broke)))

    raise RuntimeError(
        "CNPro: %s is not a control model CNPro recognises. Tried: %s."
        % (ckpt_path, ", ".join(tried)))
