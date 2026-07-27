"""CNPro's ControlNet / ControlLora / T2I-Adapter patcher.

The structural point: this patcher builds CNPro's OWN ``ControlNet`` /
``ControlLora`` / ``T2IAdapter`` classes (from ``controlnet_impl``), so the
profile-aware ``get_control`` / ``compute_controlnet_weighting`` are reached by
ordinary subclassing. Nothing in the host is patched, wrapped, or reordered --
which is why enabling CNPro cannot change behaviour for anyone using the host's
own ControlNet.

Recognizing a file is DELEGATED to the host's own loader rather than duplicated.
That decides which model zoo CNPro accepts, and a second copy of the sniffing
could only ever drift out of agreement with the host's -- it already had, once:
neo's ``cldm.ControlNet`` grew a required ``hint_width`` argument that the copy
did not pass. Delegating also means CNPro shares the host's process-lifetime
model cache instead of loading a second copy of the same weights, and inherits
its missing/unexpected-key diagnostics for free.

What CNPro still owns is the CLASS of the resulting control object: the host's
loader is asked for the model, and the loaded modules are then transplanted into
CNPro's own ``ControlNet`` / ``ControlLora`` / ``T2IAdapter`` (from
``controlnet_impl``), which are ordinary subclasses of the host's. That is what
puts the profile-aware ``control_merge`` in the path. Nothing in the host is
patched, wrapped or reordered, so enabling CNPro cannot change behaviour for
anyone using the host's own ControlNet.
"""

from __future__ import annotations

from backend.patcher.controlnet import (
    ControlLora as HostControlLora,
    ControlNet as HostControlNet,
    T2IAdapter as HostT2IAdapter,
)
from modules_forge.supported_controlnet import (
    ControlNetPatcher as HostControlNetPatcher,
)

from . import controlnet_impl as impl
from .base import CNProModelPatcher


def _as_cnpro_control(built):
    """Rebuild one of the host's control objects as CNPro's equivalent class.

    The loaded modules and weights are transplanted, never reloaded -- this is a
    re-wrap, not a second load. Returns None for a control class CNPro has no
    equivalent for, which the caller reports as a breakage rather than silently
    accepting a control whose profiles nothing would ever read.

    ControlLora is checked before ControlNet: it is a subclass of it in the host
    as well as here, so the order is what keeps a LoRA-style control from being
    flattened into a plain ControlNet.
    """
    if isinstance(built, HostControlLora):
        return impl.ControlLora(
            built.control_weights,
            global_average_pooling=built.global_average_pooling,
        )
    if isinstance(built, HostT2IAdapter):
        return impl.T2IAdapter(built.t2i_model, built.channels_in)
    if isinstance(built, HostControlNet):
        return impl.ControlNet(
            built.control_model,
            global_average_pooling=built.global_average_pooling,
            load_device=built.load_device,
            manual_cast_dtype=built.manual_cast_dtype,
        )
    return None


class ControlNetPatcher(CNProModelPatcher):
    supports_balance_profile = True
    supports_output_mask = True
    # Residuals are emitted per skip connection, so each carries a depth:
    # band and depth curves have something to scale (the UNet layout's
    # band / depth_fraction, in residual_layout.py).
    supports_band_profiles = True
    # Per-band masks go through region_masks, applied per injection
    # layer by the weighting engine - not through the `mask` argument, which this
    # patcher ignores.
    masks_via_advanced_weighting = True

    @property
    def supports_unit_prompt(self):
        # This patcher also wraps T2I adapters (see try_build_from_state_dict);
        # only the true ControlNet classes consume a text context.
        return isinstance(self.model_patcher, impl.ControlNet)

    def __init__(self, model_patcher):
        super().__init__(model_patcher)

    @staticmethod
    def try_build_from_state_dict(controlnet_data: dict, ckpt_path):
        """Recognize a ControlNet the way the host does, wrap it the way we do.

        Returning None means "not a control model I recognize" and lets the
        registry try the next candidate. A control class the host recognizes but
        CNPro cannot wrap is a different outcome and raises: silently handing
        back a control whose profiles nothing reads would be a drawn curve that
        does nothing, with no error anywhere.
        """
        built = HostControlNetPatcher.try_build_from_state_dict(controlnet_data, ckpt_path)
        if built is None:
            return None

        control = _as_cnpro_control(built.model_patcher)
        if control is None:
            raise TypeError(
                "the host loaded %s, which CNPro has no control class for - it would "
                "run with every profile ignored" % type(built.model_patcher).__name__)
        return ControlNetPatcher(control)

    def process_before_every_sampling(self, process, cond, mask, *args, control_type=None, **kwargs):
        unet = process.sd_model.forge_objects.unet
        unet = impl.apply_controlnet_advanced(
            unet=unet,
            controlnet=self.model_patcher,
            image_bchw=cond,
            strength=self.strength,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            cond_layer_weights=self.cond_layer_weights,
            uncond_layer_weights=self.uncond_layer_weights,
            frame_weights=self.frame_weights,
            sigma_weight_fn=self.sigma_weight_fn,
            region_masks=self.region_masks,
            weight_profile=self.weight_profile,
            balance_profile=self.balance_profile,
            band_weight_profiles=self.band_weight_profiles,
            depth_profile=self.depth_profile,
            unit_context=self.unit_prompt_cond,
            unit_uncond_context=self.unit_negative_prompt_cond,
            unit_emb_strength=self.unit_prompt_emb_strength,
            unit_delta_scale=self.unit_prompt_delta_scale,
            unit_uncond_emb_strength=self.unit_negative_prompt_emb_strength,
            unit_uncond_delta_scale=self.unit_negative_prompt_delta_scale,
            unit_retention=self.unit_prompt_retention,
            control_type=control_type,
        )
        process.sd_model.forge_objects.unet = unet
