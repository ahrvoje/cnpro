"""CNPro's IP-Adapter / InstantID patcher.

Split of responsibility (this is the shape point 3 of the brief asks for):

  * the MODEL definition -- projection heads, resamplers, the state-dict
    layouts -- is a legitimate host dependency and is imported from the host's
    own IP-Adapter extension. Duplicating ~1500 lines of stable, ComfyUI-derived
    model code would buy no independence.
  * the INJECTION path -- ``CrossAttentionPatch`` carrying per-sigma weight and
    balance LUTs, per-band LUT resolution at install time, per-site attn_masks --
    is CNPro's own detail and is vendored in ``ipadapter_impl.py``.

So CNPro imports what the host legitimately owns and hosts what it legitimately
owns, and the host's IP-Adapter keeps working untouched next to it.

Attention patchers resolve depth per UNET BLOCK rather than per residual, which
is why band/depth support here goes through ``band_of_unet_block`` /
``depth_fraction_of_unet_block`` instead of the residual-index twins. Both live
in cnpro_core next to the profile maths (MAINTENANCE.md invariant 20).
"""

from __future__ import annotations

from pathlib import Path

from .base import CNProModelPatcher
from . import ipadapter_impl

opIPAdapterApply = ipadapter_impl.IPAdapterApply().apply_ipadapter


class IPAdapterPatcher(CNProModelPatcher):
    # Every capability is stated here even when the answer is False -- see the
    # note in lllite.py and tests/test_patcher_contract.py.
    supports_balance_profile = True   # applied per cond/uncond chunk in the patch
    supports_output_mask = True       # forwarded as attn_mask
    # NO: IP-Adapter conditions on a CLIP-vision embedding, not on text. There is
    # no cross-attention context of its own for a unit prompt to replace.
    supports_unit_prompt = False
    # Attention sites carry the same depths a ControlNet's residuals do -- the
    # earlier "IP-Adapter has no banded layers" assumption was wrong, and is
    # exactly what per-layer IP-Adapter weighting exploits elsewhere.
    supports_band_profiles = True
    # NO: this is an ATTENTION patcher. Its per-band masks arrive through the
    # `mask` argument and become attn_masks at each patched site; it has no
    # residual list for region_masks to scale. Getting this wrong in
    # either direction silently halves the feature - True here would send masks
    # down a path this patcher ignores.
    masks_via_advanced_weighting = False

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        model = state_dict

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        if "ip_adapter" not in model.keys() or len(model["ip_adapter"]) == 0:
            return None

        o = IPAdapterPatcher(model)

        model_filename = Path(ckpt_path).name.lower()
        if "v2" in model_filename:
            o.faceid_v2 = True
            o.weight_v2 = True

        return o

    def __init__(self, state_dict):
        super().__init__()
        self.ip_adapter = state_dict
        self.faceid_v2 = False
        self.weight_v2 = False

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        # `mask` is B1HW, or a dict band -> B1HW when the unit painted per-band
        # weight masks: those bands are the same UNet depths this adapter's
        # attention sites sit at, so they are forwarded PER BAND rather than
        # unioned into one attn_mask (apply_ipadapter resolves each site).
        def as_attn(m):
            return m.squeeze(1) if m is not None else None

        attn_mask = (
            {band: as_attn(m) for band, m in mask.items()}
            if isinstance(mask, dict)
            else as_attn(mask)
        )

        unet = opIPAdapterApply(
            ipadapter=self.ip_adapter,
            model=unet,
            weight=self.strength,
            start_at=self.start_percent,
            end_at=self.end_percent,
            faceid_v2=self.faceid_v2,
            weight_v2=self.weight_v2,
            attn_mask=attn_mask,
            weight_profile=self.weight_profile,
            balance_profile=self.balance_profile,
            band_profiles=self.band_weight_profiles,
            depth_profile=self.depth_profile,
            drift_profile=self.drift_profile,
            **cond,
        )[0]

        process.sd_model.forge_objects.unet = unet
