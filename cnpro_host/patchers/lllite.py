"""CNPro's ControlLLLite patcher.

Same split as the IP-Adapter patcher: the module/loader shapes come from
``lllite_impl`` (CNPro's vendored copy, because the per-step multiplier lives
inside ``LLLiteModule.forward``), while the model files themselves are read
with the host's own state-dict handling.

KNOWN LIMITATION, carried over deliberately: LLLite evaluates the profile on an
internal STEP COUNTER, not on sigma -- it assumes one forward per sampling step,
so a batch-split cond/uncond would desync it. Band and depth curves ride the
same counter and inherit the caveat. The balance profile is NOT implemented
here (there is no cond/uncond split to bias at that point), which is why
`supports_balance_profile` stays False and the UI warns instead of silently
doing nothing.
"""

from __future__ import annotations

from .base import CNProModelPatcher
from . import lllite_impl

opLLLiteLoader = lllite_impl.LLLiteLoader().load_lllite


class ControlLLLitePatcher(CNProModelPatcher):
    # Every capability is stated here even when the answer is False. Inheriting a
    # False from the base is indistinguishable from never having considered the
    # question, and the difference matters: the first silently drops a feature the
    # user drew a curve for. Enforced by tests/test_patcher_contract.py.

    # NO: the multiplier is applied inside LLLiteModule.forward, which runs once
    # per forward with no view of which rows are cond and which are uncond. There
    # is nothing to bias, so the UI warns rather than applying a dead curve.
    supports_balance_profile = False
    # NO: LLLite has no mask route at all - neither an region_masks
    # path (it emits no residuals to scale) nor an attn_mask argument. Painted
    # masks are reported as unsupported instead of being quietly ignored.
    supports_output_mask = False
    # NO: LLLite conditions on the hint image only; it never sees text.
    supports_unit_prompt = False
    # One LLLite module per attention projection, each named after its UNet
    # block -- so a band profile scales exactly the modules of that depth
    # (lllite_module_bands). Evaluated on the step counter, see module docstring.
    supports_band_profiles = True
    # Not applicable: with supports_output_mask False there is no mask to route.
    # Still stated, because "no route" and "route not chosen yet" must not look
    # the same to the UI.
    masks_via_advanced_weighting = False

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any("lllite" in k for k in state_dict.keys()):
            return None
        return ControlLLLitePatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        from ..adapter import sampling_steps

        unet = process.sd_model.forge_objects.unet

        unet = opLLLiteLoader(
            model=unet,
            state_dict=self.state_dict,
            cond_image=cond.movedim(1, -1),
            strength=self.strength,
            steps=sampling_steps(process),
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            weight_profile=self.weight_profile,
            band_profiles=self.band_weight_profiles,
            depth_profile=self.depth_profile,
            drift_profile=self.drift_profile,
        )[0]

        process.sd_model.forge_objects.unet = unet
