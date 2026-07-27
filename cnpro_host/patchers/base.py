"""CNPro's patcher base: per-run transport + capability declarations.

Subclasses the host's own patcher class (via the adapter) so CNPro's control
models slot into the host's sampling lifecycle without the host being modified.
The original fork added these attributes to the host's ControlModelPatcher
directly; here they live on a CNPro subclass instead, which is what makes the
extension drop-in.

LIFETIME (read before adding an attribute)
------------------------------------------
Patcher instances are CACHED for the process lifetime and shared by two units
pointing at the same model file. Anything written per run is therefore both a
stale-value hazard and a retention hazard -- mask tensors are [1,1,H,W] floats
and encoded prompts are full context tensors, and they stay resident until
overwritten. `reset_run_state()` runs at the START of a unit's patch pass AND
after sampling, so "must not be stale" is enforced in one place instead of by a
comment on each assignment. ADD NEW PER-RUN ATTRIBUTES THERE, not in __init__.

CAPABILITY FLAGS
----------------
A flag answers "can this injection mechanism express X at all", never "do we
want to allow X here". The right way for a new patcher to gain band/depth
support is to resolve the depth of each of its injection sites -- not to be
added to a UI whitelist. The UI reads these flags only to warn; it never gates
on model type. (MAINTENANCE.md invariants 16 and 20.)
"""

from __future__ import annotations

from ..adapter import patcher_base


class CNProModelPatcher(patcher_base()):
    # --- capability declarations (see module docstring) --------------------
    # Honors a per-step cond/uncond balance curve.
    supports_balance_profile = False
    # Can spatially restrict its injection: either through
    # region_masks (residual patchers) or the `mask` argument
    # (attention patchers -> attn_mask).
    supports_output_mask = False
    # Has text cross-attention of its own to re-condition. True only for real
    # ControlNets/ControlLoras; T2I/IP-Adapter/LLLite never see text.
    supports_unit_prompt = False
    # Has injection sites carrying a UNet depth, so per-band and per-depth
    # curves have something to scale. False on the base so a patcher with a
    # single whole-model hook warns instead of silently ignoring a drawn curve.
    supports_band_profiles = False
    # ROUTING, not capability: does this patcher take per-band masks through
    # `region_masks` (residual patchers, which mask each injection
    # site individually) or through the `mask` argument (attention patchers,
    # where it becomes an attn_mask)?
    #
    # This exists because the UI used to ask `isinstance(model, ControlNetPatcher)`
    # to decide - which made "how are masks routed" a question about a CLASS, so
    # every new residual patcher silently took the attention path and got a
    # union mask where it should have had per-band ones. It is a flag for the
    # same reason every other flag here is one: the UI must never branch on model
    # type. (MAINTENANCE.md invariants 16 and 20.)
    masks_via_advanced_weighting = False
    # Architectures this patcher can inject into, as adapter.model_family()
    # names. Empty means "any" - which is right for the UNet-era patchers, whose
    # state-dict sniff already pins the architecture. Only families whose control
    # models are indistinguishable by shape need to set it.
    families = frozenset()

    def __init__(self, model_patcher=None):
        super().__init__(model_patcher)
        self.reset_run_state()

    def reset_run_state(self):
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0

        # --- profiles ------------------------------------------------------
        # Parsed point lists, not strings: the grammar is resolved once in the
        # UI layer so the engine never learns about serialization.
        self.weight_profile = None
        self.balance_profile = None
        self.band_weight_profiles = None
        # Step-INVARIANT per-layer multiplier over normalized model depth.
        # Multiplies the main profile: the unit runs main(step) x depth(layer),
        # a separable product. Mutually exclusive with band profiles by UI
        # contract -- combining them would count depth twice.
        self.depth_profile = None

        # --- per-site / per-row weighting inputs ---------------------------
        # CNPro's own surface, consumed by cnpro_host.patchers.weighting. The
        # host's ControlModelPatcher declares a same-shaped surface of its own
        # (positive_advanced_weighting and friends) which CNPro deliberately
        # leaves at None: our control objects subclass the host's, so sharing
        # the names would let the host's weighting pass and ours both fire on
        # one object and apply every weight twice. See weighting.py.
        self.cond_layer_weights = None
        self.uncond_layer_weights = None
        self.frame_weights = None
        self.sigma_weight_fn = None
        self.region_masks = None

        # --- per-unit prompt ----------------------------------------------
        # Encoded crossattn tensors, or None meaning "use the sampled prompt".
        self.unit_prompt_cond = None
        self.unit_negative_prompt_cond = None
        self.unit_prompt_emb_strength = 1.0
        self.unit_prompt_delta_scale = 1.0
        self.unit_negative_prompt_emb_strength = 1.0
        self.unit_negative_prompt_delta_scale = 1.0
        self.unit_prompt_retention = 0.0

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        # Drop this run's masks / encoded prompts / profiles. See LIFETIME.
        self.reset_run_state()
