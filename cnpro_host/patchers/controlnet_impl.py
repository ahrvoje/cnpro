"""ControlNet injection engine (L2, host-bound).

CNPro's control classes EXTEND the host's rather than replacing them:
``ControlBase`` subclasses ``backend.patcher.controlnet.ControlBase``, and what
lives here is the weight / balance / band / depth profile engine and the
per-unit prompt path built on top of it. Anything the host already does -- the
cond hint, timestep gating, chaining, grouping residuals, scaling by strength --
is inherited and reached through ``super()``, not copied.

Nothing in the host is patched or mutated at import time, so enabling CNPro
cannot change behaviour for anyone using the host's own ControlNet. Note the
consequence of subclassing: the host's own weighting surface
(``positive_advanced_weighting`` and friends) is inherited too, and CNPro
deliberately never writes it -- see weighting.py.

Host-bound by design - it imports backend.* freely. The host-AGNOSTIC half
(profile evaluation, LUT building, band/depth mapping) lives in
cnpro_core/weight_profile.py and is imported from there; keep it that way,
because that module is what a ComfyUI port reuses unchanged.

DiT NOTE: the per-layer weighting below indexes control['input'|'middle'|
'output'] residual lists. ComfyUI's Flux/Qwen ControlNets emit the SAME dict
shape (input->double_blocks, output->single_blocks), so this math already
transfers; what changes is n, not the algorithm. See ARCHITECTURE.md.
"""
import logging
import math

import torch

from backend.logging import setup_logger

from backend.misc import image_resize
from backend import memory_management
from backend.patcher.base import ModelPatcher
# MUST precede the backend.patcher.controlnet import below. The host has an
# import cycle -- backend.patcher.controlnet imports cldm at its top, and cldm
# imports `logger` back from backend.patcher.controlnet, which is defined AFTER
# that import. Reaching the cycle from the controlnet side therefore raises
# ImportError; reaching it from the cldm side resolves. Importing cldm first
# pins the working direction instead of depending on who imported what earlier.
import backend.nn.cnets.cldm  # noqa: F401  (import-cycle order, see above)
# The host's control classes are the bases CNPro extends. ControlLoraOps, the
# LoRA build in ControlLora.pre_run and the whole T2I forward pass come with
# them; CNPro used to carry a copy of each and now inherits them instead.
from backend.patcher.controlnet import (  # noqa: E402
    ControlBase as HostControlBase,
    ControlLora as HostControlLora,
    T2IAdapter as HostT2IAdapter,
)
from cnpro_core.weight_profile import (
    build_weight_profile_lookup,
    lookup_weight_profile_strength,
)
# The weighting engine itself now lives in `weighting.py` so the DiT patchers
# can share it (it is pure index math over residual lists and knows nothing
# about UNets). Re-exported here because this module's own call sites - and
# MAINTENANCE.md - have always named `compute_controlnet_weighting` and
# `band_of` as belonging to the ControlNet engine.
from .residual_layout import UNET_LAYOUT  # noqa: F401  (re-export)
from .weighting import (  # noqa: F401  (re-exports)
    band_of,
    compute_controlnet_weighting,
    layer_mask_for,
    resolve_band_mask,
)


logger = logging.getLogger("ControlNet")
setup_logger(logger)


def apply_controlnet_advanced(
        unet,
        controlnet,
        image_bchw,
        strength,
        start_percent,
        end_percent,
        cond_layer_weights=None,
        uncond_layer_weights=None,
        frame_weights=None,
        sigma_weight_fn=None,
        region_masks=None,
        weight_profile=None,
        balance_profile=None,
        band_weight_profiles=None,
        depth_profile=None,
        drift_profile=None,
        unit_context=None,
        unit_uncond_context=None,
        unit_emb_strength=1.0,
        unit_delta_scale=1.0,
        unit_uncond_emb_strength=1.0,
        unit_uncond_delta_scale=1.0,
        unit_retention=0.0,
        control_type=None
):
    """Clone `unet` with `controlnet` attached under CNPro's weighting inputs.

    Everything this sets lands on a per-run copy of the control object; the
    passed `controlnet` and the passed `unet` are not modified.

    The per-site / per-row inputs -- `cond_layer_weights`, `uncond_layer_weights`,
    `frame_weights`, `sigma_weight_fn`, `region_masks` -- are documented at their
    point of use in ``weighting.py``, which is the only module that reads them.
    The profile arguments below are this function's own contribution and are
    resolved into sigma lookups by ``ControlBase.pre_run``.

    # weight_profile

    A list of (x, y) points, x being relative sampling position in [0, 1] and
    y being control strength (any real, negative = repulsion). Strength at
    each step is obtained by piecewise-linear interpolation, with horizontal
    extension beyond the first and the last point. When given, it overrides
    `strength` at every step, and steps where the profile evaluates to
    exactly 0 skip control model computation.

    # depth_profile

    A list of (x, y) points where x is normalized UNET DEPTH (0 = shallowest
    injection layer = fine/texture, 1 = deepest = coarse/composition) and y a
    multiplier on that layer's injection. On its own it does not vary with the
    step, so it multiplies whatever per-step strength is in force:

        effective(step, layer) = weight_profile(step) * depth_profile(depth)

    i.e. the separable product of a time curve and a depth curve. The band
    profiles are the same depth axis quantized to three buckets, each with its
    OWN time curve - the two are alternative shapes of the same 2D field and
    the UI keeps them mutually exclusive (a per-bucket curve multiplied by a
    per-depth curve would count depth twice and no drawn value could be read
    literally any more). Layer depth comes from depth_fraction_of_residual,
    the un-quantized twin of the band mapping.

    # drift_profile

    A list of (x, y) points over the same relative sampling position as
    `weight_profile`, but y is a SHIFT along the depth axis rather than a
    weight (neutral 0, drawn on a [-1, 1] plot). It moves where the depth curve
    is read as sampling proceeds:

        effective(step, layer) =
            weight_profile(step) * depth_profile(depth - drift_profile(step))

    which is the ONE thing the separable product above cannot express: a depth
    shape that changes over time. A descending drift sweeps the control from
    composition to texture. Ignored without a `depth_profile` - there is
    nothing to move - and that falls out of the arithmetic rather than being
    guarded for; see cnpro_core.weight_profile.drifted_depth.

    # balance_profile

    A list of (x, y) points like weight_profile with y in [0, 1]: per-step
    cond/uncond balance. y = 0.5 applies control to cond and uncond equally
    (balanced), y = 1 to cond only (control matters most), y = 0 to uncond
    only (prompt matters most); in between interpolates linearly.

    # unit_context

    An already-encoded text embedding (B x tokens x dim, B normally 1) that
    the control branch uses as its cross-attention context INSTEAD of the
    sampled positive prompt's - on the COND rows of the batch only; uncond
    rows keep the sampled negative context, so the prompt's semantics live in
    the cond/uncond contrast that CFG amplifies (feeding both halves the same
    text makes the residuals cancel out of the contrast and the prompt does
    visibly nothing). Only meaningful on true ControlNets (ControlNet /
    ControlLora) - they are UNet-encoder copies with their own text
    cross-attention; T2IAdapter has no text input and ignores this.

    # unit_uncond_context

    The negative counterpart: an encoded text embedding replacing the sampled
    NEGATIVE context on the UNCOND rows only ("push this control's semantics
    away from X"). Independent of unit_context - either, both or neither may
    be set; unset sides keep the sampled context, preserving the asymmetry
    that makes the contrast work.

    # unit_emb_strength / unit_uncond_emb_strength

    Embedding-space strength of each unit context, neutral 1: the row's
    context becomes sampled + s * (unit - sampled). 1 = the plain swap,
    0 = that side off, > 1 extrapolates past the unit prompt, < 0 pushes the
    context away from it.

    # unit_delta_scale / unit_uncond_delta_scale

    Scale of the text's exact effect on the control residuals, neutral 1:
    residual = base + s * (with_text - base), where `base` is the residual
    computed with the sampled context. 0 and 1 stay a single control forward
    (structurally - 1 IS the plain swap, 0 drops the side); any other value
    runs the control model a second time per step to obtain `base`.

    # unit_retention

    One global knob (applies to both unit-context sides alike), neutral 0:
    multiplies the per-step delta scales by 1 + retention * progress, where
    progress is the relative sampling position in [0, 1]. The text's
    influence on the final image naturally decays as the latent converges
    (the cond/uncond contrast of the control residuals shrinks); the ramp
    counteracts that by amplifying the text's isolated delta more the later
    the step. x1 at the first step keeps the start of sampling untouched.
    Any value > 0 engages the base-pass isolation (second control forward)
    on every step where the ramped scale leaves 1 - i.e. all but the first.

    """

    cnet = controlnet.copy().set_cond_hint(
        image_bchw, strength, (start_percent, end_percent)
    ).set_control_type(control_type)
    cnet.cond_layer_weights = cond_layer_weights
    cnet.uncond_layer_weights = uncond_layer_weights
    cnet.frame_weights = frame_weights
    cnet.sigma_weight_fn = sigma_weight_fn
    cnet.weight_profile = weight_profile
    cnet.balance_profile = balance_profile
    cnet.depth_profile = depth_profile
    cnet.drift_profile = drift_profile
    cnet.unit_context = unit_context
    cnet.unit_uncond_context = unit_uncond_context
    cnet.unit_emb_strength = unit_emb_strength
    cnet.unit_delta_scale = unit_delta_scale
    cnet.unit_uncond_emb_strength = unit_uncond_emb_strength
    cnet.unit_uncond_delta_scale = unit_uncond_delta_scale
    cnet.unit_retention = unit_retention

    if band_weight_profiles is not None:
        assert all(band in ('coarse', 'mid', 'fine') for band in band_weight_profiles)
        cnet.band_weight_profiles = band_weight_profiles

    if region_masks is not None:
        if isinstance(region_masks, dict):
            assert all(band in ('coarse', 'mid', 'fine') for band in region_masks)
            masks = list(region_masks.values())
        else:
            masks = [region_masks]
        for m in masks:
            assert isinstance(m, torch.Tensor)
            B, C, H, W = m.shape
            assert B > 0 and C == 1 and H > 0 and W > 0

    cnet.region_masks = region_masks

    m = unet.clone()
    m.add_patched_controlnet(cnet)
    return m



def fold_previous_control(out, control_prev):
    """Add an earlier unit's residuals into `out` in place, and return it.

    Units chain: each one weights its own residuals and then adds whatever the
    previous unit in the chain produced. Keyed off the groups actually present
    rather than the UNet's fixed input/middle/output triple, so a family whose
    control emits some other group (Z-Image's single 'block') chains through
    this unchanged instead of having its residuals silently dropped.

    The wider batch has to be the accumulator: a unit that ran on cond rows only
    emits fewer rows than one that ran on both, and `narrow += wide` cannot
    broadcast. Where the shapes differ the sum is rebuilt rather than done in
    place.
    """
    if control_prev is None:
        return out

    for group, previous in control_prev.items():
        merged = out.setdefault(group, [])
        for i, prev in enumerate(previous):
            if i >= len(merged):
                merged.append(prev)
            elif prev is None:
                continue
            elif merged[i] is None:
                merged[i] = prev
            elif merged[i].shape[0] < prev.shape[0]:
                merged[i] = prev + merged[i]
            else:
                merged[i] += prev
    return out


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)


class ControlBase(HostControlBase):
    """CNPro's control object: the host's, plus the profile surface.

    Subclasses rather than replaces the host's ControlBase. Everything the host
    already provides -- the cond hint, control type, timestep gating, chaining,
    the residual merge -- is inherited and called through `super()`; the only
    members below are CNPro's own additions and the four hooks that have to
    extend a host method rather than replace it (`pre_run`, `cleanup`,
    `copy_to`, `control_merge`).

    Nothing in the host is patched or reordered by any of this: enabling CNPro
    cannot change behaviour for anyone using the host's own ControlNet.
    """

    def __init__(self, device=None):
        # HostControlBase by NAME, not super(). ControlLora and T2IAdapter
        # inherit from this class AND from a host class that also derives from
        # HostControlBase, so `super()` here resolves to HostControlLora /
        # HostT2IAdapter -- whose constructors demand their own arguments and
        # raise. Those two subclasses set their own fields themselves; what this
        # has to guarantee is only that the host's BASE state exists. The host
        # does the same thing in its own ControlLora for the same reason.
        HostControlBase.__init__(self, device)
        self.weight_profile = None
        self.weight_profile_sigmas = None
        self.weight_profile_strengths = None
        self.balance_profile = None
        self.balance_sigmas = None
        self.balance_values = None
        # per-band (coarse/mid/fine) step profiles: dict band -> point list;
        # multipliers on that band's injection layers (see band_of)
        self.band_weight_profiles = None
        self.band_profile_lookup = None
        # depth profile: point list over normalized UNet depth, a per-injection
        # layer multiplier (see apply_controlnet_advanced). Needs no sigma
        # lookup of its own - depth does not change while sampling.
        self.depth_profile = None
        # depth-drift profile: point list over the STEP axis whose y shifts
        # where the depth curve above is read, so unlike the depth curve it
        # does need a sigma lookup (built in pre_run, same as balance).
        self.drift_profile = None
        self.drift_sigmas = None
        self.drift_values = None
        # Per-unit prompt embeddings; replace the sampled prompt as the control
        # branch's cross-attention context (consumed by ControlNet.get_control
        # only - the other control types have no text input). unit_context
        # drives the cond rows, unit_uncond_context the uncond rows; either may
        # be None (that side keeps the sampled context).
        self.unit_context = None
        self.unit_uncond_context = None
        # strength pair per side (see apply_controlnet_advanced docstring):
        # embedding strength lerps the context, delta scale multiplies the
        # text's residual effect (non-0/1 values cost a second forward)
        self.unit_emb_strength = 1.0
        self.unit_delta_scale = 1.0
        self.unit_uncond_emb_strength = 1.0
        self.unit_uncond_delta_scale = 1.0
        # prompt retention (see apply_controlnet_advanced): global per-step
        # ramp on the delta scales, 1 + retention * progress. The lookup pair
        # below maps the current sigma back to relative sampling position and
        # is built in pre_run from an identity profile.
        self.unit_retention = 0.0
        self.retention_sigmas = None
        self.retention_progress = None

    def pre_run(self, model, percent_to_timestep_function):
        # the host resolves timestep_range and walks the chain; we add the
        # sigma lookups the profiles need
        super().pre_run(model, percent_to_timestep_function)
        if self.weight_profile:
            self.weight_profile_sigmas, self.weight_profile_strengths = build_weight_profile_lookup(
                self.weight_profile, percent_to_timestep_function)
        else:
            self.weight_profile_sigmas = None
            self.weight_profile_strengths = None
        if self.balance_profile:
            self.balance_sigmas, self.balance_values = build_weight_profile_lookup(
                self.balance_profile, percent_to_timestep_function)
        else:
            self.balance_sigmas = None
            self.balance_values = None
        if self.band_weight_profiles:
            self.band_profile_lookup = {
                band: build_weight_profile_lookup(points, percent_to_timestep_function)
                for band, points in self.band_weight_profiles.items() if points
            }
        else:
            self.band_profile_lookup = None
        # the drift only exists to move the depth curve, so a drift with no
        # depth curve builds no lookup at all rather than a live one nothing
        # reads - the engine's `drift_sigmas` input is then inert and
        # `_gather`'s early-out can still fire on a unit that has neither
        if self.drift_profile and self.depth_profile:
            self.drift_sigmas, self.drift_values = build_weight_profile_lookup(
                self.drift_profile, percent_to_timestep_function)
        else:
            self.drift_sigmas = None
            self.drift_values = None
        if getattr(self, 'unit_retention', 0.0):
            # identity profile: looking it up at the current sigma yields the
            # relative sampling position the retention ramp needs
            self.retention_sigmas, self.retention_progress = build_weight_profile_lookup(
                [(0.0, 0.0), (1.0, 1.0)], percent_to_timestep_function)
        else:
            self.retention_sigmas = None
            self.retention_progress = None

    def cleanup(self):
        # the host drops the cond hint, the timestep range and the chain; the
        # per-run lookups and the mask cache below are ours
        super().cleanup()
        self.weight_profile_sigmas = None
        self.weight_profile_strengths = None
        self.balance_sigmas = None
        self.balance_values = None
        self.band_profile_lookup = None
        self.drift_sigmas = None
        self.drift_values = None
        self.retention_sigmas = None
        self.retention_progress = None
        self._layer_mask_cache = None

    def current_profile_strength(self, t):
        """Strength at current timestep t from the weight profile lookup built in pre_run."""
        return lookup_weight_profile_strength(
            self.weight_profile_sigmas, self.weight_profile_strengths, float(t[0]))

    def apply_profile_strength(self, t):
        """Put the profile's value for this step on `strength`; True to skip.

        The weight profile REPLACES `strength` at every step, so a step where it
        evaluates to (near-)zero contributes nothing and the control model can be
        skipped outright rather than run and multiplied by ~0.

        The epsilon is not cosmetic: profiles mapped through a scale range rarely
        interpolate to an exact 0.0, and without it the skip never fired for
        precisely the profiles that spend many steps at "no control". Negative
        strengths past the epsilon stay valid -- that is repulsive control from a
        scale range below 0, not a rounding artefact.
        """
        if self.weight_profile_sigmas is None:
            return False
        self.strength = self.current_profile_strength(t)
        return abs(self.strength) < 1e-4

    def control_from_chain_only(self, x_noisy, t, cond, batched_number):
        """What this unit returns when it is skipping: the rest of the chain."""
        if self.previous_controlnet is None:
            return None
        return self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

    def get_control(self, x_noisy, t, cond, batched_number):
        """Gate on the weight profile, then run the host's control.

        This is the whole of CNPro's addition for control types whose forward
        pass we do not otherwise change (T2IAdapter today): the profile decides
        this step's strength, and everything else -- hint preparation, running
        the model, chaining -- is the host's, reached by `super()`. The weighting
        still happens, because the host's `get_control` finishes by calling
        `self.control_merge`, which resolves to CNPro's.

        `ControlNet` overrides this outright; it needs the per-unit prompt path
        woven through the middle of the forward pass and cannot delegate.

        ONE ORDERING NOTE: the host evaluates `controlnet_conditioning_modifiers`
        (which may rewrite `t`) before its own timestep gate, so the profile here
        is read at the PRE-modifier timestep where the old hand-copied version
        read it after. Nothing in this codebase registers such a modifier -- the
        host only exposes the setter -- so the two orders cannot currently differ,
        but a modifier that shifts `t` across the epsilon would see this.
        """
        if self.apply_profile_strength(t):
            return self.control_from_chain_only(x_noisy, t, cond, batched_number)
        return super().get_control(x_noisy, t, cond, batched_number)

    def copy_to(self, c):
        # the host copies the hint, strength, range and pooling flag
        super().copy_to(c)
        c.weight_profile = self.weight_profile
        c.balance_profile = self.balance_profile
        c.band_weight_profiles = self.band_weight_profiles
        c.depth_profile = self.depth_profile
        c.drift_profile = self.drift_profile
        c.unit_context = self.unit_context
        c.unit_uncond_context = self.unit_uncond_context
        c.unit_emb_strength = self.unit_emb_strength
        c.unit_delta_scale = self.unit_delta_scale
        c.unit_uncond_emb_strength = self.unit_uncond_emb_strength
        c.unit_uncond_delta_scale = self.unit_uncond_delta_scale
        c.unit_retention = self.unit_retention

    def control_merge(self, control_input, control_output, control_prev, output_dtype):
        """Group this unit's residuals, weight them, then fold in the chain.

        The grouping and `strength` scaling are the host's and are inherited.
        `control_prev` is deliberately withheld from the host call and folded in
        afterwards: the earlier unit was already weighted by its OWN profiles
        when it ran, and handing it over here would put it through this unit's
        weighting pass a second time.

        The host's own weighting pass runs inside that `super()` call and is a
        no-op for us by construction -- it reads `positive_advanced_weighting`
        and friends, which CNPro never sets on a control object (see
        weighting.py for why the two surfaces have different names).
        """
        out = super().control_merge(control_input, control_output, None, output_dtype)
        out = compute_controlnet_weighting(out, self)
        return fold_previous_control(out, control_prev)


class ControlNet(ControlBase):
    def __init__(self, control_model, global_average_pooling=False, device=None, load_device=None, manual_cast_dtype=None):
        super().__init__(device)
        self.control_model = control_model
        self.load_device = load_device
        self.control_model_wrapped = ModelPatcher(self.control_model, load_device=load_device, offload_device=memory_management.unet_offload_device())
        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None
        self.manual_cast_dtype = manual_cast_dtype

    def get_control(self, x_noisy, t, cond, batched_number):
        to = self.transformer_options

        for conditioning_modifier in to.get('controlnet_conditioning_modifiers', []):
            x_noisy, t, cond, batched_number = conditioning_modifier(self, x_noisy, t, cond, batched_number)

        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        # here rather than before the timestep gate above, so a step this unit
        # is not scheduled for never consults the profile at all
        if self.apply_profile_strength(t):
            return control_prev

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = image_resize.adaptive_resize(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        context = cond['c_crossattn']
        # set when delta scaling needs a second forward with the sampled
        # context (see unit_delta_scale in apply_controlnet_advanced)
        base_context = None
        delta_scale_vec = None
        if self.unit_context is not None or self.unit_uncond_context is not None:
            # Per-unit prompts: unit_context drives the COND rows of the
            # control batch, unit_uncond_context the UNCOND rows; unset sides
            # keep the sampled context. This asymmetry is the whole point:
            # cond and uncond rows share latent, timestep and hint, so if both
            # got the same text the control residuals would be (nearly)
            # identical and the prompt's semantics would cancel out of the CFG
            # contrast `uncond + scale*(cond - uncond)` - the only part CFG
            # amplifies. Verified empirically: the both-halves version
            # produced no recognizable semantic steering. y stays untouched
            # (pooled/size conds keep coming from the sampled prompts).
            def prepared(t):
                t = t.to(device=context.device, dtype=context.dtype)
                if t.shape[0] != x_noisy.shape[0]:
                    t = t.repeat(math.ceil(x_noisy.shape[0] / t.shape[0]), 1, 1)[:x_noisy.shape[0]]
                return t

            EPS = 1e-4
            s_emb_p = getattr(self, 'unit_emb_strength', 1.0)
            s_emb_n = getattr(self, 'unit_uncond_emb_strength', 1.0)
            s_dl_p = getattr(self, 'unit_delta_scale', 1.0)
            s_dl_n = getattr(self, 'unit_uncond_delta_scale', 1.0)
            # prompt retention: ramp both delta scales by 1 + r * progress so
            # the text's isolated delta grows toward the end of sampling,
            # against its natural decay. Applied BEFORE the active checks and
            # the != 1 comparisons below, so the base-pass isolation engages
            # by itself exactly on the steps where the ramp leaves 1; the
            # first step (progress 0) stays bit-identical to retention off.
            r_ret = getattr(self, 'unit_retention', 0.0)
            if abs(r_ret) > 1e-4 and self.retention_sigmas is not None:
                progress = lookup_weight_profile_strength(
                    self.retention_sigmas, self.retention_progress, float(t[0]))
                ramp = 1.0 + r_ret * progress
                s_dl_p *= ramp
                s_dl_n *= ramp
            # a side is ACTIVE when its context is set and neither strength
            # zeroes it: embedding strength 0 makes the blended context the
            # sampled one, delta scale 0 removes the effect - both mean "this
            # side contributes nothing" and are handled structurally (no
            # extra forward, no lerp), not numerically
            p_active = self.unit_context is not None and abs(s_emb_p) > EPS and abs(s_dl_p) > EPS
            n_active = self.unit_uncond_context is not None and abs(s_emb_n) > EPS and abs(s_dl_n) > EPS

            def blend(sampled, unit, s):
                # embedding strength: sampled + s*(unit - sampled). s == 1 is
                # returned as the unit tensor itself, not computed - the
                # default must stay bit-identical to the classic swap (a
                # fp16 a+(b-a) round-trip is not exactly b)
                if abs(s - 1.0) <= EPS:
                    return unit
                return sampled + s * (unit - sampled)

            cond_mark = to.get('cond_mark', None)
            if not (p_active or n_active):
                pass  # every side inert: sampled context, single pass
            elif cond_mark is None:
                # no cond/uncond bookkeeping (e.g. a bare wrapper call): only
                # a total replacement by the positive unit prompt makes sense
                if p_active:
                    unit_r = prepared(self.unit_context)
                    sampled_r = context
                    if sampled_r.shape[1] != unit_r.shape[1]:
                        t_common = sampled_r.shape[1] * unit_r.shape[1] \
                            // math.gcd(sampled_r.shape[1], unit_r.shape[1])
                        rep = lambda t: t if t.shape[1] == t_common else t.repeat(1, t_common // t.shape[1], 1)
                        sampled_r, unit_r = rep(sampled_r), rep(unit_r)
                    if abs(s_dl_p - 1.0) > EPS:
                        base_context = sampled_r
                        delta_scale_vec = s_dl_p
                    context = blend(sampled_r, unit_r, s_emb_p)
            else:
                cond_side = prepared(self.unit_context) if p_active else context
                uncond_side = prepared(self.unit_uncond_context) if n_active else context
                # token counts may differ (77-token chunking); repeat all to
                # their lcm exactly like ConditionCrossAttn.concat does
                t_common = 1
                for t_len in (context.shape[1], cond_side.shape[1], uncond_side.shape[1]):
                    t_common = t_common * t_len // math.gcd(t_common, t_len)
                rep = lambda t: t if t.shape[1] == t_common else t.repeat(1, t_common // t.shape[1], 1)
                ctx_r, cond_r, uncond_r = rep(context), rep(cond_side), rep(uncond_side)
                if p_active:
                    cond_r = blend(ctx_r, cond_r, s_emb_p)
                if n_active:
                    uncond_r = blend(ctx_r, uncond_r, s_emb_n)
                # cond_mark: 0 = cond row, 1 = uncond row (same convention as
                # compute_controlnet_weighting)
                mark = cond_mark.to(device=cond_r.device, dtype=cond_r.dtype).view(-1, 1, 1)
                if (p_active and abs(s_dl_p - 1.0) > EPS) or (n_active and abs(s_dl_n - 1.0) > EPS):
                    # delta scaling engaged: keep the sampled context for the
                    # base pass and build the per-row scale (cond rows scale
                    # by the P delta, uncond rows by the N delta; rows whose
                    # side is inert have with_text == base, so their scale is
                    # multiplied into an exact zero)
                    base_context = ctx_r
                    delta_scale_vec = s_dl_p * (1.0 - cond_mark.float()) + s_dl_n * cond_mark.float()
                context = cond_r * (1.0 - mark) + uncond_r * mark
        y = cond.get('y', None)
        if y is not None:
            y = y.to(dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        controlnet_model_function_wrapper = to.get('controlnet_model_function_wrapper', None)

        def run_control_model(ctx):
            if controlnet_model_function_wrapper is not None:
                wrapper_args = dict(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(),
                                    context=ctx.to(dtype), y=y, control_type=self.control_type)
                wrapper_args['model'] = self
                wrapper_args['inner_model'] = self.control_model
                return controlnet_model_function_wrapper(**wrapper_args)
            return self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint.to(self.device), timesteps=timestep.float(), context=ctx.to(dtype), y=y, control_type=self.control_type)

        control = run_control_model(context)
        if base_context is not None:
            # Delta scaling (unit_delta_scale != 0/1): a second forward with
            # the SAMPLED context isolates the text's exact contribution,
            # which is then scaled on its own - the hint's strength is not
            # touched. This is the only path that runs the control model
            # twice; 0 and 1 were resolved structurally above.
            control_base = run_control_model(base_context)
            if isinstance(delta_scale_vec, torch.Tensor):
                scale = delta_scale_vec.view(-1, 1, 1, 1)
            else:
                scale = delta_scale_vec
            combined = []
            for with_text, base in zip(control, control_base):
                if isinstance(with_text, torch.Tensor) and isinstance(base, torch.Tensor):
                    s = scale.to(device=base.device, dtype=base.dtype) \
                        if isinstance(scale, torch.Tensor) else scale
                    combined.append(base + s * (with_text - base))
                else:
                    combined.append(with_text)
            control = combined
        return self.control_merge(None, control, control_prev, output_dtype)

    def copy(self):
        c = ControlNet(self.control_model, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.predictor

    def cleanup(self):
        self.model_sampling_current = None
        # MUST mirror the host's ControlNet.cleanup. `copy()` builds a FRESH
        # ModelPatcher around the SAME `control_model` for every generation, and
        # `get_models()` hands it to the sampler, which registers a LoadedModel
        # holding weakrefs to both. When this copy is dropped the patcher is the
        # only thing that dies -- `control_model` stays alive in the patcher
        # cache -- so the entry becomes `LoadedModel.is_dead()`: real model
        # present, patcher gone. That entry is never removed (only the real
        # model's finalizer removes entries) and `free_memory` SKIPS dead
        # entries, so those weights can no longer be evicted from VRAM. One more
        # accumulates per generation, each logging "Memory Leak with model
        # ControlNet !" and forcing a gc.collect() on every subsequent load.
        # The getattr guard is the host's too, and it is load-bearing: ControlLora
        # inherits this cleanup but has no wrapper (its control model is built in
        # pre_run and torn down by HostControlLora.cleanup further up the chain).
        if getattr(self, 'control_model_wrapped', None) is not None:
            memory_management.unload_model(self.control_model_wrapped)
        super().cleanup()


class ControlLora(ControlNet, HostControlLora):
    """A ControlLora with CNPro's profiles.

    The two bases split cleanly: `ControlNet` (CNPro's) supplies the
    profile-aware forward pass and `control_merge`, `HostControlLora` supplies
    everything specific to a LoRA-shaped control -- building the control model
    against the running UNet in `pre_run`, the memory estimate, the teardown.
    The MRO puts CNPro's classes first, so `pre_run` runs the host's build AND
    our profile lookups.

    Two members cannot be inherited and are spelled out below; both would be
    silent if got wrong, which is why each says what it is guarding.
    """

    def __init__(self, control_weights, global_average_pooling=False, device=None):
        # NOT inheritable: the host's __init__ calls ITS OWN ControlBase.__init__
        # by name rather than through super(), which would skip every CNPro
        # attribute and leave the profiles unset on a control that looks fine.
        ControlBase.__init__(self, device)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling

    def get_models(self):
        # NOT inheritable: CNPro's ControlNet.get_models appends
        # control_model_wrapped, which a ControlLora has no equivalent of --
        # its control model does not exist until pre_run builds it.
        return HostControlBase.get_models(self)

    def copy(self):
        c = ControlLora(self.control_weights,
                        global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c


class T2IAdapter(ControlBase, HostT2IAdapter):
    """A T2I adapter with CNPro's profiles.

    Nothing here: `ControlBase.get_control` applies the weight profile and then
    defers to the host's forward pass, which ends by calling `self.control_merge`
    -- ours. `copy()` is the one member that has to name the class, and it is
    inherited from the host only because the host's builds ITS OWN T2IAdapter;
    ours has to build ours.

    T2I adapters have no text cross-attention, so none of the per-unit prompt
    machinery on ControlBase applies to them; the patcher declares that with
    `supports_unit_prompt`.
    """

    def __init__(self, t2i_model, channels_in, device=None):
        # NOT inheritable: CNPro's ControlBase precedes the host's T2IAdapter in
        # the MRO, so its one-argument __init__ shadows the host's. Route to it
        # explicitly and then set what the host's constructor would have.
        ControlBase.__init__(self, device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None

    def copy(self):
        c = T2IAdapter(self.t2i_model, self.channels_in)
        self.copy_to(c)
        return c
