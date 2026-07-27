"""The weighting engine: per-step, per-depth, per-band and per-region scaling
of whatever residuals a control model produced.

Extracted from ``controlnet_impl.py`` so it can be shared. It is the piece that
is genuinely family-agnostic: everything below is index math over lists of
residual tensors plus a lookup against the current sigma, and NONE of it knows
whether those residuals came from a UNet ControlNet, a DiT ControlNet, or
something that does not exist yet. The three things that ARE family-specific
(tensor rank, mask projection, depth of site i) are delegated to a layout
object -- see ``residual_layout.py``.

``controlnet_impl`` re-exports ``compute_controlnet_weighting`` and the UNet
``band_of`` so its own call sites and imports are unchanged.

THE INPUT SURFACE IS DECLARED ONCE
----------------------------------
``WEIGHTING_INPUTS`` below is the single list of what this engine reads off a
control object. The read, the "is anything configured at all" early-out and the
documentation all derive from that one tuple, so a newly added profile cannot be
wired into some of them and silently missed by the others. The previous version
spelled the list out three separate times -- once as locals, once as a boolean
chain, once in the docstring -- which is exactly the shape of bug AGENTS.md
section 4 is about: a curve the user had drawn would simply do nothing, with no
error anywhere.

These names are CNPro's own on purpose. The host's control objects carry a
``positive_advanced_weighting`` / ``advanced_mask_weighting`` surface of the same
shape, read by the host's own weighting pass; naming ours the same made the two
indistinguishable on one object, so anything that let the host's pass run would
have applied every weight twice with no visible symptom beyond wrong pixels.
Distinct names make that collision impossible to write.

WHAT A CALLER MAY SET ON `cnet` (all optional, all default to "no effect"):
    cond_layer_weights / uncond_layer_weights
        dict group -> per-injection-site multiplier list, applied to the cond
        and uncond rows of the batch respectively.
    frame_weights       list, one multiplier per image in the batch.
    sigma_weight_fn     callable(sigmas) -> per-row multiplier for this step.
    region_masks        a [B,1,H,W] tensor applied to every site, or a dict
                        band -> tensor. Under the dict form a band with no mask
                        of its own injects NOTHING (restrict-to-painted).
    balance_sigmas + balance_values      (from a balance profile)
    band_profile_lookup                  dict band -> (sigmas, strengths)
    depth_profile                        point list over normalized depth
    drift_sigmas + drift_values          (from a depth-drift profile; shifts
                                          where `depth_profile` is read, so it
                                          does nothing without one)
    transformer_options                  (set per step by the host sampler)
    residual_layout                      (defaults to the UNet layout)
"""

from __future__ import annotations

import torch

from cnpro_core.weight_profile import (
    balance_factors,
    drifted_depth,
    evaluate_weight_profile,
    lookup_weight_profile_strength,
)

from .residual_layout import UNET_LAYOUT


#: Every per-run input the engine reads off a control object. Order is only for
#: readability; membership is what matters. See the module docstring for why this
#: is a single declaration rather than a list repeated per use site.
WEIGHTING_INPUTS = (
    'cond_layer_weights',
    'uncond_layer_weights',
    'frame_weights',
    'sigma_weight_fn',
    'region_masks',
    'balance_sigmas',
    'band_profile_lookup',
    'depth_profile',
    'drift_sigmas',
)


def site_weight(weights, group, index):
    """One injection site's multiplier out of a {group: [per-site, ...]} dict.

    Neutral 1.0 wherever nothing is configured -- an absent group, a short list,
    or no dict at all. Sites are always addressed by their position in the
    group's residual list, so there is no negative index to defend against.
    """
    row = weights.get(group) if weights else None
    if not row or index >= len(row):
        return 1.0
    return row[index]


def layout_of(cnet):
    """The residual layout `cnet` weights through, defaulting to the UNet one.

    A patcher opts into a different geometry by setting ``residual_layout`` on
    its control object; everything else in this module stays the same.
    """
    return getattr(cnet, 'residual_layout', None) or UNET_LAYOUT


def band_of(k, i, n):
    """Band name of UNet injection layer i in block group k.

    Thin wrapper on the UNet layout, kept as a module-level function because
    that is how ``controlnet_impl`` and MAINTENANCE.md have always named it.
    """
    return UNET_LAYOUT.band(k, i, n)


def resolve_band_mask(masks, layout, k, i, n):
    """Pick the per-band mask for injection layer i of block group k.

    Returns None when the layer's band has no mask -- under restrict-to-painted
    semantics the caller then zeroes that layer's injection entirely (a band
    without its own mask contributes no control).
    """
    return masks.get(layout.band(k, i, n))


def layer_mask_for(cnet, mask, control_signal, layout=None):
    """The painted mask projected onto one injection layer, cached for the run.

    Delegates the projection to the layout: 4D residuals take a bilinear resize
    to (H, W), token residuals take a resize onto the image span of the sequence.
    The cache lives on the per-run control object, so it cannot leak across
    generations; ``cleanup()`` drops it as well.
    """
    cache = getattr(cnet, '_layer_mask_cache', None)
    if cache is None:
        cache = cnet._layer_mask_cache = {}
    return (layout or layout_of(cnet)).project_mask(control_signal, mask, cache)


def _inert(value):
    """True when this input cannot change any residual.

    An empty container counts as inert as well as None: ``region_masks={}`` and
    ``band_profile_lookup={}`` both reach here from callers that build a dict
    and find nothing to put in it, and an empty per-group weight dict looks up
    to the neutral 1.0 for every site anyway.
    """
    if value is None:
        return True
    return isinstance(value, (dict, list, tuple)) and len(value) == 0


def _gather(cnet):
    """Read the weighting surface off `cnet`.

    Returns a dict with every inert input normalized to None, or None when
    nothing at all is configured -- the caller then returns the residuals
    untouched without so much as reading `transformer_options` (which a bare
    control object may not have populated yet).
    """
    inputs = {name: getattr(cnet, name, None) for name in WEIGHTING_INPUTS}
    if all(_inert(v) for v in inputs.values()):
        return None
    return {name: (None if _inert(v) else v) for name, v in inputs.items()}


def _step_factor(cfg, cnet, to, cond_mark):
    """The multiplier that varies per batch row but not per injection site.

    Computed once per step instead of once per site per step, which is what it
    always was mathematically -- the old loop recomputed the same product for
    every residual in the model.
    """
    factor = None

    if cfg['sigma_weight_fn'] is not None:
        sigmas = to['sigmas']
        factor = torch.cat([cfg['sigma_weight_fn'](sigmas)] * len(to['cond_or_uncond']))

    if cfg['frame_weights'] is not None:
        frames = torch.Tensor(cfg['frame_weights'] * len(to['cond_or_uncond'])).to(to['sigmas'])
        if frames.shape[0] != cond_mark.shape[0]:
            # was an `assert`, which vanishes under `python -O` and would have
            # turned this into a broadcasting accident rather than an error
            raise ValueError(
                'frame_weights has %d entries after expansion but the batch has %d rows'
                % (frames.shape[0], cond_mark.shape[0]))
        factor = frames if factor is None else factor * frames

    if cfg['balance_sigmas'] is not None:
        # per-step cond/uncond balance from the balance profile: 0.5 balanced,
        # 1 -> control on cond only, 0 -> control on uncond only
        b = lookup_weight_profile_strength(
            cfg['balance_sigmas'], cnet.balance_values, float(to['sigmas'][0]))
        balance_cond, balance_uncond = balance_factors(b)
        balance = balance_cond * (1.0 - cond_mark) + balance_uncond * cond_mark
        factor = balance if factor is None else factor * balance

    return factor


def _drift_shift(cfg, cnet, sigma_now):
    """This step's shift along the depth axis, or 0.0 when no drift is set.

    Resolved ONCE per step rather than once per site: the shift is a property of
    the step, and the binary search behind it would otherwise run for every
    injection layer in the model on every step to return the same number.
    """
    if cfg['drift_sigmas'] is None:
        return 0.0
    return lookup_weight_profile_strength(
        cfg['drift_sigmas'], cnet.drift_values, sigma_now)


def _site_factor(cfg, layout, sigma_now, drift, group, index, count):
    """The multiplier that varies per injection site but not per batch row.

    Band and depth are the same depth axis -- quantized to three buckets with a
    time curve each, or continuous. The UI keeps them mutually exclusive
    (combining them would count depth twice), but nothing here depends on that:
    they simply multiply.
    """
    factor = 1.0

    if cfg['band_profile_lookup'] is not None:
        band = layout.band(group, index, count)
        lut = cfg['band_profile_lookup'].get(band)
        if lut is not None:
            factor *= lookup_weight_profile_strength(lut[0], lut[1], sigma_now)
        # a band with no profile of its own keeps its neutral 1.0

    if cfg['depth_profile'] is not None:
        # per-layer multiplier on whatever per-step strength is already in
        # force. Without a drift this is the separable product
        # strength(step) * depth(layer); the drift moves WHERE the depth curve
        # is read as sampling proceeds, which is the only thing that couples the
        # two axes (see cnpro_core.weight_profile.drifted_depth). `drift` is
        # 0.0 when none is set, and drifted_depth is then the identity.
        factor *= evaluate_weight_profile(
            cfg['depth_profile'],
            drifted_depth(layout.depth_fraction(group, index, count), drift))

    return factor


def compute_controlnet_weighting(control, cnet):
    """Apply every configured profile to `control` in place, and return it.

    `control` is ``{group_name: [residual, ...]}``. Group names are the caller's
    business: UNet controls use 'input'/'middle'/'output', the Z-Image control
    uses a single 'block' group. Only the layout interprets them.
    """
    cfg = _gather(cnet)
    if cfg is None:
        return control

    layout = layout_of(cnet)
    to = cnet.transformer_options
    cond_mark = to['cond_mark']
    sigma_now = float(to['sigmas'][0])

    step_factor = _step_factor(cfg, cnet, to, cond_mark)
    drift = _drift_shift(cfg, cnet, sigma_now)
    cond_weights = cfg['cond_layer_weights'] or {}
    uncond_weights = cfg['uncond_layer_weights'] or {}
    masks = cfg['region_masks']

    for group, residuals in control.items():
        count = len(residuals)

        for index in range(count):
            signal = residuals[index]
            if not isinstance(signal, torch.Tensor):
                continue

            mask = masks
            if isinstance(masks, dict):
                mask = resolve_band_mask(masks, layout, group, index, count)
                if mask is None:
                    # restrict-to-painted semantics: when band masks are in use,
                    # a band without its own mask gets ZERO control
                    control[group][index] = signal * 0.0
                    continue

            # cond rows take the positive per-site weight, uncond rows the
            # negative one; cond_mark is 0 on cond rows and 1 on uncond rows.
            # Always a tensor -- the layouts' broadcast_weight indexes it.
            weight = (site_weight(cond_weights, group, index) * (1.0 - cond_mark)
                      + site_weight(uncond_weights, group, index) * cond_mark)
            if step_factor is not None:
                weight = weight * step_factor
            weight = weight * _site_factor(cfg, layout, sigma_now, drift,
                                           group, index, count)

            if isinstance(mask, torch.Tensor):
                signal = signal * layer_mask_for(cnet, mask, signal, layout)

            control[group][index] = signal * layout.broadcast_weight(signal, weight)

    return control
