"""Per-step control profiles: shared evaluation math.

A profile is a piecewise-linear curve given as a list of (x, y) points sorted
by x, where x is the relative sampling position in [0, 1]. Two kinds exist:

- weight profile: y is control strength (any real, negative = repulsion);
- balance profile: y in [0, 1] is the per-step cond/uncond balance
  (0.5 balanced, 1 = control on cond only, 0 = on uncond only).

The UI serialization ('x@y;x@y;...' with an optional scale suffix and 'Mx@y'
parabola mid controls) is parsed in the ControlNet extension
(lib_cnpro/external_code.py: parse_weight_profile); every consumer below
that layer works on plain point lists and uses this module, so the evaluation
semantics cannot drift between patcher types:

- backend/patcher/controlnet.py (ControlNet / T2I adapter models)
- extensions-builtin/sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py
- extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite.py

The editor (extensions-builtin/sd_forge_controlnet/javascript/weight_profile.js)
mirrors evaluate_weight_profile for drawing; keep the two in sync.
"""


def evaluate_weight_profile(points, x):
    """Evaluate a piecewise-linear weight profile at relative position x in [0, 1].

    `points` is a list of (x, y) pairs sorted by x. The profile extends
    horizontally from the left border to the first point and from the last
    point to the right border.
    """
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            if x1 - x0 <= 0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def build_weight_profile_lookup(weight_profile, percent_to_timestep_function, n=256):
    """Sample the profile on a dense percent grid and pair each sample with its
    timestep (sigma, descending), so per-step strength can be looked up from
    the current timestep during sampling."""
    sigmas = [float(percent_to_timestep_function(i / n)) for i in range(n + 1)]
    strengths = [evaluate_weight_profile(weight_profile, i / n) for i in range(n + 1)]
    return sigmas, strengths


def lookup_weight_profile_strength(sigmas, strengths, sigma):
    """Interpolate the strength for the current sigma from a lookup built by
    build_weight_profile_lookup (sigmas descending)."""
    if sigma >= sigmas[0]:
        return strengths[0]
    if sigma <= sigmas[-1]:
        return strengths[-1]
    lo, hi = 0, len(sigmas) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sigmas[mid] >= sigma:
            lo = mid
        else:
            hi = mid
    if sigmas[lo] - sigmas[hi] <= 0:
        return strengths[hi]
    f = (sigmas[lo] - sigma) / (sigmas[lo] - sigmas[hi])
    return strengths[lo] + (strengths[hi] - strengths[lo]) * f


def drifted_depth(depth, shift):
    """The depth position a depth profile is READ AT under a drift of `shift`.

    THE definition of the drift curve, and the reason it lives here rather than
    at any of its four call sites (the weighting engine, the IP-Adapter and
    ControlLLLite injectors, and - mirrored - the editor's JS). The depth curve
    D and the drift curve S combine as

        effective(step, layer) = main(step) * D(depth(layer) - S(step))

    so a POSITIVE shift reads D further left and therefore moves whatever D
    draws toward the DEEP (coarse) end, and a descending S sweeps the control
    from composition to texture as sampling proceeds. This is the one piece of
    coupling between the step axis and the depth axis: without it main x depth
    is a rank-1 (separable) field and can only express a depth shape frozen in
    time, which is the single thing the three band profiles could say and it
    could not.

    Clamped rather than wrapped: the depth axis has two ENDS, not a period, and
    a wrapped shift would teleport the deepest layer's multiplier onto the
    shallowest. Clamping means a curve driven off the axis holds its edge value,
    which is what the plot's own horizontal extension beyond the first and last
    point already does.
    """
    return min(max(depth - shift, 0.0), 1.0)


def depth_multiplier(depth_profile, drift_profile, depth, x):
    """The depth curve's multiplier for a layer at `depth`, at step position x.

    Neutral 1.0 with no depth profile: the drift alone has nothing to move, so
    it is a no-op by construction rather than by a guard somewhere. A drift with
    no depth curve is therefore harmless everywhere, which is what lets the
    editor keep the two curves independent.
    """
    if not depth_profile:
        return 1.0
    shift = evaluate_weight_profile(drift_profile, x) if drift_profile else 0.0
    return evaluate_weight_profile(depth_profile, drifted_depth(depth, shift))


def build_depth_profile_lookup(depth_profile, drift_profile, depth,
                               percent_to_timestep_function, n=256):
    """Sigma lookup of one SITE's depth multiplier over the step axis.

    The drift makes the depth curve step-dependent, so a site's multiplier is no
    longer the single scalar the injectors used to precompute - it is a curve,
    and this builds it in exactly the shape `lookup_weight_profile_strength`
    already consumes. With no drift every entry is the same number; that is
    deliberately NOT special-cased into a scalar path, because two paths through
    a per-site weight is how one of them silently stops matching the other.
    """
    sigmas = [float(percent_to_timestep_function(i / n)) for i in range(n + 1)]
    values = [depth_multiplier(depth_profile, drift_profile, depth, i / n)
              for i in range(n + 1)]
    return sigmas, values


BANDS = ('coarse', 'mid', 'fine')


def _depth_fraction(position, count):
    """Position within a group as a 0..1 fraction (single-element group = 0.5)."""
    if count <= 1:
        return 0.5
    return min(max(position / (count - 1), 0.0), 1.0)


def depth_fraction_of_residual(k, i, n):
    """Normalized UNet depth of ControlNet residual i in block group k:
    0 = shallowest (fine, texture), 1 = deepest (coarse, composition).

    The UN-QUANTIZED twin of `band_of` (backend/patcher/controlnet.py) and it
    must keep matching it: band_of is this fraction cut into thirds, so a
    depth profile and a band profile can never disagree about where a layer
    sits. The flip is on INPUT here because control_merge stores its input
    residuals back-to-front - the opposite group from
    depth_fraction_of_unet_block, exactly as in the two band functions.
    """
    if k == 'middle':
        return 1.0
    fraction = _depth_fraction(i, n)
    return 1.0 - fraction if k == 'input' else fraction


def depth_fraction_of_unet_block(block_type, block_id, group_ids):
    """Normalized UNet depth of one ATTENTION site, addressed by block id.

    The un-quantized twin of `band_of_unet_block`, with the same `group_ids`
    contract (the split runs over the depth order of the blocks this model
    actually has) and the same direction rule: an ascending INPUT id goes down
    the U, an ascending OUTPUT id comes back up it.
    """
    if block_type == 'middle':
        return 1.0
    ids = sorted(set(group_ids))
    if not ids:
        return 0.5
    position = ids.index(block_id) if block_id in ids else 0
    fraction = _depth_fraction(position, len(ids))
    return 1.0 - fraction if block_type == 'output' else fraction


def depth_fraction_of_ordered_site(position, span):
    """Normalized model depth of an injection site addressed by ABSOLUTE position.

    For a DiT there is no U: blocks run shallow -> deep in one straight line, so
    a site's depth is simply where its block sits in the stack. `position` is
    the block index the site injects at and `span` is the model's block count,
    giving 0 at the first block and (span-1)/span-ish at the last.

    Deliberately NOT expressed through `_depth_fraction`: that helper normalizes
    a position within the GROUP OF SITES (i / (count-1)), which would stretch six
    injection points across the full 0..1 range and claim the last one sits at
    the bottom of the model. Z-Image's Fun-ControlNet injects at blocks
    [0, 5, 10, 15, 20, 25] of 30 - the deepest injection is at 0.86, not at 1.0,
    and a depth curve drawn by the user must be read against the MODEL's depth
    axis, not against the sites' own ordinal.
    """
    if span <= 1:
        return 0.5
    return min(max(position / span, 0.0), 1.0)


def band_of_depth_fraction(fraction):
    """Band ('coarse'/'mid'/'fine') of a normalized 0..1 model depth.

    The quantizer for depth-fraction-addressed sites, so band and depth agree BY
    CONSTRUCTION rather than by the two functions being kept in sync by hand
    (which is what `band_of` / `depth_fraction_of_residual` have to do below).

    Not used to re-express `band_of` or `band_of_unet_block`: those quantize an
    ORDINAL (`position * 3 // n`), and routing them through a fraction would move
    boundary layers between bands on some UNet shapes - a silent behaviour change
    to every existing SD1.x/SDXL profile. Two quantizers, each exact for its own
    addressing scheme, is the correct answer here; one "unified" quantizer would
    be wrong for one of them.

    SEMANTIC WARNING (DiT hosts): in a UNet the bands are RESOLUTION tiers, because
    the encoder downsamples - coarse really is composition and fine really is
    texture. A DiT runs every block at one token resolution, so this axis measures
    ABSTRACTION, not spatial frequency. The curve is meaningful and monotone; the
    coarse/mid/fine labels are a UNet-era presentation of it. See ARCHITECTURE.md.
    """
    return BANDS[2 - min(int(max(fraction, 0.0) * 3), 2)]


def band_of_unet_block(block_type, block_id, group_ids):
    """Band ('coarse'/'mid'/'fine') of one UNet attention site.

    This is the attention-layer face of the same rule `band_of` applies to
    ControlNet residuals (backend/patcher/controlnet.py); both mean "how deep
    in the UNet does this control act", i.e. coarse = deepest / lowest
    resolution = composition, fine = shallowest / highest resolution =
    texture. Patchers that inject through attention instead of through skip
    residuals (IP-Adapter, ControlLLLite) address their sites by UNet block,
    so they resolve the band here rather than by residual index.

    `group_ids` is every block id patched in that group, so the split is
    thirds of the DEPTH ORDER of what this model actually has - the number of
    attention blocks differs between SD1.x and SDXL (SDXL has none at the
    highest resolution at all), and a fixed id table would silently mean
    different things per architecture.

    Direction is the trap here, and it is opposite per group: an ascending
    INPUT id goes down the U (shallow -> deep, fine -> coarse) while an
    ascending OUTPUT id comes back up it (deep -> shallow, coarse -> fine).
    `band_of` looks reversed against this only because control_merge stores
    its input residuals back-to-front; both end up saying the same thing about
    the same layer.

    Exact on SD1.x, where the six input / nine output attention blocks fall
    into three resolutions three ways. On SDXL the split cannot be
    resolution-exact: attention exists at two resolutions only and the two
    sides hold different numbers of blocks, so one resolution group is always
    cut in half (measured: input 7 -> mid but input 8 -> coarse, both 1/4).
    The result stays monotone in depth, which is what the bands promise -
    do not "fix" it with a per-architecture id table, that is the thing this
    signature exists to avoid.
    """
    if block_type == 'middle':
        return 'coarse'          # the bottom of the U is always coarse
    ids = sorted(set(group_ids))
    n = len(ids)
    if n == 0:
        return 'mid'
    position = ids.index(block_id) if block_id in ids else 0
    third = min(position * 3 // n, 2)
    if block_type == 'output':
        third = 2 - third
    return ('fine', 'mid', 'coarse')[third]


def balance_factors(balance):
    """(cond, uncond) scale factors for a balance value, clamped to [0, 1].

    0.5 -> (1, 1) applies control to cond and uncond equally (balanced);
    1 -> (1, 0) control on cond only (control matters most); 0 -> (0, 1)
    control on uncond only (prompt matters most); linear in between. Single
    source of the formula for every patcher type that honors balance.
    """
    b = min(max(balance, 0.0), 1.0)
    return min(2.0 * b, 1.0), min(2.0 * (1.0 - b), 1.0)
