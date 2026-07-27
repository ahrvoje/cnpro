from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import List, Optional, Tuple, Union, Dict, TypedDict
import numpy as np
from lib_cnpro.logging import logger
from lib_cnpro.enums import HiResFixOption
from modules.api import api
# single source of the piecewise-linear evaluation (see that module's
# docstring); this file must never re-implement the interpolation
from cnpro_core.weight_profile import evaluate_weight_profile


class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"


class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"


resize_mode_aliases = {
    'Inner Fit (Scale to Fit)': 'Crop and Resize',
    'Outer Fit (Shrink to Fit)': 'Resize and Fill',
    'Scale to Fit (Inner Fit)': 'Crop and Resize',
    'Envelope (Outer Fit)': 'Resize and Fill',
}


def resize_mode_from_value(value: Union[str, int, ResizeMode]) -> ResizeMode:
    if isinstance(value, str):
        return ResizeMode(resize_mode_aliases.get(value, value))
    elif isinstance(value, int):
        assert value >= 0
        if value == 3:  # 'Just Resize (Latent upscale)'
            return ResizeMode.RESIZE

        if value >= len(ResizeMode):
            logger.warning(f'Unrecognized ResizeMode int value {value}. Fall back to RESIZE.')
            return ResizeMode.RESIZE

        return [e for e in ResizeMode][value]
    else:
        return value


def control_mode_from_value(value: Union[str, int, ControlMode]) -> ControlMode:
    if isinstance(value, str):
        return ControlMode(value)
    elif isinstance(value, int):
        return [e for e in ControlMode][value]
    else:
        return value


def visualize_inpaint_mask(img):
    if img.ndim == 3 and img.shape[2] == 4:
        result = img.copy()
        mask = result[:, :, 3]
        mask = 255 - mask // 2
        result[:, :, 3] = mask
        return np.ascontiguousarray(result.copy())
    return img


def pixel_perfect_resolution(
        image: np.ndarray,
        target_H: int,
        target_W: int,
        resize_mode: ResizeMode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT:
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    logger.debug(f"Pixel Perfect Computation:")
    logger.debug(f"resize_mode = {resize_mode}")
    logger.debug(f"raw_H = {raw_H}")
    logger.debug(f"raw_W = {raw_W}")
    logger.debug(f"target_H = {target_H}")
    logger.debug(f"target_W = {target_W}")
    logger.debug(f"estimation = {estimation}")

    return int(np.round(estimation))


PROFILE_SCALE_MIN = -1.0
PROFILE_SCALE_MAX = 2.0


def _finite(value) -> float:
    """float() that rejects nan/inf.

    float('nan') raises nothing, min/max propagate it, and abs(nan) < eps is
    False - so without this guard a hand-edited 'nan@1' infotext (or API
    string) sails through parsing into per-step strengths and silently
    corrupts the whole image. The editor rejects non-finite input; the python
    side must match it.
    """
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"non-finite profile value: {value!r}")
    return f


def _flatten_profile_mids(points, mids, samples=24):
    """Replace segments that carry an active mid control with a dense polyline.

    A mid control (editor: green segment midpoint, serialized as 'Mx@y') turns
    its segment into the parabola (Lagrange quadratic) through the two segment
    vertices and the control; the curve is sampled here and clamped to [0, 1]
    to match the editor's drawing (javascript/weight_profile.js; the `samples`
    default equals MID_CURVE_STEPS there), so callers keep receiving plain
    piecewise-linear points. `points` must be sorted by x and normalized
    (scale not yet applied). Orphaned mids are ignored.
    """
    out = []
    for i, (x0, y0) in enumerate(points):
        out.append((x0, y0))
        if i + 1 >= len(points):
            break
        x1, y1 = points[i + 1]
        if x1 - x0 <= 0:
            continue
        mid = next((m for m in mids if x0 < m[0] < x1), None)
        if mid is None:
            continue
        xm, ym = mid
        for x in sorted({x0 + (x1 - x0) * k / samples for k in range(1, samples)} | {xm}):
            l0 = (x - xm) * (x - x1) / ((x0 - xm) * (x0 - x1))
            lm = (x - x0) * (x - x1) / ((xm - x0) * (xm - x1))
            l1 = (x - x0) * (x - xm) / ((x1 - x0) * (x1 - xm))
            y = y0 * l0 + ym * lm + y1 * l1
            out.append((x, min(max(y, 0.0), 1.0)))
    return out


PROFILE_COSINE_MAX_OSCILLATIONS = 4.0
PROFILE_COSINE_SAMPLES = 512
PROFILE_GAMMA_MAX = 10.0


def _apply_profile_gamma(points, exponent, samples=PROFILE_COSINE_SAMPLES):
    """Bend a normalized profile with y -> y**exponent, as a dense polyline.

    Runs on the NORMALIZED [0, 1] values, BEFORE the scale mapping - that is
    the editor's contract (the response slider is independent of the range
    selects; the range maps the already-bent curve). The power of a piecewise-
    linear function is not piecewise-linear, so the curve is resampled densely
    exactly like _apply_profile_cosine; running after it on already-dense
    points just re-samples them. Exponent 1 is a structural no-op (the caller
    skips this entirely). Matches the editor's gammaAt (weight_profile.js).
    """
    out = []
    for i in range(samples + 1):
        x = i / samples
        y = min(max(evaluate_weight_profile(points, x), 0.0), 1.0)
        out.append((x, y ** exponent))
    return out


def _apply_profile_cosine(points, oscillations, phase, samples=PROFILE_COSINE_SAMPLES):
    """Modulate a profile polyline with a cosine, returning a dense polyline.

    In cosine mode the drawn profile is the ENVELOPE of the wave, not the wave:
    the value at x is envelope(x) * (0.5 + 0.5 * cos(2*pi*n*x - phase)), which
    stays within [0, envelope] - the wave pulses the control on and off rather
    than flipping its sign. The phase is SUBTRACTED so that moving the editor's
    parameter point right shifts the wave right.
    Sampled here for the same reason mid controls are
    flattened: every consumer keeps receiving plain piecewise-linear points
    (see _flatten_profile_mids). The editor mirrors this in
    javascript/weight_profile.js (cosFactor / evaluate); the envelope is the
    shared evaluate_weight_profile - do not re-implement it here.
    """
    if not points:
        return points

    out = []
    for i in range(samples + 1):
        x = i / samples
        factor = 0.5 + 0.5 * math.cos(2.0 * math.pi * oscillations * x - phase)
        out.append((x, evaluate_weight_profile(points, x) * factor))
    return out


def parse_weight_profile(profile, phase_offset=0.0) -> Optional[List[Tuple[float, float]]]:
    """Parse a control weight profile into a list of (x, y) points sorted by x.

    Accepts the UI/infotext string format 'x@y;x@y;...' with an optional scale
    suffix, or a list of [x, y] pairs (API). x is relative sampling position in
    [0, 1]; for strings y is in [0, 1] before scaling. An 'Mx@y' token marks an
    active segment mid control: its segment is evaluated as the parabola through
    the two vertices and the control, flattened here into a dense clamped
    polyline (see _flatten_profile_mids). The scale suffix maps the normalized
    profile onto an output range: '|hi' maps [0, 1] -> [0, hi] (legacy single
    scale) and '|lo~hi' maps [0, 1] -> [lo, hi], both clamped to [-1, 2], so
    callers always get calculation-ready weights (possibly negative).
    List input has no scale channel, so its y values are taken as
    calculation-ready weights directly and clamped to [-1, 2] - an API caller
    can express weight > 1 without learning the string grammar.
    Returns None if the profile is empty or invalid.

    `phase_offset` (radians) is added to the cosine phase when the profile is
    in cosine mode and ignored otherwise - the multi-phase feature ('P'
    marker, see weight_profile_is_multiphase) parses one variant per Input
    with offsets k * 2*pi/n. The bare 'P' token itself is tolerated and
    skipped here: whether to fan out is the CALLER's decision, so plain
    parses of a multi-phase string keep returning input 1's profile.

    A 'G<e>' token (editor: the vertical response slider in the range column,
    e in [0.1, 10]) bends the normalized profile with y -> y**e after mids
    and wave, before the scale mapping (_apply_profile_gamma).
    """
    if profile is None:
        return None
    is_string = isinstance(profile, str)
    points = []
    mids = []
    cosine = None
    gamma = None
    lo, hi = 0.0, 1.0
    try:
        if is_string:
            # per-band segments ('#C...#M...#F...', see parse_band_profiles)
            # ride in the same string; the main profile is everything before
            # the first '#', so every existing consumer stays band-agnostic
            profile = profile.split('#', 1)[0]
            if not profile.strip():
                return None
            parts = profile.split('|')
            if len(parts) > 1:
                clamp = lambda v: min(max(_finite(v), PROFILE_SCALE_MIN), PROFILE_SCALE_MAX)
                bounds = parts[1].split('~')
                if len(bounds) > 1:
                    lo, hi = clamp(bounds[0]), clamp(bounds[1])
                else:
                    lo, hi = 0.0, clamp(bounds[0])
                if lo > hi:
                    lo, hi = hi, lo
            pairs = []
            for pair in parts[0].split(';'):
                token = pair.strip()
                if not token:
                    continue
                if token == 'P':
                    # multi-phase marker, handled by the caller (see docstring)
                    continue
                if token.startswith('M'):
                    xm, ym = token[1:].split('@')
                    mids.append((min(max(_finite(xm), 0.0), 1.0), min(max(_finite(ym), 0.0), 1.0)))
                elif token.startswith('C'):
                    # cosine mode: 'C<oscillations>@<phase>', the drawn points
                    # are the envelope of that wave
                    n, ph = token[1:].split('@')
                    cosine = (
                        min(max(_finite(n), 0.0), PROFILE_COSINE_MAX_OSCILLATIONS),
                        _finite(ph) % (2.0 * math.pi),
                    )
                elif token.startswith('G'):
                    # response exponent 'G<e>': the normalized profile is bent
                    # with y -> y**e before the scale mapping
                    gamma = min(max(_finite(token[1:]), 1.0 / PROFILE_GAMMA_MAX),
                                PROFILE_GAMMA_MAX)
                else:
                    pairs.append(token.split('@'))
        else:
            pairs = list(profile)
        # string y is normalized (scale applied below); list y is already
        # calculation-ready and only clamped to the representable range
        y_lo, y_hi = (0.0, 1.0) if is_string else (PROFILE_SCALE_MIN, PROFILE_SCALE_MAX)
        for pair in pairs:
            x, y = pair
            points.append((min(max(_finite(x), 0.0), 1.0), min(max(_finite(y), y_lo), y_hi)))
    except (TypeError, ValueError):
        logger.warning(f"Invalid control weight profile: {profile}")
        return None
    if not points:
        return None
    points.sort(key=lambda point: point[0])
    if mids:
        points = _flatten_profile_mids(points, mids)
    if cosine is not None:
        # after the mids, so the wave rides the envelope the editor draws
        points = _apply_profile_cosine(points, cosine[0], cosine[1] + phase_offset)
    if gamma is not None and abs(gamma - 1.0) > 1e-4:
        # after the wave (the bend applies to the effective normalized value,
        # waves included), before the scale mapping below
        points = _apply_profile_gamma(points, gamma)
    if lo != 0.0 or hi != 1.0:
        points = [(x, lo + y * (hi - lo)) for x, y in points]
    return points


def weight_profile_is_multiphase(profile) -> bool:
    """True when the MAIN weight profile carries the multi-phase marker.

    A bare 'P' token in the main segment (editor: the multi preset toggle)
    means "with several Inputs, run one phase-shifted variant of this cosine
    profile per input" - the fan-out itself happens in scripts/controlnet.py
    via parse_weight_profile(phase_offset=...). Only the main profile can be
    multi-phase; band segments are not inspected. Only meaningful alongside a
    cosine token ('C...'), which the editor guarantees; without one the
    variants would all be identical anyway (phase_offset is ignored).
    """
    if not isinstance(profile, str):
        return False
    main = profile.split('#', 1)[0].split('|', 1)[0]
    return any(token.strip() == 'P' for token in main.split(';'))


def serialize_weight_profile(points) -> str:
    """Serialize profile points to the 'x@y;x@y;...' format.

    The format intentionally avoids ',' and ':' so that the profile can be
    embedded in infotext.
    """
    return ";".join(f"{round(x, 4):g}@{round(y, 4):g}" for x, y in points)


def weight_profile_support(points) -> Tuple[float, float]:
    """Relative step range (start, end) outside which the profile is zero.

    Negative weights (from a scale range reaching below 0) count as active
    control, so 'nonzero' is what matters, not 'positive'.
    """
    eps = 1e-9
    active = lambda y: abs(y) > eps
    if not any(active(y) for _, y in points):
        return 0.0, 0.0
    if active(points[0][1]):
        start = 0.0
    else:
        first_active = next(i for i, (_, y) in enumerate(points) if active(y))
        start = points[first_active - 1][0]
    if active(points[-1][1]):
        end = 1.0
    else:
        last_active = next(i for i in reversed(range(len(points))) if active(points[i][1]))
        end = points[last_active + 1][0]
    return start, end


PROFILE_BAND_PREFIXES = {'C': 'coarse', 'M': 'mid', 'F': 'fine'}
PROFILE_BAND_MODE_PREFIX = 'B'
PROFILE_DEPTH_PREFIX = 'D'


def band_mode_active(profile) -> bool:
    """True when the packed profile string says the BANDS drive the weights.

    Main and bands are exclusive and the switch is the editor's band selector,
    not any property of the curves: whichever selector is pressed is what the
    unit runs on, so a band left at flat 1 in band mode means "this band does
    nothing", not "fall back to the main profile". The editor writes the
    pressed selector as a '#B<band>' segment (absent = main); both profile
    parsers ignore segments they do not know, so the marker rides along
    without touching the rest of the grammar.
    """
    if not isinstance(profile, str) or '#' not in profile:
        return False
    return any(segment[:1] == PROFILE_BAND_MODE_PREFIX
               for segment in profile.split('#')[1:])


def masks_in_force(global_mask, band_masks, band_selected: bool):
    """Which painted weight masks a unit runs on: (global, {band: mask}).

    THE PROFILE SELECTOR DECIDES, not which masks happen to be painted. The
    four mask slots are the four profiles' spatial half: the G slot belongs to
    the MAIN profile and the C/M/F slots to the band profiles, so whichever
    selector is pressed picks the mask(s) with it. Main (or depth, which
    multiplies main) runs on G alone; a band selection runs on C/M/F alone.

    Before this, the masks had their own precedence - "a painted global mask
    governs everything, otherwise any painted band does" - which is a SECOND
    switch for one decision. It could disagree with the first: with a band
    selector pressed and an old global mask still painted, the unit ran band
    profiles through a global mask, and nothing on screen said so. Same class
    of bug as computing weights from a curve the user is not looking at
    (MAINTENANCE.md invariant 21) - here, masking with a mask they are not
    looking at.

    Returns the masks in force; the ones this mode does not use are dropped,
    not merged. Restrict-to-painted still applies WITHIN the chosen mode: in
    band mode a band with no mask contributes nothing.
    """
    if band_selected:
        return None, dict(band_masks or {})
    return global_mask, {}


def band_points_are_neutral(points) -> bool:
    """True when parsed band-profile points are flat 1.0, i.e. a no-op.

    Band profiles are per-step MULTIPLIERS of their band's injection, so the
    neutral element is 1 (unlike the balance profile's 0.5). Neutral bands are
    neither serialized by the editor nor forwarded to the patchers.
    """
    return points is None or all(abs(y - 1.0) <= 5e-4 for _, y in points)


def parse_band_profiles(profile) -> Optional[dict]:
    """Per-band (coarse/mid/fine) step profiles from a packed profile string.

    The UI packs them after the main weight profile as
    'main#C<profile>#M<profile>#F<profile>' where each <profile> uses the
    standard 'x@y;...' grammar (scale suffix, mids and cosine tokens
    included). Only non-neutral bands are present. Returns a dict
    band-name -> calculation-ready point list, or None when the string
    carries no (non-neutral) band profiles. The values multiply the band's
    injection layers per step (backend band_of mapping).
    """
    if not isinstance(profile, str) or '#' not in profile:
        return None
    bands = {}
    for segment in profile.split('#')[1:]:
        segment = segment.strip()
        if not segment or segment[0] not in PROFILE_BAND_PREFIXES:
            continue
        points = parse_weight_profile(segment[1:])
        if points is None or band_points_are_neutral(points):
            continue
        bands[PROFILE_BAND_PREFIXES[segment[0]]] = points
    return bands or None


def parse_depth_profile(profile) -> Optional[List[Tuple[float, float]]]:
    """Depth profile from a packed profile string ('#D<profile>' segment).

    X is normalized UNet DEPTH (0 = shallowest injection layer = fine/texture,
    1 = deepest = coarse/composition) and y a per-layer MULTIPLIER, so the
    neutral element is 1 exactly like a band's - a flat-1 curve is a no-op and
    is neither serialized by the editor nor forwarded here.

    Unlike the band profiles this one does NOT replace the main profile: it
    has no step dimension, so it multiplies whatever per-step strength the
    main profile produces (effective = main(step) * depth(layer)). The two
    depth objects - bands and this - stay mutually exclusive in the editor,
    because a per-bucket curve times a per-depth curve would count depth twice
    and no drawn value could be read literally any more.
    """
    if not isinstance(profile, str) or '#' not in profile:
        return None
    for segment in profile.split('#')[1:]:
        segment = segment.strip()
        if not segment or segment[0] != PROFILE_DEPTH_PREFIX:
            continue
        points = parse_weight_profile(segment[1:])
        if points is None or band_points_are_neutral(points):
            return None
        return points
    return None


def balance_points_are_neutral(points) -> bool:
    """True when parsed balance-profile points are flat 0.5, i.e. a no-op.

    Single source of the neutrality test: a neutral profile must neither be
    forwarded to the patchers (needless per-layer weighting every step) nor
    written into infotext (meaningless token in every image).
    """
    return points is None or all(abs(y - 0.5) <= 5e-4 for _, y in points)


def weight_profile_from_scalars(weight, guidance_start=0.0, guidance_end=1.0) -> str:
    """Build a profile string approximating legacy weight + timestep range controls.

    Profile points are normalized to [0, 1], so a legacy weight above 1 (the
    old slider reached 2) is carried by the '|hi' scale suffix instead of
    being clamped away.
    """
    w = min(max(float(weight), 0.0), PROFILE_SCALE_MAX)
    s = min(max(float(guidance_start), 0.0), 1.0)
    e = min(max(float(guidance_end), 0.0), 1.0)
    if e <= s:
        # degenerate legacy window: control active nowhere
        return "0@0;1@0"
    scale = w if w > 1.0 else 1.0
    y = w / scale
    # a fixed ramp inverts the profile when the window is narrower than
    # 2*ramp (the ramped inner points sort OUTSIDE the window bounds), so it
    # shrinks with the window
    ramp = min(0.001, (e - s) / 2.0)
    points = []
    if s > 0.0:
        points.append((s, 0.0))
    points.append((min(s + ramp, 1.0) if s > 0.0 else 0.0, y))
    points.append((max(e - ramp, 0.0) if e < 1.0 else 1.0, y))
    if e < 1.0:
        points.append((e, 0.0))
    points.sort(key=lambda point: point[0])
    profile = serialize_weight_profile(points)
    if scale != 1.0:
        profile += f"|{round(scale, 4):g}"
    return profile


class GradioImageMaskPair(TypedDict):
    """Represents the dict object from Gradio's image component if `tool="sketch"`
    is specified.
    {
        "image": np.ndarray,
        "mask": np.ndarray,
    }
    """
    image: np.ndarray
    mask: np.ndarray


# Input slots a single unit can hold. Gradio cannot create components after the
# page is built, so every slot is pre-rendered (hidden until the "+" tab opens
# it) and every slot needs its own dataclass field - which is why this is a
# constant and not a setting. Raising it means adding image_<n>/image_<n>_fg
# (and image_<n>_enabled) fields below AND the matching entries in
# controlnet_ui_group.unit_fields. Page-build cost scales with it: every slot
# is a full ForgeCanvas + 4 mask channels + a strip channel, per unit, per
# tab.
MAX_INPUT_IMAGES = 5


@dataclass
class ControlNetUnit:
    use_preview_as_input: bool = False
    generated_image: Optional[np.ndarray] = None
    mask_image: Optional[GradioImageMaskPair] = None
    mask_image_fg: Optional[GradioImageMaskPair] = None
    hr_option: Union[HiResFixOption, int, str] = HiResFixOption.BOTH
    enabled: bool = True
    module: str = "None"
    model: str = "None"
    weight: float = 1.0
    image: Optional[GradioImageMaskPair] = None
    image_fg: Optional[GradioImageMaskPair] = None
    # Input slots 2..MAX_INPUT_IMAGES. Each filled slot is preprocessed on its
    # own and becomes its own control; their residuals are summed into the
    # unit's single output (see ControlNetUnit.input_images).
    image_2: Optional[GradioImageMaskPair] = None
    image_2_fg: Optional[GradioImageMaskPair] = None
    image_3: Optional[GradioImageMaskPair] = None
    image_3_fg: Optional[GradioImageMaskPair] = None
    image_4: Optional[GradioImageMaskPair] = None
    image_4_fg: Optional[GradioImageMaskPair] = None
    image_5: Optional[GradioImageMaskPair] = None
    image_5_fg: Optional[GradioImageMaskPair] = None
    # Per-slot mute (the "use this input" checkbox on each Input tab): a muted
    # slot keeps its image and masks but contributes nothing to the generation,
    # so inputs can be A/B-ed without destructively clearing the canvas.
    image_enabled: bool = True
    image_2_enabled: bool = True
    image_3_enabled: bool = True
    image_4_enabled: bool = True
    image_5_enabled: bool = True
    # CNPro default: "Resize and Fill" rather than the historical "Crop and
    # Resize". Cropping silently DISCARDS part of the control image - a canny or
    # depth hint whose subject runs to the frame edge loses that edge, and the
    # loss is invisible in the UI because the thumbnail still shows the whole
    # image. Filling keeps every pixel of the hint and pads instead; the
    # preprocessors that care already have `fill_mask_with_one_when_resize_and_fill`
    # / `expand_mask_when_resize_and_fill` to handle the padded border.
    resize_mode: Union[ResizeMode, int, str] = ResizeMode.OUTER_FIT
    # -1 means "resolve from the preprocessor at generation time"; it stays the
    # sentinel for API callers that pass it explicitly, and now resolves to
    # global_state.default_processor_res (1024) rather than to the host's
    # SD1.5-era 512. See bound_check_params in scripts/cnpro.py.
    processor_res: int = -1
    threshold_a: float = -1
    threshold_b: float = -1
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    pixel_perfect: bool = False
    control_mode: Union[ControlMode, int, str] = ControlMode.BALANCED
    # Piecewise-linear control strength over relative sampling steps, serialized
    # as 'x@y;x@y;...' with an optional '|hi' or '|lo~hi' scale-range suffix
    # (range clamped to [-1, 2]). When set, it overrides
    # weight / guidance_start / guidance_end.
    weight_profile: str = ""
    # Rainbow-hue weight mask painted over the source image (RGBA, HWC uint8).
    # Hue encodes local control strength; decoded in scripts/controlnet.py.
    # The global mask takes priority over the per-band layer masks below.
    weight_mask: Optional[np.ndarray] = None
    # Per-band layer masks (same rainbow encoding): coarse = composition band
    # (deepest injection layers + middle), mid = form band, fine = texture band.
    # Used only when no global weight_mask is painted; an absent band means
    # full weight everywhere for that band.
    weight_mask_coarse: Optional[np.ndarray] = None
    weight_mask_mid: Optional[np.ndarray] = None
    weight_mask_fine: Optional[np.ndarray] = None
    # Same four slots for input 2..MAX_INPUT_IMAGES: every input is its own
    # control, so it carries its own masks - painting on input 3 restricts what
    # input 3 contributes and never touches the others.
    weight_mask_2: Optional[np.ndarray] = None
    weight_mask_2_coarse: Optional[np.ndarray] = None
    weight_mask_2_mid: Optional[np.ndarray] = None
    weight_mask_2_fine: Optional[np.ndarray] = None
    weight_mask_3: Optional[np.ndarray] = None
    weight_mask_3_coarse: Optional[np.ndarray] = None
    weight_mask_3_mid: Optional[np.ndarray] = None
    weight_mask_3_fine: Optional[np.ndarray] = None
    weight_mask_4: Optional[np.ndarray] = None
    weight_mask_4_coarse: Optional[np.ndarray] = None
    weight_mask_4_mid: Optional[np.ndarray] = None
    weight_mask_4_fine: Optional[np.ndarray] = None
    weight_mask_5: Optional[np.ndarray] = None
    weight_mask_5_coarse: Optional[np.ndarray] = None
    weight_mask_5_mid: Optional[np.ndarray] = None
    weight_mask_5_fine: Optional[np.ndarray] = None
    # Output-side weight mask (same rainbow encoding), painted on the "Output
    # mask" tab over a throwaway reference image: it is registered with the
    # GENERATED image, not with the control input, so it only scales the
    # control injection per output region and never gates what the control
    # model gets to see. Empty = control applies to the whole output.
    output_mask: Optional[np.ndarray] = None
    # Per-step cond/uncond balance profile, same 'x@y;...' serialization as
    # weight_profile. y = 0.5 is balanced (control on cond and uncond), y = 1
    # applies control to cond only (control matters most), y = 0 to uncond
    # only (prompt matters most); in between interpolates. Empty = balanced.
    # Replaces the legacy control_mode chooser (the field is kept for API
    # back-compat but the UI no longer exposes it).
    balance_profile: str = ""
    # Per-unit prompt for the control branch: encoded with the model's text
    # encoder and fed to the ControlNet's OWN cross-attention in place of the
    # main positive prompt, on the cond rows only (uncond keeps the negative
    # context, so the prompt's semantics ride the CFG contrast; the main unet
    # still sees the main prompt). Empty = main prompt. Only true ControlNet
    # models consume it - T2I-Adapter, IP-Adapter and ControlLLLite have no
    # text input and log a warning instead.
    # Deliberately NOT in infotext: free text would break the ','/':'-free
    # serialization format (same reason masks stay out).
    unit_prompt: str = ""
    # Negative counterpart of unit_prompt: encoded the same way and fed to the
    # control branch's cross-attention on the UNCOND rows in place of the
    # sampled negative context - "push this control's semantics away from X".
    # Empty = the sampled negative prompt (the default asymmetry that makes
    # unit_prompt work). Same model gating and infotext exclusion as
    # unit_prompt.
    unit_negative_prompt: str = ""
    # Strength controls of the unit prompts, one pair per side, both in
    # [-1, 3] with neutral 1. Deliberately NOT in infotext, like the prompts
    # they modify - a strength without its text is meaningless.
    # - embedding strength: how far the control branch's context moves from
    #   the sampled prompt toward (past) the unit prompt, applied in embedding
    #   space: context = sampled + s * (unit - sampled). 1 = exactly the unit
    #   prompt (the classic swap), 0 = unit prompt off, > 1 extrapolates,
    #   < 0 pushes away from the unit prompt.
    # - delta scale: scales the text's exact effect on the control residuals:
    #   residual = base + s * (with_text - base), where base is the residual
    #   with the sampled context. 1 = the swap's natural effect (single pass),
    #   0 = off (single pass); any other value costs a SECOND control-model
    #   forward per step to obtain `base`.
    # The DEFAULTS are deliberately off neutral (1.6, not 1): the plain swap
    # moves the control residuals by only ~3%, too subtle to steer with, so a
    # freshly opened unit starts at a strength that actually shows. Nothing is
    # spent until a prompt is typed - both fields are only read when a unit
    # context exists.
    unit_prompt_emb_strength: float = 1.6
    unit_prompt_delta_scale: float = 1.6
    unit_negative_prompt_emb_strength: float = 1.6
    unit_negative_prompt_delta_scale: float = 1.6
    # Prompt retention, ONE global knob for both sides, in [0, 3] with
    # neutral 0: the per-step delta scales are multiplied by
    # 1 + retention * progress (progress = relative sampling position), so
    # the text's effect ramps up toward the end of sampling instead of
    # decaying with the cond/uncond contrast as the latent converges. The
    # first step is always untouched (x1). Any value > 0 costs the second
    # control forward on every step after the first (the ramped scale
    # leaves 1, engaging the base-pass isolation). Same infotext exclusion
    # as the strengths above. Default 0.75, off neutral for the same reason
    # as the strengths: the decay is the common case worth compensating.
    unit_prompt_retention: float = 0.75
    # Visual order of the Input tabs as slot digits (e.g. "10234"), "" =
    # natural. The order is generation-relevant (multi-phase profiles assign
    # input k the phase k*2pi/n), so it is a unit field: get_input_data sorts
    # the slots by it, and the strip mirrors it via CSS order (tab_marks.js).
    # Content never moves between slots - a "tab move" is only this string
    # changing. Same infotext exclusion as masks/mute (pure UI-session state).
    input_order: str = ""
    # Which input slot the preprocessor PREVIEW was generated from. Only read
    # when use_preview_as_input is on: the preview replaces that slot's image,
    # so it must be gated by that slot's weight masks - hardcoding slot 0 made
    # a preview of input 3 arrive with input 1's paint. Pure session state,
    # same infotext exclusion as input_order.
    preview_slot: int = 0
    save_detected_map: bool = True

    def input_images(self):
        """(image, foreground, enabled) of every input slot, in tab order.

        Slots the user never opened - or closed again, which clears them - hold
        None and are skipped by the caller; the unit is driven by whatever
        images are actually present, so UI tab visibility never has to be
        mirrored on this side. `enabled` is the per-slot mute checkbox: a muted
        slot keeps its image but must be skipped exactly like an empty one.
        Per-input WEIGHTING is the weight masks' job: a flat-painted global
        mask at value v scales exactly that input's contribution.
        """
        triples = [(self.image, self.image_fg, self.image_enabled)]
        for slot in range(2, MAX_INPUT_IMAGES + 1):
            triples.append((getattr(self, f"image_{slot}"),
                            getattr(self, f"image_{slot}_fg"),
                            getattr(self, f"image_{slot}_enabled")))
        return triples

    @staticmethod
    def weight_mask_field(slot, band=None):
        """Field name of a weight mask channel. slot is 0-based (tab order),
        band is None for the global mask or coarse/mid/fine."""
        name = "weight_mask" if slot == 0 else f"weight_mask_{slot + 1}"
        return name if band is None else f"{name}_{band}"

    @staticmethod
    def input_order_permutation(order):
        """Full slot permutation encoded by an input_order string.

        FAILS OPEN: invalid characters and duplicates are dropped, missing
        slots are appended in natural order - any garbage degrades to the
        natural order instead of losing inputs. tab_marks.js applies the same
        lenient rule; keep them aligned."""
        perm = []
        for ch in str(order or ""):
            if ch.isdigit():
                slot = int(ch)
                if slot < MAX_INPUT_IMAGES and slot not in perm:
                    perm.append(slot)
        perm.extend(slot for slot in range(MAX_INPUT_IMAGES) if slot not in perm)
        return perm

    def input_weight_masks(self, slot):
        """(global, {band: mask}) of one input slot, raw RGBA as painted."""
        bands = {}
        for band in ("coarse", "mid", "fine"):
            bands[band] = getattr(self, ControlNetUnit.weight_mask_field(slot, band))
        return getattr(self, ControlNetUnit.weight_mask_field(slot)), bands

    @staticmethod
    def infotext_fields():
        """Fields that should be included in infotext.
        You should define a Gradio element with exact same name in ControlNetUiGroup
        as well, so that infotext can wire the value to correct field when pasting
        infotext.
        """
        return (
            "module",
            "model",
            "weight_profile",
            "balance_profile",
            "resize_mode",
            "processor_res",
            "threshold_a",
            "threshold_b",
            "pixel_perfect",
            "hr_option",
        )

    @staticmethod
    def from_dict(d: Dict) -> "ControlNetUnit":
        """Create ControlNetUnit from dict. This is primarily used to convert
        API json dict to ControlNetUnit."""
        unit = ControlNetUnit(
            **{k: v for k, v in d.items() if k in vars(ControlNetUnit)}
        )
        for image_field in ['image'] + [f"image_{i}" for i in range(2, MAX_INPUT_IMAGES + 1)]:
            value = getattr(unit, image_field)
            if isinstance(value, str):
                setattr(unit, image_field, np.array(api.decode_base64_to_image(value)).astype('uint8'))
        if isinstance(unit.mask_image, str):
            unit.mask_image = np.array(api.decode_base64_to_image(unit.mask_image)).astype('uint8')
        if isinstance(unit.weight_mask, str):
            unit.weight_mask = np.array(api.decode_base64_to_image(unit.weight_mask)).astype('uint8')
        mask_fields = ['output_mask']
        for slot in range(MAX_INPUT_IMAGES):
            for band in (None, 'coarse', 'mid', 'fine'):
                field = ControlNetUnit.weight_mask_field(slot, band)
                if field != 'weight_mask':  # decoded above
                    mask_fields.append(field)
        for band_field in mask_fields:
            band_value = getattr(unit, band_field)
            if isinstance(band_value, str):
                setattr(unit, band_field, np.array(api.decode_base64_to_image(band_value)).astype('uint8'))
        # Legacy control_mode from API callers: the generation path only reads
        # balance_profile now, so convert here exactly like infotext paste does
        # - otherwise a caller's Balanced/Prompt/Control choice would be
        # silently ignored.
        if not unit.balance_profile and 'control_mode' in d:
            try:
                mode = control_mode_from_value(unit.control_mode).value
            except Exception:
                mode = None
            unit.balance_profile = {
                ControlMode.CONTROL.value: "0@1;1@1",
                ControlMode.PROMPT.value: "0@0.25;1@0.25",
            }.get(mode, "")
        return unit


# Backward Compatible
UiControlNetUnit = ControlNetUnit


def _slot_field_names():
    """Every per-input-slot field MAX_INPUT_IMAGES implies, in dataclass order."""
    names = []
    for slot in range(MAX_INPUT_IMAGES):
        image = "image" if slot == 0 else f"image_{slot + 1}"
        names += [image, f"{image}_fg", f"{image}_enabled"]
        names += [ControlNetUnit.weight_mask_field(slot, band)
                  for band in (None, "coarse", "mid", "fine")]
    return names


def _check_slot_fields():
    """Raising MAX_INPUT_IMAGES means adding fields; say exactly which ones.

    The per-slot fields are written out by hand (they carry documentation that
    a generated field list would lose) and gradio applies unit_fields
    POSITIONALLY, so a missing one is not a cosmetic problem. The ui group's
    startup check already catches a mismatch against its components; this one
    catches the case before that, naming the fields instead of showing two
    lists to diff. A hard raise for the same reason: `python -O` strips
    asserts.
    """
    present = {f.name for f in fields(ControlNetUnit)}
    missing = [name for name in _slot_field_names() if name not in present]
    if missing:
        raise RuntimeError(
            f"ControlNetUnit is missing per-slot fields for MAX_INPUT_IMAGES="
            f"{MAX_INPUT_IMAGES}: {missing}. Add them to the dataclass in slot "
            f"order (image/_fg/_enabled then the four weight_mask channels) and "
            f"extend ControlNetUiGroup.unit_fields to match.")


_check_slot_fields()

