"""Scaling one line of a CNPro weight profile.

Shared by the two search scripts - `scripts/CNPro_XY.py` walks a list of factors
along an axis, `scripts/CNPro_DNA.py` searches over one - because the arithmetic
below is subtle enough that two copies of it would eventually disagree, and the
disagreement would be invisible: both copies produce a plausible curve.

HOW THE MULTIPLIER IS APPLIED
-----------------------------
Not by moving the drawn points. A CNPro profile is a NORMALIZED polyline plus a
`|lo~hi` scale range, and the effective weight is

    y = lo + f(y_norm) * (hi - lo)

where f folds in the segment mids, the oscillatory wave, the multi-phase
partition and the response exponent. Multiplying BOTH lo and hi by m gives
exactly `m * y` at every step, whatever f is - so the factor is an exact
scaling that leaves every drawn feature (mids, waves, phase hand-off,
convergence, gamma) intact. Scaling the points instead would be wrong the
moment a response exponent is set, because the exponent applies before the
range mapping - `(m*y)**e` is not `m * y**e`.

WHY A MULTIPLIER RATHER THAN AN OFFSET
--------------------------------------
Because a profile is a SHAPE, and the useful question is almost always "the
same shape, more or less of it" rather than "the same shape, lifted". Adding
0.2 to a curve that ramps 0 -> 1 changes its ramp from 5:1 to 6:1 and adds
control where the user drew none; multiplying it by 1.2 keeps every ratio the
curve expresses, including the zeros - a step the user deliberately left at no
control stays at no control. It is also the operation that composes: two
factors multiply to one factor, so a scaled profile is still on the same
one-parameter family.

NEGATIVE FACTORS ARE REFUSED, and this is a grammar limit rather than a
judgement. Inverting a curve means `lo, hi -> -lo, -hi`, i.e. an upper bound
BELOW the lower one, and `parse_weight_profile` sorts those two back into
order - so `|0~-1` is read as `|-1~0`, which is the curve slid down rather
than flipped. Writing an inverted profile would mean mirroring every drawn
point, which is exactly the "move the points" approach the first paragraph
rules out. So `parse_factors` raises instead of quietly producing a curve
nobody asked for.

A line with no curve drawn on it is scaled from its NEUTRAL element - flat 1
for the bands and depth, flat 0 for drift, and the unit's weight/step-range
scalars for Main. That makes "scale the depth curve by 1.5" mean something
before a depth curve exists (a flat 1 becomes a flat 1.5), and it makes the
same instruction on DRIFT mean nothing at all - 0 * m is 0 - which is warned
about rather than left to be discovered in a grid of identical images.

Consequence worth knowing: lo and hi are each clamped to [-1, 2] by the
parser, so a factor that pushes a bound past those does not scale the curve,
it squashes it. That case is logged rather than passing silently.

THE POINT OFFSET IS THE DELIBERATE EXCEPTION to "not by moving the drawn
points". `offset_profile_point` moves exactly ONE drawn point of one line
vertically - the same edit dragging that point in the editor would make - and
it exists because the multiplier above cannot ask its question: "this curve,
but with the step-0.75 knee a little higher" is a question about a point, not
about the shape. Three rules keep it honest:

* the offset is expressed in the PLOT'S OWN AXIS UNITS (the `|lo~hi` range),
  so "+0.1" moves the point by 0.1 of whatever the y axis shows - the same
  distance on a 0~1 weight plot and a 0~2 depth plot;
* an offset of exactly 0 returns the profile VERBATIM - byte-identical, no
  synthesized segments, no re-serialization - because "0 is the current
  profile" is the contract the search scripts print in their legends;
* everything that is not the chosen point survives verbatim: the other
  points, the mid/wave/phase/gamma tokens, the range suffix. A point pushed
  past the plot's range is clamped and WARNED about, exactly like a factor
  that squashes against the parser's bounds.

Unlike a factor, a point offset may be NEGATIVE - moving a point down is half
the point - which is why `parse_offsets` exists beside `parse_factors` instead
of sharing its refusal.

This module imports CNPro lazily, for the same reason the scripts do: either
script can live in the webui's own `scripts/` directory, where the extension's
basedir is not on sys.path and the only reason `lib_cnpro` imports at all is
that cnpro.py already put it in sys.modules.
"""

import logging
import re

logger = logging.getLogger("CNPro")

#: A drawn point inside a segment body: a NUMBER followed by '@'. Everything
#: else on the ';' list is a token (Mx@y mids, C/P/A wave tokens, G exponent)
#: and is never a point - the letter prefix is the tell.
RE_POINT = re.compile(r"^\s*-?(?:\d+\.?\d*|\.\d+)\s*@")

#: The six lines the profile editor draws, mapped to the prefix of their segment
#: in the packed profile string ('main#C..#M..#F..#D..#S..'). Main is everything
#: before the first '#', hence the None.
#:
#: IN THE EDITOR'S OWN ORDER (weight_profile.js SELECTOR_ORDER, the order the
#: six selector bars sit in under the presets): main | depth | drift, then the
#: separator, then coarse | mid | fine. That row's reading order IS the model -
#: the first three assemble into one field, main(step) x depth(layer -
#: drift(step)), and the last three each replace it - so a dropdown in a
#: different order would quietly teach the wrong thing about which curves
#: compose and which exclude each other.
PROFILE_LINES = {
    "Main": None,
    "Depth": "D",
    "Drift": "S",
    "Coarse": "C",
    "Mid": "M",
    "Fine": "F",
}

#: Neutral element of each line, i.e. what an ABSENT segment already means.
#: Bands and depth are multipliers (1); drift is a shift along the depth axis
#: (0). Main is absent-means-scalars and is handled on its own.
PROFILE_NEUTRAL = {"C": 1.0, "M": 1.0, "F": 1.0, "D": 1.0, "S": 0.0}

#: The bounds the profile parser clamps a scale range to.
SCALE_MIN, SCALE_MAX = -1.0, 2.0


def _external_code():
    """CNPro's profile grammar module, or None if CNPro is not loaded."""
    try:
        from lib_cnpro import external_code
    except Exception:
        return None
    return external_code


def _parse_numbers(text, what):
    """Comma-separated numbers, each bare or an `a:b:n` linspace.

    Colons rather than xyz_grid's `a-b` range notation, so a negative number
    and a range cannot be confused - which matters twice over now that
    offsets, which are negative half the time, read through this too.
    """
    values = []
    for token in str(text or "").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            values.append(float(token))
            continue
        parts = [t.strip() for t in token.split(":")]
        if len(parts) != 3:
            raise ValueError(f"'{token}' is not {what}: use a number, or "
                             f"'start:end:count' for an evenly spaced run.")
        start, end, count = float(parts[0]), float(parts[1]), int(float(parts[2]))
        if count < 1:
            raise ValueError(f"'{token}': count must be at least 1.")
        if count == 1:
            values.append(start)
        else:
            step = (end - start) / (count - 1)
            values.extend(start + step * i for i in range(count))
    return values


def parse_factors(text):
    """Comma-separated factors, each a number or an `a:b:n` linspace.

    `1` is the profile untouched, `0` is the line switched off, `2` is twice
    the control.
    """
    values = _parse_numbers(text, "a factor")
    negative = [v for v in values if v < 0]
    if negative:
        # See the module docstring: the range suffix cannot hold an inverted
        # curve, and the parser would silently read it back as a lowered one.
        raise ValueError(
            f"a profile factor cannot be negative ({', '.join(f'{v:g}' for v in negative)}). "
            f"Inverting a curve is not something the scale range can express - "
            f"it would be read back as the curve slid down instead. Use 0 to "
            f"switch the line off, or draw the inverted curve in the editor.")
    return values


def parse_offsets(text):
    """Comma-separated point offsets, each a number or an `a:b:n` linspace.

    `0` is the profile exactly as drawn. Negative is as legitimate as
    positive - moving a point DOWN is half the reason to move one - which is
    the whole difference from `parse_factors`.
    """
    return _parse_numbers(text, "an offset")


def _num(value):
    return f"{round(float(value), 4):g}"


def _read_range(tail, what, who="CNPro"):
    """The (lo, hi) a `|` suffix names - `lo~hi`, legacy bare `hi`, or the
    default 0~1 when there is no suffix at all (tail is None) or it is
    unreadable (warned, exactly as the parser would fall back)."""
    if tail is None:
        return 0.0, 1.0
    bounds = tail.split("~")
    try:
        if len(bounds) > 1:
            lo, hi = float(bounds[0]), float(bounds[1])
        else:
            lo, hi = 0.0, float(bounds[0])
    except ValueError:
        logger.warning(f"{who}: unreadable scale range {tail!r} on "
                       f"{what}; reading it as the default 0~1.")
        return 0.0, 1.0
    return (hi, lo) if lo > hi else (lo, hi)


def scale_segment(body, factor, neutral, what, who="CNPro"):
    """One profile segment's body, scaled by `factor`.

    `body` is the standard 'x@y;...' grammar with an optional '|lo~hi' (or
    legacy '|hi') suffix. Everything before the '|' is preserved VERBATIM -
    points, segment mids, the cosine and multi-phase tokens, the response
    exponent, the convergence marker - because the scaling is expressed
    entirely in the scale range. See the module docstring for why that is
    exact.
    """
    body = (body or "").strip()
    if not body:
        body = f"0@{_num(neutral)};1@{_num(neutral)}"

    head, sep, tail = body.partition("|")
    lo, hi = _read_range(tail if sep else None, what, who)

    scaled_lo, scaled_hi = lo * factor, hi * factor
    clamped_lo = min(max(scaled_lo, SCALE_MIN), SCALE_MAX)
    clamped_hi = min(max(scaled_hi, SCALE_MIN), SCALE_MAX)
    if abs(clamped_lo - scaled_lo) > 1e-6 or abs(clamped_hi - scaled_hi) > 1e-6:
        # A clamped bound is NOT the factor that was asked for: the curve is
        # squashed against the parser's range instead of scaled. Loud, because
        # the image it produces is plausible and wrong.
        logger.warning(
            f"{who}: factor x{factor:g} takes {what} from range "
            f"{lo:g}~{hi:g} to {scaled_lo:g}~{scaled_hi:g}, outside the "
            f"representable [{SCALE_MIN:g}, {SCALE_MAX:g}] - it is CLAMPED to "
            f"{clamped_lo:g}~{clamped_hi:g}, so this is not a pure scaling.")
    return f"{head}|{_num(clamped_lo)}~{_num(clamped_hi)}"


def _rewrite_line(unit, unit_index, line, rewrite):
    """`unit.weight_profile` with `rewrite(body, neutral, what)` applied to
    ONE line's segment body.

    The shared walk under scale_profile and offset_profile_point, because
    the segment bookkeeping - which prefix, an absent segment, Main's
    legacy-scalars case - must not fork between the two operations: both
    copies would still produce a plausible string, just not the same one.
    """
    external_code = _external_code()
    prefix = PROFILE_LINES[line]
    main, *segments = (unit.weight_profile or "").split("#")
    what = f"unit {unit_index}'s {line.lower()} profile"

    if prefix is None:
        if not main.strip():
            # No main curve drawn: the unit is running the legacy scalars, so
            # those are what gets rewritten - the same three numbers the
            # editor itself would start from. They put the weight in the
            # range suffix, so a range rewrite rewrites the weight exactly.
            if external_code is None:
                return unit.weight_profile or ""
            main = external_code.weight_profile_from_scalars(
                unit.weight, unit.guidance_start, unit.guidance_end)
        return "#".join([rewrite(main, 0.0, what)] + segments)

    for i, segment in enumerate(segments):
        if segment[:1] == prefix:
            segments[i] = prefix + rewrite(
                segment[1:], PROFILE_NEUTRAL[prefix], what)
            break
    else:
        segments.append(prefix + rewrite("", PROFILE_NEUTRAL[prefix], what))
    return "#".join([main] + segments)


def scale_profile(unit, unit_index, line, factor, who="CNPro"):
    """`unit.weight_profile` with one of its six lines scaled by `factor`."""
    return _rewrite_line(
        unit, unit_index, line,
        lambda body, neutral, what: scale_segment(body, factor, neutral,
                                                  what, who))


def offset_segment_point(body, point_index, offset, neutral, what,
                         who="CNPro"):
    """One segment's body with drawn point `point_index` moved vertically by
    `offset`, expressed in the plot's own axis units (the `|lo~hi` range).

    Negative indices count from the end, -1 being the rightmost point. An
    index that names no point leaves the body UNCHANGED and warns: the
    variant then produces the same image as the original, and the reason is
    invisible in the picture. An offset of exactly 0 returns the body
    verbatim - see the module docstring.
    """
    original = body or ""
    if abs(float(offset)) < 1e-12:
        return original

    body = original.strip()
    if not body:
        body = f"0@{_num(neutral)};1@{_num(neutral)}"
    head, sep, tail = body.partition("|")
    lo, hi = _read_range(tail if sep else None, what, who)
    span = hi - lo
    if span < 1e-9:
        logger.warning(
            f"{who}: {what} has a zero-width scale range ({lo:g}~{hi:g}), so "
            f"a vertical offset cannot be expressed on it - the profile is "
            f"left unchanged. Widen the range on the plot first.")
        return original

    tokens = head.split(";")
    slots = [i for i, token in enumerate(tokens) if RE_POINT.match(token)]
    index = point_index + len(slots) if point_index < 0 else point_index
    if not (0 <= index < len(slots)):
        logger.warning(
            f"{who}: {what} has {len(slots)} point(s), so point index "
            f"{point_index} names none of them - the profile is left "
            f"unchanged, and every offset of this variation produces the "
            f"same image.")
        return original

    slot = slots[index]
    x_text, _at, y_text = tokens[slot].partition("@")
    try:
        y = float(y_text)
    except ValueError:
        logger.warning(f"{who}: {what}'s point {point_index} has an "
                       f"unreadable value {y_text!r} - left unchanged.")
        return original

    moved = y + float(offset) / span
    clamped = min(max(moved, 0.0), 1.0)
    if abs(clamped - moved) > 1e-9:
        # A clamped point is NOT the offset that was asked for - the point
        # is pressed against the plot's range instead of moved. Loud,
        # because the image it produces is plausible and wrong.
        logger.warning(
            f"{who}: offset {offset:+g} takes {what}'s point {point_index} "
            f"from {lo + y * span:g} to {lo + moved * span:g}, outside the "
            f"plot's {lo:g}~{hi:g} range - it is CLAMPED to "
            f"{lo + clamped * span:g}, so this is not the offset asked for. "
            f"Widen the range on the plot to make room.")
    tokens[slot] = f"{x_text.strip()}@{_num(clamped)}"
    head = ";".join(tokens)
    return f"{head}|{tail}" if sep else head


def offset_profile_point(unit, unit_index, line, point_index, offset,
                         who="CNPro"):
    """`unit.weight_profile` with ONE drawn point of one line moved
    vertically by `offset` plot units - the same edit dragging that point in
    the editor would make. 0 is the profile exactly as it stands."""
    if abs(float(offset)) < 1e-12:
        return unit.weight_profile or ""
    return _rewrite_line(
        unit, unit_index, line,
        lambda body, neutral, what: offset_segment_point(
            body, point_index, offset, neutral, what, who))


def profile_point_count(unit, line):
    """How many drawn points the chosen line currently has.

    A line with no curve counts the neutral flat it would be synthesized
    from (two points), and Main with no curve counts the legacy-scalars
    profile the editor itself would start from. This is what lets the
    search scripts refuse a bad point index BEFORE spending generations on
    variants that are all the same image.
    """
    prefix = PROFILE_LINES[line]
    main, *segments = (unit.weight_profile or "").split("#")
    if prefix is None:
        body = main
        if not body.strip():
            external_code = _external_code()
            if external_code is None:
                return 2
            body = external_code.weight_profile_from_scalars(
                unit.weight, unit.guidance_start, unit.guidance_end)
    else:
        body = next((s[1:] for s in segments if s[:1] == prefix), "")
    head = (body or "").strip().partition("|")[0]
    if not head:
        return 2
    return sum(1 for token in head.split(";") if RE_POINT.match(token))


def warn_if_inert(unit, index, line, who="CNPro"):
    """Say when a chosen line is not the one the unit is actually running.

    Band mode and main mode are exclusive, and so are the bands and the depth
    curve; a drift with a flat depth curve moves nothing, and an ABSENT drift
    curve cannot be scaled at all. Varying any of those produces identical
    images, and the reason is invisible in the picture.
    """
    external_code = _external_code()
    if external_code is None:
        return
    profile = unit.weight_profile or ""
    band_mode = external_code.band_mode_active(profile)
    prefix = PROFILE_LINES[line]
    head = f"{who}: unit {index} is in"

    if prefix is None and band_mode:
        logger.warning(f"{head} BAND mode, so its main profile is not applied "
                       f"- scaling it will not change anything. Use "
                       f"Coarse/Mid/Fine, or leave band mode on the plot.")
    elif prefix in ("C", "M", "F") and not band_mode:
        logger.warning(f"{head} MAIN mode, so its band profiles are not applied "
                       f"- scaling one will not change anything. Press a band "
                       f"selector on the unit's plot first.")
    elif prefix == "D" and band_mode:
        logger.warning(f"{head} BAND mode, so its depth profile is not applied "
                       f"- bands and depth are exclusive. Scaling it will not "
                       f"change anything.")
    elif prefix == "S":
        if band_mode:
            logger.warning(f"{head} BAND mode, so its depth-drift profile is "
                           f"not applied. Scaling it will not change anything.")
        elif f"#{prefix}" not in profile:
            # Drift is the one line whose neutral is 0, and 0 times anything
            # is 0. Every other absent line starts at a flat 1 and scales.
            logger.warning(
                f"{who}: unit {index} has no depth-drift curve, and drift is "
                f"the one line whose neutral element is 0 - scaling it is 0 x "
                f"factor = 0 for every factor, so nothing will change. Draw a "
                f"drift curve first.")
        elif external_code.parse_depth_profile(profile) is None:
            logger.warning(f"{who}: unit {index} has a flat depth profile. "
                           f"Drift shifts WHERE the depth curve is read, so "
                           f"with nothing to move a scaled drift will not "
                           f"change anything - draw a depth curve first.")
