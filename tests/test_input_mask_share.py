"""A mask painted with ONE weight reduces to that weight, and warns about nothing.

THE CONTRACT
------------
An input mask is a SHAPE plus one NUMBER. `scripts/cnpro.py
decode_weight_mask_parts` reads the two channels the painter ships - the VALUE
that was painted and the COVERAGE it was painted with - and
`external_code.input_mask_share` reduces them to that one number, plus a flag
saying whether the paint really held more than one value.

WHAT WENT WRONG, AND WHY THIS FILE EXISTS
-----------------------------------------
The two channels used to arrive already multiplied together, because that
product is what a mask TENSOR wants and one decode served both readers. A
reduction cannot use it. The canvas anti-aliases every fill it draws - there is
no way to ask it not to - so the edge of any stroke is a ring of pixels holding
the painted value at a fractional coverage, and the product therefore sweeps
from ~0 up to the painted weight along every stroke the user ever draws. Two
things followed, both silent in the sense that the picture on screen looked
exactly right:

  * a mask painted with a single weight reported `is_graded`, and the user was
    told the painted values had been averaged - a gradient nobody painted; and
  * the share came out BELOW the value painted, by roughly the stroke's
    perimeter-to-area ratio: 0.98 for a broad blob, 0.86 for a thin line, and
    further with the feather slider up. Two inputs painted at the SAME weight
    then divided the unit unevenly, purely from the shapes of their strokes.

So the masks here are built the way a browser hands one back - the value stored
PREMULTIPLIED by coverage in 8 bits and un-premultiplied on readback, which is
where the noise floor in `input_mask_share` comes from - and pushed through the
real decode and the real reduction. Every case states the weight it was painted
with, and the share has to be that number.

Run:  python tests/test_input_mask_share.py
Needs numpy + cv2; without them it SKIPS, loudly. Exit code 0 = pass.
"""
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []
SKIPS = []

# What the painter's weight slider can express (canvas_tools.js wmaskWeight,
# step 0.01). Two weights a user can actually pick differ by at least this, and
# every "one weight" case here has to stay well inside it.
PAINT_STEP = 0.01
# Tolerance on a recovered share: the value channel is 8-bit, so a painted
# weight cannot round-trip closer than half a level, and the coverage-weighted
# mean adds a little of the same. Anything looser would hide the bias this file
# is about (0.98 for a blob, 0.86 for a line - both an order above this).
TOL = 0.5 * PAINT_STEP


def fail(msg):
    FAILURES.append(msg)


def read(rel):
    with open(os.path.join(EXTENSION, rel), encoding="utf-8") as fh:
        return fh.read()


def extract(rel, names, namespace):
    """exec the named top-level functions out of a module, without the host.

    Same extraction tests/test_coverage_map.py uses on utils.py: scripts/cnpro.py
    imports gradio, torch and the whole webui, and none of that has anything to
    do with decoding four bytes per pixel.
    """
    src = read(rel)
    parts = []
    for name in names:
        m = re.search(r"^def %s\(.*?(?=\n\ndef |\n\nclass )" % name, src, re.M | re.S)
        if not m:
            fail("%s no longer defines %s where this test can reach it - the "
                 "input-mask reduction has nothing left to be checked against"
                 % (rel, name))
            return None
        parts.append(m.group(0))
    exec("\n\n".join(parts), namespace)
    return namespace


# ---------------------------------------------------------------------------
# the wire format, as a browser produces it
# ---------------------------------------------------------------------------

def paint(coverage, weight, np, chromatic=False):
    """One painted mask, in the RGBA the painter puts on the wire.

    `coverage` is the per-pixel paint coverage in [0, 1] - 1 inside a stroke,
    the anti-aliased (or feathered, or resampled) ramp along its edge - and
    `weight` is the single value the whole stroke was painted with.

    The round trip is the browser's: a canvas keeps colours PREMULTIPLIED by
    alpha in 8 bits and divides them back out on readback, so the value a
    fringe pixel reports is a rounded product divided by a rounded coverage.
    Painting the un-premultiplied colour directly would be a cleaner input than
    the code will ever see, and the noise floor this exercises is the reason
    `input_mask_share` has one.
    """
    coverage = np.clip(np.asarray(coverage, np.float64), 0.0, 1.0)
    colour = hue_rgb(weight, np) if chromatic else np.array(
        [round(weight * 255.0)] * 3, np.float64)
    alpha_u8 = np.round(coverage * 255.0)
    stored = np.round(colour[None, None, :] * (alpha_u8 / 255.0)[:, :, None])
    out = np.zeros(coverage.shape + (4,), np.uint8)
    lit = alpha_u8 > 0
    recovered = np.zeros_like(stored)
    recovered[lit] = np.round(
        stored[lit] * 255.0 / alpha_u8[lit][:, None])
    out[:, :, :3] = np.clip(recovered, 0, 255).astype(np.uint8)
    out[:, :, 3] = alpha_u8.astype(np.uint8)
    return out


def hue_rgb(weight, np):
    """The LEGACY rainbow colour for a weight: hue (1 - w) * 270, full S and V.

    Old sessions and API callers still send this, and `decode_weight_mask_parts`
    keeps the hue path for them, so the reduction has to survive it too.
    """
    import cv2
    hsv = np.array([[[round((1.0 - weight) * 270.0 / 2.0), 255, 255]]], np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float64)


def disc(np, size=256, cx=128.0, cy=128.0, r=70.0):
    """An anti-aliased filled circle: coverage 1 inside, a 1 px ramp at the rim.

    What `ctx.arc(...); ctx.fill()` puts down for a single click of the brush.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    return np.clip(r + 0.5 - np.hypot(yy - cy, xx - cx), 0.0, 1.0)


def line(np, size=128, half_width=3.0):
    """A thin anti-aliased stroke - the shape the old share was worst on."""
    yy, _ = np.mgrid[0:size, 0:size]
    return np.clip(half_width + 0.5 - np.abs(yy - size / 2.0), 0.0, 1.0)


def feathered(coverage, radius, np):
    """The FEATHER slider: a Gaussian on the coverage channel.

    The painter blurs the exported mask, and canvas blur runs on premultiplied
    alpha - for paint of a single colour that is exactly a blur of coverage,
    which is why the value channel comes back holding the painted weight.
    """
    import cv2
    k = int(radius) * 6 + 1
    return cv2.GaussianBlur(coverage, (k, k), radius, borderType=cv2.BORDER_CONSTANT)


def wire_tolerance(coverage, np):
    """How close a share CAN come, given how thinly the paint covers.

    A fact about the wire format, not about the reduction: the value is stored
    premultiplied in 8 bits, so recovering it where the paint covers `c` of a
    pixel cannot beat half a level rounding in plus half a level rounding out,
    divided by c. Paint with a solid core round-trips exactly; a stroke
    feathered until nothing is half covered has only its faint peak to speak
    from, and 0.50 painted there comes back as 0.506 no matter who reads it.
    """
    best = float(np.max(coverage))
    least = 0.5 if best >= 0.5 else best
    return TOL + (0.0 if least >= 1.0 else (0.5 / least + 0.5) / 255.0)


def downscaled(coverage, to, np):
    """The oversized-export path: the same paint, resampled smaller.

    weight_mask.js shrinks a mask whose data-url exceeds 8 MB, and a change of
    canvas dimensions rescales one too. Both interpolate, so the edge ramp
    stops being one pixel wide and becomes several.
    """
    import cv2
    return cv2.resize(coverage.astype("float32"), (to, to),
                      interpolation=cv2.INTER_AREA).astype("float64")


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def main():
    try:
        import numpy as np
        import cv2  # noqa: F401
    except ImportError as exc:
        print("SKIPPED - numpy/cv2 are not importable here (%s), so NOTHING "
              "about the input-mask share was verified" % exc)
        return 0

    warnings = []
    ns = extract("scripts/cnpro.py",
                 ("decode_weight_mask_parts", "decode_weight_mask"),
                 {"np": np, "cv2": cv2, "WEIGHT_MASK_HUE_SPAN": 270.0,
                  "logger": types.SimpleNamespace(warning=warnings.append)})
    if ns is None:
        return report()
    ec = extract("lib_cnpro/external_code.py", ("input_mask_share",), {"np": np})
    if ec is None:
        return report()

    decode_parts = ns["decode_weight_mask_parts"]
    decode = ns["decode_weight_mask"]
    share_of = ec["input_mask_share"]

    def reduce_mask(rgba):
        return share_of(*decode_parts(rgba))

    # -- 1. ONE weight, whatever shape it was painted in -------------------
    #
    # Every one of these is a single click or drag of the brush at one slider
    # position. The share is that slider position and there is no gradient to
    # report, no matter how much of the stroke is edge.
    shapes = [
        ("a broad anti-aliased disc", disc(np)),
        ("a thin anti-aliased stroke", line(np)),
        ("a feathered disc", feathered(disc(np), 6, np)),
        ("a disc downscaled to a third", downscaled(disc(np), 85, np)),
        ("a thin stroke, feathered until nothing is solid",
         feathered(line(np, half_width=1.5), 4, np)),
    ]
    for label, coverage in shapes:
        allowed = wire_tolerance(coverage, np)
        for weight in (1.0, 0.75, 0.5, 0.25):
            share, graded = reduce_mask(paint(coverage, weight, np))
            if share is None:
                fail("%s painted at %.2f decoded to nothing at all" % (label, weight))
                continue
            if abs(share - weight) > allowed:
                fail("%s painted at %.2f reduced to %.4f (off by %.4f, and the "
                     "wire format allows %.4f) - the share of a mask painted "
                     "with one weight IS that weight, whatever the shape's "
                     "perimeter" % (label, weight, share, abs(share - weight),
                                    allowed))
            if graded:
                fail("%s painted at %.2f was reported as painted with more than "
                     "one weight - it holds exactly one, and the user gets a "
                     "warning about a gradient they did not paint"
                     % (label, weight))

    # -- 2. ...and the old reduction really did fail these ------------------
    #
    # AGENTS.md section 5: prove the fix by breaking it again. This is the
    # previous implementation - mean and spread over the per-pixel weight, the
    # two channels already multiplied - and it has to be visibly wrong on the
    # very first case, or nothing above is evidence of anything.
    def old_reduction(rgba):
        values = decode(rgba)
        kept = values[values > 0]
        return float(kept.mean()), float(kept.max() - kept.min()) > 5e-3

    for label, coverage in shapes[:2]:
        old_share, old_graded = old_reduction(paint(coverage, 1.0, np))
        if not old_graded:
            fail("the pre-fix reduction does not report %s at weight 1 as "
                 "graded, so this test no longer discriminates - check the "
                 "shapes still have anti-aliased edges" % label)
        if abs(old_share - 1.0) <= TOL:
            fail("the pre-fix reduction returns %.4f for %s at weight 1, which "
                 "is within tolerance - this test would have passed on the bug"
                 % (old_share, label))

    # -- 3. genuinely two weights still says so ----------------------------
    #
    # The warning is not being deleted, only aimed. Two regions painted at
    # different slider positions is what it was written for, and the share is
    # their coverage-weighted mean.
    two = np.zeros((128, 128), np.float64)
    two[20:60, 20:108] = 1.0
    two[70:110, 20:108] = 1.0
    rgba = paint(two, 1.0, np)
    lower = paint(two, 0.25, np)
    rgba[70:110] = lower[70:110]          # the second region at a lower weight
    share, graded = reduce_mask(rgba)
    if not graded:
        fail("a mask painted 1.0 over one region and 0.25 over another was NOT "
             "reported as holding more than one weight - the warning that "
             "tells the user their per-region paint was flattened is gone")
    if abs(share - 0.625) > TOL:
        fail("two equal regions at 1.0 and 0.25 averaged to %.4f, expected "
             "0.625" % share)

    # ...including a difference of one slider step, which is the smallest thing
    # the painter can express and still has to clear the noise floor.
    stepped = paint(two, 0.5, np)
    nudged = paint(two, 0.5 + PAINT_STEP, np)
    stepped[70:110] = nudged[70:110]
    _, graded = reduce_mask(stepped)
    if not graded:
        fail("two regions one slider step apart (0.50 and 0.51) were reported "
             "as one weight - the noise floor has grown past what the painter "
             "can express")

    # -- 4. an explicitly painted 0 is excluded, not averaged in ------------
    #
    # Painting 0 is how a region is cut OUT of what an input contributes (the
    # eraser removes paint entirely; a 0 brush marks it). It must not drag the
    # share of everything else down - the value is what was painted where the
    # input counts, not an average over the frame.
    holed = paint(two, 1.0, np)
    zeroed = paint(two, 0.0, np)
    holed[70:110] = zeroed[70:110]
    share, graded = reduce_mask(holed)
    if share is None or abs(share - 1.0) > TOL:
        fail("a region painted at 1.0 beside a region painted at 0 gave share "
             "%s, expected 1.0 - painted zero excludes, it does not dilute"
             % (None if share is None else "%.4f" % share))
    if graded:
        fail("a region painted at 1.0 beside a region painted at 0 was called "
             "multi-weight - the 0 is a hole in the shape, not a second value")

    # -- 5. the legacy rainbow wire format reduces the same -----------------
    for weight in (1.0, 0.6, 0.3):
        share, graded = reduce_mask(paint(disc(np), weight, np, chromatic=True))
        # the hue channel is stored in 2-degree steps, so a weight round-trips
        # through it no closer than 2/270; that is the format's own resolution
        # and not something the reduction can improve on
        if share is None or abs(share - weight) > TOL + 2.0 / 270.0:
            fail("a LEGACY rainbow disc painted at %.2f reduced to %s"
                 % (weight, None if share is None else "%.4f" % share))
        if graded:
            fail("a LEGACY rainbow disc painted at %.2f was reported as holding "
                 "more than one weight" % weight)

    # -- 6. nothing painted is nothing, not zero ----------------------------
    empty = np.zeros((32, 32, 4), np.uint8)
    if decode_parts(empty) != (None, None):
        fail("an unpainted mask decodes to something - an input with no mask "
             "must fall back to a full share, not to weight 0")
    if reduce_mask(empty) != (None, False):
        fail("an unpainted mask does not reduce to (None, False)")
    if share_of(None) != (None, False):
        fail("input_mask_share(None) must be (None, False) - it is the "
             "no-mask path every unpainted input takes")

    # a mask that IS painted, but painted 0 everywhere: still no share to give
    all_zero = paint(disc(np), 0.0, np)
    if reduce_mask(all_zero) != (None, False):
        fail("a mask painted 0 everywhere must give no share (the caller reads "
             "that as 'unpainted'), not a share of 0")

    # -- 7. the SPATIAL reading keeps the ramp ------------------------------
    #
    # The split must not have quietly hardened mask edges: a tensor and a gate
    # read the product, and the feather has to survive into it. This is the
    # half of the decode that was always right, pinned so the fix cannot have
    # traded one for the other.
    soft = paint(feathered(disc(np), 6, np), 1.0, np)
    values, coverage = decode_parts(soft)
    product = decode(soft)
    if values is None or product is None:
        fail("a feathered disc decoded to nothing")
    else:
        if abs(float(np.abs(product - values * coverage).max())) > 1e-6:
            fail("decode_weight_mask is no longer the product of the two parts "
                 "it splits into - the tensor path and the share path have "
                 "stopped describing the same mask")
        ramp = product[(product > 0.02) & (product < 0.98)]
        if ramp.size < 100:
            fail("the feathered edge did not survive into the per-pixel weight "
                 "(%d intermediate pixels) - mask edges are hard again"
                 % ramp.size)
        core = values[coverage >= 0.5]
        if float(core.max() - core.min()) > PAINT_STEP:
            fail("the VALUE channel of a feathered single-weight mask is not "
                 "constant where it is covered (%.4f..%.4f) - the falloff is "
                 "supposed to live in coverage alone"
                 % (float(core.min()), float(core.max())))

    # -- 8. the generation path still hands over the PARTS ------------------
    #
    # The one thing above cannot see: `input_share_for` lives inside the unit
    # loop and is not extractable, and the whole bug was a call site passing
    # the product to a reducer. Collapsing `*decode_slot_parts(i)` back to
    # `decode_slot_mask(i)` type-checks, runs, and silently restores every
    # failure this file lists.
    src = read("scripts/cnpro.py")
    call = re.search(r"external_code\.input_mask_share\(\s*([^)]*)\)", src)
    if not call:
        fail("scripts/cnpro.py no longer calls external_code.input_mask_share")
    elif "decode_slot_parts" not in call.group(1):
        fail("the share is computed from `%s` instead of the mask's two parts "
             "- a reducer given the per-pixel weight reads every anti-aliased "
             "stroke edge as a different painted value" % call.group(1).strip())
    for reader, what in (("gate_values = decode_slot_mask", "the input gate"),
                         ("global_values = decode_slot_mask", "the mask tensor")):
        if reader not in src:
            fail("%s no longer reads the per-pixel weight (decode_slot_mask) - "
                 "the SPATIAL readers want the two channels multiplied, and "
                 "only the reduction wants them apart" % what)

    # -- 9. a mask with no alpha is refused, out loud -----------------------
    warnings.clear()
    if decode_parts(np.full((8, 8, 3), 200, np.uint8)) != (None, None):
        fail("an RGB mask with no alpha channel was accepted - alpha is what "
             "marks painted pixels and there is nothing to read without it")
    if not any("RGBA" in str(w) for w in warnings):
        fail("an RGB mask with no alpha was dropped without saying so")

    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    for s in SKIPS:
        print("SKIPPED -", s)
    print("ok - a mask painted with one weight reduces to that weight and "
          "warns about nothing, whatever its shape, feather or scale; two "
          "painted weights (down to one slider step) still say so; painted "
          "zero excludes rather than dilutes; and the per-pixel weight the "
          "tensors read is still the product of the two channels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
