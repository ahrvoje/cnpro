"""The coverage panel must draw the geometry and the arithmetic the RUN uses.

WHAT THIS EXISTS FOR
--------------------
The panel under the CNPro title claims to show, in output geometry, what every
enabled unit together will do to the image. It is a prediction, and a prediction
nobody checks is decoration: a wrong one is not visibly wrong - it is a
plausible picture of a run that will not happen, and the user's next move is
made from it.

Two of its three halves are checkable here, and both have a twin that already
exists somewhere else:

1. GEOMETRY. `coverage_map.js fitRect` says where a mask painted at some input
   resolution lands inside the output rectangle. `lib_cnpro/utils.py
   crop_and_resize_image` is what actually puts it there (via
   crop_and_resize_mask, which shares its geometry exactly). They are compared
   through an OBSERVABLE: a marker painted at a known source pixel is pushed
   through the real python function, its centroid is measured in the output, and
   fitRect has to predict that position under all three resize modes.
1b. THE PAD COLOUR. "Resize and Fill" letterboxes with the MEDIAN of the source
   border, and `borderMedian` has to be numpy's median, not the middle element:
   a border is 2w + 2h samples, always even, numpy averages the two middle ones
   and `.astype` truncates. The first version took the upper middle, and a
   half-painted mask filled its letterbox with 255 instead of 127 - measured in
   a browser as "62% of the frame covered" for paint that covers 37%, the map
   claiming control over two bands the source does not reach.
2. AGGREGATION. `aggregate()` folds the per-unit contributions into the mean and
   the peak field. The two answer different questions (max of a sum is not the
   sum of maxima) and the panel offers both, so both are pinned - including the
   property the whole panel is for: contributions ADD, so two units at weight 1
   over the same pixel read as 2 and land above the orange contour.

Plus the contract the legend depends on: level 1 is orange, everything above it
is red, and the ramp past 1 keeps darkening instead of flattening into one red.

The third half - reading the live DOM - is not checkable without a browser and
is covered by the same Playwright flow as the rest of the UI (AGENTS.md §1).

Run:  python tests/test_coverage_map.py
Needs `node` (the JS math is loaded headless). The geometry parity half also
needs numpy + cv2; without them it SKIPS LOUDLY and says what went unverified.
Exit code 0 = pass.
"""
import json
import os
import re
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []
SKIPS = []


def fail(msg):
    FAILURES.append(msg)


def read(rel):
    with open(os.path.join(EXTENSION, rel), encoding="utf-8") as fh:
        return fh.read()


def run_js(payload):
    harness = os.path.join(HERE, "coverage_map_js.js")
    try:
        proc = subprocess.run(["node", harness], input=json.dumps(payload),
                              capture_output=True, text=True, cwd=HERE)
    except (OSError, FileNotFoundError):
        return None
    if proc.returncode != 0:
        fail("the node harness failed:\n%s" % proc.stderr.strip()[:2000])
        return None
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# the real python geometry, without the host (same extraction as
# tests/test_mask_resize.py: utils.py imports external_code for the enum only)
# ---------------------------------------------------------------------------

def load_geometry():
    try:
        import numpy as np
        import cv2  # noqa: F401
    except ImportError as exc:
        SKIPS.append("numpy/cv2 are not importable here (%s), so the coverage "
                     "panel's geometry was NOT compared against "
                     "crop_and_resize_image - only the JS half ran" % exc)
        return None
    src = read("lib_cnpro/utils.py")
    parts = []
    for name in ("mask_resize", "safe_numpy", "high_quality_resize",
                 "crop_and_resize_image", "crop_and_resize_mask"):
        m = re.search(r"^def %s\(.*?(?=\n\ndef |\n\nclass )" % name, src, re.M | re.S)
        if not m:
            fail("lib_cnpro/utils.py no longer defines %s - the coverage panel's "
                 "geometry has nothing left to be compared against" % name)
            return None
        parts.append(m.group(0))

    class _Mode:
        def __init__(self, v):
            self.value = v

    stub = types.SimpleNamespace(ResizeMode=types.SimpleNamespace(
        RESIZE=_Mode("Just Resize"),
        OUTER_FIT=_Mode("Resize and Fill"),
        INNER_FIT=_Mode("Crop and Resize"),
    ))
    import cv2 as _cv2
    ns = {"np": np, "cv2": _cv2, "external_code": stub}
    exec("\n\n".join(parts), ns)
    return ns


# source rasters and outputs worth asking about: square -> wide, wide -> square,
# a scale-down and a scale-up, and one pair that is neither aspect nor multiple
# of the other (the rounding cases)
FIT_CASES = [
    (512, 512, 1024, 1024),
    (1024, 512, 1024, 1024),
    (512, 1024, 1024, 1024),
    (900, 600, 1024, 768),
    (333, 777, 640, 512),
    (2048, 1152, 512, 512),
]
MODES = ["Just Resize", "Resize and Fill", "Crop and Resize"]

# Border samples whose median the letterbox is filled with. The second is the
# shape that caught the bug: a mask painted over exactly half the frame, whose
# border is half 255 and half 0 - numpy answers 127, "the middle element"
# answers 255, and the difference is a quarter of the output frame claiming to
# be under full control.
MEDIAN_CASES = [
    [0] * 8,
    [255] * 4 + [0] * 4,
    [255] * 5 + [0] * 3,
    [0, 10, 20, 30, 40, 200],
    [7],
    [3, 4],
]


def marker_centroid(np, out):
    """Centre of the bright marker in a rendered output, or None.

    In PIXEL INDICES, which is half a pixel below the continuous centre - the
    prediction below subtracts that 0.5 rather than this adding it, so the two
    coordinate conventions are converted in exactly one place.
    """
    gray = out[:, :, 0].astype(float)
    ys, xs = (gray > 128).nonzero()
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def predict(rect, cx, cy, sw, sh):
    """Where fitRect says a source point lands, as an output pixel index.

    `cx`/`cy` are CONTINUOUS source coordinates: a marker filling source pixels
    [c - half, c + half) has its centre at exactly c.
    """
    return (rect["dx"] + cx * rect["dw"] / sw - 0.5,
            rect["dy"] + cy * rect["dh"] / sh - 0.5)


def check_medians(js_medians):
    """The pad colour, against numpy's own median - the code it mirrors."""
    import numpy as np
    for values, got in zip(MEDIAN_CASES, js_medians):
        want = int(np.median(np.array(values, dtype=np.uint8)).astype(np.uint8))
        if got != want:
            fail("borderMedian(%r) = %r, numpy's median (which is what "
                 "crop_and_resize_image pads with) = %r. The letterbox of a "
                 "'Resize and Fill' mask would be filled with a weight the "
                 "generation does not use." % (values, got, want))


def check_geometry(ns, js_fits):
    import numpy as np
    modes = ns["external_code"].ResizeMode
    by_name = {"Just Resize": modes.RESIZE,
               "Resize and Fill": modes.OUTER_FIT,
               "Crop and Resize": modes.INNER_FIT}
    index = 0
    for (sw, sh, w, h) in FIT_CASES:
        for mode in MODES:
            rect = js_fits[index]
            index += 1
            # a small bright square at the source centre: it survives every
            # mode's crop, which a corner marker would not
            half = max(2, min(sw, sh) // 32)
            # the centre, which every mode keeps, and one off-centre point,
            # which is the only thing that can see the SCALE (a wrong k still
            # puts the centre in the middle)
            probes = [(sw // 2, sh // 2), (int(sw * 0.75), int(sh * 0.25))]
            for (cx, cy) in probes:
                want = predict(rect, cx, cy, sw, sh)
                # a marker the mode CROPS in half has a centroid that measures
                # the surviving half, not the marker: that says nothing about
                # either implementation, so it is skipped rather than fudged
                margin = half * max(rect["dw"] / sw, rect["dh"] / sh) + 2
                if not (margin <= want[0] < w - margin
                        and margin <= want[1] < h - margin):
                    continue
                src = np.zeros((sh, sw, 3), dtype=np.uint8)
                src[cy - half:cy + half, cx - half:cx + half] = 255
                out = ns["crop_and_resize_mask"](src, by_name[mode], h, w)
                got = marker_centroid(np, out)
                if got is None:
                    fail("%dx%d -> %dx%d (%s): fitRect places the marker at "
                         "(%.1f, %.1f), the real resize dropped it from the "
                         "output entirely" % (sw, sh, w, h, mode, want[0], want[1]))
                    continue
                # one output pixel of slack: both sides round, and the centroid
                # is measured off thresholded, resampled pixels
                if abs(got[0] - want[0]) > 1.0 or abs(got[1] - want[1]) > 1.0:
                    fail("%dx%d -> %dx%d (%s): coverage_map.js fitRect puts the "
                         "source point (%d, %d) at (%.1f, %.1f), "
                         "crop_and_resize_mask puts it at (%.1f, %.1f). The "
                         "coverage map would show masks somewhere the generation "
                         "does not."
                         % (sw, sh, w, h, mode, cx, cy,
                            want[0], want[1], got[0], got[1]))


def main():
    steps_probe = run_js({"levels": [0]})
    if steps_probe is None:
        print("SKIPPED - node is not available (or the harness failed), so the "
              "coverage panel's math was NOT verified")
        return 1 if FAILURES else 0
    steps = steps_probe["steps"]

    # ---- 1. the fit rectangles, asked of JS once for every case/mode pair
    fits = [{"sw": sw, "sh": sh, "w": w, "h": h, "mode": mode}
            for (sw, sh, w, h) in FIT_CASES for mode in MODES]

    # ---- 2. aggregation. Four pixels, two units:
    #   A covers everything at weight 1 for the first half of the schedule,
    #   B covers pixels 0 and 3 at weight 1 for the SECOND half.
    # mean: they share the schedule, so the overlap is 1, not 2.
    # peak: they never overlap in time, so the peak is also 1 - which is the
    #       whole reason both exist. The second case makes them simultaneous
    #       and the peak has to become 2 while the mean stays 1.
    half = steps // 2
    a_w = [1.0 if t < half else 0.0 for t in range(steps)]
    b_w = [0.0 if t < half else 1.0 for t in range(steps)]
    both_w = [1.0] * steps
    payload = {
        "fits": fits,
        "aggregates": [
            {"npx": 4, "contributions": [
                {"mask": None, "w": a_w},
                {"mask": [255, 0, 0, 255], "w": b_w},
            ]},
            {"npx": 4, "contributions": [
                {"mask": None, "w": both_w},
                {"mask": [255, 0, 0, 255], "w": both_w},
            ]},
            # a half-strength mask: 128/255 of the weight, not all or nothing -
            # the feathered edge of every painted mask depends on it
            {"npx": 2, "contributions": [
                {"mask": [128, 255], "w": both_w},
            ]},
        ],
        # a 3x3 field with a plateau in the middle: the level-0.5 contour must
        # exist and must enclose the plateau rather than run off the grid
        "contour": {"w": 3, "h": 3, "level": 0.5,
                    "field": [0, 0, 0, 0, 1, 0, 0, 0, 0]},
        "levels": [0.4, 1.0, 1.7],
        "colors": [0.0, 0.5, 1.0, 1.5, 3.0],
        "medians": MEDIAN_CASES,
    }
    got = run_js(payload)
    if got is None:
        print("FAIL - the node harness did not answer")
        return 1

    check_js(got, steps, half)

    ns = load_geometry()
    if ns is not None:
        check_medians(got["medians"])
        check_geometry(ns, got["fits"])

    return report()


def check_js(got, steps, half):
    # --- aggregation
    mean0, peak0 = got["aggregates"][0]["mean"], got["aggregates"][0]["peak"]
    a_mean = half / steps
    b_mean = (steps - half) / steps
    want_mean0 = [a_mean + b_mean, a_mean, a_mean, a_mean + b_mean]
    for i, (g, w) in enumerate(zip(mean0, want_mean0)):
        if abs(g - w) > 1e-6:
            fail("mean field pixel %d is %.4f, expected %.4f - the mean is the "
                 "average over the WHOLE schedule (a unit that runs half the "
                 "steps at 1 contributes half), and the map's 'total influence' "
                 "reading depends on exactly that" % (i, g, w))
    if max(peak0) > 1 + 1e-6:
        fail("two units that never run at the same step produced a peak of %.3f; "
             "taking turns is precisely what must NOT read as oversaturation"
             % max(peak0))

    mean1, peak1 = got["aggregates"][1]["mean"], got["aggregates"][1]["peak"]
    if abs(peak1[0] - 2.0) > 1e-6 or abs(peak1[1] - 1.0) > 1e-6:
        fail("two units covering the same pixel at weight 1 at the same step "
             "gave a peak of %r, expected 2 where they overlap and 1 where they "
             "do not - contributions ADD, and the oversaturation the panel "
             "exists to show is exactly that sum" % (peak1[:2],))
    if abs(mean1[0] - 2.0) > 1e-6:
        fail("the same two units gave a mean of %.3f over the pixel they share, "
             "expected 2" % mean1[0])

    mean2 = got["aggregates"][2]["mean"]
    if abs(mean2[0] - 128 / 255) > 1e-3 or abs(mean2[1] - 1.0) > 1e-6:
        fail("a mask painted at half weight contributed %r, expected [0.502, 1] "
             "- mask values are WEIGHTS, not a painted/unpainted flag, and a "
             "feathered edge is nothing but those in-between values" % (mean2,))

    # --- contours
    segments = got["contour"]
    if not segments:
        fail("no level-0.5 contour was traced around a plateau that crosses it - "
             "the contour tracer found nothing to draw")
    else:
        xs = [c for s in segments for c in (s[0], s[2])]
        ys = [c for s in segments for c in (s[1], s[3])]
        if not (0 < min(xs) and max(xs) < 2 and 0 < min(ys) and max(ys) < 2):
            fail("the level-0.5 contour of a centred plateau runs to x %r y %r; "
                 "it must close around the plateau, inside the grid"
                 % ([min(xs), max(xs)], [min(ys), max(ys)]))

    # --- level colours: the legend and the stylesheet name these three
    levels_low, levels_one, levels_over = got["levels"]
    one = [l for l in levels_one if abs(l["value"] - 1.0) < 1e-9]
    if not one or one[0]["color"] != "#ff9800":
        fail("the level-1 contour is %r, not the orange #ff9800 the legend and "
             "style.css both name" % (one,))
    reds = [l for l in levels_over if l["value"] > 1.0]
    if not reds or any(l["color"] != "#ff1744" for l in reds):
        fail("oversaturation contours above 1 are %r, expected every one of them "
             "red (#ff1744)" % (reds,))
    if [l for l in levels_low if l["value"] > 1.0]:
        fail("a field whose maximum is 0.4 was given contours above 1: %r"
             % (levels_low,))
    if not any(abs(l["value"] - 0.25) < 1e-9 for l in levels_one):
        fail("the fixed 0.25 contour is gone from %r" % (levels_one,))

    # --- the ramp past 1 must keep moving, or 1.0 and 3.0 paint identically
    colors = got["colors"]
    if colors[0][2] <= colors[0][0]:
        fail("weight 0 painted %r - the ramp must start at the violet end"
             % (colors[0],))
    if colors[3] == colors[2] or colors[4] == colors[3]:
        fail("weights 1, 1.5 and 3 paint as %r, %r, %r - past 1 the colour has "
             "to keep darkening, or every oversaturated region looks exactly as "
             "saturated as a correct one" % (colors[2], colors[3], colors[4]))


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    for s in SKIPS:
        print("PARTIAL SKIP -", s)
    # The summary states what RAN. A skip that still prints the full claim is
    # how "nothing was verified" reads as a pass (AGENTS.md §1).
    if SKIPS:
        print("ok (partial) - the coverage map sums contributions the way the "
              "units do, separates mean from peak and keeps the orange 1 / red "
              ">1 contract; its geometry was NOT compared against the real "
              "crop_and_resize_mask - see the skip above")
    else:
        print("ok - the coverage map places masks where crop_and_resize_mask "
              "does, sums contributions the way the units do, separates mean "
              "from peak, and keeps the orange 1 / red >1 contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
