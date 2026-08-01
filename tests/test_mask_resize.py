"""Weight/output masks must survive resizing with their VALUES intact.

THE BUG THIS EXISTS FOR
-----------------------
Painted masks used to travel through `high_quality_resize`, which was written
for DETECTED MAPS and reads a mask's statistics as if it were one:

  * a uniform paint at weight >= ~0.945 (gray >= 241, plus the transparent 0
    background) has two levels -> "binary" -> OTSU-thresholded to exactly 0/255
    after the resize. The painted 0.95 ran as 1.00 whenever generation dims
    differed from canvas dims, silently.
  * a light feather ramp (3..199 unique levels) -> INTER_NEAREST, stair-
    stepping the very edge the feather exists to smooth.

The fix routes masks through `mask_resize` (AREA down / LINEAR up, nothing
thresholded) via `crop_and_resize_mask` - the SAME geometry code as the hint
(`crop_and_resize_image`), only the resampler swapped, so the paint stays
aligned with the source under every resize mode.

WHAT IS PINNED HERE
-------------------
1. A uniform 0.95-weight mask survives a resize at 0.95 (the OTSU snap).
2. A feather ramp stays monotone and smooth (the INTER_NEAREST staircase).
3. Geometry parity: the mask path and the hint path agree about where a
   painted spot LANDS under RESIZE / OUTER_FIT / INNER_FIT.
4. The mask path can never fall back into the detected-map resampler (the
   extracted code is executed with a high_quality_resize that raises).
5. The generation path actually calls crop_and_resize_mask for weight and
   output masks (declared here, honoured there).

Run:  python tests/test_mask_resize.py     (numpy + cv2 only, no host)
Exit code 0 = pass.
"""
import os
import re
import sys
import types

import numpy as np

try:
    import cv2
except ImportError:
    print("SKIPPED - cv2 is not importable here; the mask resample behaviour "
          "was NOT verified")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def read(rel):
    with open(os.path.join(EXTENSION, rel), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# extract the real functions (utils.py imports the host; same pattern as
# test_defaults.py). external_code is stubbed to its two enum members, and
# high_quality_resize is replaced by a tripwire: if the mask path ever reaches
# the detected-map resampler again, this test fails by construction.
# ---------------------------------------------------------------------------

def load_functions():
    src = read("lib_cnpro/utils.py")
    parts = []
    for name in ("mask_resize", "safe_numpy", "crop_and_resize_image",
                 "crop_and_resize_mask"):
        m = re.search(r"^def %s\(.*?(?=\n\ndef |\n\nclass )" % name, src, re.M | re.S)
        if not m:
            fail("lib_cnpro/utils.py no longer defines %s" % name)
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

    def hq_tripwire(x, size):
        raise AssertionError(
            "high_quality_resize ran on the MASK path - the detected-map "
            "heuristics (OTSU snap, INTER_NEAREST) are back on painted weights")

    ns = {"np": np, "cv2": cv2, "external_code": stub,
          "high_quality_resize": hq_tripwire}
    exec("\n\n".join(parts), ns)
    return ns


def hwc3(gray):
    return np.stack([gray] * 3, axis=2)


def main():
    ns = load_functions()
    if ns is None:
        return report()
    # the tripwire raises; a harness must SURVIVE a broken module and report
    # it, so the raise is converted into a named failure, not a stack trace
    def guarded(fn):
        def call(*args):
            try:
                return fn(*args)
            except AssertionError as exc:
                fail(str(exc))
                return np.zeros((2, 2, 3), dtype=np.uint8)
        return call
    mask_resize = guarded(ns["mask_resize"])
    crop_and_resize_mask = guarded(ns["crop_and_resize_mask"])
    modes = ns["external_code"].ResizeMode

    # 1. the OTSU snap: uniform 0.95 paint (gray 242 on black) must stay ~242.
    #    Pre-fix, high_quality_resize saw two levels and thresholded to 255 -
    #    re-introduce the old routing and this fails with "snapped to 1.0".
    m = np.zeros((512, 512), dtype=np.uint8)
    m[128:384, 128:384] = 242
    out = mask_resize(hwc3(m), (768, 768))
    center = int(out[384, 384, 0])
    if abs(center - 242) > 1:
        fail("uniform weight 0.95 came back as %d/255 = %.3f after resize - "
             "painted weights must survive (the OTSU snap ran it at 1.0)"
             % (center, center / 255.0))
    values = np.unique(out[out > 0])
    if len(values) and int(values.max()) == 255 and 242 not in values:
        fail("resize produced pure 255 from 242-valued paint - thresholded")

    # 2. the feather staircase: a smooth ramp must stay smooth. Pre-fix the
    #    3..199-unique-level heuristic resized it with INTER_NEAREST, which
    #    copies source pixels - a STEEP ramp (adjacent levels 13 apart)
    #    upscaled 16x keeps those 13-level cliffs, while LINEAR splits them
    #    into ~1-level slopes. The first version of this check used a shallow
    #    ramp whose nearest-jumps slid under the threshold, and it passed on
    #    the broken code (rule 5: this line was proven by breaking it).
    ramp = np.tile(np.linspace(0, 195, 16).astype(np.uint8), (16, 1))
    up = mask_resize(hwc3(ramp), (256, 256))
    row = up[128, :, 0].astype(int)
    steps = np.abs(np.diff(row))
    if steps.max() > 6:
        fail("feather ramp came back stair-stepped (max adjacent jump %d, "
             "linear gives ~1-2) - nearest-neighbour is back on the mask path"
             % steps.max())

    # 3. geometry parity across the three modes: a spot painted at a known
    #    place must land where the shared geometry puts the hint. The values
    #    below are computed from crop_and_resize_image's own fit math (k =
    #    min/max of the axis ratios, centered pad/crop), which the mask path
    #    shares by construction - a divergence means the wrapper stopped
    #    sharing the geometry.
    spot = np.zeros((100, 200), dtype=np.uint8)  # h=100, w=200
    spot[50, 100] = 255

    out = crop_and_resize_mask(hwc3(spot), modes.RESIZE, 200, 200)
    if out.shape[:2] != (200, 200):
        fail("RESIZE: expected 200x200, got %r" % (out.shape,))
    ys, xs = np.nonzero(out[:, :, 0])
    if len(ys) == 0 or abs(ys.mean() - 100) > 3 or abs(xs.mean() - 100) > 3:
        fail("RESIZE: the painted spot moved (found at y~%s x~%s, expected "
             "100,100)" % (ys.mean() if len(ys) else None,
                           xs.mean() if len(xs) else None))

    # OUTER_FIT to a taller target: k = min(200/100, 200/200) = 1, image
    # centered vertically with 50px pads -> spot at (100, 100)
    out = crop_and_resize_mask(hwc3(spot), modes.OUTER_FIT, 200, 200)
    ys, xs = np.nonzero(out[:, :, 0] > 128)
    if len(ys) == 0 or abs(ys.mean() - 100) > 3 or abs(xs.mean() - 100) > 3:
        fail("OUTER_FIT: spot expected at (100,100) after centered padding, "
             "found y~%s x~%s" % (ys.mean() if len(ys) else None,
                                  xs.mean() if len(xs) else None))

    # INNER_FIT to a square: k = max(2, 1) = 2 -> 200x400, centered crop of
    # 100 px per side -> spot at (100, 100)
    out = crop_and_resize_mask(hwc3(spot), modes.INNER_FIT, 200, 200)
    ys, xs = np.nonzero(out[:, :, 0] > 128)
    if len(ys) == 0 or abs(ys.mean() - 100) > 3 or abs(xs.mean() - 100) > 3:
        fail("INNER_FIT: spot expected at (100,100) after centered crop, "
             "found y~%s x~%s" % (ys.mean() if len(ys) else None,
                                  xs.mean() if len(xs) else None))

    # 4. (implicit) high_quality_resize is a tripwire in this namespace: if any
    #    of the calls above had reached it, they would have raised.

    # 5. the generation path uses the mask route for every mask it prepares
    cnpro = read("scripts/cnpro.py")
    wm = re.search(r"def prepare_weight_mask\(.*?return tensor, tensor_hr",
                   cnpro, re.S)
    if not wm:
        fail("scripts/cnpro.py no longer defines prepare_weight_mask where "
             "this test can see it")
    elif "crop_and_resize_mask(" not in wm.group(0):
        fail("prepare_weight_mask does not use crop_and_resize_mask - painted "
             "weights are back on the detected-map heuristics")
    out_mask = re.search(r"output_mask_values = decode_weight_mask.*?params\.output_mask_for_hr_fix = ",
                         cnpro, re.S)
    if out_mask and "crop_and_resize_mask(" not in out_mask.group(0):
        fail("the output-mask path does not use crop_and_resize_mask")

    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - painted weights survive every resize (no OTSU snap, no "
          "nearest staircase), the mask path shares the hint path's geometry "
          "in all three modes, cannot reach the detected-map resampler, and "
          "the generation path routes every mask through it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
