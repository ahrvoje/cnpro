"""API base64 image decode must yield true pixel values for every PNG shape.

THE BUG THIS EXISTS FOR
-----------------------
The host's decode (Image.open + exif_transpose + fix_png_transparency)
normalizes nothing, and `np.array(...).astype('uint8')` then mangled two
real-world shapes into hints that still generated plausible images:

  * a palette-mode PNG without transparency (optipng/pngcrush output for
    binary edge maps) decodes to its palette INDICES - a black/white canny map
    with palette {0: black, 1: white} became an image of 0s and 1s, near
    black, the control near-inert;
  * a 16-bit grayscale PNG (MiDaS depth exports) decodes to uint16 0..65535,
    and .astype('uint8') WRAPS mod 256 into banded noise.

Both read as "the unit is badly configured". external_code.decode_base64_image_array
is the fix; this file pins its behaviour and that every decode site uses it.

WHAT IS PINNED HERE
-------------------
1. Palette PNG -> true colors (white stays 255, not index 1).
2. Palette PNG with transparency -> RGBA, alpha preserved.
3. 16-bit grayscale -> rescaled by 255/65535, never wrapped.
4. RGBA stays RGBA byte-exact (mask feather rides the alpha channel).
5. Plain RGB stays byte-exact.
6. Every decode site (from_dict image/mask/weight-mask loops, api.py detect)
   routes through decode_base64_image_array - no raw np.array(...).astype
   remains.

Run:  python tests/test_api_images.py     (numpy + PIL only, no host)
Exit code 0 = pass.
"""
import base64
import io
import os
import re
import sys
import types

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:
    print("SKIPPED - PIL is not importable here; the decode behaviour was "
          "NOT verified")
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
# extract the real function (external_code.py imports the host; same pattern
# as test_defaults.py). The stub `api` reproduces the host decode faithfully:
# Image.open + exif_transpose, NO mode normalization - that is the exact
# behaviour the function under test exists to compensate for.
# ---------------------------------------------------------------------------

def load_decoder():
    src = read("lib_cnpro/external_code.py")
    m = re.search(r"^def decode_base64_image_array\(.*?(?=\n\ndef |\n\nclass |\n\n@)",
                  src, re.M | re.S)
    if not m:
        fail("external_code.py no longer defines decode_base64_image_array")
        return None

    def host_decode(encoding):
        img = Image.open(io.BytesIO(base64.b64decode(encoding)))
        return ImageOps.exif_transpose(img)

    ns = {"np": np, "api": types.SimpleNamespace(decode_base64_to_image=host_decode)}
    exec(m.group(0), ns)
    return ns["decode_base64_image_array"]


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    decode = load_decoder()
    if decode is None:
        return report()

    # 1. palette without transparency: a binary edge map, palette {0, 1}
    edge = Image.new("P", (32, 32), 0)
    edge.putpalette([0, 0, 0, 255, 255, 255] + [0] * (254 * 3))
    for x in range(32):
        edge.putpixel((x, 16), 1)  # a white line
    arr = decode(png_b64(edge))
    if arr.ndim == 2:
        fail("palette PNG decoded to a 2-D index array - the palette was "
             "never applied and a canny map is an image of 0s and 1s")
    else:
        line = int(arr[16, 16, 0]) if arr.ndim == 3 else int(arr[16, 16])
        if line != 255:
            fail("palette white decoded to %d, expected 255 - the control "
                 "would be near-inert" % line)

    # 2. palette WITH transparency: alpha must survive (masks ride alpha)
    pal = Image.new("P", (8, 8), 1)
    pal.putpalette([0, 0, 0, 200, 100, 50] + [0] * (254 * 3))
    pal.info["transparency"] = bytes([0] + [255] * 255)
    arr = decode(png_b64(pal))
    if arr.ndim != 3 or arr.shape[2] != 4:
        fail("palette-with-transparency decoded to shape %r - the alpha "
             "channel (mask feather) was dropped" % (arr.shape,))

    # 3. 16-bit grayscale: full-range gradient must rescale, not wrap.
    #    Pre-fix, 32768 (mid gray) wrapped to 0 and 65535 (white) to 255 while
    #    65280 wrapped to 0 - banded noise.
    g16 = np.zeros((16, 16), dtype=np.uint16)
    g16[:, 8:] = 32768   # mid gray
    g16[:, 12:] = 65535  # white
    img16 = Image.fromarray(g16)  # mode I;16
    arr = decode(png_b64(img16))
    mid = int(arr[8, 10]) if arr.ndim == 2 else int(arr[8, 10, 0])
    white = int(arr[8, 14]) if arr.ndim == 2 else int(arr[8, 14, 0])
    if not (120 <= mid <= 135):
        fail("16-bit mid gray (32768) decoded to %d, expected ~128 - the "
             "value wrapped mod 256 instead of rescaling" % mid)
    if white != 255:
        fail("16-bit white (65535) decoded to %d, expected 255" % white)

    # 4. RGBA byte-exact (weight masks: value in RGB, feather in alpha)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 242
    rgba[..., 3] = np.linspace(0, 255, 8).astype(np.uint8)[None, :]
    arr = decode(png_b64(Image.fromarray(rgba)))
    if arr.shape != (8, 8, 4) or not np.array_equal(arr, rgba):
        fail("RGBA did not round-trip byte-exact (shape %r) - the previous "
             "behaviour must be preserved for every already-working mode"
             % (arr.shape,))

    # 5. plain RGB byte-exact
    rgb = (np.arange(8 * 8 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(8, 8, 3)
    arr = decode(png_b64(Image.fromarray(rgb)))
    if not np.array_equal(arr, rgb):
        fail("plain RGB did not round-trip byte-exact")

    # 6. every decode site routes through the function (both directions of the
    #    declared/honoured check: the fixed decoder exists AND nothing bypasses
    #    it)
    ext = read("lib_cnpro/external_code.py")
    api_src = read("lib_cnpro/api.py")
    for name, src in (("external_code.py", ext), ("api.py", api_src)):
        if re.search(r"np\.array\(api\.decode_base64_to_image\([^)]*\)\)\.astype", src):
            fail("%s still decodes with raw np.array(...).astype('uint8') - "
                 "palette and 16-bit PNGs are mangled on that path" % name)
    if ext.count("decode_base64_image_array(") < 5:
        fail("external_code.from_dict no longer routes its image/mask fields "
             "through decode_base64_image_array")
    if "decode_base64_image_array(" not in api_src:
        fail("api.py /controlnet/detect does not use decode_base64_image_array")

    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - palette PNGs decode to true colors (alpha preserved), 16-bit "
          "grayscale rescales instead of wrapping, RGB/RGBA round-trip "
          "byte-exact, and every decode site routes through the fixed decoder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
