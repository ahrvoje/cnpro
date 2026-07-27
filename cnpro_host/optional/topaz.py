"""1-click Topaz tools (Upscale, Enhance, Denoise, 1 Mpx resize) for ForgeCanvas images.

Exposes the locally installed Topaz Photo AI CLI (tpai.exe) to the canvas
toolbar: canvas_extra.js posts the current image as a PNG data url to
/forge-canvas/topaz/process and loads the result back into the canvas. The
toolbar buttons appear only when the status endpoint reports an executable
was found.

Photo AI is used instead of the new Gigapixel app on purpose: gigapixel.exe
refuses CLI use without an enterprise license, while tpai.exe works with a
regular (logged-in) license.

Empirical notes on the tpai settings API (3.1.1, everything A/B-verified by
comparing outputs - the parser swallows ANY key=value silently, so only an
output diff proves a key is live):

- `--override` does NOT stop autopilot from running its own enhancement
  picks (a denoise-only run came back 4x upscaled); every enhancement the
  tool does not want must be disabled explicitly with `enabled=false`.
- Upscale (the "Enhance" block): the named keys (denoise/deblur/compression
  and their minor_/fix_ variants) are dead; the generic slots param1/2/3
  are live, in GUI slider order - param1 = Minor denoise, param2 = Minor
  deblur, param3 = Fix compression, values 0..1 (GUI shows 0..100).
  Direction-verified: param1/param3 up -> smoother, param2 up -> sharper.
- Denoise (the "Noise" block): NO strength control works - param1/param2,
  strength/minor_deblur/original_detail, auto=false, explicit model= are
  all inert (or, for wrong model names, silently produce no output at
  all). The enhancement always runs at the model's auto-computed
  strengths; the only external influence is the Autopilot preferences
  panel in the Photo AI app ("Denoise (non-RAW) Strength").
- `scale` accepts FRACTIONAL values and values below 1 (A/B-verified:
  200x150 with scale=1.7 -> 340x255, with scale=0.5 -> 100x75), which is
  what the 1 Mpx tool resamples with. The named `width`/`height` keys are
  dead like all other named keys - tpai logs "Overwriting Enhance width to
  640" and then ignores it (200x150 came back 800x600, the autopilot
  scale), so target dimensions can only be reached through `scale`.

Route registration lives in scripts/forge_canvas_topaz.py: an import-time
on_app_started registration here would be erased by clear_callbacks() in
load_scripts() (modules/scripts.py), which runs after module imports but
before the app starts.
"""

import base64
import glob
import math
import os
import shutil
import subprocess
import tempfile
import threading
from io import BytesIO

from PIL import Image

from modules import shared

TPAI_CANDIDATES = [
    r"C:\Program Files\Topaz Labs LLC\Topaz Photo AI\tpai.exe",
    r"C:\Program Files\Topaz Photo AI\tpai.exe",
]

RUN_TIMEOUT = 600  # tpai loads its models on first run; big images take a while

# 1 Mpx tool: same megapixel convention as the resolution_1Mpx extension, so
# the canvas and the img2img sliders agree on what "1 Mpx" means
MPX_TARGET_PIXELS = 1024 * 1024
MPX_MULTIPLE = 16      # both dimensions must be divisible by this
MPX_SCALE_LIMITS = (0.2, 6.0)

# tpai exit codes (from --help)
TPAI_INVALID_LOGIN = 254

_run_lock = threading.Lock()  # one tpai process at a time


def find_tpai():
    """Full path of tpai.exe, or None. Checked per call - it is cheap and the
    tool appearing/disappearing without a server restart is a feature."""
    for path in TPAI_CANDIDATES:
        if os.path.isfile(path):
            return path
    return shutil.which("tpai")


def upscale_scale():
    # CNPro key first; the fork's key is honoured as a fallback so an
    # existing config.json carries over unchanged.
    d = shared.opts.data
    return int(d.get("cnpro_topaz_scale", d.get("forge_canvas_topaz_scale", 2)))


def target_dimensions(width, height, target_pixels=MPX_TARGET_PIXELS, multiple=MPX_MULTIPLE):
    """Closest width x height to target_pixels with the source aspect ratio,
    under the hard constraint that BOTH dimensions are divisible by `multiple`.

    Divisibility is the constraint, not a tie-breaker: the search only ever
    considers multiples of 16, and among those it minimizes aspect ratio error
    first and area error second. The aspect term is weighted heavily (8x) so
    that when the source ratio IS reachable exactly on the 16 grid (3:4, 1:1,
    ...) that candidate wins even though it sits a few percent below the pixel
    target. Both errors are logarithmic, i.e. symmetric in over/undershoot.
    """
    if width <= 0 or height <= 0:
        return multiple, multiple
    aspect = width / height
    ideal_w = math.sqrt(target_pixels * aspect)
    base = max(1, int(round(ideal_w / multiple)))
    best = None
    for step in range(-4, 5):
        cw = (base + step) * multiple
        if cw < multiple:
            continue
        ch = max(multiple, int(round(cw / aspect / multiple)) * multiple)
        err = 8 * abs(math.log((cw / ch) / aspect)) + abs(math.log((cw * ch) / target_pixels))
        if best is None or err < best[0]:
            best = (err, cw, ch)
    return best[1], best[2]


def mpx_plan(png_bytes):
    """(source size, target size, tpai scale) for the 1 Mpx tool. The scale
    covers both dimensions (max of the two ratios), so tpai's own rounding can
    only overshoot - the exact target is then reached by a small resample."""
    with Image.open(BytesIO(png_bytes)) as im:
        source = im.size
    target = target_dimensions(*source)
    scale = max(target[0] / source[0], target[1] / source[1])
    scale = min(max(scale, MPX_SCALE_LIMITS[0]), MPX_SCALE_LIMITS[1])
    return source, target, scale


def tool_args(tool, mpx_scale=None):
    """tpai argument list for a tool, or None for unknown tools. See the
    module docstring for why the lists look the way they do."""
    off_except = lambda *keep: [flag for name in ("upscale", "noise", "sharpen", "lighting", "color")
                                if name not in keep
                                for flag in (f"--{name}", "enabled=false")]
    if tool == "mpx1":
        # same enhancement as 'upscale' at the scale that lands on ~1 Mpx;
        # fractional and below-1 scales are both honored by tpai
        return ["--upscale", f"scale={mpx_scale:.4f}",
                "param1=1.0", "param2=0.01", "param3=0.9"] + off_except("upscale")
    if tool == "upscale":
        # Minor denoise 100, Minor deblur 1, Fix compression 90 (GUI scale)
        return ["--upscale", f"scale={upscale_scale()}",
                "param1=1.0", "param2=0.01", "param3=0.9"] + off_except("upscale")
    if tool == "hq":
        # same enhancement as 'upscale' at scale=1: dimensions unchanged, the
        # enhance model still runs and still honors paramN (A/B-verified)
        return ["--upscale", "scale=1",
                "param1=1.0", "param2=0.01", "param3=0.9"] + off_except("upscale")
    if tool == "denoise":
        # runs at auto-computed strengths - no CLI knob works (see docstring)
        return ["--noise"] + off_except("noise")
    return None


def fit_to(png_bytes, size):
    """Resample to exactly `size` (no-op when tpai already landed on it). The
    scale tpai gets is rounded to its own pixel grid, so the result is usually
    a few pixels off the divisible-by-16 target."""
    with Image.open(BytesIO(png_bytes)) as im:
        if im.size == tuple(size):
            return png_bytes
        out = im.resize(tuple(size), Image.LANCZOS)
        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()


def run_tool(png_bytes, tool):
    """Run a tpai tool on PNG bytes, return result PNG bytes. Raises
    RuntimeError with a user-presentable message on any failure."""
    exe = find_tpai()
    if exe is None:
        raise RuntimeError("Topaz Photo AI (tpai.exe) not found.")
    target = None
    if tool == "mpx1":
        source, target, scale = mpx_plan(png_bytes)
        print(f"[forge canvas] Topaz 1 Mpx: {source[0]}x{source[1]} -> "
              f"{target[0]}x{target[1]} (tpai scale {scale:.4f})")
        args = tool_args(tool, mpx_scale=scale)
    else:
        args = tool_args(tool)
    if args is None:
        raise RuntimeError(f"Unknown Topaz tool: {tool!r}")
    workdir = tempfile.mkdtemp(prefix="forge_topaz_")
    try:
        in_path = os.path.join(workdir, "input.png")
        out_dir = os.path.join(workdir, "out")
        with open(in_path, "wb") as f:
            f.write(png_bytes)
        cmd = [exe, in_path, "--output", out_dir, "--override",
               "--format", "png", "--compression", "2"] + args
        with _run_lock:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        if proc.returncode == TPAI_INVALID_LOGIN:
            raise RuntimeError("Topaz login token invalid - open Topaz Photo AI once to log in.")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(f"tpai exited with code {proc.returncode}: {' | '.join(tail)}")
        results = glob.glob(os.path.join(out_dir, "*"))
        if not results:
            # tpai also exits 0 on inputs it cannot process - the output
            # directory is the only reliable success signal
            raise RuntimeError("tpai reported success but produced no output file.")
        with open(results[0], "rb") as f:
            result = f.read()
        if target is not None:
            result = fit_to(result, target)
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"tpai timed out after {RUN_TIMEOUT}s.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def register_routes(_demo, app):
    from fastapi import Body
    from fastapi.responses import JSONResponse

    @app.get("/forge-canvas/topaz/status")
    def topaz_status():
        exe = find_tpai()
        return {"available": exe is not None, "scale": upscale_scale()}

    @app.post("/forge-canvas/topaz/process")
    def topaz_process(payload: dict = Body()):
        image = payload.get("image", "")
        tool = payload.get("tool", "upscale")
        prefix = "base64,"
        if prefix not in image:
            return JSONResponse(status_code=400, content={"error": "expected a data url in 'image'"})
        try:
            png = base64.b64decode(image.split(prefix, 1)[1])
            result = run_tool(png, tool)
        except RuntimeError as e:
            print(f"[forge canvas] Topaz {tool} failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
        return {"image": "data:image/png;base64," + base64.b64encode(result).decode()}
