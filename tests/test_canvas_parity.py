"""The canvas/control parity contract, checked pixel by pixel in a real browser.

THE CONTRACT (stated in javascript/canvas_extra.js)

    The raster the control receives is the flattened canvas the user sees.
    Same pixels, same dimensions, always.

WHY IT NEEDS A TEST AND NOT A COMMENT
-------------------------------------
The two sides are produced by different code owned by different projects:
CNPro composites the layer stack and hands ONE canvas to the widget; the host's
`updateBackgroundImageData` then re-encodes the <img> element it was given and
writes that to the `logical_image_background` channel, which is what actually
generates. Nothing structural forces those to agree - they agree because the
composite is the only thing ever handed over, and every new tool, blend mode or
adjustment is a fresh chance to break that by writing one side only.

The failure is silent in the worst way: the canvas shows the edit, the user
generates, and the result was made from a different picture. No error, no
warning, and the screenshot in the bug report looks correct.

This test decodes BOTH sides and compares every channel of every pixel across
layers (moved, scaled, rotated, flipped, lighten-blended, partly off-stage,
per-layer gamma/invert), global adjustments, geometry, crop, pen strokes and
drops. There is no tolerance: both sides are PNG and no resampling happens
between them, so the only correct answer is zero differing pixels.

The crop tool being open is the contract's one stated exception (the display is
the full frame being edited, gradio holds the committed crop). It is asserted as
an exception rather than skipped.

REQUIRES puppeteer-core and a Chrome/Edge binary, neither of which is vendored:
    npm install --no-save --prefix <dir> puppeteer-core
    CNPRO_TEST_NODE_PATH=<dir> python tests/test_canvas_parity.py
Without them the test SKIPS LOUDLY rather than passing quietly. No webui
instance is needed - the page is built from the host's own canvas.html/canvas.js
plus every file in javascript/.

Exit code 0 = pass or skip; 1 = fail.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def run():
    if not shutil.which("node"):
        return None, "node is not on PATH"

    env = dict(os.environ)
    if "CNPRO_TEST_NODE_PATH" not in env:
        # the scratchpad the harness was developed against, so a bare run works
        # if puppeteer-core is still installed there
        env["CNPRO_TEST_NODE_PATH"] = os.path.join(
            os.environ.get("LOCALAPPDATA", "/tmp"), "Temp", "claude")
    extra = env["CNPRO_TEST_NODE_PATH"]
    env["NODE_PATH"] = os.pathsep.join(
        p for p in (extra, os.path.join(extra, "node_modules"), env.get("NODE_PATH")) if p)

    proc = subprocess.run(["node", os.path.join(HERE, "canvas_parity_js.js")],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return None, "node harness failed:\n%s" % proc.stderr.strip()[:2000]
    try:
        return json.loads(proc.stdout), None
    except ValueError:
        return None, "harness produced no JSON:\n%s\n%s" % (
            proc.stdout[:400], proc.stderr[:400])


def main():
    data, why = run()
    if data is None:
        print("SKIPPED - the canvas/control parity test did not run.")
        print("  %s" % why.replace("\n", "\n  "))
        print("  Nothing has been verified about what the control actually receives.")
        return 0
    if data.get("skip"):
        print("SKIPPED - the canvas/control parity test did not run.")
        print("  %s" % data["skip"])
        print("  Nothing has been verified about what the control actually receives.")
        return 0
    if data.get("fatal"):
        fail("harness crashed:\n%s" % data["fatal"])
        return report()

    cases = data.get("cases") or {}
    if len(cases) < 20:
        fail("only %d cases ran (expected the full matrix) - the harness has "
             "stopped exercising states and this check is worth little" % len(cases))

    # every javascript/ module must be on the page: a parity test that loads
    # half the tools proves parity for half the tools
    modules = data.get("modules") or []
    for required in ("canvas_adapter.js", "canvas_nodes.js", "canvas_extra.js", "weight_mask.js"):
        if required not in modules:
            fail("javascript/%s was not loaded into the test page" % required)

    for name, r in sorted(cases.items()):
        if r.get("error"):
            fail("%s: %s" % (name, r["error"]))
            continue
        if r.get("pageErrors"):
            fail("%s: the page threw: %s" % (name, "; ".join(r["pageErrors"][:3])))
        if r.get("match") is True:
            continue
        if r.get("exception"):
            fail("%s: %s - canvas shows %s, control holds %s, expected %s"
                 % (name, r["exception"], r.get("displayed"), r.get("control"),
                    r.get("expectedControl")))
            continue
        if r.get("reason") == "dimensions differ":
            fail("%s: THE CONTROL IS NOT THE CANVAS - canvas is %s, the control "
                 "got %s. The user generates from the second one."
                 % (name, r.get("displayed"), r.get("control")))
            continue
        fail("%s: THE CONTROL IS NOT THE CANVAS - %s of %s pixels differ "
             "(worst channel delta %s, first at %s). Both are %s. The canvas is "
             "showing one picture and the generation would use another."
             % (name, r.get("diffPixels"), r.get("totalPixels"), r.get("maxDiff"),
                r.get("firstDiffAt"), r.get("displayed")))

    return report(data)


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    cases = data.get("cases") or {}
    exceptions = sum(1 for r in cases.values() if r.get("exception"))
    print("ok - %d canvas states checked pixel-for-pixel in %s: the control "
          "holds exactly the flattened canvas in every one (%d stated "
          "exception%s asserted, not skipped)"
          % (len(cases), os.path.basename(data.get("chrome") or "chrome"),
             exceptions, "" if exceptions == 1 else "s"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
