"""Inserting an image into a canvas must NOT destroy the weight mask painted on
it - driven end to end, on a real ForgeCanvas, through the real insert path.

THE REPORT, TWICE
-----------------
"Inserting output into canvas still removes the mask." The first time, the dims
watchdog was taught to carry the paint onto the new geometry. The action still
destroyed the mask, because a second branch two hundred lines away - the
`forge-image-info` handler's upload-sequence check - went on calling
`clearMask(slot, true)` for every genuinely new image, and `forgeCanvasPush`, the
insert path, is genuinely new content by definition.

Two tests came out of that, and they are deliberately different in kind:

  * `test_mask_clear_reasons.py` reads the SOURCE. It forbids the shape - paint
    can only be destroyed for a declared reason, no declared reason may describe
    an image changing, and every image-change site is funnelled through
    `onImageReplaced`. It runs anywhere, in a second, with no dependencies.
  * THIS ONE drives the FEATURE, because "the structure is right" and "the mask
    is still there" are different claims and the user only has the second one.

WHAT IS PINNED
--------------
1. A mask painted on a canvas survives an insert at a DIFFERENT size, and the
   exported channel is re-registered at the new geometry (not left stale at the
   old one, which would reach python as a mask for a frame that no longer
   exists).
2. It survives an insert at the SAME size with different content - the branch
   the first fix missed. No dimension moves here, so nothing but the upload
   counter can see it.
3. Roughly the same fraction of the frame is painted afterwards: the mask is
   RESCALED, not silently blanked to an empty canvas that still counts as
   "present".
4. NEGATIVE CONTROL: removing the image entirely DOES end the mask. Without this
   the file would pass on a painter whose clear path had been deleted, which is
   the other way to make checks 1-3 green.

REQUIRES playwright + chromium + the host's canvas.html:
    npm install --no-save --prefix <dir> playwright
    npx playwright install chromium
    CNPRO_TEST_NODE_PATH=<dir> CNPRO_WEBUI_DIR=<webui> python tests/test_mask_survives_insert.py
Without them it SKIPS LOUDLY rather than passing quietly.

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
        env["CNPRO_TEST_NODE_PATH"] = os.path.join(
            os.environ.get("LOCALAPPDATA", "/tmp"), "Temp", "claude")
    proc = subprocess.run(["node", os.path.join(HERE, "mask_insert_js.js")],
                          capture_output=True, text=True, env=env)
    # A CRASHED HARNESS IS A FAILURE, NOT A SKIP. Every genuine "cannot run
    # here" path exits 0 with `unavailable`, so a non-zero status means the
    # harness itself broke - and reporting that as SKIPPED is how a syntax error
    # in this file came out as a green run that had verified nothing.
    if proc.returncode != 0:
        return {"fatal": "the node harness exited %d:\n%s"
                         % (proc.returncode, proc.stderr.strip()[:900])}, None
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return {"fatal": "the harness produced no JSON:\n%s\n%s"
                         % (proc.stdout[:400], proc.stderr[:400])}, None
    if data.get("unavailable"):
        return None, data["unavailable"]
    return data, None


def main():
    data, why = run()
    if data is None:
        print("SKIPPED - the mask-survives-insert test did not run.")
        print("  %s" % why.replace("\n", "\n  "))
        print("  Whether inserting an image destroys a painted mask is UNVERIFIED.")
        return 0

    if data.get("fatal"):
        fail("the harness threw:\n%s" % data["fatal"])
        return report(data)

    if data.get("console"):
        fail("the page logged errors:\n  %s" % "\n  ".join(data["console"][:6]))

    painted = data.get("painted") or {}
    if not painted.get("hasValue"):
        fail("no mask was exported after painting, so nothing below tests "
             "anything: the stroke did not reach the painter (%r)" % painted)
        return report(data)
    if not painted.get("painted", 0) > 0.001:
        fail("the exported mask is empty after a stroke across the frame (%r) - "
             "the harness painted nothing" % painted.get("painted"))
        return report(data)

    base = painted["painted"]

    # ---- 1 & 3. the reported action, at a new size
    resized = data.get("afterResize") or {}
    if not resized.get("hasValue"):
        fail("THE REPORTED BUG IS BACK: inserting a 640x384 image destroyed the "
             "mask painted on the 512x512 one. The exported channel is empty, so "
             "the generation would run with no mask at all.")
    else:
        if not resized.get("marked"):
            fail("after the insert the G button no longer reads as modified, so "
                 "the painter has dropped the mask even though the channel still "
                 "holds a value - the two have come apart (invariant 15)")
        if resized.get("maskDims") != resized.get("imageDims"):
            fail("after the insert the exported mask is %s against a %s image. "
                 "It was not re-registered onto the new geometry, so python "
                 "receives a mask for a frame that no longer exists."
                 % (resized.get("maskDims"), resized.get("imageDims")))
        after = resized.get("painted", 0)
        if abs(after - base) > 0.05:
            fail("the painted fraction moved from %.3f to %.3f across the "
                 "insert. The mask survived as an OBJECT but not as a picture - "
                 "a blank canvas still counts as 'present' everywhere else."
                 % (base, after))

    # ---- 2. the branch the first fix missed
    same = data.get("afterSameSize") or {}
    if not same.get("hasValue"):
        fail("inserting a different image at the SAME size destroyed the mask. "
             "This is the branch the earlier fix missed: no dimension moves, so "
             "the dims watchdog never sees it and only the upload-sequence path "
             "does - which is exactly where the surviving clearMask was.")
    else:
        if abs(same.get("painted", 0) - base) > 0.05:
            fail("the painted fraction moved from %.3f to %.3f across a same-size "
                 "insert, which should not touch the mask at all"
                 % (base, same.get("painted", 0)))

    # ---- 4. the negative control
    removed = data.get("afterRemove") or {}
    if removed.get("hasValue") or removed.get("marked"):
        fail("removing the image left the mask in place (%r). The mask is "
             "registered with the FRAME - with no frame there is nothing for it "
             "to be registered with - and until this fails, checks 1-3 above are "
             "also satisfied by a painter that has simply stopped clearing."
             % removed)

    return report(data)


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        if data:
            for step in data.get("steps", []):
                print("     %s: %s" % (step.get("step"), step))
        return 1
    p = data["painted"]
    r = data["afterResize"]
    s = data["afterSameSize"]
    print("ok - a painted mask survives both inserts and dies with the image "
          "(host %s): painted %.1f%% of a %s frame -> %.1f%% of %s after a "
          "resized insert -> %.1f%% after a same-size one -> cleared on remove"
          % (data.get("host"), 100 * p["painted"], p["imageDims"],
             100 * r["painted"], r["maskDims"], 100 * s["painted"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
