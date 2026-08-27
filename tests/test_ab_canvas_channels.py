"""The A/B panel's Canvas layer row, browser half (javascript/cnpro_ab.js).

The row's value lives in the page - a canvas' layer stack never reaches
python - so the search depends on three channels between the two: the
inventory the row's dropdowns are built from, the request/reply that renders
a duel side's composite off-screen, and the set channel Set/Reset write
through. `tests/ab_canvas_channels_js.js` drives the page half in jsdom with
the two canvas_extra.js hooks stubbed, and this file reads the verdict.

Needs node and jsdom (CNPRO_TEST_NODE_PATH=<dir with node_modules/jsdom>).
Exit code 0 = pass or skip; 1 = fail.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FAILURES = []


def fail(message):
    FAILURES.append(message)


def main():
    if not shutil.which("node"):
        print("SKIPPED - node is not on PATH.")
        print("  The A/B canvas channels have NOT been verified.")
        return 0
    proc = subprocess.run(["node", os.path.join(HERE, "ab_canvas_channels_js.js")],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("FAIL (1)")
        print("  - node harness failed:\n%s" % proc.stderr.strip())
        return 1
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        print("FAIL (1)")
        print("  - harness produced no JSON:\n%s\n%s"
              % (proc.stdout[:400], proc.stderr[:400]))
        return 1
    if data.get("unavailable"):
        print("SKIPPED - %s" % data["unavailable"])
        print("  The A/B canvas channels have NOT been verified.")
        return 0
    if data.get("loadError"):
        fail("cnpro_ab.js threw at load:\n%s" % data["loadError"])
        return report()

    replies = [(r["tab"], json.loads(r["text"])) for r in data["replies"]]
    composites = data["composites"]

    # 1. the unit canvas is found by where it lives, and the override reaches
    #    the compositor as {index: alpha} for that canvas' uuid
    if composites[:1] != [{"uuid": "uuid_unit", "opacities": {"1": 0.5}}]:
        fail("the request for u0.in2 layer 2 did not reach "
             "forgeCanvasComposite as expected: %r" % (composites[:1],))
    if not replies or replies[0] != ("txt2img", {
            "seq": 7, "images": {"u0.in2": "data:image/png;base64,uuid_unit"}}):
        fail("the reply to request 7 is wrong: %r" % (replies[:1],))

    # 2. the same message painted twice is served once
    if len(composites) > 1 and composites[1] == composites[0]:
        fail("a repainted request rendered its composite again")

    # 3. a missing layer is an ERROR reply, not a composite of what is there
    error_8 = next((r for tab, r in replies if r.get("seq") == 8), None)
    if not error_8 or not error_8.get("error") or "images" in error_8:
        fail("a request for a layer the canvas does not have was not refused: "
             "%r" % (error_8,))
    if any(c["opacities"].get("5") is not None for c in composites):
        fail("a layer the canvas does not have was composited anyway")

    # 4/5. the img2img canvas belongs to the img2img tab only
    error_9 = next((r for tab, r in replies if r.get("seq") == 9), None)
    if not error_9 or not error_9.get("error"):
        fail("the txt2img tab served a request for the img2img canvas: %r"
             % (error_9,))
    reply_10 = next((r for tab, r in replies
                     if tab == "img2img" and r.get("seq") == 10), None)
    if reply_10 != {"seq": 10, "images": {"img2img": "data:image/png;base64,uuid_host"}}:
        fail("the img2img tab did not serve its own canvas: %r" % (reply_10,))
    if not any(c == {"uuid": "uuid_host", "opacities": {"0": 0.25}}
               for c in composites):
        fail("the img2img composite was not asked for at 25%%: %r"
             % (composites,))

    # 6. Set reaches the live setter with the right layer
    if data["sets"] != [{"uuid": "uuid_unit", "index": 1, "alpha": 0.5}]:
        fail("the set channel did not reach forgeCanvasSetLayerOpacity as "
             "expected: %r" % (data["sets"],))
    # ...and republishes the inventory at once, keyed by where each canvas
    # lives, layers with their opacity in percent
    if not data["inventories"]:
        fail("Set did not republish the inventory")
    else:
        last = json.loads(data["inventories"][-1]["text"])
        if [c["key"] for c in last] != ["u0.in2"] \
                or [layer["opacity"] for layer in last[0]["layers"]] != [100, 40]:
            fail("the txt2img inventory is not the unit canvas with its "
                 "layers' opacities in percent: %r" % (last,))

    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("ok - a canvas is keyed by where it lives and by tab, a request's "
          "overrides reach the compositor and its reply carries the seq, a "
          "missing layer or another tab's canvas is an error reply, and Set "
          "reaches the live setter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
