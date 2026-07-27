"""End-to-end: generate with the Z-Image ControlNet through the running webui.

THE ONLY TEST THAT PROVES IT WORKS
----------------------------------
Everything else in this directory verifies a layer: the config sniff matches the
checkpoint, the module tree matches its tensors, the hooks fire at the right
blocks, the toolbar has pixels. All necessary; none of them can tell you the
ControlNet actually steers an image.

This runs the same prompt and seed three ways -- no control, v1 Union, v2.1
Union -- against the live webui and compares pixels. A ControlNet that loads,
runs, and injects nothing produces an image IDENTICAL to the no-control
baseline, and every other test in the suite would still pass. That is the
failure this exists to catch.

The control image is a line drawing of a house (generated below, no asset
needed): a roof triangle, a square body, two windows, a door, a sun. Structure
that is unmistakable in the output if the ControlNet is working, and
unmistakably absent if it is not.

REQUIRES:
  * the webui running WITH --api            (else every request 404s)
  * a Z-Image checkpoint loaded             (the ControlNets refuse other families)
  * the ControlNet files in --controlnet-dir
Skips loudly when any of those is missing - it is a live-system diagnostic, not
a hermetic unit test.

Run:  CNPRO_URL=http://127.0.0.1:7870 <webui python> tests/test_generate_live.py
      (7870, NOT 7860 - see AGENTS.md section 0)
Exit code 0 = pass or skip; 1 = fail.
"""
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
#: Port 7860 is the USER'S instance. Never attach to it - this test GENERATES
#: images and would consume their GPU mid-session. Default to an agent-owned
#: instance and refuse 7860 even when passed explicitly (AGENTS.md section 0).
DEFAULT_URL = "http://127.0.0.1:7870"
#: rstrip('/'), because every route below is joined with a leading slash and the
#: form AGENTS.md tells you to export is `http://127.0.0.1:7870/`. With the
#: trailing slash the join produced `...7870//sdapi/v1/sd-models`, gradio
#: answered 404, and this file printed "the webui API is not reachable, start it
#: with --api" - about an instance that WAS running with --api. A skip that
#: names the wrong cause is worse than a skip, because the advice it gives
#: cannot work and the real reason is invisible.
URL = os.environ.get("CNPRO_URL", DEFAULT_URL).rstrip("/")
CONTROL = os.path.join(HERE, "_control_house.png")


def make_control_image():
    """A line drawing with structure that is obvious in the output."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1024, 1024), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([250, 480, 780, 900], outline="black", width=6)      # body
    d.polygon([(220, 480), (515, 220), (810, 480)], outline="black", width=6)  # roof
    d.rectangle([330, 560, 460, 680], outline="black", width=6)      # window L
    d.rectangle([570, 560, 700, 680], outline="black", width=6)      # window R
    d.rectangle([460, 740, 570, 900], outline="black", width=6)      # door
    d.ellipse([820, 90, 960, 230], outline="black", width=6)         # sun
    im.save(CONTROL)

PROMPT = ("a photograph of a small suburban house on a sunny day, "
          "clear blue sky, sharp focus, high detail")
SEED = 12345
STEPS = 8


def b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def post(route, payload, timeout=900):
    req = urllib.request.Request(
        URL + route, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def unit(model, module="canny", weight=1.0):
    return {
        "enabled": True, "module": module, "model": model,
        "weight": weight, "image": b64(CONTROL),
        "resize_mode": "Resize and Fill", "processor_res": 1024,
        "guidance_start": 0.0, "guidance_end": 1.0,
        "control_mode": "Balanced", "pixel_perfect": False,
    }


def run(label, units, out_name):
    payload = {
        "prompt": PROMPT, "negative_prompt": "", "seed": SEED,
        "steps": STEPS, "cfg_scale": 1.0, "width": 1024, "height": 1024,
        "sampler_name": "Euler", "scheduler": "Simple",
        "save_images": False,
    }
    if units:
        payload["alwayson_scripts"] = {"CNPro": {"args": units}}
    t0 = time.time()
    try:
        res = post("/sdapi/v1/txt2img", payload)
    except Exception as exc:
        body = ""
        if hasattr(exc, "read"):
            try:
                body = exc.read().decode()[:1500]
            except Exception:
                pass
        print("  %-14s FAILED: %s %s" % (label, exc, body))
        run.last_error = "%s %s" % (exc, body)
        return None
    dt = time.time() - t0
    img = res.get("images", [None])[0]
    if not img:
        print("  %-14s no image returned: %r" % (label, str(res)[:300]))
        return None
    raw = base64.b64decode(img.split(",", 1)[-1])
    path = os.path.join(HERE, out_name)
    with open(path, "wb") as fh:
        fh.write(raw)
    print("  %-14s ok in %5.1fs -> %s (%d KB)" % (label, dt, out_name, len(raw) // 1024))
    return path


def compare(a, b):
    """Mean absolute difference, 0-255, between two saved PNGs."""
    from PIL import Image
    import numpy as np
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    if ia.shape != ib.shape:
        return None
    return float(np.abs(ia - ib).mean())


def main():
    if ":7860" in URL:
        print("SKIPPED - refusing port 7860: that is the user's instance, and this "
              "test generates images on it (AGENTS.md section 0).")
        print("  Start your own:  python launch.py --port 7870 --api ...")
        return 0
    # THREE OUTCOMES, NOT ONE. "nothing is listening", "something is listening
    # but has no API" and "the API answered" are different problems with
    # different fixes, and collapsing them is how this file spent a session
    # telling someone to add --api to an instance that already had it.
    try:
        urllib.request.urlopen(URL + "/sdapi/v1/sd-models", timeout=10).read()
    except urllib.error.HTTPError as exc:
        reachable = False
        try:
            urllib.request.urlopen(URL, timeout=10).read()
            reachable = True
        except Exception:
            pass
        print("SKIPPED - %s answered HTTP %s for /sdapi/v1/sd-models."
              % (URL, exc.code))
        if reachable:
            print("  The webui IS running there, so this is the API and not the "
                  "server: start it with --api (and check the URL has no "
                  "trailing slash - the routes below are joined with one).")
        else:
            print("  Nothing served the root either, so the address is wrong.")
        print("  Nothing has been generated, so nothing is verified.")
        return 0
    except Exception as exc:
        print("SKIPPED - the webui is not reachable at %s (%s)." % (URL, exc))
        print("  Nothing has been generated, so nothing is verified.")
        return 0

    # THE BASE CHECKPOINT HAS TO BE A Z-IMAGE ONE.
    #
    # These are Z-Image ControlNets, and CNPro refuses them on any other family
    # (patchers/zimage.py, MAINTENANCE invariant 26) - correctly, loudly, with a
    # RuntimeError naming both sides. But the HOST catches script exceptions and
    # finishes the generation anyway, so what arrives back is a perfectly good
    # image with no control in it. This file then measured "the output equals
    # the baseline" and reported "the ControlNet is not injecting" - a real
    # regression and a wrong checkpoint producing the identical verdict, which
    # is the failure-collapsing AGENTS.md section 4 is about. Measured: an
    # instance whose current checkpoint was SDXL, three confident failures, and
    # nothing wrong with the code under test.
    #
    # So the family is established BEFORE anything is generated: switch to a
    # Z-Image checkpoint if one is available, and skip loudly if not.
    restore_checkpoint = None
    try:
        opts = json.loads(urllib.request.urlopen(URL + "/sdapi/v1/options", timeout=30).read().decode())
        current = opts.get("sd_model_checkpoint") or ""
        checkpoints = [m["title"] for m in json.loads(urllib.request.urlopen(
            URL + "/sdapi/v1/sd-models", timeout=60).read().decode())]
    except Exception as exc:
        print("SKIPPED - could not read the loaded checkpoint (%s), so this test "
              "cannot tell a real regression from a Z-Image ControlNet pointed "
              "at an SDXL base." % exc)
        return 0

    def is_zimage(title):
        low = title.lower()
        # a ControlNet file sitting in the checkpoint list is not a base model
        if "controlnet" in low or "\\cn\\" in low or "/cn/" in low:
            return False
        return "z-image" in low or "zimage" in low or "z_image" in low

    if not is_zimage(current):
        candidate = next((c for c in checkpoints if is_zimage(c)), None)
        if candidate is None:
            print("SKIPPED - the loaded checkpoint is %r and no Z-Image checkpoint "
                  "is available to switch to." % current)
            print("  A Z-Image ControlNet cannot steer a non-Z-Image base (CNPro "
                  "refuses it by design), so generating here would report "
                  "'the ControlNet is not injecting' about working code.")
            print("  Nothing has been generated, so nothing is verified.")
            return 0
        print("base checkpoint: %r is not Z-Image; switching to %r" % (current, candidate))
        post("/sdapi/v1/options", {"sd_model_checkpoint": candidate}, timeout=600)
        restore_checkpoint = current
    else:
        print("base checkpoint:", current)

    try:
        return _generate_and_compare()
    finally:
        # leave the instance as it was found - it may be someone's session
        if restore_checkpoint:
            print("restoring the base checkpoint to %r" % restore_checkpoint)
            try:
                post("/sdapi/v1/options",
                     {"sd_model_checkpoint": restore_checkpoint}, timeout=600)
            except Exception as exc:
                print("  WARNING: could not restore it (%s)" % exc)


def _generate_and_compare():
    make_control_image()
    models = json.loads(urllib.request.urlopen(
        URL + "/controlnet/model_list", timeout=60).read().decode())["model_list"]
    v1 = next((m for m in models if m.startswith("Z-Image-Turbo-Fun-Controlnet-Union [")), None)
    v2 = next((m for m in models if "Union-2.1" in m), None)
    print("v1 model:", v1)
    print("v2 model:", v2)
    if not v1 and not v2:
        print("SKIPPED - no Z-Image ControlNet in the model list; put the "
              "Fun-Controlnet-Union files in --controlnet-dir.")
        return 0

    print("\ngenerating (%d steps, seed %d, 1024x1024):" % (STEPS, SEED))
    run.last_error = ""
    base = run("no control", None, "_gen_none.png")

    # THE BASELINE CARRIES NO CONTROLNET AT ALL. If it cannot generate, the
    # instance cannot generate, and that is not a fact about CNPro - reporting
    # it as a failure of the code under test is how a missing text encoder
    # ("You do not have Qwen3 state dict!", measured) turns into three
    # confident, wrong accusations. Skip, loudly, quoting the server.
    if base is None:
        print("\nSKIPPED - the instance cannot generate even WITHOUT a ControlNet,")
        print("  so nothing here is a statement about CNPro. The server said:")
        print("    %s" % (run.last_error or "(no message)").strip()[:400])
        print("  A Z-Image checkpoint also needs its text encoder selected in the")
        print("  UI's additional modules; a headless instance has none by default.")
        print("  Nothing has been verified.")
        return 0

    g1 = run("v1 Union", [unit(v1)], "_gen_v1.png") if v1 else None
    g2 = run("v2.1 Union", [unit(v2)], "_gen_v21.png") if v2 else None

    problems = []
    for label, model, g in (("v1", v1, g1), ("v2.1", v2, g2)):
        if model and g is None:
            # the baseline succeeded, so the instance CAN generate: a failure
            # here is about this control model and nothing else
            problems.append("%s generation failed while the plain baseline "
                            "succeeded, so it is the ControlNet that broke it"
                            % label)

    if base and g1:
        d = compare(base, g1)
        print("\n  mean |diff| baseline vs v1   : %.2f / 255" % d)
        if d is not None and d < 1.0:
            problems.append("v1 output is essentially identical to the no-control "
                            "baseline (mean diff %.2f) - the ControlNet is not "
                            "affecting the image" % d)
    if base and g2:
        d = compare(base, g2)
        print("  mean |diff| baseline vs v2.1 : %.2f / 255" % d)
        if d is not None and d < 1.0:
            problems.append("v2.1 output is essentially identical to the baseline "
                            "(mean diff %.2f) - not injecting" % d)
    if g1 and g2:
        d = compare(g1, g2)
        print("  mean |diff| v1 vs v2.1       : %.2f / 255" % d)
        if d is not None and d < 1.0:
            problems.append("v1 and v2.1 produced the same image (mean diff %.2f); "
                            "they have different architectures and 6 vs 15 "
                            "injection sites, so this means one of them is not "
                            "really running" % d)

    print()
    if problems:
        print("FAIL (%d)" % len(problems))
        for p in problems:
            print("  -", p)
        return 1
    print("ok - both ControlNets generated, and each measurably changed the image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
