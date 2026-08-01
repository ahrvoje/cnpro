"""The hires-fix dimensions CNPro predicts must be the ones the host samples at.

THE BUG THIS EXISTS FOR
-----------------------
Scripts run BEFORE p.init(), so `hr_upscale_to_x/y` do not exist yet and
`get_target_dimensions` must REPLICATE the host's formula. Replicas drift:
the host moved from int-truncation to `sRound` (round-half-up to
`opts.res_step`, default 64), and the old copy - `int(dim * hr_scale)` aligned
to 8 - kept passing every test while preparing every hires cond at the wrong
size. At the first hires step `get_control` noticed the mismatch and silently
rebuilt the carefully cubic-resized hint with nearest-exact + a center crop;
region masks were bilinear-stretched without the crop. No error anywhere - the
hires pass just came out softer and a sliver off its masks, which reads as bad
configuration. The one-sided hr_resize case (host: aspect-preserving) was not
replicated at all and crashed cv2 with a zero dsize.

WHAT IS PINNED HERE
-------------------
1. `predict_hires_dimensions` (lib_cnpro/utils.py) reproduces the host formula
   on hand-derived cases: sRound to the step, both one-sided aspect branches,
   and it can never return a zero dimension.
2. A TRIPWIRE against the next drift: the host's own source still says what
   the replica assumes - sRound's body, res_step's default, and
   calculate_target_resolution's branch structure. When the host changes any
   of them this fails by name, instead of the hint quietly degrading.
3. scripts/cnpro.py actually calls the replica (a fixed formula that nothing
   uses is the silent-miss shape all over again).

Run:  python tests/test_hires_dims.py
Needs numpy only. The tripwire half needs the webui source (CNPRO_WEBUI_DIR or
<webui>/extensions/<name> layout) and skips LOUDLY without it.
Exit code 0 = pass.
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []
SKIPS = []


def fail(msg):
    FAILURES.append(msg)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def webui_dir():
    env = os.environ.get("CNPRO_WEBUI_DIR")
    if env and os.path.isdir(env):
        return env
    two_up = os.path.dirname(os.path.dirname(EXTENSION))
    if os.path.isfile(os.path.join(two_up, "modules", "processing.py")):
        return two_up
    return None


# ---------------------------------------------------------------------------
# the replica, executed for real (utils.py imports the host, so the pure
# function is extracted from its own source - the same pattern as
# test_defaults.py's helper)
# ---------------------------------------------------------------------------

def load_predict():
    src = read(os.path.join(EXTENSION, "lib_cnpro", "utils.py"))
    m = re.search(r"^def predict_hires_dimensions\(.*?(?=\n\ndef |\n\nclass )",
                  src, re.M | re.S)
    if not m:
        fail("lib_cnpro/utils.py no longer defines predict_hires_dimensions - "
             "the hires formula went back to being inlined (or renamed), and "
             "this parity net no longer covers it")
        return None
    ns = {}
    exec(m.group(0), ns)  # pure: needs no imports
    return ns["predict_hires_dimensions"]


def s_round(value, step):
    return math.floor(value / step + 0.5) * step


def test_formula(predict):
    # the drift example, in numbers: 832x1216 at 1.5 with the host default 64
    hr_h, hr_w = predict(832, 1216, 1.5, 0, 0, 64)
    if (hr_h, hr_w) != (1856, 1280):
        fail("832x1216 @ 1.5 (step 64) predicted %dx%d, host samples 1856x1280 "
             "- the old int()+align8 replica said 1824x1248 and every hires "
             "hint was silently re-resized onto the real latent"
             % (hr_h, hr_w))

    # generic sRound agreement on scale cases across steps
    for w, h, scale, step in [(1024, 1024, 2.0, 64), (896, 1152, 1.3, 64),
                              (512, 768, 1.7, 8), (832, 1216, 1.25, 32)]:
        hr_h, hr_w = predict(w, h, scale, 0, 0, step)
        want = (s_round(h * scale, step), s_round(w * scale, step))
        if (hr_h, hr_w) != want:
            fail("%dx%d @ %s (step %d): predicted %r, sRound says %r"
                 % (w, h, scale, step, (hr_h, hr_w), want))

    # one-sided resize = the host's aspect-preserving branch. The old replica
    # passed the 0 straight through, and a zero dimension is a cv2 crash at
    # Generate (RESIZE/OUTER_FIT) or a zero-height cond (INNER_FIT).
    hr_h, hr_w = predict(832, 1216, 1.0, 1024, 0, 64)
    want = (s_round(1024 * (1216 / 832), 64), s_round(1024, 64))
    if (hr_h, hr_w) != want:
        fail("hr_resize_x=1024, y=0: predicted %r, host computes %r"
             % ((hr_h, hr_w), want))
    hr_h, hr_w = predict(832, 1216, 1.0, 0, 1024, 64)
    want = (s_round(1024, 64), s_round(1024 * (832 / 1216), 64))
    if (hr_h, hr_w) != want:
        fail("hr_resize_y=1024, x=0: predicted %r, host computes %r"
             % ((hr_h, hr_w), want))

    # both set: taken verbatim (then rounded), like the host
    if predict(832, 1216, 1.0, 1024, 1536, 64) != (1536, 1024):
        fail("explicit hr_resize 1024x1536 must round-trip unchanged")

    # a zero dimension must be impossible for any positive input
    for args in [(832, 1216, 1.5, 0, 0, 64), (832, 1216, 1.0, 64, 0, 64),
                 (832, 1216, 1.0, 0, 64, 64), (64, 64, 1.0, 0, 0, 64)]:
        hr_h, hr_w = predict(*args)
        if hr_h <= 0 or hr_w <= 0:
            fail("predict_hires_dimensions%r returned a zero dimension (%d, %d)"
                 % (args, hr_h, hr_w))


# ---------------------------------------------------------------------------
# the tripwire: the host's source still says what the replica assumes
# ---------------------------------------------------------------------------

def test_host_tripwire():
    webui = webui_dir()
    if webui is None:
        SKIPS.append("webui source not found (set CNPRO_WEBUI_DIR) - the "
                     "host-drift tripwire did NOT run, so a host formula "
                     "change would not be caught here")
        return

    ui = read(os.path.join(webui, "modules", "ui.py"))
    if not re.search(r"def sRound\(.*?\n\s*return math\.floor\(val / _STEP \+ 0\.5\) \* _STEP",
                     ui, re.S):
        fail("the host's modules/ui.py sRound no longer reads "
             "`math.floor(val / _STEP + 0.5) * _STEP` - "
             "predict_hires_dimensions replicates exactly that and has now "
             "drifted from the host AGAIN. Update the replica and this pin "
             "together.")

    options = read(os.path.join(webui, "modules", "shared_options.py"))
    if not re.search(r'"res_step":\s*OptionInfo\(64', options):
        fail("the host's res_step default is no longer 64 - the fallback in "
             "scripts/cnpro.py (getattr(shared.opts, 'res_step', 64)) is now "
             "wrong on hosts without the option")

    processing = read(os.path.join(webui, "modules", "processing.py"))
    block = re.search(r"def calculate_target_resolution\(self\):.*?(?=\n    def |\n\nclass )",
                      processing, re.S)
    if not block:
        fail("the host's calculate_target_resolution is gone or renamed - the "
             "replica's model of hires sizing no longer has a counterpart")
    else:
        body = block.group(0)
        for marker, why in [
            ("sRound(self.width * self.hr_scale)", "the scale branch rounds through sRound"),
            ("self.hr_resize_x * (self.height / self.width)", "the y-from-aspect branch"),
            ("self.hr_resize_y * (self.width / self.height)", "the x-from-aspect branch"),
            ("sRound(self.hr_upscale_to_x)", "the resize branch rounds through sRound"),
        ]:
            if marker not in body:
                fail("calculate_target_resolution no longer contains %r (%s) - "
                     "the replica in lib_cnpro/utils.py must be re-verified "
                     "against the host and this pin updated with it"
                     % (marker, why))


def test_replica_is_used():
    src = read(os.path.join(EXTENSION, "scripts", "cnpro.py"))
    if "predict_hires_dimensions(" not in src:
        fail("scripts/cnpro.py does not call predict_hires_dimensions - the "
             "replica exists but the generation path computes hires dims some "
             "other way (declared here, honoured nowhere)")
    if re.search(r"hr_y = int\(p\.height \* p\.hr_scale\)", src):
        fail("scripts/cnpro.py still contains the old int() hires formula")


def main():
    predict = load_predict()
    if predict is not None:
        test_formula(predict)
    test_host_tripwire()
    test_replica_is_used()

    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    for s in SKIPS:
        print("PARTIAL SKIP -", s)
    print("ok - predict_hires_dimensions matches the host formula on every "
          "case (sRound, both aspect branches, no zero dims), the host source "
          "still says what the replica assumes, and the generation path calls it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
