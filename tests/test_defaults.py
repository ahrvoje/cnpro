"""CNPro's own defaults, pinned where they are actually decided.

TWO DEFAULTS, THREE DECISION POINTS
-----------------------------------
"Resize and Fill" and "resolution 1024" sound like one-line changes. They are
not, because the effective value of each is decided in more than one place, and
setting only the obvious one leaves the others to quietly disagree:

  resize mode   -> ControlNetUnit.resize_mode (dataclass default), which the UI
                   dropdown reads via `self.default_unit.resize_mode.value`.

  resolution    -> ControlNetUnit.processor_res (the -1 sentinel);
                   the slider's initial `value=`;
                   `build_sliders`, which fires on EVERY preprocessor change and
                   pushes that preprocessor's own value into the slider;
                   `bound_check_params`, which resolves the sentinel at
                   generation time.

Setting only the dataclass would give 1024 on first render and silently drop
back to the host's 512 the moment the user picked a module -- the same
declared-here / honoured-there shape as ARCHITECTURE.md section 8. So all four
are pinned here, by reading the source where behaviour cannot be imported
without a running gradio app.

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_defaults.py
Needs nothing for the source checks; the clamping check runs the real helper
with stub preprocessors and needs no host.
Exit code 0 = pass.
"""
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def read(rel):
    with open(os.path.join(EXTENSION, rel), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# the helper, exercised for real (global_state imports the host, so the pure
# function is re-created here from its own source rather than imported)
# --------------------------------------------------------------------------

def load_helper():
    """Execute just `default_processor_res` + its constant, with no host."""
    src = read("lib_cnpro/global_state.py")
    const = re.search(r"^DEFAULT_PROCESSOR_RES = (\d+)", src, re.M)
    func = re.search(r"^def default_processor_res\(.*?(?=\n\n\n|\nclass |\ndef )", src, re.M | re.S)
    if not const or not func:
        fail("global_state no longer defines DEFAULT_PROCESSOR_RES / "
             "default_processor_res - the resolution default has moved and this "
             "test can no longer see it")
        return None, None
    ns = {}
    exec("DEFAULT_PROCESSOR_RES = %s\n\n%s" % (const.group(1), func.group(0)), ns)
    return int(const.group(1)), ns["default_processor_res"]


class Slider:
    def __init__(self, **kw):
        base = dict(minimum=0.0, maximum=1.0, step=0.01, value=0.5, visible=False)
        base.update(kw)
        self.gradio_update_kwargs = base


class Prep:
    def __init__(self, **kw):
        self.slider_resolution = Slider(**kw)


def test_resolution_helper():
    want, fn = load_helper()
    if fn is None:
        return

    if want != 1024:
        fail("DEFAULT_PROCESSOR_RES is %d, expected 1024" % want)

    # the host's own Preprocessor base: 128..2048 step 8, value 512
    host_default = Prep(minimum=128, maximum=2048, step=8, value=512, visible=True)
    got = fn(host_default)
    if got != 1024:
        fail("a standard preprocessor should get 1024, got %r" % got)

    # a hidden slider means "this preprocessor has no resolution" - and its range
    # is the base 0.0..1.0, so forcing 1024 would clamp to 1 and write nonsense
    # into the infotext. Its own value must come back untouched.
    hidden = Prep(visible=False, value=0.5)
    got = fn(hidden)
    if got != 0:
        fail("a hidden resolution slider should keep its own value (int(0.5) == 0), got %r" % got)

    # narrower range than 1024: must clamp, never exceed the maximum
    narrow = Prep(minimum=64, maximum=768, step=64, value=512, visible=True)
    got = fn(narrow)
    if got > 768 or got < 64:
        fail("clamping failed for a 64..768 slider: got %r" % got)
    if got != 768:
        fail("a 64..768 slider should clamp 1024 to its maximum 768, got %r" % got)

    # step snapping must not push the value past the maximum
    odd = Prep(minimum=100, maximum=1000, step=7, value=500, visible=True)
    got = fn(odd)
    if got > 1000:
        fail("step snapping pushed the value past the maximum: %r > 1000" % got)

    # a range that contains 1024 exactly on a step boundary
    exact = Prep(minimum=64, maximum=2048, step=64, value=512, visible=True)
    if fn(exact) != 1024:
        fail("64..2048 step 64 should give exactly 1024, got %r" % fn(exact))

    # missing/malformed metadata must not raise
    try:
        fn(types.SimpleNamespace())
    except Exception as exc:
        fail("default_processor_res raised on a preprocessor with no slider: %s" % exc)


# --------------------------------------------------------------------------
# the decision points, pinned in source
# --------------------------------------------------------------------------

def test_resize_mode_default():
    src = read("lib_cnpro/external_code.py")
    m = re.search(r"^\s*resize_mode: Union\[ResizeMode, int, str\] = ResizeMode\.(\w+)",
                  src, re.M)
    if not m:
        fail("ControlNetUnit.resize_mode default not found")
        return
    if m.group(1) != "OUTER_FIT":
        fail("ControlNetUnit.resize_mode defaults to ResizeMode.%s; CNPro's "
             "default is OUTER_FIT ('Resize and Fill')" % m.group(1))

    # and OUTER_FIT must still be spelled the way the infotext and the dropdown
    # expect, or old images restore to something else
    if not re.search(r'OUTER_FIT = "Resize and Fill"', src):
        fail("ResizeMode.OUTER_FIT is no longer 'Resize and Fill' - saved "
             "infotext and the dropdown label would disagree")

    # The UI builds its own `default_unit` and passes resize_mode explicitly, so
    # the default is declared TWICE. They disagreed for a long time - the UI said
    # OUTER_FIT while the dataclass said INNER_FIT - which meant the dropdown
    # showed "Resize and Fill" but any unit built without going through the UI
    # (the API, an infotext restore that omits the field) silently got
    # "Crop and Resize", and cropped the hint. Same shape as every other bug in
    # ARCHITECTURE.md section 8: two declarations, no check that they agree.
    ui = read("scripts/cnpro.py")
    m2 = re.search(r"default_unit = ControlNetUnit\((.*?)\n\s*\)", ui, re.S)
    if m2 and "resize_mode" in m2.group(1):
        m3 = re.search(r"resize_mode=external_code\.ResizeMode\.(\w+)", m2.group(1))
        if not m3:
            fail("the UI's default_unit sets resize_mode in a form this test "
                 "cannot read; make the two declarations checkable")
        elif m3.group(1) != m.group(1):
            fail("the UI's default_unit uses ResizeMode.%s but ControlNetUnit "
                 "defaults to ResizeMode.%s - the dropdown and every non-UI code "
                 "path would disagree about the default"
                 % (m3.group(1), m.group(1)))


def test_resolution_decision_points():
    ui = read("lib_cnpro/controlnet_ui/controlnet_ui_group.py")
    script = read("scripts/cnpro.py")

    # 1. build_sliders must override the preprocessor's own value
    if not re.search(r"slider_resolution_kwargs\['value'\]\s*=\s*global_state\.default_processor_res",
                     ui):
        fail("build_sliders does not override slider_resolution's value - picking "
             "a preprocessor would silently reset the resolution to the host's 512")

    # 2. the slider's initial value must not be the -1 sentinel
    if re.search(r"label=\"Preprocessor resolution\",\s*\n\s*value=self\.default_unit\.processor_res,", ui):
        fail("the resolution slider is initialised to the -1 sentinel, which is "
             "below its own minimum of 64")
    if "global_state.DEFAULT_PROCESSOR_RES" not in ui:
        fail("the resolution slider does not fall back to CNPro's default")

    # 3. generation time must resolve the sentinel through the helper, not
    #    through the preprocessor's own slider value
    if re.search(r"unit\.processor_res\s*=\s*int\(preprocessor\.slider_resolution", script):
        fail("bound_check_params still reads the preprocessor's own slider value; "
             "that is the host's 512, not CNPro's default")
    if not re.search(r"unit\.processor_res\s*=\s*global_state\.default_processor_res", script):
        fail("bound_check_params does not resolve the resolution through "
             "global_state.default_processor_res")


def main():
    for fn in (test_resolution_helper, test_resize_mode_default,
               test_resolution_decision_points):
        try:
            fn()
        except Exception as exc:
            import traceback
            fail("%s raised %s: %s\n%s" % (fn.__name__, type(exc).__name__, exc,
                                           traceback.format_exc()))

    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - resize mode defaults to 'Resize and Fill', and resolution "
          "defaults to 1024 at all four decision points (clamped per preprocessor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
