"""What the user actually sees: the running webui, in a real browser.

THE LAST LAYER
--------------
Three toolbar tests already existed when this was written, and each had missed
something the next one caught:

  test_toolbar_contract.py  reads the JS as text   -> a contract exists
  toolbar_contract_js.js    runs it on a DOM stub  -> reveal/audit logic works,
                                                      on nodes the TEST created
  test_toolbar_dom.py       jsdom + real template  -> injection works, but with
                                                      NO stylesheet and no layout

Only a real browser has the CSS cascade and a layout engine. Three things that
matter here are invisible to everything above:

  * `.forge-toolbar { opacity: 0 }` - the toolbar is invisible until hovered. A
    screenshot taken without hovering is indistinguishable from a broken toolbar,
    which is exactly how two rounds of this investigation went wrong.
  * `.cnet-output-mask-group ... { display: none !important }` deliberately hides
    thirteen controls. jsdom loads no CSS, so it could not see this - and the
    audit consequently reported 13 false problems on every output-mask canvas.
  * an element can be `display: inline-block` and still measure 0x0 because an
    ancestor is collapsed. "Declared visible" and "has pixels" are different
    questions and only one of them is the user's.

This test SKIPS when the webui is not running. That is deliberate: it is a
diagnostic against a live system. The skip is loud, because a quiet skip is the
same failure mode as everything else in this file's history.

Setup:
    npm install --no-save --prefix <dir> playwright
    npx playwright install chromium
    CNPRO_TEST_NODE_PATH=<dir> CNPRO_URL=http://127.0.0.1:7870/ python tests/test_toolbar_live.py

Port 7860 belongs to the user and is refused by the harness (AGENTS.md section 0).

Exit code 0 = pass or skip; 1 = fail.
"""
import json
import re
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
    extra = env.get("CNPRO_TEST_NODE_PATH")
    if extra:
        node_path = os.path.join(extra, "node_modules")
        env["NODE_PATH"] = (env.get("NODE_PATH", "") + os.pathsep + node_path).strip(os.pathsep)
    proc = subprocess.run(["node", os.path.join(HERE, "toolbar_live_js.js")],
                          capture_output=True, text=True, env=env, encoding="utf-8")
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if "Cannot find module 'playwright'" in err:
            return None, ("playwright is not installed:\n"
                          "    npm install --no-save --prefix <dir> playwright\n"
                          "    npx playwright install chromium\n"
                          "  then set CNPRO_TEST_NODE_PATH=<dir>")
        return None, "harness failed:\n%s" % err
    try:
        return json.loads(proc.stdout), None
    except ValueError:
        return None, "harness produced no JSON:\n%s" % (proc.stdout or proc.stderr)[:500]


def main():
    data, why = run()
    if data is None:
        print("SKIPPED - the live-browser toolbar test did not run.")
        print("  %s" % why.replace("\n", "\n  "))
        print("  Nothing has been verified against a real browser.")
        return 0
    if data.get("skip"):
        print("SKIPPED - %s" % data["skip"])
        print("  Start the webui and re-run to check the toolbar as a user sees it.")
        return 0
    if data.get("fatal"):
        fail("harness crashed: %s" % data["fatal"])
        return report()

    mod = data.get("module") or {}
    if not mod.get("present"):
        fail("window.cnproCanvasNodes is not defined in the live page - "
             "canvas_nodes.js did not load or threw")
        return report()

    # The registry loads AFTER the renderer (filename order) and the renderer
    # resolves it lazily. If it never loaded, every derived list is empty and the
    # toolbar is blank - which looks identical to "the accordion is closed".
    if not mod.get("registryLoaded"):
        fail("window.cnproCanvasTools is not defined in the live page - the tool "
             "registry did not load, so nothing was injected into any canvas")
    if not mod.get("toolbarIds"):
        fail("the live page reports zero toolbar buttons - the registry is empty "
             "or the renderer could not read it")

    # The two shipped bugs, checked against the LIVE objects rather than source.
    if mod.get("idsTruncated"):
        fail("the live toolbar ids are truncated (ids ending in 'Butto') - the "
             "id parser regression is back, and no control will resolve")
    if not mod.get("auditUsesComputedStyle"):
        fail("the live audit is not using the computed-style visibility test - "
             "the offsetParent version reports every control broken whenever the "
             "accordion is collapsed")

    tb = data.get("toolbar") or {}

    # The toolbar fades in on hover; if it is still transparent the harness did
    # not really hover, and every measurement below would be meaningless.
    if tb.get("opacity") not in ("1", 1):
        fail("the toolbar is still at opacity %r after hovering - measurements "
             "below cannot be trusted" % tb.get("opacity"))

    if tb.get("missing"):
        fail("%d controls are absent from the live DOM: %r"
             % (len(tb["missing"]), tb["missing"]))

    visible = tb.get("visible") or []
    zero = tb.get("zeroSized") or []
    expected = [i for i in mod.get("toolbarIds", []) if i not in mod.get("deferred", [])]

    for wm in ("wmaskButton", "wmaskCoarseButton", "wmaskMidButton", "wmaskFineButton"):
        if wm not in visible:
            fail("%s has no pixels on screen in the live app (zero-sized: %r)" % (wm, zero))

    for need in expected:
        if need not in visible:
            fail("%s should be visible but measures 0x0 in the live app" % need)

    if not visible:
        fail("NOT ONE control has pixels on screen - the toolbar is empty")

    # The audit must be quiet on a healthy canvas...
    if tb.get("auditNow"):
        fail("audit() reports problems on the live, laid-out canvas: %r"
             % tb["auditNow"][:6])

    # ...and quiet on the output-mask canvases, where 13 controls are hidden by
    # design. This was 13 false errors per canvas until measured here.
    om = data.get("outputMask") or {}
    if om.get("auditProblems"):
        fail("audit() reports %d problems across %d output-mask canvases, where "
             "the tool chrome is hidden ON PURPOSE by style.css. False alarms at "
             "this volume bury the real ones. Sample: %r"
             % (om["auditProblems"], om.get("count"), om.get("sample")))

    # Any CNPro console error in a healthy session is a finding in itself.
    errors = [c for c in data.get("console", []) if c.startswith(("error", "pageerror"))]
    if errors:
        fail("the live page logged %d CNPro error(s):\n  %s"
             % (len(errors), "\n  ".join(e[:220] for e in errors[:5])))

    check_server_is_current(data.get("defaults") or {})
    return report(data)


def check_server_is_current(live):
    """Is the running server actually running the code on disk?

    JS is served from disk per request; scripts/ and lib_cnpro/ are imported once
    at startup. So a JS fix can be live while a Python one is not, and the UI then
    shows values whose source lines no longer exist. That is not a wrong default,
    it is a STALE SERVER, and telling the two apart by hand costs a round trip
    every time - so the test does it.
    """
    on_disk = {}
    ext = os.path.dirname(HERE)
    try:
        with open(os.path.join(ext, "lib_cnpro", "external_code.py"), encoding="utf-8") as fh:
            src = fh.read()
        m = re.search(r"resize_mode: Union\[ResizeMode, int, str\] = ResizeMode\.(\w+)", src)
        m2 = re.search(r'(\w+) = "Resize and Fill"', src)
        on_disk["resize_mode_enum"] = m.group(1) if m else None
        on_disk["fill_enum"] = m2.group(1) if m2 else None
        with open(os.path.join(ext, "lib_cnpro", "global_state.py"), encoding="utf-8") as fh:
            m3 = re.search(r"^DEFAULT_PROCESSOR_RES = (\d+)", fh.read(), re.M)
        on_disk["res"] = m3.group(1) if m3 else None
    except OSError as exc:
        fail("could not read the on-disk defaults to compare: %s" % exc)
        return

    stale = []
    want_fill = on_disk["resize_mode_enum"] == on_disk["fill_enum"]
    if want_fill and live.get("resizeMode") not in (None, "Resize and Fill"):
        stale.append("Resize Mode: live %r, on disk 'Resize and Fill'" % live.get("resizeMode"))
    if live.get("processorRes") == "-1":
        stale.append("Preprocessor resolution: live '-1' (a sentinel the source no "
                     "longer produces), on disk %s" % on_disk["res"])

    if not stale:
        return

    # Two causes produce identical symptoms, and blaming the wrong one wastes a
    # restart. Check the more specific one first.
    #
    # `ui-config.json` records component defaults BY LABEL, not by elem_id -
    # "txt2img/Resize Mode/value", not anything containing "controlnet". Once
    # written it overrides the code, permanently, and grepping it for "controlnet"
    # finds nothing and looks reassuring. (It looked reassuring to me, twice.)
    overrides = []
    ui_cfg = os.path.join(os.path.dirname(os.path.dirname(ext)), "ui-config.json")
    try:
        with open(ui_cfg, encoding="utf-8") as fh:
            cfg = json.load(fh)
        for key in ("txt2img/Resize Mode/value", "img2img/Resize Mode/value",
                    "txt2img/Preprocessor resolution/value",
                    "img2img/Preprocessor resolution/value"):
            if key in cfg:
                overrides.append("%s = %r" % (key, cfg[key]))
    except (OSError, ValueError):
        pass

    if overrides:
        fail("ui-config.json IS OVERRIDING CNPro's defaults - the code is fine, the "
             "recorded values win. Delete these keys and restart (note they are "
             "keyed by LABEL, so searching that file for 'controlnet' finds "
             "nothing):\n  %s\n  observed: %s"
             % ("\n  ".join(overrides), "; ".join(stale)))
    else:
        fail("THE RUNNING SERVER IS STALE - it is not executing the Python on disk. "
             "Restart the webui (JS reloads on a browser refresh; scripts/ and "
             "lib_cnpro/ do not):\n  %s" % "\n  ".join(stale))


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    tb = data["toolbar"]
    print("ok - live browser: toolbar fades in on hover (opacity %s), %d controls "
          "have pixels on screen%s, audit clean here and across %d output-mask "
          "canvases" % (tb["opacity"], len(tb["visible"]),
                        " incl. all 4 weight-mask buttons" if "wmaskFineButton" in tb["visible"] else "",
                        (data.get("outputMask") or {}).get("count", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
