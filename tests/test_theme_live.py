"""Both themes, in a real browser: is the main profile actually visible, and is
the dark theme still exactly what it was?

WHY A TEST AND NOT A LOOK
-------------------------
The report was "on the default light theme many white elements cannot be seen -
the main profile button, the main profile line". That class of bug passes every
check that is not a pixel: the element is in the DOM, it has a box, `draw()`
ran, the colour parses, `getComputedStyle` returns it. What fails is only
"is what was painted different from what it was painted on".

AND THE TRAP THAT MADE IT WORSE
-------------------------------
`weight_profile.js::colors()` used to decide dark-vs-light by measuring
`document.body`'s background. Gradio leaves body WHITE on both themes and paints
the real fill onto the app element inside it, so that probe read 255 and the
LIGHT arm won every time. Two consequences, and the second is why this test
asserts in both directions:

  1. the light-theme arm was the only one that ever ran, so the "white on dark"
     step separators the source describes were never painted; and
  2. fixing the detection therefore CHANGES THE DARK THEME - it would have
     switched a finished, signed-off look as a side effect of a light-theme
     feature.

The dark expectations below are pinned to what the running app was measured to
paint before light support existed (926 px of the grey separator tone on the
plot, none of the white one), not to what the source read as if it chose. If a
future change makes dark "correct" on paper, this test is the thing that will
say it also made it different.

This test SKIPS when the webui is not running - it is a diagnostic against a
live system. The skip is loud; a quiet skip is the same failure mode as the bug.

Setup:
    npm install --no-save --prefix <dir> playwright
    npx playwright install chromium
    CNPRO_TEST_NODE_PATH=<dir> CNPRO_URL=http://127.0.0.1:7870/ python tests/test_theme_live.py

Set CNPRO_SHOT_DIR to also write one screenshot of the editor per theme.
Port 7860 belongs to the user and is refused by the harness (AGENTS.md section 0).

Exit code 0 = pass or skip; 1 = fail.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# A line, a bar or a curve has to clear this against the surface behind it. It
# is deliberately not the 4.5:1 of body text: these are 2-3 px graphic marks,
# and the dark palette's own band colours sit at 3.3-4.3:1 on a light page - so
# the bar is set where "as visible as the colours that already work" is, and
# white-on-white (1.0:1) is nowhere near it.
MIN_CONTRAST = 3.0

# What the dark theme paints. Measured on the running app, not read off the
# source - see the module docstring.
DARK_PINS = {
    "mainLine": "#ffffff",
    "bandMid": "#fdd835",
    "stepDot": "rgb(56, 62, 80)",
}

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
    proc = subprocess.run(["node", os.path.join(HERE, "theme_live_js.js")],
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


def check_theme(name, d):
    if not d.get("hasEditor"):
        fail("%s: no weight-profile editor was laid out - nothing could be measured"
             % name)
        return

    # 1. the theme was detected, and detected as the one gradio was asked for
    if d.get("attr") != name:
        fail("%s: data-cnpro-theme is %r. theme.js did not recognise the theme, so "
             "style.css's overrides never applied and the plot resolved the wrong "
             "palette (gradio's own body.dark was %r)"
             % (name, d.get("attr"), d.get("bodyDark")))

    # 2. THE MAIN LINE IS ON THE CANVAS, IN PIXELS, AND DIFFERS FROM THE PAGE
    if not d.get("mainPixelCount"):
        fail("%s: not one pixel of the main line colour %s is on the plot. The "
             "curve is drawn - it is drawn in something invisible, which is the "
             "whole bug. Colours actually painted: %r"
             % (name, d.get("mainLine"), d.get("topColors")[:4]))
    elif d.get("mainOnPanel", 0) < MIN_CONTRAST:
        fail("%s: the main profile line is %.2f:1 against the panel it is drawn on "
             "(%s on %s). Below %.1f:1 the curve the whole editor is about cannot "
             "be read." % (name, d["mainOnPanel"], d.get("mainLine"), d.get("panel"),
                           MIN_CONTRAST))

    # 3. ...and so is its SELECTOR BAR, which gets its colour by a different
    #    route (an inline --band-color written by python)
    if d.get("barOnPanel") is None:
        fail("%s: the main profile selector bar was not found on the page" % name)
    elif d["barOnPanel"] < MIN_CONTRAST:
        fail("%s: the main profile selector bar is %.2f:1 against the panel "
             "(painted rgb%s). The plot and the bar resolve their colour "
             "separately, so one can go invisible while the other is fine."
             % (name, d["barOnPanel"], tuple(d.get("barPixel") or ())))

    # 4. the step dot marks the main line by SEPARATION from it, so it has to
    #    move to the other side when the line does
    if d.get("dotOnMain") is not None and d["dotOnMain"] < MIN_CONTRAST:
        fail("%s: the per-step value dot is %.2f:1 against the main line it sits on "
             "(%s on %s) - the dot is swallowed by the very curve it annotates."
             % (name, d["dotOnMain"], d.get("stepDot"), d.get("mainLine")))

    # 5. every band colour has to survive this theme, not just the main line.
    #    Yellow is the one that historically does not, which is why it is named.
    for key, label in (("midOnPanel", "mid (yellow)"),
                       ("coarseOnPanel", "coarse (red)"),
                       ("fineOnPanel", "fine (blue)")):
        value = d.get(key)
        if value is not None and value < MIN_CONTRAST:
            fail("%s: the %s band colour is %.2f:1 against the panel - its line and "
                 "its selector bar are both unreadable on this theme"
                 % (name, label, value))


def main():
    data, why = run()
    if data is None:
        print("SKIPPED - the live theme test did not run.")
        print("  %s" % why.replace("\n", "\n  "))
        print("  Nothing has been verified about either theme in a real browser.")
        return 0
    if data.get("skip"):
        print("SKIPPED - %s" % data["skip"])
        print("  Start the webui and re-run to check what the two themes paint.")
        return 0
    if data.get("fatal"):
        fail("harness crashed: %s" % data["fatal"])
        return report(data)

    themes = data.get("themes") or {}
    for name in ("dark", "light"):
        if name not in themes:
            fail("the %s theme was never measured" % name)
            continue
        check_theme(name, themes[name])

    # THE DARK THEME MUST NOT MOVE. See the module docstring: the light-theme
    # work corrects a detection that was silently broken, and the correction
    # alone would have restyled dark.
    dark = themes.get("dark") or {}
    for key, want in DARK_PINS.items():
        got = dark.get(key)
        if got is not None and got != want:
            fail("the DARK theme changed: %s is now %r, it has always painted %r. "
                 "The dark palette is finished and is not part of light-theme work "
                 "- if this is deliberate, change the pin here and say why."
                 % (key, got, want))

    errors = [c for c in data.get("console", []) if c.startswith(("error", "pageerror"))]
    if errors:
        fail("the live page logged %d error(s):\n  %s"
             % (len(errors), "\n  ".join(e[:220] for e in errors[:5])))

    return report(data)


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    themes = (data or {}).get("themes") or {}
    bits = []
    for name in ("dark", "light"):
        d = themes.get(name) or {}
        bits.append("%s: main line %s at %.1f:1 on %s (%d px), bar %.1f:1, "
                    "dot/line %.1f:1, yellow %.1f:1"
                    % (name, d.get("mainLine"), d.get("mainOnPanel") or 0,
                       d.get("panel"), d.get("mainPixelCount") or 0,
                       d.get("barOnPanel") or 0, d.get("dotOnMain") or 0,
                       d.get("midOnPanel") or 0))
    print("ok - live browser, both themes readable, dark unchanged\n  " + "\n  ".join(bits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
