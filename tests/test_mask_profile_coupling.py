"""The weight-mask slots follow the profile selector - in BOTH implementations.

WHAT THIS TEST IS FOR
---------------------
The four mask slots are the four weight profiles' spatial half: G belongs to the
MAIN profile, C/M/F to the coarse/mid/fine band profiles. Which slots a
generation uses is therefore decided by the editor's band selector, and by
nothing else.

That one rule is implemented twice, because it has to be:

    javascript/weight_mask.js   liveSlotKeys()   -> which slots the toolbar
                                                    shows as live (the rest are
                                                    dimmed)
    lib_cnpro/external_code.py  masks_in_force() -> which masks are actually
                                                    applied to the generation

and the only thing that travels between them is the profile string
('#B<band>', written by the editor, read by band_mode_active). A disagreement
is silent by construction: the toolbar would dim a slot the backend still
applies, or leave a slot bright whose paint the backend drops - and painting is
slow enough that the user finds out several minutes in.

So this runs BOTH and compares them, on every selector, over the real
serialized strings rather than on hand-written ones.

WHAT IS PINNED
--------------
1. main and depth run on G; coarse/mid/fine run on C/M/F. (depth MULTIPLIES the
   main profile instead of replacing it, so it is main mode as far as masks are
   concerned.)
2. The two implementations agree on all five selectors.
3. The selector survives serialize -> parse, and the string the editor writes is
   the one python reads the mode out of - including when every curve is neutral,
   which is the common case and used to be the one that omits segments.
4. The masks the active mode does not use are DROPPED, not merged: no
   precedence, no fallback, no "if it is painted it must have been meant".
5. An unknown selector value falls back to main, never to bands.

Needs node for half of it; without node the python half still runs and the JS
half SKIPS LOUDLY.

Exit code 0 = pass or skip; 1 = fail.
"""
import json
import os
import shutil
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
WEBUI = os.path.dirname(os.path.dirname(EXTENSION))

FAILURES = []
SKIPS = []


def fail(msg):
    FAILURES.append(msg)


def _import_external_code():
    """external_code without pulling in the whole webui.

    Same preamble as tests/test_profile_parity.py: it imports modules.shared /
    modules.api for logging and the API decode helper, neither of which the
    mask rule touches, so both are stubbed and this stays runnable from a bare
    checkout.
    """
    sys.path.insert(0, WEBUI)
    sys.path.insert(0, EXTENSION)
    modules = types.ModuleType("modules")
    modules.__path__ = []
    sys.modules.setdefault("modules", modules)
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace(data={})
    shared.cmd_opts = types.SimpleNamespace(cnpro_loglevel="INFO")
    sys.modules.setdefault("modules.shared", shared)
    modules.shared = shared
    api_pkg = types.ModuleType("modules.api")
    api_pkg.__path__ = []
    api_mod = types.ModuleType("modules.api.api")
    api_pkg.api = api_mod
    sys.modules.setdefault("modules.api", api_pkg)
    sys.modules.setdefault("modules.api.api", api_mod)
    from lib_cnpro import external_code
    return external_code


def run_js():
    if not shutil.which("node"):
        return None, "node is not on PATH"
    proc = subprocess.run(["node", os.path.join(HERE, "mask_profile_js.js")],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None, "node harness failed:\n%s" % proc.stderr.strip()
    try:
        return json.loads(proc.stdout), None
    except ValueError:
        return None, ("harness produced no JSON:\n%s\n%s"
                      % (proc.stdout[:400], proc.stderr[:400]))


# The expectation, written out once, in the plainest form there is. Everything
# below is compared against THIS rather than against the other implementation,
# so a matched pair of wrong answers cannot pass.
EXPECTED_LIVE = {
    "main": ["global"],
    "depth": ["global"],
    "coarse": ["coarse", "mid", "fine"],
    "mid": ["coarse", "mid", "fine"],
    "fine": ["coarse", "mid", "fine"],
}


def check_python(js):
    external_code = _import_external_code()

    # 1. the truth table of masks_in_force, on its own terms
    g = "G-MASK"
    bands = {"coarse": "C", "mid": "M", "fine": "F"}

    kept_g, kept_bands = external_code.masks_in_force(g, bands, False)
    if kept_g != g:
        fail("masks_in_force in MAIN mode dropped the global mask (%r)" % (kept_g,))
    if kept_bands:
        fail("masks_in_force in MAIN mode kept band masks %r - the band masks "
             "belong to the band profiles, which are not running" % (kept_bands,))

    kept_g, kept_bands = external_code.masks_in_force(g, bands, True)
    if kept_g is not None:
        fail("masks_in_force in BAND mode kept the global mask (%r) - it belongs "
             "to the main profile, which is not running" % (kept_g,))
    if kept_bands != bands:
        fail("masks_in_force in BAND mode returned %r, expected every painted "
             "band %r" % (kept_bands, bands))

    # ...and it must not alias the caller's dict: the returned bands are mutated
    # downstream (tensors are substituted per input), and aliasing would write
    # those back into the decode cache.
    kept_g, kept_bands = external_code.masks_in_force(g, bands, True)
    kept_bands["coarse"] = "mutated"
    if bands["coarse"] != "C":
        fail("masks_in_force returns the caller's own dict - mutating the result "
             "corrupts the decoded-mask cache it came from")

    # 2. nothing painted stays nothing painted, in both modes
    for band_selected in (False, True):
        kept_g, kept_bands = external_code.masks_in_force(None, {}, band_selected)
        if kept_g is not None or kept_bands:
            fail("masks_in_force invented masks out of nothing in %s mode: %r / %r"
                 % ("band" if band_selected else "main", kept_g, kept_bands))

    # 3. a mode with no mask of its own does NOT fall back to the other one.
    #    This is the whole behaviour change: it used to be "global if painted,
    #    else bands", so a leftover global mask silently governed band mode.
    kept_g, kept_bands = external_code.masks_in_force(g, {}, True)
    if kept_g is not None or kept_bands:
        fail("band mode with no band masks fell back to the painted global mask "
             "(%r) - that is the old precedence, and it means the unit masks "
             "with a mask the user is not looking at" % (kept_g,))
    kept_g, kept_bands = external_code.masks_in_force(None, bands, False)
    if kept_g is not None or kept_bands:
        fail("main mode with no global mask fell back to the painted band masks "
             "(%r) - same bug, other direction" % (kept_bands,))


def check_agreement(js):
    external_code = _import_external_code()

    live_js = js["js"]

    # 1. JS matches the stated rule
    for band, expected in EXPECTED_LIVE.items():
        got = live_js.get(band)
        if sorted(got or []) != sorted(expected):
            fail("weight_mask.js liveSlotKeys(%r) = %r, expected %r"
                 % (band, got, expected))

    # 2. an unknown selector must mean MAIN. The attribute is written by another
    #    module and read from the DOM; garbage there must not switch the masks.
    for label in ("(unknown)", "(empty)"):
        if live_js.get(label) != ["global"]:
            fail("liveSlotKeys for an %s selector returned %r - an unreadable "
                 "selector must fall back to main (G), never to the bands"
                 % (label, live_js.get(label)))
    pub = js["published"]
    for label, value in pub.items():
        if label != "set" and value != "main":
            fail("selectedProfileBand() returned %r for the %r case - a unit "
                 "whose profile mode has not been published yet must read as "
                 "main" % (value, label))
    if pub.get("set") != "mid":
        fail("selectedProfileBand() did not read the published attribute (got %r)"
             % pub.get("set"))

    # 3. THE TWO IMPLEMENTATIONS AGREE, over the strings the editor really
    #    writes. This is the part neither side can check alone.
    for band in js["selectors"]:
        for kind in ("strings", "neutral"):
            text = js[kind][band]
            band_selected = external_code.band_mode_active(text)
            expected_selected = band in ("coarse", "mid", "fine")
            if band_selected != expected_selected:
                fail("band_mode_active(%r) = %s for the %r selector (%s curves) "
                     "- python reads a different mode out of the string the "
                     "editor wrote for it"
                     % (text, band_selected, band, kind))
                continue
            kept_g, kept_bands = external_code.masks_in_force(
                "G", {"coarse": "C", "mid": "M", "fine": "F"}, band_selected)
            python_live = sorted(([("global")] if kept_g else []) + sorted(kept_bands))
            if python_live != sorted(EXPECTED_LIVE[band]):
                fail("python applies %r for the %r selector, the rule says %r"
                     % (python_live, band, sorted(EXPECTED_LIVE[band])))
            if python_live != sorted(js["js"][band]):
                fail("THE TOOLBAR AND THE BACKEND DISAGREE for the %r selector "
                     "(%s curves): weight_mask.js shows %r as live, python "
                     "applies %r. One of them is lying to the user."
                     % (band, kind, sorted(js["js"][band]), python_live))

    # 4. the selector survives serialize -> parse. Losing it on reload would
    #    silently move the live mask slots on every page load.
    for band, got in js["roundTrip"].items():
        expected = band if band in ("coarse", "mid", "fine") else "main"
        if got != expected:
            fail("the %r selector serializes to %r and parses back as %r - the "
                 "mode does not survive a reload, so neither do the live mask "
                 "slots" % (band, js["strings"][band], got))


def main():
    check_python(None)

    js, why = run_js()
    if js is None:
        SKIPS.append("the JS half did not run (%s), so weight_mask.js's view of "
                     "which slots are live has NOT been compared against "
                     "python's" % why)
    else:
        check_agreement(js)

    return report(js)


def report(js=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    for s in SKIPS:
        print("PARTIAL SKIP -", s)
    if js:
        print("ok - the mask slots follow the profile selector: %d selectors, "
              "toolbar and backend agree on every one, over the strings the "
              "editor actually writes (curved and neutral), and the selector "
              "survives serialize -> parse" % len(js["selectors"]))
    else:
        print("ok - masks_in_force keeps main -> G and bands -> C/M/F apart "
              "with no fallback between them (python half only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
