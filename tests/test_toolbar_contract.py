"""Every declared toolbar control must be revealed, or explicitly deferred.

THE BUG THIS EXISTS FOR
-----------------------
The tool chrome is injected, and every top-level button is
`style="display: none"` when it lands, so something has to turn it back on. For
a long time that something was one line in `canvas_extra.js`:

    container.querySelectorAll('.forge-adjust-control').forEach(n => n.style.display = '')

The four weight-mask buttons (G / C / M / F) carry `.forge-wmask-control`
instead. They were injected, styled, wired to working handlers -- and never
revealed. The whole weight-mask feature was absent from the toolbar.

Nothing failed. `querySelectorAll` on a class nobody uses returns an empty
NodeList, which is not an error. There was no exception, no warning, no failed
selector, no test. The feature was just gone, and the only way to notice was to
look at the toolbar and remember it should be there.

WHAT IS PINNED HERE
-------------------
1. Every button the registry declares is either revealed or carries a stated
   `deferred` reason. (The regression itself.)
2. No deferred reason names a button that has left the registry.
3. `canvas_extra.js` actually calls the contract, and does not rely on the class
   selector as its primary path.
4. The weight-mask buttons specifically are revealed ON THEIR HOME SURFACE --
   named, so that if the generic rule is ever weakened this still fails. Since
   the GCMF move (2026-08-02) that is two surfaces: G reveals on the input
   canvas AND the output-mask canvas, C/M/F on the output-mask canvas only.
5. The renderer emits exactly the ids the registry claims, with the classes the
   reveal and the stylesheet key off.
6. The reverse of 4, on the host's own canvases (no group wrapper): the
   weight-mask buttons carry a registry `scope` and stay hidden there, where
   nothing wires them -- revealed there they are visible-but-inert chrome --
   and the audit does not cry wolf about it.

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_toolbar_contract.py
Needs node for the half that matters; without it the source checks still run and
the skip is stated. Exit code 0 = pass.
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
JS = os.path.join(EXTENSION, "javascript")

FAILURES = []
SKIPS = []


def fail(msg):
    FAILURES.append(msg)


def read(name):
    with open(os.path.join(JS, name), encoding="utf-8") as fh:
        return fh.read()


def run_harness():
    """EXECUTE the renderer and the registry, and report what they do.

    This is the half of the test that matters. Source checks can only show that
    a mechanism is PRESENT; they cannot show it WORKS, and the difference is not
    academic: the first version of the contract derived its ids with
    `s.slice(4, -14)` when the suffix is thirteen characters long. Every id was
    truncated by one letter, every lookup missed, the whole toolbar disappeared
    -- and every source-level check passed, because a regex was indeed being
    used to derive ids from the markup.
    """
    if not shutil.which("node"):
        SKIPS.append("node is not on PATH - only the source-level checks ran, and "
                     "those cannot tell you whether the toolbar actually works")
        return None
    harness = os.path.join(HERE, "toolbar_contract_js.js")
    proc = subprocess.run(["node", harness], capture_output=True, text=True)
    if proc.returncode != 0:
        fail("node harness failed:\n%s" % proc.stderr.strip())
        return None
    try:
        return json.loads(proc.stdout)
    except ValueError:
        fail("node harness produced no JSON:\n%s\n%s" % (proc.stdout[:400], proc.stderr[:400]))
        return None


def test_runtime_behaviour(data):
    ids = data["toolbarIds"]
    owned = set(data["ownedIds"])
    deferred = set(data["deferred"])

    if not ids:
        fail("the registry declares no toolbar buttons - revealToolbar would "
             "reveal nothing and the toolbar would be blank")
    if data["inject"]["error"]:
        fail("inject() threw:\n%s" % data["inject"]["error"])

    # 1. THE INDEPENDENT DERIVATION. The registry says which ids exist; the
    #    renderer emits markup. Parsing the ids back out of that markup is a
    #    second route to the same fact, so a renderer that drops, duplicates or
    #    mangles a control disagrees with the list that claims it exists.
    rendered = data["renderedIds"]
    if sorted(rendered) != sorted(owned):
        only_declared = sorted(owned - set(rendered))
        only_rendered = sorted(set(rendered) - owned)
        fail("the rendered markup and the registry disagree.\n"
             "  declared but never rendered: %r\n"
             "  rendered but not declared:   %r" % (only_declared, only_rendered))
    dupes = sorted({i for i in rendered if rendered.count(i) > 1})
    if dupes:
        fail("the renderer emitted duplicate ids %r - getElementById returns the "
             "FIRST match, so one node is wired and the other is dead chrome" % dupes)

    # 2. the classes the reveal sweep and the stylesheet key off must survive
    #    rendering. This is the original bug expressed as one missing class name.
    classes = data["renderedClasses"]
    for button in ids:
        cls = classes.get(button)
        if cls is None:
            fail("button %r rendered without a class attribute at all" % button)
            continue
        if "forge-btn" not in cls:
            fail("button %r does not carry .forge-btn - it will not be styled as "
                 "a toolbar button" % button)
    for wm in ("wmaskButton", "wmaskCoarseButton", "wmaskMidButton", "wmaskFineButton"):
        if wm in classes and "forge-wmask-control" not in classes[wm]:
            fail("%s lost .forge-wmask-control - that class is what the class "
                 "sweep in canvas_extra.js uses as its fallback path" % wm)

    # 3. the module's own self-check must be quiet on good input
    if data["selfCheck"]:
        fail("the registry self-check reported problems on its own data:\n  %s"
             % "\n  ".join(data["selfCheck"]))
    if data["selfCheckErrors"]:
        fail("the canvas modules logged problems at load:\n  %s"
             % "\n  ".join(data["selfCheckErrors"]))

    # 4. a clean attach ON THE INPUT CANVAS reveals everything except the
    #    deferred few and the buttons scoped to another surface. Since the
    #    GCMF move (2026-08-02) the C/M/F band masks belong to the OUTPUT
    #    mask canvas only and G to both surfaces, so "everything visible"
    #    stopped being true on any single canvas. The expected sets are
    #    DERIVED from the registry's own scope strings, so reveal and
    #    registry cannot drift apart without failing here.
    scopes_map = data.get("scopesMap", {})

    def scoped_off(surface):
        return {b for b, sel in scopes_map.items()
                if "." + surface not in [s.strip() for s in sel.split(",")]}

    clean = data["scenarios"]["clean"]
    input_hidden = deferred | scoped_off("cnet-input-image-group")
    expected_shown = len(ids) - len(input_hidden)
    if clean["shown"] != expected_shown:
        fail("revealToolbar reported %d controls shown on the input canvas, "
             "expected %d" % (clean["shown"], expected_shown))
    if sorted(clean["hidden"]) != sorted(input_hidden):
        fail("after reveal on the input canvas, hidden = %r but expected "
             "deferred + output-side scoped = %r"
             % (sorted(clean["hidden"]), sorted(input_hidden)))
    if clean["audit"]:
        fail("audit() complains about a correctly revealed toolbar: %r" % clean["audit"])

    # The most direct symptom of a mangled id list: the module asks the DOM for
    # names that were never created. This is what "the entire toolbar is
    # missing" looks like from inside.
    if clean["unresolved"]:
        fail("%d of %d toolbar ids resolve to NO element even though every "
             "declared control exists: %r. The id derivation is producing names "
             "that were never injected."
             % (len(clean["unresolved"]), len(ids), clean["unresolved"][:6]))

    # 5. the controls that started all of this, per surface, BY NAME. The
    #    derived check above would bless a registry whose scopes were edited
    #    by accident; this one pins the current design: G is the one slot
    #    both surfaces share, C/M/F live on the output-mask canvas only.
    if "wmaskButton" not in clean["visible"]:
        fail("wmaskButton (G) is not visible on the input canvas - it is "
             "scoped to both surfaces and must reveal here")
    for wm in ("wmaskCoarseButton", "wmaskMidButton", "wmaskFineButton"):
        if wm not in clean["hidden"]:
            fail("%s is revealed on the INPUT canvas - the band masks are "
                 "output-side since the GCMF move and nothing wires them here" % wm)
    output = data["scenarios"].get("outputSurface")
    if output is None:
        fail("the harness has no outputSurface scenario - the output-mask "
             "canvas reveal is asserted nowhere")
    else:
        output_hidden = deferred | scoped_off("cnet-output-mask-group")
        if sorted(output["hidden"]) != sorted(output_hidden):
            fail("after reveal on the output-mask canvas, hidden = %r but "
                 "expected %r" % (sorted(output["hidden"]), sorted(output_hidden)))
        for wm in ("wmaskButton", "wmaskCoarseButton", "wmaskMidButton", "wmaskFineButton"):
            if wm not in output["visible"]:
                fail("%s is not visible on the output-mask canvas" % wm)
        if output["audit"]:
            fail("audit() cries wolf on the output-mask canvas, where style.css "
                 "suppresses the chrome on purpose and the audit must skip the "
                 "visibility half: %r" % output["audit"])

    # 6. audit must DETECT each failure mode, not merely run
    hidden = data["scenarios"]["oneHidden"]
    if not any(hidden["victim"] in p and "not visible" in p for p in hidden["audit"]):
        fail("audit did not report a control that was hidden after reveal "
             "(victim %r, audit %r)" % (hidden["victim"], hidden["audit"]))

    missing = data["scenarios"]["oneMissing"]
    if not any(missing["victim"] in p and "absent" in p for p in missing["audit"]):
        fail("audit did not report a control missing from the DOM "
             "(victim %r, audit %r)" % (missing["victim"], missing["audit"]))

    # A MENU node, not a button. The audit used to walk buttons only, so a slider
    # that never landed was invisible to it -- a menu opens empty and nothing
    # says why.
    menu = data["scenarios"]["menuNodeMissing"]
    if menu["victim"] and not any(menu["victim"] in p and "absent" in p for p in menu["audit"]):
        fail("audit did not report a MENU node missing from the DOM (victim %r, "
             "audit %r). Auditing buttons alone leaves every slider, label and "
             "overlay unchecked." % (menu["victim"], menu["audit"]))

    none_at_all = data["scenarios"]["nothingInjected"]
    if none_at_all["shown"] != 0 or not none_at_all["audit"]:
        fail("with nothing injected, reveal claimed %r shown and audit said %r - "
             "a completely empty toolbar must be reported, not passed"
             % (none_at_all["shown"], none_at_all["audit"]))

    # 7. the same toolbar OUTSIDE CNPro's input group (the host's own
    #    img2img/inpaint canvases): registry-`scope`d buttons stay hidden -
    #    revealed there they are visible-but-inert chrome, rule 8c's exact
    #    shape - and the audit stays QUIET about them (a check that cries wolf
    #    trains everyone to ignore the one real failure).
    scoped = set(data.get("scoped", []))
    if not scoped:
        fail("the registry declares no scoped buttons - the weight-mask slots "
             "lost their `scope` and are revealed on the host's own canvases "
             "where nothing wires them")
    out = data["scenarios"].get("outOfScope")
    if out is None:
        fail("the harness has no outOfScope scenario - the scoped-reveal rule "
             "is honoured nowhere it can be seen failing")
    else:
        if sorted(out["hidden"]) != sorted(deferred | scoped):
            fail("out of scope, hidden = %r but expected deferred + scoped = %r"
                 % (sorted(out["hidden"]), sorted(deferred | scoped)))
        if out["audit"]:
            fail("audit() cries wolf about scoped buttons hidden off their home "
                 "container: %r" % out["audit"])


def main():
    nodes = read("canvas_nodes.js")
    tools = read("canvas_tools.js")
    extra = read("canvas_extra.js")

    # ---- source-level POLICY checks. Grep is the right tool for "does this call
    # site exist"; it is the wrong tool for "does this work", which is why the
    # harness above exists and why these are deliberately few.

    # 1. the reveal must not be opt-in through a CSS class again
    if "revealToolbar" not in nodes or "deferred" not in nodes:
        fail("canvas_nodes.js no longer exposes revealToolbar / deferred; the "
             "reveal has gone back to being opt-in, which is how the mask "
             "buttons were lost")

    # 2. NO HAND-KEPT PARALLEL ID LIST. The whole point of the registry is that
    #    ids exist once. A literal array of quoted ids in the renderer is the
    #    reintroduction of the thing that used to drift.
    for name, src in (("canvas_nodes.js", nodes),):
        for literal in re.findall(r"=\s*\[([^\]]{80,})\]", src):
            if literal.count("'") > 8 and "Button" in literal:
                fail("%s contains a hand-written id list - ids must be derived "
                     "from the registry, not restated: %s..." % (name, literal[:80]))

    # 3. canvas_extra must actually use the contract
    if "cnproCanvasNodes.revealToolbar" not in extra:
        fail("canvas_extra.js does not call revealToolbar - the contract is "
             "declared but not enforced")
    if "cnproCanvasNodes.audit" not in extra:
        fail("canvas_extra.js does not call audit() - a silent miss would stay silent")

    # The old class-only reveal must not be the primary path any more.
    # Checked against CODE, not prose: the comment explaining the bug quotes the
    # very selector being banned, and a check that cannot tell those apart would
    # fire on its own documentation.
    code = "\n".join(re.sub(r"//.*$", "", line) for line in extra.split("\n"))
    if re.search(r"querySelectorAll\('\.forge-adjust-control'\)", code):
        fail("canvas_extra.js still reveals by '.forge-adjust-control' alone - "
             "that selector is exactly what missed the weight-mask buttons")

    # 4. the audit must test real visibility, not just the inline style: a
    #    stylesheet loses a control just as completely.
    #    NOT offsetParent -- that was tried and reported all fourteen controls
    #    broken on every attach, because CNPro's canvas lives in a gradio
    #    accordion that is closed by default.
    if "getComputedStyle" not in nodes:
        fail("audit() does not consult getComputedStyle - it would miss a control "
             "hidden by a stylesheet rather than by an inline style")
    # Comments stripped first. The note explaining WHY offsetParent was abandoned
    # quotes the expression it bans, so a check that cannot tell code from prose
    # fires on its own documentation -- which it did, first run.
    nodes_code = "\n".join(re.sub(r"//.*$", "", line) for line in nodes.split("\n"))
    if re.search(r"offsetParent\s*===\s*null", nodes_code):
        fail("audit() is back to testing offsetParent, which is null for every "
             "element inside a collapsed accordion - it would cry wolf on every "
             "single attach")

    # 5. every deferred reason must be a real reason (a bare '' is not one)
    reasons = re.findall(r"deferred:\s*'([^']*)'", tools)
    if not reasons:
        fail("canvas_tools.js declares no deferred reasons at all - either the "
             "Topaz tools stopped being deferred, or the key was renamed and "
             "every deferred button is now audited as broken")
    for reason in reasons:
        if len(reason.strip()) < 10:
            fail("a deferred entry has no real reason (%r); a control hidden "
                 "without a stated reason is indistinguishable from this bug" % reason)

    # 6. the module must self-check its own data
    if "selfCheck" not in nodes:
        fail("canvas_nodes.js has no self-check; a malformed registry would only "
             "show up as a blank toolbar with no explanation")

    # 7. the Topaz probe must never cache a failure as an answer. The status
    #    route is registered from on_app_started, AFTER the page is already
    #    being served: the first probe can 404 by pure startup timing, and the
    #    old one-shot promise cached that miss as unavailable for the whole
    #    session - the buttons showed or not on the SAME machine and config
    #    depending on a millisecond race, and only a reload rolled the dice
    #    again. Checked against CODE (comments stripped above): the comments
    #    documenting the bug quote the very patterns banned here.
    if "onTopazAvailable" not in code:
        fail("canvas_extra.js no longer routes the Topaz reveal through "
             "onTopazAvailable - a one-shot status fetch reintroduces the "
             "startup race where a 404 before route registration hides the "
             "tools until the next full reload")
    if not re.search(r"setTimeout\(\s*topazProbeRun", code):
        fail("the Topaz probe does not reschedule itself after a failed status "
             "fetch - a transient failure (404 during on_app_started, server "
             "busy) becomes a permanent 'unavailable'")
    if re.search(r"catch\b[^}]*available:\s*false", code, re.S):
        fail("the Topaz status .catch coerces failure to {available:false} - "
             "'could not ask' must stay distinct from 'server said no'; "
             "conflating them is the exact cache-the-race bug")

    # --- the part that actually runs the code ---
    data = run_harness()
    if data is not None:
        if data.get("fatal"):
            fail(data["fatal"])
        else:
            test_runtime_behaviour(data)

    return report(data)


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    for s in SKIPS:
        print("PARTIAL SKIP -", s)
    if data is None:
        print("ok - source-level policy checks only (see the skip above)")
        return 0
    print("ok - %d tools declaring %d controls, %d toolbar buttons, %d deliberately "
          "deferred (%s); renderer output matches the registry id for id, all "
          "others revealed by contract and audited after attach"
          % (data["toolCount"], len(data["ownedIds"]), len(data["toolbarIds"]),
             len(data["deferred"]), ", ".join(sorted(data["deferred"]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
