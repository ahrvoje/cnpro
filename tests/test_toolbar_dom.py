"""Attach CNPro to the host's REAL canvas.html, in a real DOM, and look.

WHY THIS EXISTS
---------------
Two toolbar outages shipped in a row, both silent, both past a green test suite:

  1. the four weight-mask buttons were never revealed (wrong CSS class);
  2. every button vanished, because the id parser truncated each name by one
     character (`slice(4, -14)` on a 13-character suffix).

After the second one, `tests/toolbar_contract_js.js` began EXECUTING the module
-- a real improvement, and still not enough. That harness feeds the module a DOM
stub that the test itself populates from OWNED_IDS, so it verifies reveal/audit
logic *given that the nodes exist*. It assumes the very thing most likely to
break.

Nothing in the suite ran `inject()` against the host's actual template. So no
test could answer the only question the user is ever asking: after CNPro attaches
to a real ForgeCanvas, is each control in the DOM and visible?

This one does, from the two real inputs -- `modules_forge/forge_canvas/canvas.html`
and `javascript/canvas_nodes.js` -- in jsdom.

It immediately found a third bug: `audit()` tested visibility with
`offsetParent === null`, which is null for ANY element in a collapsed container.
CNPro's canvas lives in a gradio accordion that is closed by default, so the
audit would have reported all fourteen controls as broken on every attach --
a check that cries wolf, which is worse than no check.

REQUIRES jsdom, which is not vendored:
    npm install --no-save --prefix <dir> jsdom
    CNPRO_TEST_NODE_PATH=<dir> python tests/test_toolbar_dom.py
Without it the test SKIPS LOUDLY rather than passing quietly.

Exit code 0 = pass or skip; 1 = fail.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []
# Checks that ran but could only cover part of what they are about. Printed on
# success, because "verified" and "partly verified" must not look the same.
SKIPS = []


def fail(msg):
    FAILURES.append(msg)


def run():
    if not shutil.which("node"):
        return None, "node is not on PATH"

    env = dict(os.environ)
    # Default to the scratchpad the harness was developed against, so a bare run
    # works if jsdom is still installed there.
    if "CNPRO_TEST_NODE_PATH" not in env:
        guess = os.path.join(
            os.environ.get("LOCALAPPDATA", "/tmp"), "Temp", "claude")
        env["CNPRO_TEST_NODE_PATH"] = env.get("CNPRO_TEST_NODE_PATH", guess)

    proc = subprocess.run(["node", os.path.join(HERE, "toolbar_dom_js.js")],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if "Cannot find module 'jsdom'" in err:
            return None, ("jsdom is not installed. Install it with:\n"
                          "    npm install --no-save --prefix <dir> jsdom\n"
                          "then re-run with CNPRO_TEST_NODE_PATH=<dir>")
        return None, "node harness failed:\n%s" % err
    try:
        return json.loads(proc.stdout), None
    except ValueError:
        return None, "harness produced no JSON:\n%s\n%s" % (proc.stdout[:400], proc.stderr[:400])


def main():
    data, why = run()
    if data is None:
        # A skip must be IMPOSSIBLE to mistake for a pass. This whole file exists
        # because things that quietly did nothing looked fine.
        print("SKIPPED - the real-DOM toolbar test did not run.")
        print("  %s" % why.replace("\n", "\n  "))
        print("  Nothing about the toolbar has been verified against a real DOM.")
        return 0

    if data.get("loadError"):
        fail("canvas_nodes.js threw at load:\n%s" % data["loadError"])
    if data.get("fatal"):
        fail(data["fatal"])
    if data.get("logs"):
        fail("the canvas modules logged an error or warning:\n  %s"
             % "\n  ".join(data["logs"]))
    if data.get("selfCheck"):
        fail("the tool registry's self-check found problems:\n  %s"
             % "\n  ".join(data["selfCheck"]))

    if FAILURES:
        return report()

    # 1. injection must work against the host's own template
    inj = data["inject"]
    if inj["error"]:
        fail("inject() threw against the real canvas.html:\n%s" % inj["error"])
    if inj["returned"] is not True:
        fail("inject() returned %r - it could not find the host's anchors %r"
             % (inj["returned"], data["anchors"]))
    for k, present in data["anchors"].items():
        if not present:
            fail("the host template has no %r anchor - injection cannot work" % k)

    # 2. every declared node must actually exist afterwards
    if data["missingAfterInject"]:
        fail("%d declared nodes are absent after inject(): %r"
             % (len(data["missingAfterInject"]), data["missingAfterInject"][:10]))

    # 3. attach() must not hit its all-or-nothing gate.
    #    The gate is parsed out of canvas_extra.js rather than restated here, so
    #    an empty parse means the check silently covers nothing - refuse it.
    if data.get("attachGateParsed", 0) < 20:
        fail("only %d gate ids were parsed out of canvas_extra.js's REQUIRED map "
             "(expected all 23). The parser has stopped matching, so this check "
             "is verifying almost nothing." % data.get("attachGateParsed", 0))
    absent_gate = [k for k, v in data["attachGate"].items() if not v]
    if absent_gate:
        fail("attach() would abort on its required-node gate (missing: %s). That "
             "aborts EVERYTHING - toolbar, Topaz probe, listeners - for this canvas."
             % ", ".join(absent_gate))

    # 4. reveal must show exactly the non-deferred controls
    rev = data["reveal"]
    if rev["error"]:
        fail("revealToolbar() threw:\n%s" % rev["error"])
    expected_visible = [i for i in data["toolbarIds"] if i not in data["deferred"]]
    if sorted(rev["visible"]) != sorted(expected_visible):
        fail("visible after reveal = %r\nexpected = %r"
             % (sorted(rev["visible"]), sorted(expected_visible)))
    if sorted(rev["hidden"]) != sorted(data["deferred"]):
        fail("hidden after reveal = %r but only the deferred set %r should be hidden"
             % (sorted(rev["hidden"]), sorted(data["deferred"])))

    # 5. the controls this whole saga is about
    for wm in ("wmaskButton", "wmaskCoarseButton", "wmaskMidButton", "wmaskFineButton"):
        if wm not in rev["visible"]:
            fail("%s is not visible after attaching to a real canvas" % wm)

    # 6. THE AUDIT MUST BE QUIET when nothing is wrong. This is the regression
    #    that offsetParent caused: 14 false alarms on every attach.
    if data["audit"]:
        fail("audit() reports %d problems on a correctly attached canvas - it is "
             "crying wolf, which trains everyone to ignore it:\n  %s"
             % (len(data["audit"]), "\n  ".join(data["audit"][:6])))

    # 7. attaching twice must not duplicate nodes
    if data["duplicateIds"]:
        fail("inject() is not idempotent - duplicate ids after a second call: %r"
             % data["duplicateIds"])

    # 8. THE REVERSE DIRECTION the audit cannot see.
    #    audit() walks the registry and asks "is this in the DOM". Nothing asked
    #    the other way round, and that gap is exactly how `levelsBox` shipped:
    #    markup, ids and wired handlers on both sides, no button that opened it,
    #    two working sliders unreachable, in the fork AND here, silently.
    if data["wantedButUndeclared"]:
        fail("canvas_extra.js resolves %d id(s) that neither the registry nor the "
             "host template declares - those lookups return null and their tool is "
             "dead: %r"
             % (len(data["wantedButUndeclared"]), data["wantedButUndeclared"]))
    if data["declaredButUnused"]:
        fail("the registry declares %d id(s) that canvas_extra.js never resolves - "
             "chrome nothing wires, which is how a menu ends up unreachable: %r"
             % (len(data["declaredButUnused"]), data["declaredButUnused"]))

    # 9. THE LABEL SIZERS DESCRIBE THE LABELS THAT ARE ACTUALLY WRITTEN.
    #
    #    The menu's label column is sized from `labelMax` in canvas_tools.js; the
    #    text comes from canvas_extra.js. Declared in one place, honoured in
    #    another - so the drift has to raise (ARCHITECTURE.md section 8), and a
    #    browser cannot raise it: measuring the DECLARED strings in a real layout
    #    engine asks the declaration about itself and passes while the app shows
    #    something wider.
    check_label_sizers(data)

    return report(data)


def check_label_sizers(data):
    rows = {r["labelId"]: r for r in data.get("rowSpecs", [])}
    writes = {w["id"]: w for w in data.get("labelWrites", [])}
    if not rows:
        fail("no range rows were read out of the registry - the label-sizer check "
             "verified nothing")
        return

    # COVERAGE, CHECKED. Walking the registry's rows and rendering them are two
    # code paths, and when rows gained the ability to be nested in a group() the
    # walk stopped seeing three of them while the renderer still emitted their
    # sizers - so this check quietly covered less and said nothing. Comparing
    # what was walked against what is in the DOM makes that impossible.
    in_dom = data.get("sizerLabelsInDom")
    if in_dom is not None and in_dom != len(rows):
        fail("the registry walk found %d range row(s) but %d label(s) carry a "
             "sizer in the DOM - the walk is missing rows (nested in a group?), "
             "so they are unchecked" % (len(rows), in_dom))

    unchecked = []
    for label_id, row in sorted(rows.items()):
        declared = row.get("labelMax")
        if not declared:
            fail("range %r declares no labelMax" % row["id"])
            continue
        write = writes.get(label_id)
        if not write:
            # Not fatal on its own (a label could be static), but it means this
            # row's declaration is unverifiable, and that must be SAID.
            unchecked.append("%s (nothing in canvas_extra.js/weight_mask.js "
                             "writes it, so its labelMax is unverified)" % label_id)
            continue

        # 9a. THE PREFIX, verbatim. This is what a rename breaks.
        for prefix in write["prefixes"]:
            if not declared.startswith(prefix):
                fail("labelMax %r for %s does not start with %r, which is what "
                     "%s actually writes into it. The label column is sized for "
                     "text the tool never shows."
                     % (declared, label_id, prefix, "/".join(write["files"])))
                break
        else:
            # 9b. THE VALUE, by width. The prefix can be right while the value is
            #     declared one digit short, which is the whole failure this sizing
            #     scheme can produce: "mask opacity 1" reserves 16px too little
            #     for "mask opacity 100".
            prefix = write["prefixes"][0]
            suffix = next((s for s in write["suffixes"] if s), "")
            value = declared[len(prefix):]
            if suffix and value.endswith(suffix):
                value = value[:-len(suffix)]

            if write["toFixed"] is not None:
                decimals = write["toFixed"]
                if "." not in value or len(value.split(".")[-1]) != decimals:
                    fail("labelMax %r for %s carries %r, but %s prints it with "
                         "toFixed(%d) - the declared width is %d decimal(s) out"
                         % (declared, label_id, value, "/".join(write["files"]),
                            decimals, decimals - len(value.split(".")[-1])))
            elif write["rawValue"]:
                # the slider's own value, printed as-is: the widest it can get is
                # the longest of the range ends the registry itself declares
                ends = [row["min"], row["max"]]
                widest = max(len(_render(e)) for e in ends)
                if len(value) < widest:
                    fail("labelMax %r for %s leaves room for %d character(s) of "
                         "value, but %s prints the slider directly and its range "
                         "%s..%s needs %d"
                         % (declared, label_id, len(value), "/".join(write["files"]),
                            _render(row["min"]), _render(row["max"]), widest))
            else:
                unchecked.append("%s (value is computed: %s - only its prefix %r "
                                 "could be checked)"
                                 % (label_id, "; ".join(write["exprs"])[:60], prefix))

    # Never silently. A row whose value the parser cannot reason about is stated,
    # so "checked" and "could not be checked" stay two different outcomes.
    if unchecked:
        SKIPS.append("%d label(s) had only a partial sizer check:\n    %s"
                     % (len(unchecked), "\n    ".join(unchecked)))


def _render(value):
    """How JS would stringify a range endpoint (1 not 1.0, -45 not -45.0)."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    for s in SKIPS:
        print("PARTIAL -", s)
    n = len(data["reveal"]["visible"])
    print("ok - injected into the host's real canvas.html: all %d registry nodes "
          "present, %d/%d attach-gate ids resolved, %d toolbar buttons visible, "
          "%d deferred, audit clean, no id declared-but-unwired or wired-but-"
          "undeclared" % (data["ownedCount"], len(data["attachGate"]),
                          data["attachGateParsed"], n, len(data["deferred"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
