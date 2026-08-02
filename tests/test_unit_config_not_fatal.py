"""A half-configured unit takes only ITSELF out of the run. A bug still shouts.

THE CRASH THIS EXISTS FOR
-------------------------
    File "scripts/cnpro.py", line 854, in process_unit_after_click_generate
        assert unit.model != 'None', 'You have not selected any control model!'
    AssertionError: You have not selected any control model!

Enable a unit, press Generate before picking its model, and that assertion came
out of `Script.process`. The host catches it per script and carries on - so the
consequences were:

  * the hook died at the FIRST unconfigured unit, so every unit AFTER it was
    never prepared either, however complete its own setup was;
  * the generation then finished with no control from ANY unit;
  * and the result is a perfectly ordinary-looking image. Nothing in the gallery
    distinguishes "ControlNet ran" from "ControlNet was skipped entirely" - the
    single most expensive kind of wrong output this codebase has (invariant 26's
    rationale: a plausible wrong image costs far more to notice than an error).

WHAT IS PINNED HERE
-------------------
1. `UnitConfigurationError` exists. It is the type that separates "the user has
    not finished setting this unit up" from "something is broken", which design
    rule 4.2 requires be two outcomes rather than one.
2. No `assert` in the file carries a user-facing selection message any more.
    An assert is the wrong instrument twice over: it aborts the whole hook, and
    `python -O` deletes it.
3. `process()` catches `UnitConfigurationError` and does NOT re-raise - the run
    continues with the remaining units.
4. It appends to `current_units` OUTSIDE that try, so a skipped unit still pairs
    with its own params and the list cannot renumber (invariant behind
    `current_units`).
5. THE OTHER DIRECTION, and the one that keeps this from becoming a blanket
    swallow: any broader `except` in `process()` MUST re-raise. Without this
    check, "no crash" is also satisfied by catching everything and hiding real
    failures, which is the bug this fix is supposed to be the opposite of.
6. The skip reaches the INFOTEXT, not just a console log.

Needs nothing but a stdlib `ast`. Exit code 0 = pass.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
SOURCE = os.path.join(EXTENSION, "scripts", "cnpro.py")

# Words that only ever appear in a message about the USER's half-finished setup.
# An assert carrying one of these is a configuration problem wired to a fatal
# instrument.
USER_FACING = ("not selected", "have not selected", "no input image",
               "no control model", "select any control")

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def handler_names(handler):
    """The exception names one `except` clause catches."""
    t = handler.type
    if t is None:
        return ["<bare except>"]
    parts = t.elts if isinstance(t, ast.Tuple) else [t]
    out = []
    for p in parts:
        if isinstance(p, ast.Name):
            out.append(p.id)
        elif isinstance(p, ast.Attribute):
            out.append(p.attr)
    return out


def reraises(handler):
    return any(isinstance(n, ast.Raise) for n in ast.walk(handler))


def main():
    if not os.path.exists(SOURCE):
        print("FAIL - %s is missing" % SOURCE)
        return 1
    src = open(SOURCE, encoding="utf-8").read()
    tree = ast.parse(src)

    # ---- 1. the type
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if "UnitConfigurationError" not in classes:
        fail("UnitConfigurationError is gone from scripts/cnpro.py. It is what "
             "separates 'the user has not picked a model' from 'something is "
             "broken'; without it those two collapse into one outcome again and "
             "only one of them can be right.")
        return report()

    # ---- 2. no user-facing assert
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        text = (ast.get_source_segment(src, node) or "").lower()
        for word in USER_FACING:
            if word in text:
                fail("line %d asserts on a USER-FACING configuration problem "
                     "(%r). An assert is the wrong instrument twice: it aborts "
                     "the whole process hook - taking every OTHER unit down with "
                     "it and producing an uncontrolled image that looks normal - "
                     "and `python -O` removes it outright. Raise "
                     "UnitConfigurationError instead."
                     % (node.lineno, word))
                break

    # ---- the loop
    process = find_function(tree, "process")
    if process is None:
        fail("Script.process is gone from scripts/cnpro.py")
        return report()

    tries = [n for n in ast.walk(process) if isinstance(n, ast.Try)]
    if not tries:
        fail("process() has no try/except at all, so ANY unit that is not fully "
             "set up still aborts the hook for every other unit. This is the "
             "reported crash: `assert unit.model != 'None'` reaching the host as "
             "an AssertionError.")
        return report()

    caught_config = False
    for node in tries:
        for handler in node.handlers:
            names = handler_names(handler)
            if "UnitConfigurationError" in names:
                caught_config = True
                # ---- 3. it must not re-raise
                if reraises(handler):
                    fail("process() catches UnitConfigurationError and re-raises "
                         "it - which is the crash again, one indirection later.")
            elif "Exception" in names or "BaseException" in names or names == ["<bare except>"]:
                # ---- 5. THE OTHER DIRECTION
                if not reraises(handler):
                    fail("process() catches %s WITHOUT re-raising. 'No crash' is "
                         "then also satisfied by swallowing genuine failures - a "
                         "broken loader, a bad state dict - and every one of them "
                         "would produce a silently uncontrolled image, which is "
                         "the exact outcome this fix exists to remove. Only "
                         "UnitConfigurationError may be absorbed."
                         % "/".join(names))

    if not caught_config:
        fail("process() never catches UnitConfigurationError, so a unit missing "
             "its model still takes the whole hook down")

    # ---- 4. pairing survives a skip. The append must not be INSIDE the try, or
    #         a skipped unit drops out of current_units and every later hook
    #         iterates a list whose positions no longer match the units.
    appends = [n for n in ast.walk(process)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "append"
               and isinstance(n.func.value, ast.Attribute)
               and n.func.value.attr == "current_units"]
    if not appends:
        fail("process() no longer appends to self.current_units - the pairing of "
             "a unit with its own params is what the later hooks iterate")
    else:
        in_try = set()
        for node in tries:
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if sub in appends:
                        in_try.add(sub.lineno)
        if in_try:
            fail("self.current_units.append(...) is inside the try at line(s) %s. "
                 "A skipped unit then never joins the list, so current_units "
                 "renumbers and every later hook pairs a unit with another unit's "
                 "params - the exact failure the list was introduced to make "
                 "impossible." % sorted(in_try))

    # ---- 6. the skip is visible in the artifact, not only in a log
    if "extra_generation_params" not in ast.get_source_segment(src, process):
        fail("process() does not record skipped units in extra_generation_params. "
             "A console line is not enough: an image generated without a unit the "
             "user believed was running is otherwise indistinguishable from one "
             "generated with it.")

    # ---- and the raises themselves still carry an actionable message
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        text = ast.get_source_segment(src, node) or ""
        if "UnitConfigurationError" not in text:
            continue
        if not re.search(r"[Pp]ick|[Cc]hoose|[Dd]rop|[Dd]isable|[Rr]efresh", text):
            fail("a UnitConfigurationError at line %d says what is wrong but not "
                 "what to do about it. These are the only errors in this file the "
                 "user is expected to act on themselves." % node.lineno)

    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - a half-configured unit skips itself with an actionable message "
          "and reaches the infotext; the other units still run; a genuine "
          "failure still raises")
    return 0


if __name__ == "__main__":
    sys.exit(main())
