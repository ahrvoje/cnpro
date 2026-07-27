"""Nothing may write the unit State by rebuilding it from a snapshot.

THE INVARIANT

    Every handler whose `outputs` is the unit gr.State writes the fields it
    owns, in place, on the object it was handed. None of them constructs a new
    unit.

WHY
---
`dataclasses.replace(current, image=x)` reads as "set one field". It is not: it
builds a NEW unit whose other ~60 fields are copies of values read when THAT
request started. So two handlers overlapping is a lost update by construction -
the later writer's snapshot predates the earlier writer's field and silently
reverts it. Gradio gives every `.change` its own dependency and runs them
concurrently, so overlapping is the normal case, not the pathological one.

This bit for real, and it was not subtle in effect, only in cause: ONE image
landing on a canvas moves two channels in the same tick (background, and the
scribble layer that is resized with it), so `image` - megabytes of base64 -
races `image_fg`, which is a blank canvas. Whichever lands second wins the whole
unit. When that was `image_fg`, the canvas showed the new picture, the gradio
channel held the new picture, and the generation used the OLD one, permanently.
The only path that un-stuck it was closing and reopening the Input tab, because
`close_active_slot` happened to write the unit atomically.

`setattr` on the live State object touches exactly one attribute. Concurrent
handlers write disjoint attributes, cannot revert each other, and need no lock.
gr.State passes its value through by reference, so the object mutated is the
stored one.

This test is AST-based on purpose: it needs no gradio, no torch and no host, so
it runs anywhere and cannot be skipped into uselessness. The behavioural half
below then demonstrates, on the two strategies themselves, that the difference
is real rather than stylistic.

Exit code 0 = pass; 1 = fail.
"""
import ast
import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
UI_GROUP = os.path.join(EXTENSION, "lib_cnpro", "controlnet_ui", "controlnet_ui_group.py")

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


# --------------------------------------------------------------- structural
def unit_state_writers(tree):
    """Name of every function passed as `fn=` to an event whose `outputs`
    mentions the unit State.

    `outputs` is often a local (`outputs=close_outputs`), so a plain walk of the
    argument misses those writers entirely - and a writer this scan cannot see
    is a writer this test does not protect. Names are therefore expanded through
    their assignment before being searched.
    """
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value

    def mentions_unit(node, seen=None):
        if node is None:
            return False
        seen = set() if seen is None else seen
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and n.attr == "unit":
                return True
            if isinstance(n, ast.Name):
                if n.id == "unit":
                    return True
                if n.id in assigned and n.id not in seen:
                    seen.add(n.id)           # names, not nodes: no cycles, no depth cap
                    if mentions_unit(assigned[n.id], seen):
                        return True
        return False

    writers = {}
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        kwargs = {k.arg: k.value for k in call.keywords if k.arg}
        fn, outputs = kwargs.get("fn"), kwargs.get("outputs")
        if fn is None or outputs is None or not mentions_unit(outputs):
            continue
        if isinstance(fn, ast.Name):
            writers[fn.id] = None
        elif isinstance(fn, ast.Call) and isinstance(fn.func, ast.Name):
            writers[fn.func.id] = None       # e.g. fn=field_updater("image")
    return writers


def check_structure():
    src = open(UI_GROUP, encoding="utf-8").read()
    tree = ast.parse(src)

    writers = unit_state_writers(tree)
    # the wiring must still be recognisable; an empty match means this test has
    # quietly stopped covering anything
    if len(writers) < 2:
        fail("only %d unit-State writer(s) were found by the AST scan (%r). The "
             "parser has stopped matching the wiring, so this check verifies "
             "nothing." % (len(writers), sorted(writers)))
        return

    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in sorted(writers):
        node = funcs.get(name)
        if node is None:
            fail("unit-State writer %r is wired up but its definition was not "
                 "found - cannot verify how it writes." % name)
            continue
        # rebuilding the unit, by any construction, is the defect
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            qual = None
            if isinstance(f, ast.Attribute):
                qual = "%s.%s" % (getattr(f.value, "id", "?"), f.attr)
            elif isinstance(f, ast.Name):
                qual = f.id
            if qual in ("dataclasses.replace", "replace"):
                fail("%s() rebuilds the unit with %s(). That is a snapshot "
                     "write: it carries ~60 stale fields and reverts whatever a "
                     "concurrent handler set. Assign the fields it owns instead."
                     % (name, qual))
        # ...and it must actually assign something
        assigns = [n for n in ast.walk(node)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "setattr"]
        if not assigns:
            fail("%s() is wired to write the unit State but never assigns a "
                 "field - check how it is producing its output." % name)
    return sorted(writers)


# -------------------------------------------------------------- behavioural
@dataclasses.dataclass
class _Unit:
    image: object = None
    image_fg: object = None


def _interleaved(strategy):
    """The interleaving two overlapping gradio dependencies produce.

    Both handlers are given the State value when their request starts, and both
    write when their request finishes. Overlap means BOTH READ BEFORE EITHER
    WRITES - that ordering is the whole hazard, so it is asserted directly
    rather than raced for. (Threads would only add scheduler noise: the reads
    and writes below are the same four steps either way, and a flaky
    reproduction of a lost update is worse than none.)
    """
    state = {"unit": _Unit()}

    handler_a = state["unit"]                                # `image` starts
    handler_b = state["unit"]                                # `image_fg` starts
    state["unit"] = strategy(handler_a, "image", "NEW-IMAGE")  # the big one lands
    state["unit"] = strategy(handler_b, "image_fg", "blank")   # the small one lands
    return state["unit"]


def _snapshot_write(current, field, value):
    return dataclasses.replace(current, **{field: value})


def _in_place_write(current, field, value):
    setattr(current, field, value)
    return current


def check_behaviour():
    lost = _interleaved(_snapshot_write)
    if lost.image == "NEW-IMAGE":
        fail("the snapshot-write model did NOT lose the image, so this test is "
             "not demonstrating the hazard it claims to and the structural "
             "check above is unmotivated.")
    kept = _interleaved(_in_place_write)
    if kept.image != "NEW-IMAGE" or kept.image_fg != "blank":
        fail("in-place writes lost a field (%r) - the fix does not hold under "
             "the same interleaving." % (kept,))


def main():
    writers = check_structure()
    check_behaviour()
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - %d unit-State writer(s) (%s) assign their own fields in place; "
          "the snapshot write they replaced provably loses one under the same "
          "interleaving" % (len(writers), ", ".join(writers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
