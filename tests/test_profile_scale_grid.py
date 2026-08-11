"""The profile plot's range selects offer stops that fit what the axis MEANS.

A weight and a multiplier are different quantities and the editor draws both.
The main profile and the three band profiles are WEIGHTS - a share of the
control - and their axis is [0, 1]: a weight above 1 is a unit pulling harder
than the whole of itself, which is what the coverage panel calls oversaturation,
and a negative one is repulsion nobody asked the weight plot for. The depth
curve is a per-layer MULTIPLIER whose neutral is 1, and the drift is a SHIFT
whose useful half is negative; capping either at 1 would put "leave these layers
alone" at the very top of its plot.

So the grid follows the axis on screen, and this pins that:

1. the weight axis offers exactly (0, 0.05, 0.1, 0.25, 0.5, 0.75, 1) - the
   same list for main and for every band, because they share one plot. The
   intermediate 0.15/0.2/0.35 stops were removed from the picker on
   2026-08-11;
2. depth and drift keep -1 .. 2 in steps of 0.25;
3. the balance editor keeps its own [0, 1] quarters;
4. a profile STRING carrying an off-grid weight range still parses to that
   range. This is the half that is easy to get wrong in the other direction:
   clamping a legacy '|0~2' band range to 1 on load would silently halve every
   weight in every saved profile and every docs example. Unreachable from the
   picker is not the same as rewritten on sight.

Run:  python tests/test_profile_scale_grid.py
Needs `node` (the editor is loaded headless - see profile_scale_grid_js.js).
Exit code 0 = pass.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

WEIGHT_GRID = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0]
MULTIPLIER_GRID = [2 - 0.25 * i for i in range(13)]          # 2 .. -1
BALANCE_GRID = [1, 0.75, 0.5, 0.25, 0]

WEIGHT_AXES = ["main", "coarse", "mid", "fine"]
OWN_AXES = ["depth", "drift"]

# (string, expected lo, expected hi). The last two are what a saved profile or
# an infotext paste can carry.
PARSES = [
    ("0@1;1@0", 0.0, 1.0),
    ("0@1;1@0|0.5", 0.0, 0.5),
    ("0@1;1@0|0.05~0.35", 0.05, 0.35),
    ("0@1;1@0|0.2", 0.0, 0.2),
    ("0@1;1@0|0.15~0.35", 0.15, 0.35),
    ("0@0.5;1@0.5|0~2", 0.0, 2.0),          # docs/example_1.html's band range
    ("0@1;1@0|-1~1", -1.0, 1.0),
]

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def close(a, b):
    return abs(a - b) < 1e-9


def main():
    harness = os.path.join(HERE, "profile_scale_grid_js.js")
    payload = json.dumps({
        "axes": WEIGHT_AXES + OWN_AXES,
        "parses": [p[0] for p in PARSES],
    })
    try:
        proc = subprocess.run(["node", harness], input=payload, capture_output=True,
                              text=True, cwd=HERE)
    except (OSError, FileNotFoundError):
        print("SKIPPED - node is not on PATH, so the range grids were NOT verified")
        return 0
    if proc.returncode != 0:
        print("FAIL - the node harness failed:\n%s" % proc.stderr.strip()[:2000])
        return 1
    got = json.loads(proc.stdout)

    for axis in WEIGHT_AXES:
        grid = got["grids"].get(axis)
        if grid != WEIGHT_GRID:
            fail("the %s axis offers %r; a weight plot is [0, 1] and its stops "
                 "are %r - anything else lets a single unit be given a weight "
                 "the coverage map has to draw as oversaturation"
                 % (axis, grid, WEIGHT_GRID))

    for axis in OWN_AXES:
        grid = got["grids"].get(axis)
        if not grid or len(grid) != len(MULTIPLIER_GRID) \
                or any(not close(a, b) for a, b in zip(grid, MULTIPLIER_GRID)):
            fail("the %s axis offers %r, expected the -1 .. 2 grid in quarters. "
                 "It is a multiplier/shift, not a weight: 1 is its NEUTRAL and "
                 "has to sit inside the range, not at the top of it"
                 % (axis, grid))

    balance = got["balance"]
    if not balance or len(balance) != len(BALANCE_GRID) \
            or any(not close(a, b) for a, b in zip(balance, BALANCE_GRID)):
        fail("the balance editor offers %r, expected %r (balance_factors clamps "
             "it to [0, 1])" % (balance, BALANCE_GRID))

    for (text, lo, hi), parsed in zip(PARSES, got["parses"]):
        if parsed is None:
            fail("%r no longer parses at all" % text)
            continue
        if not close(parsed["lo"], lo) or not close(parsed["hi"], hi):
            fail("%r parsed to [%s, %s], expected [%s, %s]. A range that is not "
                 "on the picker's grid must still be READ as written - clamping "
                 "it on load rewrites saved profiles, and halving a '|0~2' band "
                 "is a silent change to every weight in it"
                 % (text, parsed["lo"], parsed["hi"], lo, hi))

    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - the weight axis offers %d stops in [0, 1], depth and drift keep "
          "-1 .. 2, balance keeps its quarters, and off-grid ranges still parse "
          "as written" % len(WEIGHT_GRID))
    return 0


if __name__ == "__main__":
    sys.exit(main())
