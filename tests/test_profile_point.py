"""The profile POINT OFFSET - the arithmetic under the "Profile point"
channel of CNPro X/Y and CNPro A/B.

What has to hold, and why each failure would be invisible in the images:

* the offset is expressed in the PLOT'S AXIS UNITS (the `|lo~hi` range), so
  "+0.1" moves the point by 0.1 of whatever the y axis shows - an offset
  applied to the normalized value instead would move twice as far on a 0~2
  plot as its label claims;
* an offset of exactly 0 returns the profile VERBATIM - "0 is the current
  profile" is what both scripts print in their legends, and a re-serialized
  lookalike would defeat render caching and recipe identity;
* everything that is not the chosen point survives byte-for-byte: the other
  points, the M/C/P/A/G tokens, the range suffix - the point index counts
  DRAWN POINTS only, never tokens;
* a point pushed past the plot's range is clamped (and warned about), an
  index that names no point leaves the profile unchanged (and warns), and
  negative indices count from the end;
* offsets may be negative - parse_offsets accepts what parse_factors
  refuses, and parse_factors keeps refusing it.

Run directly: python tests/test_profile_point.py
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []


def fail(message):
    FAILURES.append(message)


def check(condition, message):
    if not condition:
        fail(message)
    return condition


def unit(profile, weight=1.0, start=0.0, end=1.0):
    return types.SimpleNamespace(weight_profile=profile, weight=weight,
                                 guidance_start=start, guidance_end=end)


def main():
    sys.path.insert(0, EXTENSION)
    from lib_cnpro import profile_scale as ps

    # -- the parsers -------------------------------------------------------
    check(ps.parse_offsets("-0.2, 0, 0.2") == [-0.2, 0.0, 0.2],
          "parse_offsets did not read a plain negative/zero/positive list")
    run = ps.parse_offsets("-0.2:0.2:5")
    check(len(run) == 5 and abs(run[0] + 0.2) < 1e-9
          and abs(run[-1] - 0.2) < 1e-9 and abs(run[2]) < 1e-9,
          "parse_offsets did not expand a linspace symmetric around 0: %r"
          % (run,))
    try:
        ps.parse_factors("-0.5")
    except ValueError:
        pass
    else:
        fail("parse_factors accepted a negative factor - the range suffix "
             "cannot hold an inverted curve, and the refactor that gave "
             "offsets their own parser must not have relaxed it")

    # -- axis units --------------------------------------------------------
    moved = ps.offset_profile_point(unit("0@0;0.5@0.5;1@1|0~2"), 0, "Main",
                                    1, 0.5, who="test")
    check(moved == "0@0;0.5@0.75;1@1|0~2",
          "an offset is not in the plot's axis units: +0.5 on a 0~2 plot is "
          "+0.25 normalized, got %r" % (moved,))

    # -- 0 is the current profile, verbatim --------------------------------
    quirky = "0@1.00;0.50@0.500;1@0|0~1"
    check(ps.offset_profile_point(unit(quirky), 0, "Main", 1, 0.0,
                                  who="test") == quirky,
          "offset 0 re-serialized the profile - '0 is the current profile' "
          "must mean byte-identical, or the cell/duel labelled 0 is a "
          "lookalike that defeats render caching and recipe identity")
    check(ps.offset_profile_point(unit("0@1;1@0"), 0, "Coarse", 0, 0.0,
                                  who="test") == "0@1;1@0",
          "offset 0 on an ABSENT line synthesized a segment - absent and "
          "neutral mean the same thing, and the string must stay put")

    # -- tokens are not points, and they survive ---------------------------
    tokened = "0@1;M0.5@0.3;1@0;C3@0.5;G2|0~1"
    moved = ps.offset_profile_point(unit(tokened), 0, "Main", 1, 0.2,
                                    who="test")
    check(moved == "0@1;M0.5@0.3;1@0.2;C3@0.5;G2|0~1",
          "point index 1 did not name the second DRAWN point (tokens must "
          "not count, and must survive verbatim): %r" % (moved,))

    # -- negative index counts from the end --------------------------------
    moved = ps.offset_profile_point(unit("0@0;0.5@0.5;1@1"), 0, "Main",
                                    -1, -0.5, who="test")
    check(moved == "0@0;0.5@0.5;1@0.5",
          "-1 did not name the rightmost point: %r" % (moved,))

    # -- out of range: unchanged, never a guess ----------------------------
    check(ps.offset_profile_point(unit("0@0;1@1"), 0, "Main", 5, 0.3,
                                  who="test") == "0@0;1@1",
          "an index past the last point changed the profile - it must leave "
          "it alone (and warn), not guess a point")

    # -- clamped at the plot's range ---------------------------------------
    moved = ps.offset_profile_point(unit("0@0.95;1@1|0~1"), 0, "Main",
                                    0, 0.5, who="test")
    check(moved == "0@1;1@1|0~1",
          "a point pushed past the range was not clamped to it: %r"
          % (moved,))

    # -- an absent band line offsets from its neutral flat -----------------
    moved = ps.offset_profile_point(unit("0@1;1@0"), 0, "Coarse", 0, -0.5,
                                    who="test")
    check(moved == "0@1;1@0#C0@0.5;1@1",
          "offsetting a point of an ABSENT band did not start from the "
          "neutral flat 1: %r" % (moved,))

    # -- the count the scripts validate against ----------------------------
    check(ps.profile_point_count(unit(tokened), "Main") == 2,
          "profile_point_count counted tokens as points")
    check(ps.profile_point_count(unit("0@1;1@0"), "Depth") == 2,
          "an absent line must count its synthesized neutral two points")
    check(ps.profile_point_count(unit("0@0;0.3@1;0.7@1;1@0"), "Main") == 4,
          "profile_point_count miscounted plain points")

    # -- CNPro absent: Main with no curve stays untouched ------------------
    bare = unit("")
    check(ps.offset_profile_point(bare, 0, "Main", 0, 0.3, who="test") == "",
          "with CNPro absent and no main curve there is nothing to rebuild "
          "the scalars from - the profile must come back unchanged")
    check(ps.profile_point_count(bare, "Main") == 2,
          "the legacy-scalars Main must still report a workable point count")

    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("ok - offsets are in plot units, 0 is byte-identical, tokens are "
          "not points, clamps and misses warn instead of guessing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
