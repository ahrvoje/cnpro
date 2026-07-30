"""The profile grammar is implemented twice - keep the two implementations equal.

`lib_cnpro/external_code.parse_weight_profile` (python, what RUNS) and
`javascript/weight_profile.js` (the editor, what you SEE) parse and evaluate the
same string format, and MAINTENANCE.md invariant 2 lists the alignment points
that have had to be fixed by hand: 4-decimal serialization, first-mid-wins,
non-finite rejection, band neutrality epsilon, the cosine factor, the gamma
bend, the shared plot range. Until this file existed, the only thing holding
them together was prose - a divergence shows up as "the plot draws one curve
and the image is generated from another", which is invisible until someone
compares pixels.

Run:  D:/store/forge/system/python/python.exe extensions-builtin/sd_forge_controlnet/tests/test_profile_parity.py
Needs `node` on PATH (the editor is loaded headless - see profile_parity_js.js).
Exit code 0 = the two agree on every case.
"""
import json
import math
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
WEBUI = os.path.dirname(os.path.dirname(EXTENSION))


def _import_external_code():
    """external_code without pulling in the whole webui.

    It imports modules.shared / modules.api for logging and the API decode
    helper, neither of which the grammar touches, so both are stubbed. This is
    the same preamble the ad-hoc verification scripts used; keeping it here
    makes the check runnable from a bare checkout.
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


SAMPLES = 21

# (case, tolerance). Exact agreement is expected for plain piecewise-linear
# profiles. Two documented approximations need slack, and ONLY these two:
#   - mid controls: python flattens the parabola into 24 clamped chords while
#     the editor draws the exact parabola (MAINTENANCE: divergence < ~1e-3);
#   - cosine / gamma: python resamples into a 512-point polyline and
#     interpolates linearly between samples, the editor evaluates the closed
#     form.
#
# A case is (string, tolerance) or (string, tolerance, phase_index,
# phase_count): the multi-phase families split the wave between phase_count
# Inputs, so exercising them means naming the Input whose share is compared.
# Without those two the profile is parsed the way a non-fanning-out caller
# sees it - Input 1 of a unit that does not split - which is exactly what the
# bare marker has always meant.
CASES = [
    ("0@1;1@1", 1e-9),
    ("0@1;1@0", 1e-9),
    ("0@0;0.5@1;1@0", 1e-9),
    ("0@1;1@0.5|2", 1e-9),
    ("0@1;1@0|-1~1", 1e-9),
    ("0@0.25;1@0.75|0.5~1.5", 1e-9),
    ("0.25@1;0.75@1", 1e-9),                      # horizontal extension
    ("0@1;M0.5@0.2;1@1", 5e-3),                   # active mid control
    ("0@1;M0.3@0.9;0.6@0.2;1@0.8|2", 5e-3),
    ("0@1;1@1;C1@0", 5e-3),                       # cosine mode
    ("0@1;1@1;C2.5@1.5|2", 5e-3),
    # a marked profile parsed WITHOUT a fan-out (count 1): the degenerate
    # single-Input case, where a family is not a partition of anything but
    # still has a shape - the cosine and Fejer lobes are the plain wave, von
    # Mises the exp(kappa*cos) pulse. Both sides must draw and run the same one.
    ("0@1;1@1;C1@0;P", 5e-3),
    ("0@1;1@1;C1@0;PF", 5e-3),
    ("0@1;1@1;C1@0;PV4", 5e-3),
    ("0@1;1@1;C1@0.8;PV10|0~2", 5e-3),            # the narrowest single lobe
    ("0@1;1@0.5;C2@0;PV1;G0.6", 5e-3),
    # the fan-out itself: every Input's share, in each family. Input 0 of a
    # 2-way cosine split must equal the plain wave (offset 0), and Fejer at
    # N=2 must equal the cosine everywhere - the identity that makes 'PF' a
    # drop-in for 'P' on a two-Input unit.
    ("0@1;1@1;C1@0;P", 5e-3, 0, 2),
    ("0@1;1@1;C1@0;P", 5e-3, 1, 3),
    ("0@1;1@0.4;C2@1;P", 5e-3, 2, 4),
    ("0@1;1@1;C1@0;PF", 5e-3, 0, 2),
    ("0@1;1@1;C1@0;PF", 5e-3, 1, 2),
    ("0@1;1@0.4;C2@1;PF", 5e-3, 3, 5),
    ("0@1;1@1;C1@0.7;PF|0.5~1.5", 5e-3, 1, 3),
    ("0@1;1@1;C1@0;PV0", 5e-3, 1, 4),             # kappa 0 = every input equally on
    ("0@1;1@1;C1@0;PV3", 5e-3, 2, 5),
    ("0@1;1@1;C1@0;PV10", 5e-3, 0, 3),            # kappa max = near-hard switching
    ("0@1;M0.5@0.3;1@1;C2@2;PV6;G1.7|-1~2", 5e-3, 4, 6),  # every feature at once
    # convergence ('A<at>@<e>'): the waves slide onto the flat share, reaching
    # it at `at` and holding it. Every family, both sides of the pad's middle
    # row, and the degenerate single wave (which converges onto the envelope).
    ("0@1;1@1;C2@0;P;A1@1", 5e-3, 1, 3),
    ("0@1;1@1;C2@0;PF;A0.5@1", 5e-3, 2, 4),
    ("0@1;1@1;C2@0;PF;A0.5@1", 5e-3, 0, 4),
    ("0@1;1@1;C2@0;PV5;A0.75@0.25", 5e-3, 3, 5),
    ("0@1;1@1;C2@0;PV5;A0.75@4", 5e-3, 3, 5),
    ("0@1;1@0.6;C3@1;A0.6@2", 5e-3),              # no family: one wave, fades to the envelope
    ("0@1;1@1;C2@0;P;A0@1", 5e-3, 1, 3),          # arrives at 0: flat share throughout
    ("0@1;1@1;C1@0.4;PV3;A0.8@0.5;G1.5|0.5~1.5", 5e-3, 2, 3),
    ("0@1;1@0.2;G2", 5e-3),                       # response exponent
    ("0@1;1@0.2;G0.5|2", 5e-3),
    ("0@1;1@1;C1@0.5;G3|-1~2", 5e-3),             # wave + bend + scale
    # band segments ride along and must not disturb the main profile
    ("0@1;1@0.5|2#C0@0.75;1@0.75|2#M0@0.25;1@0.25|2", 1e-9),
    ("0@1;1@1#C0@0.5;1@0.9;C1@0#BC", 5e-3),
    # ...including a band the ladder put on a family rung or gave a
    # convergence: ONLY the main profile fans out, so python parses these with
    # a count of 1 and the editor has to preview the same single kernel
    ("0@1;1@1#C0@1;1@1;C2@0;PV5", 5e-3),
    ("0@1;1@1#M0@1;1@1;C2@0;PF;A0.5@1", 5e-3),
    ("0@1;1@1;C1@0;PF;A0.5@2#F0@1;1@1;C2@0;P;A0.3@0.5", 5e-3, 1, 3),
    # depth segment: same grammar, and equally invisible to the main parse
    ("0@1;1@0.5|2#D0@0.25;1@0.75|2", 1e-9),
    ("0@1;1@1#D0@0;M0.5@0.9;1@1", 5e-3),
    ("0@1;1@0.5|2#D0@0.25;1@0.75|2#C0@0.75;1@0.75|2#BC", 1e-9),
    # ...and the depth segment carries its OWN range: depth is a second plot
    # with its own Y axis, so its suffix is read on its own and neither pulls
    # the step-domain segments onto it nor gets folded into theirs (python
    # always parsed per segment - this pins the editor to that)
    ("0@1;1@0.5|2#D0@0.5;1@1", 1e-9),             # depth narrower than main
    ("0@1;1@1#D0@0;1@1|0~2", 1e-9),               # depth wider than main
    ("0@0.5;1@1|0.5~1.5#C0@1;1@0.5|0.5~1.5#D0@0.25;1@0.75|-1~2", 1e-9),
    # drift segment: the third plot. Same grammar again, its own range again,
    # and - the part worth pinning - a NEUTRAL OF 0 rather than the multiplier 1,
    # so the two sides have to agree about which strings are omitted as no-ops
    # and which are real curves. Its default axis is [-1, 1], where a drawn 0.5
    # is the neutral shift 0 and the flat line sits mid-plot.
    ("0@1;1@1#S0@1;1@0|-1~1", 1e-9),              # full sweep coarse -> fine
    ("0@1;1@1#S0@0.75;1@0.25|-1~1", 1e-9),        # half sweep
    ("0@1;1@0.5|2#D0@0.25;1@0.75|2#S0@1;1@0|-1~1", 1e-9),
    ("0@1;1@1#S0@0;M0.5@0.9;1@0.5|-1~1", 5e-3),   # mid control on the drift
    ("0@1;1@1#S0@1;1@0;G2|-1~1", 5e-3),           # ...and the response exponent
    ("0@1;1@1#S0@0.5;1@1|0~2", 1e-9),             # drift on a non-default axis
    # every segment at once, each on a different axis, plus band mode: the '#S'
    # segment must survive a fold it takes no part in and stay readable by python
    ("0@0.5;1@1|0.5~1.5#C0@1;1@0.5|0.5~1.5#D0@0.25;1@0.75|-1~2"
     "#S0@0.75;1@0.25|-1~1#BC", 1e-9),
]


def js_results(cases):
    harness = os.path.join(HERE, "profile_parity_js.js")
    payload = json.dumps({"cases": cases, "samples": SAMPLES})
    proc = subprocess.run(["node", harness], input=payload, capture_output=True,
                          text=True, cwd=HERE)
    if proc.returncode != 0:
        raise SystemExit(f"node harness failed:\n{proc.stderr}")
    return json.loads(proc.stdout)["results"]


def sampled(external_code, points):
    if points is None:
        return None
    from cnpro_core.weight_profile import evaluate_weight_profile
    return [evaluate_weight_profile(points, i / (SAMPLES - 1)) for i in range(SAMPLES)]


def depth_field(external_code, case):
    """Depth-under-drift on the same (depth, step) grid the JS harness uses.

    The composite is what RUNS - every injector multiplies by exactly this - and
    it is the only thing the per-curve comparisons cannot check: both sides can
    parse the depth curve and the drift curve identically and still disagree
    about the sign of the shift, the clamp at the axis ends, or which of the two
    arguments the drift is measured in. Compared through the shared
    cnpro_core.depth_multiplier, so a divergence here is a real divergence and
    not this test's own arithmetic.
    """
    from cnpro_core.weight_profile import depth_multiplier
    depth = external_code.parse_depth_profile(case)
    if depth is None:
        return None
    drift = external_code.parse_drift_profile(case)
    out = []
    for i in range(SAMPLES):
        d = i / (SAMPLES - 1)
        for j in range(SAMPLES):
            out.append(depth_multiplier(depth, drift, d, j / (SAMPLES - 1)))
    return out


def compare(name, case, py_values, js_values, tol, failures):
    if (py_values is None) != (js_values is None):
        failures.append(f"{case!r}: {name} present on one side only "
                        f"(python={py_values is not None}, editor={js_values is not None})")
        return
    if py_values is None:
        return
    if len(py_values) != len(js_values):
        failures.append(f"{case!r}: {name} sampled at {len(py_values)} points by python "
                        f"and {len(js_values)} by the editor")
        return
    worst = max(abs(a - b) for a, b in zip(py_values, js_values))
    if not math.isfinite(worst) or worst > tol:
        # index over the WHOLE series, not over SAMPLES: the depth field is a
        # SAMPLES x SAMPLES grid, and a fixed range would have reported the
        # worst point of its first row as if it were the worst point overall
        bad = max(range(len(py_values)), key=lambda i: abs(py_values[i] - js_values[i]))
        where = (f"x={bad / (SAMPLES - 1):.2f}" if len(py_values) == SAMPLES
                 else f"depth={(bad // SAMPLES) / (SAMPLES - 1):.2f}, "
                      f"x={(bad % SAMPLES) / (SAMPLES - 1):.2f}")
        failures.append(
            f"{case!r}: {name} diverges by {worst:.2g} (tolerance {tol:g}) "
            f"at {where}: python {py_values[bad]:.6f} "
            f"vs editor {js_values[bad]:.6f}")


def main():
    external_code = _import_external_code()
    specs = [(c[0], c[1], c[2] if len(c) > 2 else 0, c[3] if len(c) > 3 else 1)
             for c in CASES]
    results = js_results([
        {"text": case, "phaseIndex": index, "phaseCount": count}
        for case, _, index, count in specs
    ])

    failures = []
    for (case, tol, index, count), js in zip(specs, results):
        # no phase_offset: a marked profile derives its own shift from the
        # ordinal, and passing both would apply it twice (external_code)
        py_main = external_code.parse_weight_profile(
            case, phase_index=index, phase_count=count)
        if js.get("main") is None or py_main is None:
            failures.append(f"{case!r}: parsed by only one side "
                            f"(python={py_main is not None}, editor={js.get('main') is not None})")
            continue
        compare("main profile", case, sampled(external_code, py_main), js["main"], tol, failures)

        py_bands = external_code.parse_band_profiles(case) or {}
        js_bands = js.get("bands") or {}
        for band in set(py_bands) | set(js_bands):
            compare(f"band {band}", case, sampled(external_code, py_bands.get(band)),
                    js_bands.get(band), tol, failures)

        compare("depth profile", case,
                sampled(external_code, external_code.parse_depth_profile(case)),
                js.get("depth"), tol, failures)

        compare("drift profile", case,
                sampled(external_code, external_code.parse_drift_profile(case)),
                js.get("drift"), tol, failures)

        # ...and the composite the two of them produce, which is the thing that
        # actually multiplies residuals
        compare("depth field (depth x drift)", case,
                depth_field(external_code, case), js.get("depthField"), tol, failures)

    print(f"{len(CASES)} profile strings compared, "
          f"{SAMPLES} samples each, python vs editor")
    if failures:
        for line in failures:
            print("FAIL", line)
        print(f"\n{len(failures)} divergence(s) - the two implementations of the "
              f"profile grammar have drifted apart (MAINTENANCE.md invariant 2).")
        return 1
    print("OK - the editor and the generation path agree everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
