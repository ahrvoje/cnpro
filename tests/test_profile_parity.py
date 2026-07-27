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
    ("0@1;1@1;C1@0;P", 5e-3),                     # multi-phase marker tolerated
    ("0@1;1@0.2;G2", 5e-3),                       # response exponent
    ("0@1;1@0.2;G0.5|2", 5e-3),
    ("0@1;1@1;C1@0.5;G3|-1~2", 5e-3),             # wave + bend + scale
    # band segments ride along and must not disturb the main profile
    ("0@1;1@0.5|2#C0@0.75;1@0.75|2#M0@0.25;1@0.25|2", 1e-9),
    ("0@1;1@1#C0@0.5;1@0.9;C1@0#BC", 5e-3),
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


def compare(name, case, py_values, js_values, tol, failures):
    if (py_values is None) != (js_values is None):
        failures.append(f"{case!r}: {name} present on one side only "
                        f"(python={py_values is not None}, editor={js_values is not None})")
        return
    if py_values is None:
        return
    worst = max(abs(a - b) for a, b in zip(py_values, js_values))
    if not math.isfinite(worst) or worst > tol:
        bad = max(range(SAMPLES), key=lambda i: abs(py_values[i] - js_values[i]))
        failures.append(
            f"{case!r}: {name} diverges by {worst:.2g} (tolerance {tol:g}) "
            f"at x={bad / (SAMPLES - 1):.2f}: python {py_values[bad]:.6f} "
            f"vs editor {js_values[bad]:.6f}")


def main():
    external_code = _import_external_code()
    cases = [c for c, _ in CASES]
    results = js_results(cases)

    failures = []
    for (case, tol), js in zip(CASES, results):
        py_main = external_code.parse_weight_profile(case)
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
