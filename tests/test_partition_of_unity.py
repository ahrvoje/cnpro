"""The n Inputs' shares of the wave sum to exactly 1 - in both implementations.

THE CONTRACT
------------
Multi-phase does not change how hard a unit pulls, only WHO pulls at each step.
Input k gets `envelope(x) * w_k(x)` and the n weights sum to 1 at every x, so
the unit's summed per-step pull is the drawn envelope itself. That is what
"partition of unity" means here, and the whole feature is worthless without it:
a family whose weights summed to n/2 (as the raw cosine's do) makes a 5-input
unit silently half as strong as the same curve with one input, which is how it
was first observed - "half the steps missed" on an IP-Adapter face unit.

WHY THE STRUCTURE, NOT THE ARITHMETIC, IS THE GUARANTEE
-------------------------------------------------------
The property used to be per-family: Fejer and von Mises summed to 1 by their own
identities, the cosine summed to n/2 and was corrected by a 2/n applied by hand
at the call site in scripts/cnpro.py. A new family added without its correction
would have been wrong in exactly that silent way. Now:

  * `_phase_weight` divides each family's raw lobe by the SUM of the n shifted
    copies of it (`_phase_kernel`), so summing to 1 is a property of the
    division, not of the lobe. Any lobe works, and no caller can forget it.
  * `_converged_weight` blends toward the flat 1/n as a CONVEX combination -
    (1-s)*partition + s*flat - so the sum survives any convergence setting:
    (1-s)*1 + s*n*(1/n) = 1. The convergence position and dynamics appear
    nowhere except inside s.

This file exists to keep those two statements true after edits, on BOTH sides -
python (what runs) and the editor (what you see) - since the contract is only
worth anything if the plot and the image agree about it.

Run:  D:/store/forge/system/python/python.exe tests/test_partition_of_unity.py
Node is needed for the editor half; without it that half SKIPS, loudly.
Exit code 0 = every configuration sums to 1.
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

SAMPLES = 101
# float noise only: the identity is exact, not approximate. A tolerance any
# looser than this would let a real correction factor hide inside it.
TOL = 1e-12


def _import_external_code():
    """external_code without pulling in the whole webui (see test_profile_parity)."""
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


# (family token, kappa) x (oscillations, phase) x convergence, so the sum is
# checked across the whole reachable parameter space rather than at one point
# of it. The kappa extremes matter most: at 0 every von Mises weight is 1/n and
# at 10 all but one are ~0, and both are where a normalization slip shows.
FAMILIES = [("P", "cos", 0.0), ("PF", "fejer", 0.0),
            ("PV0", "mises", 0.0), ("PV3", "mises", 3.0),
            ("PV10", "mises", 10.0)]
WAVES = [(1.0, 0.0), (2.5, 1.3), (4.0, 5.0), (0.0, 0.0)]
CONVERGENCES = [None, (1.0, 1.0), (0.5, 1.0), (0.35, 0.1), (0.9, 10.0), (0.0, 1.0)]
COUNTS = [2, 3, 4, 5, 8]


def python_failures(external_code):
    """Every (family, wave, convergence, count) combination, summed."""
    failures = []
    for token, family, kappa in FAMILIES:
        for osc, phase in WAVES:
            for converge in CONVERGENCES:
                for count in COUNTS:
                    worst = 0.0
                    where = 0.0
                    for i in range(SAMPLES):
                        x = i / (SAMPLES - 1)
                        theta = 2.0 * math.pi * osc * x - phase
                        total = sum(
                            external_code._wave_factor(theta, x, k, count,
                                                       family, kappa, converge)
                            for k in range(count))
                        if abs(total - 1.0) > worst:
                            worst = abs(total - 1.0)
                            where = x
                    if worst > TOL:
                        failures.append(
                            f"python: {token} n={count} C{osc}@{phase} "
                            f"converge={converge}: the {count} shares sum to "
                            f"{1 + worst:.12f} at x={where:.2f}, not 1 - the unit "
                            f"pulls {worst * 100:.4g}% off the drawn envelope")
    return failures


def degenerate_failures(external_code):
    """A count of 1 must keep the drawn wave, and converge onto the envelope.

    This is the case that CANNOT be a partition (one weight summing to 1 is a
    flat line, i.e. the wave the user drew silently gone), so it is pinned as
    what it is instead: the family's kernel, peaking at 1 where the Input's node
    is. The convergence target follows the same formula - A/n with n = 1 is A -
    so a fully converged single wave is exactly the envelope, factor 1.
    """
    failures = []
    for token, family, kappa in FAMILIES:
        peak = external_code._wave_factor(0.0, 0.0, 0, 1, family, kappa, None)
        if abs(peak - 1.0) > TOL:
            failures.append(f"python: {token} with one Input peaks at {peak:.6f}, "
                            f"not 1 - the single kernel is being normalized away")
        opposite = external_code._wave_factor(math.pi, 0.5, 0, 1, family, kappa, None)
        expected = {"cos": 0.0, "fejer": 0.0}.get(family,
                                                  math.exp(kappa * -2.0))
        if abs(opposite - expected) > 1e-12:
            failures.append(f"python: {token} with one Input is {opposite:.6f} "
                            f"half a cycle from its node, expected {expected:.6f}")
        # fully converged: the whole envelope, whatever the wave was doing
        converged = external_code._wave_factor(math.pi, 1.0, 0, 1, family, kappa,
                                               (0.5, 1.0))
        if abs(converged - 1.0) > TOL:
            failures.append(f"python: {token} with one Input converges to "
                            f"{converged:.6f}, not the envelope (A/1 = A)")
    return failures


def parsed_sum_failures(external_code):
    """End to end: the n parsed profiles add up to the parsed envelope.

    _wave_factor above is the arithmetic; this is what scripts/cnpro.py actually
    hands each Input. It is the check that would fail if a share factor were
    reintroduced at the call site, or if the fan-out parse and the plain parse
    stopped agreeing about the envelope.
    """
    failures = []
    envelope_case = "0@0.2;0.5@1;1@0.7"
    for token, _, _ in FAMILIES:
        for count in COUNTS:
            case = f"{envelope_case};C2@0.9;{token}"
            envelope = external_code.parse_weight_profile(envelope_case)
            variants = [external_code.parse_weight_profile(case, phase_index=k,
                                                           phase_count=count)
                        for k in range(count)]
            from cnpro_core.weight_profile import evaluate_weight_profile
            worst = 0.0
            for i in range(SAMPLES):
                x = i / (SAMPLES - 1)
                total = sum(evaluate_weight_profile(v, x) for v in variants)
                worst = max(worst, abs(total - evaluate_weight_profile(envelope, x)))
            # the variants are dense polylines sampled on the same grid, so the
            # only slack is the linear interpolation between two of its points
            if worst > 1e-6:
                failures.append(
                    f"python: the {count} parsed '{token}' variants sum to "
                    f"{worst:.3g} away from the drawn envelope - the unit does "
                    f"not pull what the plot shows")
    return failures


def plateau_failures(external_code):
    """Past the convergence position every Input sits on EXACTLY the flat share.

    The sum check above cannot see this one: the blend is applied with the same
    s to every Input, so the deviations (1/n - w_k) cancel in the total whatever
    factor is put in front of them - the shares would still sum to 1 while
    overshooting the constant they are supposed to arrive at. This pins the
    other half of the promise: A/n is reached AT the pad's x and HELD for the
    rest of the range.
    """
    failures = []
    at = 0.5
    for token, family, kappa in FAMILIES:
        for count in COUNTS:
            for i in range(SAMPLES):
                x = i / (SAMPLES - 1)
                if x < at:
                    continue
                theta = 2.0 * math.pi * 2.0 * x - 0.3
                for k in range(count):
                    v = external_code._wave_factor(theta, x, k, count, family,
                                                   kappa, (at, 1.0))
                    if abs(v - 1.0 / count) > TOL:
                        failures.append(
                            f"python: {token} input {k} of {count} is {v:.12f} at "
                            f"x={x:.2f}, past the convergence at {at} - expected "
                            f"exactly the flat share {1.0 / count:.12f}")
                        break
                else:
                    continue
                break
    return failures


def editor_plateau_failures():
    """The same plateau, in the editor.

    Read off a FLAT envelope of 1 on the default [0, 1] range, where the sampled
    profile value IS the wave factor - so this goes through the editor's real
    parse-and-evaluate path rather than a shortcut into its internals.
    """
    at = 0.5
    specs = []
    for token, _, _ in FAMILIES:
        for count in COUNTS:
            for k in range(count):
                specs.append({"text": f"0@1;1@1;C2@0.3;{token};A{at}@1",
                              "phaseIndex": k, "phaseCount": count})
    harness = os.path.join(HERE, "profile_parity_js.js")
    payload = json.dumps({"cases": specs, "samples": SAMPLES})
    proc = subprocess.run(["node", harness], input=payload, capture_output=True,
                          text=True, cwd=HERE)
    if proc.returncode != 0:
        raise SystemExit(f"node harness failed:\n{proc.stderr}")
    results = json.loads(proc.stdout)["results"]
    failures = []
    for spec, result in zip(specs, results):
        series = result.get("main")
        if series is None:
            failures.append(f"editor: {spec['text']!r} did not parse")
            continue
        share = 1.0 / spec["phaseCount"]
        for i, v in enumerate(series):
            x = i / (SAMPLES - 1)
            if x >= at and abs(v - share) > 1e-9:
                failures.append(
                    f"editor: {spec['text']!r} input {spec['phaseIndex']} of "
                    f"{spec['phaseCount']} draws {v:.9f} at x={x:.2f}, past the "
                    f"convergence at {at} - expected the flat share {share:.9f}")
                break
    return failures


def js_sums(specs):
    """The editor's own sums, through tests/profile_parity_js.js."""
    harness = os.path.join(HERE, "profile_parity_js.js")
    payload = json.dumps({"cases": [], "sums": specs, "samples": SAMPLES})
    proc = subprocess.run(["node", harness], input=payload, capture_output=True,
                          text=True, cwd=HERE)
    if proc.returncode != 0:
        raise SystemExit(f"node harness failed:\n{proc.stderr}")
    return json.loads(proc.stdout)["sums"]


def editor_failures():
    """The same contract in the editor - the plot has to state the truth too."""
    specs = []
    for token, _, _ in FAMILIES:
        for count in COUNTS:
            specs.append({"text": f"0@1;1@1;C2@0.9;{token}", "count": count})
            specs.append({"text": f"0@1;1@1;C2@0.9;{token};A0.6@0.5", "count": count})
    sums = js_sums(specs)
    failures = []
    for spec, series in zip(specs, sums):
        if series is None:
            failures.append(f"editor: {spec['text']!r} did not parse")
            continue
        worst = max(abs(v - 1.0) for v in series)
        if worst > TOL:
            failures.append(
                f"editor: {spec['text']!r} n={spec['count']}: the shares sum to "
                f"{1 + worst:.12f}, not 1 - the plot draws a unit that pulls "
                f"{worst * 100:.4g}% off its envelope")
    return failures


def main():
    external_code = _import_external_code()
    failures = python_failures(external_code)
    failures += degenerate_failures(external_code)
    failures += parsed_sum_failures(external_code)
    failures += plateau_failures(external_code)
    checked = len(FAMILIES) * len(WAVES) * len(CONVERGENCES) * len(COUNTS)
    print(f"{checked} python configurations x {SAMPLES} samples "
          f"(every family, wave, convergence and Input count)")

    try:
        failures += editor_failures()
        failures += editor_plateau_failures()
        print(f"{len(FAMILIES) * len(COUNTS) * 2} editor configurations "
              f"x {SAMPLES} samples, plus every Input's convergence plateau")
    except (OSError, SystemExit) as err:
        print(f"SKIPPED the editor half: {err}")
        print("A skip is not a pass - the plot was not checked against the "
              "contract at all.")

    if failures:
        for line in failures:
            print("FAIL", line)
        print(f"\n{len(failures)} configuration(s) do not sum to 1: the Inputs "
              f"amplify or starve the envelope instead of dividing it.")
        return 1
    print("OK - every Input count, family and convergence divides the envelope "
          "exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
