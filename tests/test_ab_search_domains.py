"""Focused regressions for A/B quality domains and perceptual geometry.

This file is intentionally small.  The full stochastic bench lives in
``test_ab_search.py``; these checks pin the structural facts that bench can
average past: ordered choices are ordered, conditional weights only have a
meaning under their own pick, the two on-track rows anchor quality, and a
collage never promotes a configuration the only absolute verdict called bad.
"""

import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib_cnpro import ab_search as search


FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def conditional_space():
    pick = search.Dimension.choice("LoRA", ["A", "B"])
    weight = search.Dimension.choice(
        "LoRA weight", ["0", "0.25", "0.5", "0.75", "1"])
    weight.parent = pick
    return search.Space([pick, weight])


def test_perceptual_geometry():
    space = conditional_space()
    check(abs(space.separation((0, 0), (0, 1)) - 0.25) < 1e-12,
          "adjacent numeric choices are not a quarter-span apart")
    check(abs(space.separation((0, 0), (0, 4)) - 1.0) < 1e-12,
          "the full numeric choice span is not one distance unit")
    check(abs(space.separation((0, 0), (1, 0))
              - space.separation((0, 0), (1, 4))) < 1e-12,
          "a conditional weight adds phantom distance across different picks")
    points = [(0, 0), (0, 1), (1, 4)]
    matrix = space.separations(points)
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            check(abs(float(matrix[i, j]) - space.separation(a, b)) < 1e-12,
                  "batch and scalar perceptual distances disagree")


def test_contextual_similarity_and_kernel():
    engine = search.PreferenceSearch(conditional_space(), seed=3)
    # Under A the weight is visually inert; under B it is vivid.  A global
    # child weight cannot express this, but the row is explicitly conditional.
    for _ in range(24):
        engine.observe((0, 0), (0, 4), 5, similar=True)
        engine.observe((1, 0), (1, 4), 5, similar=False)
    conditional = getattr(engine.space, "conditional_weights", {})
    rates = conditional.get(1)
    check(rates is not None and rates[0] < 0.6 * rates[1],
          "the visual metric did not learn a conditional weight per LoRA pick")

    encoded_a = engine.space.encode([(0, 0), (0, 4)])
    encoded_b = engine.space.encode([(1, 0), (1, 4)])
    ka = float(engine._kernel(encoded_a[:1], encoded_a[1:], 0.3, 1.5)[0, 0])
    kb = float(engine._kernel(encoded_b[:1], encoded_b[1:], 0.3, 1.5)[0, 0])
    check(ka > kb + 1e-6,
          "similarity feedback changes the selection metric but not the utility kernel")


def test_absolute_quality_rows():
    space = search.Space([search.Dimension.choice("x", ["a", "b"])])
    on_track = search.PreferenceSearch(space, seed=0)
    on_track.observe((0,), (1,), 5, similar=False)
    means, _ = on_track._posterior([(0,), (1,)], with_covariance=False)
    check(float(min(means)) > 0.05,
          "an on-track verdict did not move both samples above par")

    bad = search.PreferenceSearch(
        search.Space([search.Dimension.choice("x", ["a", "b"])]), seed=0)
    bad.observe((0,), (1,), 5, disliked=True)
    bad_means, _ = bad._posterior([(0,), (1,)], with_covariance=False)
    check(float(max(bad_means)) < -0.05,
          "a bad verdict did not move both samples below par")
    check(bad.population(2) == [],
          "N-GOOD returned a configuration whose only absolute verdict was bad")


def test_quality_uses_difference_covariance():
    engine = search.PreferenceSearch(
        search.Space([search.Dimension.choice("x", ["a", "b", "c"])]),
        seed=0)
    means = np.array([0.7, 0.5, 0.5])
    covariance = np.array([
        [1.0, 0.99, 0.0],
        [0.99, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    quality = engine._quality(means, covariance)
    check(quality.tolist() == [True, True, False],
          "quality did not distinguish a champion-correlated basin member "
          "from an equally scored remote uncertain point")


def test_collage_rejects_numeric_near_duplicates():
    # The large case from the UI: seven LoRA pick/weight pairs plus one
    # profile-point choice = 15 dimensions.  Moving every numeric setting by
    # one adjacent notch totals only 2 normalized units, below the collage's
    # 90%-different bar.  Treating those lists categorically instead gives a
    # distance of 8 and buys an entire sheet of almost-identical images.
    dimensions = []
    for slot in range(7):
        pick = search.Dimension.choice("pick%d" % slot, ["A", "B"])
        weight = search.Dimension.choice(
            "weight%d" % slot, ["0", "0.25", "0.5", "0.75", "1"])
        weight.parent = pick
        dimensions.extend((pick, weight))
    dimensions.append(search.Dimension.choice(
        "profile point", ["-1", "-0.5", "0", "0.5", "1"]))
    space = search.Space(dimensions)
    base = tuple([0, 0] * 7 + [0])
    near = tuple([0, 1] * 7 + [1])
    far = tuple([1, 0] * 7 + [4])
    pairwise = space.separations([base, near, far])
    engine = search.PreferenceSearch(space, seed=0)
    picked = engine._distinct_indices(
        pairwise, [3.0, 2.0, 1.0], 3, search.POPULATION_SEPARATION)
    check(picked == [0, 2],
          "collage diversity was paid for by adjacent numeric weight notches")


def test_interesting_transplants_atomic_gene():
    engine = search.PreferenceSearch(conditional_space(), seed=9)
    donor, base = (1, 4), (0, 0)
    children = {engine._hybrid(donor, base) for _ in range(20)}
    check(children == {donor},
          "an interesting LoRA donor was split into a pick without its weight")


def test_internal_basin_archive_exceeds_report_limit():
    space = search.Space([
        search.Dimension.choice("look", ["a", "b", "c", "d",
                                          "e", "f", "g", "h"])
    ])
    engine = search.PreferenceSearch(space, seed=4)
    # Equal positive evidence for eight isolated categorical looks. The STOP
    # report should stay concise, but acquisition must remember all eight.
    for level in range(8):
        row = engine._row((level,))
        engine.observations.append((None, row, search.ON_TRACK_P))
    engine.duels = 8
    public = engine.frontier()
    check(len(public) == search.FRONTIER_SIZE,
          "the public frontier no longer respects its report-size limit")
    check(len(engine._attractors) > search.FRONTIER_SIZE,
          "the four-entry STOP report is still truncating the internal basin memory")


def main():
    test_perceptual_geometry()
    test_contextual_similarity_and_kernel()
    test_absolute_quality_rows()
    test_quality_uses_difference_covariance()
    test_collage_rejects_numeric_near_duplicates()
    test_interesting_transplants_atomic_gene()
    test_internal_basin_archive_exceeds_report_limit()
    if FAILURES:
        print("FAIL: %d A/B domain regression(s)" % len(FAILURES))
        for failure in FAILURES:
            print(" - " + failure)
        raise SystemExit(1)
    print("PASS: A/B quality domains and perceptual geometry")


if __name__ == "__main__":
    main()
