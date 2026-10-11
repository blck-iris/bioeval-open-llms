import math

import numpy as np

from bioeval.metrics import (cohen_kappa, cyclic_permutation, expected_calibration_error,
                             softmax, wilson_interval)


def test_wilson_contains_point_estimate():
    lo, hi = wilson_interval(60, 100)
    assert lo < 0.6 < hi and 0 <= lo and hi <= 1


def test_wilson_edge_cases():
    assert math.isnan(wilson_interval(0, 0)[0])
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and hi > 0


def test_cyclic_permutation_is_permutation_and_covers_all_slots():
    n = 4
    for s in range(n):
        assert sorted(cyclic_permutation(n, s)) == list(range(n))
    # the original option 0 lands in a different slot for each shift
    slots = [cyclic_permutation(n, s).index(0) for s in range(n)]
    assert sorted(slots) == list(range(n))


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.25, 1.0, 20000)
    correct = rng.uniform(size=conf.size) < conf
    ece, _ = expected_calibration_error(conf, correct)
    assert ece < 0.02


def test_ece_overconfident_is_large():
    conf = np.full(1000, 0.95)
    correct = np.zeros(1000)
    ece, _ = expected_calibration_error(conf, correct)
    assert ece > 0.9


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 0, 1, 1], [1, 0, 0, 1]) - 0.5) < 1e-9


def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0, 4.0]))
    assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 3
