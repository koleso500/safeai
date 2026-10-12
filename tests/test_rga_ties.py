"""Tied predictions: RGA must not depend on the input row order."""

import numpy as np

from safeai.cramer import concordance_curve, cvm1_concordance_weighted
from safeai.rga import rga_curve, rga_score


def _stable_sort_cvm(y, yhat):
    """Previous implementation: ties in yhat broken by row order."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ord_y = np.argsort(y, kind='stable')
    l = np.cumsum(y[ord_y]) / np.sum(y)
    ord_yhat = np.argsort(yhat, kind='stable')
    c = np.cumsum(y[ord_yhat]) / np.sum(y)
    weights = y[ord_y] / np.sum(y)
    return np.sum(np.abs(c - l) * weights)


def _permutations(n, count=50, seed=0):
    gen = np.random.default_rng(seed)
    return [gen.permutation(n) for _ in range(count)]


def test_constant_score_same_value_in_every_row_order():
    y = np.array([1] * 143 + [0] * 85)
    const = np.full(len(y), 0.5)
    orders = [
        np.argsort(y, kind='stable'),   # zeros first: was 1.0
        np.argsort(-y, kind='stable'),  # ones first: was -0.115
        *_permutations(len(y)),
    ]
    values = {rga_score(y[p], const) for p in orders}
    assert len(values) == 1
    assert abs(values.pop() - 0.5) < 0.01


def test_coarse_scores_order_invariant():
    gen = np.random.default_rng(1)
    y = gen.integers(0, 2, size=300)
    score = np.round(np.clip(0.3 * y + gen.uniform(0, 0.7, size=300), 0, 1), 1)
    perms = _permutations(len(y))
    old = {_stable_sort_cvm(y[p], score[p]) for p in perms}
    assert len(old) > 1
    values = {rga_score(y[p], score[p]) for p in perms}
    assert len(values) == 1


def test_continuous_target_order_invariant():
    # RGE and RGR call cvm1_concordance_weighted with continuous y. The total
    # np.sum(y) still depends on the row order in the last bit.
    gen = np.random.default_rng(2)
    y = gen.uniform(size=200)
    yhat = np.round(y + gen.normal(scale=0.2, size=200), 1)
    perms = _permutations(len(y))
    old = [_stable_sort_cvm(y[p], yhat[p]) for p in perms]
    assert np.ptp(old) > 1e-4
    values = [cvm1_concordance_weighted(y[p], yhat[p]) for p in perms]
    assert np.ptp(values) < 1e-12


def test_multiclass_order_invariant():
    # A shallow tree: few distinct probability rows, each shared by both
    # correctly and wrongly classified cases.
    leaves = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
        [0.4, 0.4, 0.2],
    ])
    gen = np.random.default_rng(3)
    y = gen.integers(0, 3, size=240)
    prob = leaves[gen.integers(0, len(leaves), size=240)]
    values = {
        rga_score(y[p], prob[p], class_order=[0, 1, 2])
        for p in _permutations(len(y))
    }
    assert len(values) == 1


def test_binary_removal_curve_starts_at_rga_score():
    # The removal curve regroups rows by class, which used to put negatives
    # first inside every tie block, so its first point differed from RGA.
    gen = np.random.default_rng(4)
    y = gen.integers(0, 2, size=200)
    score = np.round(np.clip(0.3 * y + gen.uniform(0, 0.7, size=200), 0, 1), 1)
    ref = rga_curve(y, score, curve_method='removal')
    for p in _permutations(len(y), count=10):
        res = rga_curve(y[p], score[p], curve_method='removal')
        assert res['curve'][0] == res['rga'] == ref['rga']
        assert np.array_equal(res['curve'], ref['curve'])


def test_without_ties_unchanged():
    gen = np.random.default_rng(5)
    y_binary = gen.integers(0, 2, size=500).astype(float)
    y_continuous = gen.uniform(size=500)
    yhat = gen.normal(size=500)
    assert len(np.unique(yhat)) == len(yhat)
    for y in (y_binary, y_continuous):
        assert cvm1_concordance_weighted(y, yhat) == _stable_sort_cvm(y, yhat)


def test_concordance_curve_matches_published_example():
    # Agosto, Giudici and Raffinetti (2020), Example 2: the y values of tied
    # predictions are replaced by their mean before ordering.
    y = [15, 10, 26, 21, 32, 45]
    yhat = [21.97, 24.72, 21.97, 27.81, 27.81, 24.72]
    expected = np.array([20.5, 41, 68.5, 96, 122.5, 149]) / 149
    assert np.allclose(concordance_curve(y, yhat), expected)
