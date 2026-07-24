"""Seeding behaviour of the tabular RGE random-masking path."""

import numpy as np

from safeai.rge import rge_curve


def _random_curve(model, x, feature_names, random_seed):
    result = rge_curve(
        model,
        x,
        method='tabular',
        feature_names=feature_names,
        masking_method='random',
        n_steps=5,
        random_seed=random_seed,
        plot=False,
        verbose=False,
    )
    return np.asarray(result['rge_scores'], dtype=float)


def test_same_seed_identical(fitted_model, tabular_data):
    x, _, feature_names = tabular_data
    a = _random_curve(fitted_model, x, feature_names, random_seed=1)
    b = _random_curve(fitted_model, x, feature_names, random_seed=1)
    assert np.array_equal(a, b)


def test_different_seed_differs(fitted_model, tabular_data):
    x, _, feature_names = tabular_data
    a = _random_curve(fitted_model, x, feature_names, random_seed=1)
    b = _random_curve(fitted_model, x, feature_names, random_seed=2)
    assert not np.array_equal(a, b)


def test_default_is_seed_42(fitted_model, tabular_data):
    # The documented effective default for rge random masking is seed 42.
    x, _, feature_names = tabular_data
    default = _random_curve(fitted_model, x, feature_names, random_seed=None)
    explicit = _random_curve(fitted_model, x, feature_names, random_seed=42)
    assert np.array_equal(default, explicit)
