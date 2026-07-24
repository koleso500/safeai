"""
Shared fixtures and test helpers for the safeai seeding tests.

These tests exercise the RNG plumbing only. They use tiny synthetic data and a
trivial sklearn model, require neither ART nor network access, and run in well
under a second.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

# Make the in-repo safeai package importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_art_stub():
    """
    Register a minimal fake ``art`` package so ``safeai.rgr`` can be imported
    when the real adversarial-robustness-toolbox is not installed.

    ``safeai.rgr`` imports several ART symbols at module top. None of them are
    used by the Gaussian-noise path under test, so empty placeholder classes are
    sufficient. If the real ART is present it is used unchanged.
    """
    try:
        import art  # noqa: F401
        return
    except ImportError:
        pass

    art = types.ModuleType('art')
    attacks = types.ModuleType('art.attacks')
    evasion = types.ModuleType('art.attacks.evasion')
    estimators = types.ModuleType('art.estimators')
    classification = types.ModuleType('art.estimators.classification')

    for name in (
        'FastGradientMethod',
        'ProjectedGradientDescent',
        'SquareAttack',
        'HopSkipJump',
        'SimBA',
        'Wasserstein',
        'SpatialTransformation',
    ):
        setattr(evasion, name, type(name, (), {}))

    for name in ('PyTorchClassifier', 'SklearnClassifier'):
        setattr(classification, name, type(name, (), {}))

    attacks.evasion = evasion
    estimators.classification = classification
    art.attacks = attacks
    art.estimators = estimators

    sys.modules['art'] = art
    sys.modules['art.attacks'] = attacks
    sys.modules['art.attacks.evasion'] = evasion
    sys.modules['art.estimators'] = estimators
    sys.modules['art.estimators.classification'] = classification


# Installed at conftest import time (before test modules are collected) so that
# `import safeai.rgr` succeeds without the optional ART dependency.
install_art_stub()


@pytest.fixture
def tabular_data():
    """Small deterministic 3-class tabular dataset."""
    gen = np.random.default_rng(0)
    n, d = 60, 6
    x = gen.normal(size=(n, d))
    y = (x[:, 0] + 0.5 * x[:, 1] - x[:, 2] > 0).astype(int) + (x[:, 3] > 0).astype(int)
    feature_names = [f'f{i}' for i in range(d)]
    return x, y, feature_names


@pytest.fixture
def fitted_model(tabular_data):
    """Trivial fitted sklearn classifier for the tabular dataset."""
    x, y, _ = tabular_data
    model = LogisticRegression(max_iter=500)
    model.fit(x, y)
    return model
