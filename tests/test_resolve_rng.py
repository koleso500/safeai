"""Unit tests for the resolve_rng helper."""

import numpy as np

from safeai.utils import resolve_rng


def test_passes_through_existing_generator():
    gen = np.random.default_rng(123)
    assert resolve_rng(gen) is gen
    assert resolve_rng(gen, default_seed=42) is gen


def test_int_seed_is_reproducible():
    a = resolve_rng(7).integers(0, 1_000_000, size=10)
    b = resolve_rng(7).integers(0, 1_000_000, size=10)
    assert np.array_equal(a, b)


def test_default_seed_used_when_none():
    a = resolve_rng(None, default_seed=42).integers(0, 1_000_000, size=10)
    b = np.random.default_rng(42).integers(0, 1_000_000, size=10)
    assert np.array_equal(a, b)


def test_none_and_no_default_is_nondeterministic():
    a = resolve_rng(None).integers(0, 1_000_000, size=10)
    b = resolve_rng(None).integers(0, 1_000_000, size=10)
    assert not np.array_equal(a, b)
