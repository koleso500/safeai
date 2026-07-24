"""Defect-3 fix: the pytorch Gaussian-noise path must honour the passed rng.

safeai.rgr imports ART at module top; the real toolkit is optional here, so a
minimal stub is installed before import. None of the ART symbols are used by the
noise path under test.
"""

import numpy as np
import pytest
import torch

# conftest installs a minimal ART stub at collection time so this import works.
rgr = pytest.importorskip('safeai.rgr')
_add_gaussian_noise = rgr._add_gaussian_noise
_rgr_curve_noise_core = rgr._rgr_curve_noise_core


def test_pytorch_noise_same_seed_identical():
    x = torch.zeros((5, 4), dtype=torch.float32)
    a = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(11))
    b = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(11))
    assert torch.equal(a, b)


def test_pytorch_noise_different_seed_differs():
    x = torch.zeros((5, 4), dtype=torch.float32)
    a = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(11))
    b = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(12))
    assert not torch.equal(a, b)


def test_pytorch_noise_unseeded_is_nondeterministic():
    x = torch.zeros((5, 4), dtype=torch.float32)
    a = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(None))
    b = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(None))
    assert not torch.equal(a, b)


def test_pytorch_noise_preserves_dtype_and_device():
    for dtype in (torch.float32, torch.float64):
        x = torch.zeros((3, 2), dtype=dtype)
        out = _add_gaussian_noise(x, 0.3, model_type='pytorch', rng=np.random.default_rng(1))
        assert out.dtype == dtype
        assert out.device == x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_pytorch_noise_cuda_device_and_reproducible():
    # Exercises the generate-on-CPU-then-.to(device) branch on a real GPU. This
    # is skipped in the CPU-only build env and is meant for GPU CI coverage.
    x = torch.zeros((5, 4), dtype=torch.float32, device='cuda')
    a = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(11))
    b = _add_gaussian_noise(x, 0.5, model_type='pytorch', rng=np.random.default_rng(11))
    assert a.device == x.device
    assert a.dtype == x.dtype
    assert torch.equal(a, b)


def _tiny_torch_model(in_dim, n_classes):
    torch.manual_seed(0)
    model = torch.nn.Linear(in_dim, n_classes)
    model.eval()
    return model


def test_noise_curve_reproducible_under_seed():
    x = torch.tensor(np.random.default_rng(0).normal(size=(40, 4)) * 2.0, dtype=torch.float32)
    model = _tiny_torch_model(4, 3)
    class_order = np.array([0, 1, 2])
    prob = torch.softmax(model(x), dim=1).detach().numpy()
    noise_levels = np.array([0.5, 1.0, 2.0, 4.0])

    kwargs = dict(
        model=model, x_data=x, prob_original=prob, noise_levels=noise_levels,
        model_class_order=class_order, class_order=class_order,
        model_type='pytorch', plot=False, verbose=False,
    )

    a = _rgr_curve_noise_core(random_seed=42, **kwargs)['rgr_scores']
    b = _rgr_curve_noise_core(random_seed=42, **kwargs)['rgr_scores']
    assert np.array_equal(a, b)

    c = _rgr_curve_noise_core(random_seed=7, **kwargs)['rgr_scores']
    assert not np.array_equal(a, c)
