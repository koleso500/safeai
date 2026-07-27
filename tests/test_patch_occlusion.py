"""apply_patch_occlusion must be reproducible when seeded and must never
mutate the caller's global random / torch RNG streams."""

import random

import torch

from safeai.utils import apply_patch_occlusion

PATCH = 16
NUM = 4


def _images():
    # Non-constant content so that occlusion positions actually affect output
    # (masking writes 0.0 onto a field of ones).
    return torch.ones((2, 3, 64, 64), dtype=torch.float32)


def test_same_seed_identical():
    a = apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=5)
    b = apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=5)
    assert torch.equal(a, b)


def test_different_seed_differs():
    a = apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=5)
    b = apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=6)
    assert not torch.equal(a, b)


def test_does_not_perturb_global_random_stream():
    random.seed(999)
    baseline = [random.random() for _ in range(5)]

    random.seed(999)
    apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=5)
    after = [random.random() for _ in range(5)]

    assert baseline == after


def test_does_not_perturb_global_torch_stream():
    torch.manual_seed(999)
    baseline = torch.randn(5)

    torch.manual_seed(999)
    apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=5)
    after = torch.randn(5)

    assert torch.equal(baseline, after)


def test_unseeded_consumes_global_random_stream():
    # With random_seed=None the function must still draw from the global random
    # module, so the pre/post global state differs from an untouched baseline.
    random.seed(321)
    untouched = [random.random() for _ in range(5)]

    random.seed(321)
    apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=None)
    consumed = [random.random() for _ in range(5)]

    assert untouched != consumed


def test_seeded_output_matches_legacy_global_seeding():
    # The seeded result must equal the pre-fix behaviour, which called
    # random.seed(seed) on the global module. random.Random(seed) yields the
    # identical Mersenne-Twister stream, so outputs must match bit-for-bit.
    got = apply_patch_occlusion(_images(), num_patches=NUM, patch_size=PATCH, random_seed=7)

    random.seed(7)
    expected = _images()
    _, _, h, w = expected.shape
    for i in range(expected.shape[0]):
        for _ in range(NUM):
            y0 = random.randint(0, h - PATCH)
            x0 = random.randint(0, w - PATCH)
            expected[i, :, y0:y0 + PATCH, x0:x0 + PATCH] = 0.0

    assert torch.equal(got, expected)
