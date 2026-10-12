"""
Backward-compatibility: patched unseeded RGE must match upstream bit-for-bit.

Upstream sources are materialised from ``git show upstream/main:...`` into a
temporary package and executed in an isolated subprocess, so the pristine and
patched implementations never share module state. The test is skipped when the
upstream ref is unavailable (e.g. a source checkout without the remote).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REF = 'upstream/main'
UPSTREAM_FILES = ('__init__.py', 'cramer.py', 'utils.py', 'rge.py')

WORKER = '''
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from sklearn.linear_model import LogisticRegression
from safeai.rge import rge_curve

gen = np.random.default_rng(0)
x = gen.normal(size=(60, 6))
y = (x[:, 0] + 0.5 * x[:, 1] - x[:, 2] > 0).astype(int) + (x[:, 3] > 0).astype(int)
feature_names = [f'f{i}' for i in range(6)]
model = LogisticRegression(max_iter=500).fit(x, y)

greedy = rge_curve(
    model, x, method='tabular', feature_names=feature_names,
    masking_method='greedy', n_steps=6, random_seed=None,
    plot=False, verbose=False,
)['rge_scores']

rand = rge_curve(
    model, x, method='tabular', feature_names=feature_names,
    masking_method='random', n_steps=6, random_seed=None,
    plot=False, verbose=False,
)['rge_scores']

np.savez(sys.argv[2], greedy=np.asarray(greedy, float), rand=np.asarray(rand, float))
'''


def _git_show(ref_path):
    return subprocess.run(
        ['git', 'show', ref_path],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout


def _upstream_available():
    result = subprocess.run(
        ['git', 'rev-parse', '--verify', UPSTREAM_REF],
        cwd=REPO_ROOT, capture_output=True,
    )
    return result.returncode == 0


def _run_worker(package_root, out_path):
    subprocess.run(
        [sys.executable, '-c', WORKER, str(package_root), str(out_path)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    return np.load(out_path)


def test_unseeded_rge_matches_upstream(tmp_path):
    if not _upstream_available():
        pytest.skip(f'{UPSTREAM_REF} ref not available')

    upstream_root = tmp_path / 'upstream'
    pkg = upstream_root / 'safeai'
    pkg.mkdir(parents=True)
    for fname in UPSTREAM_FILES:
        (pkg / fname).write_bytes(_git_show(f'{UPSTREAM_REF}:safeai/{fname}'))

    up = _run_worker(upstream_root, tmp_path / 'upstream.npz')
    new = _run_worker(REPO_ROOT, tmp_path / 'patched.npz')

    # The last point masks every feature, so all reduced predictions tie.
    # Its value depended on the row order before tied predictions were
    # averaged in cvm1_concordance_weighted, so it is excluded here.
    for key in ('greedy', 'rand'):
        up_key, new_key = up[key][:-1], new[key][:-1]
        assert np.array_equal(up_key, new_key), f'{key} curve diverged from upstream'
        assert up_key.tobytes() == new_key.tobytes(), f'{key} curve not bit-identical'
