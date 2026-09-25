"""RANSAC candidate search: batched scoring, n_jobs invariance, early stopping."""
from __future__ import annotations

import numpy as np
import pytest

from cylfit import fit_cylinder, generate_noisy_cylinder, residuals_to_cylinder
from cylfit import core
from cylfit.core import _batch_candidate_scores, _candidate_score


def _candidates(rng, n):
    out = []
    for _ in range(n):
        axis = rng.normal(size=3)
        out.append((rng.normal(size=3), axis / np.linalg.norm(axis), float(rng.uniform(0.1, 3.0))))
    return out


@pytest.mark.parametrize("offset", [0.0, 1e4])
def test_batch_scores_match_single_candidate_scores(offset):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(3000, 3)) * 2.0 + offset
    centroid = points.mean(axis=0)
    candidates = [(p0 + offset, axis, r) for p0, axis, r in _candidates(rng, 7)]

    batch = _batch_candidate_scores(points - centroid, centroid, candidates, 0.2, None)

    for (p0, axis, r), (soft, hard) in zip(candidates, batch):
        ref_soft, ref_hard = _candidate_score(residuals_to_cylinder(points, p0, axis, r), 0.2)
        assert soft == pytest.approx(ref_soft, rel=1e-9)
        assert hard == ref_hard


def test_batch_scores_independent_of_block_split(monkeypatch):
    rng = np.random.default_rng(1)
    points = rng.normal(size=(5000, 3))
    centroid = points.mean(axis=0)
    candidates = _candidates(rng, 5)
    whole = _batch_candidate_scores(points - centroid, centroid, candidates, 0.3, None)

    monkeypatch.setattr(core, "_SCORE_BLOCK_ELEMS", 5 * 97)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        blocked = _batch_candidate_scores(points - centroid, centroid, candidates, 0.3, pool)

    for (soft_a, hard_a), (soft_b, hard_b) in zip(whole, blocked):
        assert soft_a == pytest.approx(soft_b, rel=1e-12)
        assert hard_a == hard_b


@pytest.mark.parametrize("n_jobs", [2, 4, -1])
def test_n_jobs_does_not_change_the_fit(monkeypatch, n_jobs):
    # Shrink the scoring blocks so a small cloud still spans many blocks and
    # the thread pool is actually exercised.
    monkeypatch.setattr(core, "_SCORE_BLOCK_ELEMS", 16 * 256)
    syn = generate_noisy_cylinder(n_points=6000, noise=0.02, outlier_fraction=0.3, random_state=5)

    serial = fit_cylinder(syn.points, random_state=5, n_jobs=1)
    parallel = fit_cylinder(syn.points, random_state=5, n_jobs=n_jobs)

    np.testing.assert_array_equal(serial.axis_point, parallel.axis_point)
    np.testing.assert_array_equal(serial.axis_direction, parallel.axis_direction)
    assert serial.radius == parallel.radius
    np.testing.assert_array_equal(serial.inlier_mask, parallel.inlier_mask)


def test_early_stop_leaves_rng_after_stopping_trial(monkeypatch):
    """An early stop mid-batch must not consume draws for the unused candidates."""
    syn = generate_noisy_cylinder(n_points=2000, noise=0.0, outlier_fraction=0.0, random_state=2)
    kwargs = dict(
        threshold=0.05, sample_size=64, known_radius=None, initial_axis=None,
        orientation_axis=None, min_axis_cos=None,
    )

    rng_stop = np.random.default_rng(9)
    core._ransac_or_pca_initial(syn.points, trials=64, stop_inlier_fraction=0.5, rng=rng_stop, **kwargs)

    # Replay: find the trial that triggered the stop, one candidate per batch.
    rng_ref = np.random.default_rng(9)
    monkeypatch.setattr(core, "_SCORE_BATCH", 1)
    core._ransac_or_pca_initial(syn.points, trials=64, stop_inlier_fraction=0.5, rng=rng_ref, **kwargs)

    assert rng_stop.bit_generator.state == rng_ref.bit_generator.state
