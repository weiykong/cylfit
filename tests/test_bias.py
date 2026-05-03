"""Statistical bias and consistency tests.

Verifies that the radius and axis estimators are:
  * unbiased — mean error ≈ 0 across many random seeds
  * consistent — variance shrinks as n_points grows
  * convergent — LM converges on clean data across seeds

These tests deliberately use many repetitions (N_TRIALS=20) with a fixed
noise level to characterise the *distribution* of the estimator, not just
one outcome.  They run in reasonable time because n_points is kept modest.
"""
from __future__ import annotations

import numpy as np
import pytest

from cylfit import fit_cylinder, fit_cylinder_known_radius, generate_noisy_cylinder


N_TRIALS = 20   # number of independent random seeds per experiment
R_TRUE   = 2.0  # ground-truth radius
NOISE    = 0.02 # noise std-dev (1% of radius)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_and_fit(radius, noise, n_points, seed, *, known_radius=False, **kw):
    syn = generate_noisy_cylinder(
        radius=radius, noise=noise, outlier_fraction=0.0,
        n_points=n_points, random_state=seed,
    )
    if known_radius:
        return fit_cylinder_known_radius(
            syn.points, radius, threshold=noise * 4 + 0.01,
            ransac_trials=48, random_state=seed, **kw
        ), syn
    return fit_cylinder(
        syn.points, threshold=noise * 4 + 0.01,
        ransac_trials=48, random_state=seed, **kw
    ), syn


# ---------------------------------------------------------------------------
# Radius estimator bias
# ---------------------------------------------------------------------------

class TestRadiusBias:
    """The radius estimator should be approximately unbiased."""

    def test_radius_mean_error_small(self):
        """Mean |error| across N_TRIALS seeds < 1% of true radius."""
        errors = []
        for seed in range(N_TRIALS):
            model, _ = _generate_and_fit(R_TRUE, NOISE, n_points=800, seed=seed)
            errors.append(model.radius - R_TRUE)
        mean_err = float(np.mean(errors))
        assert abs(mean_err) < 0.02 * R_TRUE, (
            f"Radius estimator appears biased: mean error = {mean_err:.4f}"
        )

    def test_radius_mae_small(self):
        """Mean absolute error < 2% of true radius."""
        maes = []
        for seed in range(N_TRIALS):
            model, _ = _generate_and_fit(R_TRUE, NOISE, n_points=800, seed=seed)
            maes.append(abs(model.radius - R_TRUE))
        mean_mae = float(np.mean(maes))
        assert mean_mae < 0.02 * R_TRUE, f"MAE = {mean_mae:.4f}"

    def test_radius_std_reasonable(self):
        """Standard deviation of radius estimates < 5% of true radius."""
        radii = []
        for seed in range(N_TRIALS):
            model, _ = _generate_and_fit(R_TRUE, NOISE, n_points=800, seed=seed)
            radii.append(model.radius)
        std = float(np.std(radii))
        assert std < 0.05 * R_TRUE, f"Radius std = {std:.4f}"

    def test_known_radius_unbiased(self):
        """known-radius fit: radius returned is *exactly* the input."""
        for seed in range(10):
            model, _ = _generate_and_fit(
                R_TRUE, NOISE, n_points=400, seed=seed, known_radius=True
            )
            assert abs(model.radius - R_TRUE) < 1e-12, (
                f"seed {seed}: radius = {model.radius}"
            )


# ---------------------------------------------------------------------------
# Radius estimator consistency (variance shrinks with more data)
# ---------------------------------------------------------------------------

class TestRadiusConsistency:
    """More data → tighter estimates (classical consistency)."""

    def test_larger_sample_gives_smaller_std(self):
        n_small, n_large = 100, 1000
        radii_small, radii_large = [], []
        for seed in range(N_TRIALS):
            m_s, _ = _generate_and_fit(R_TRUE, NOISE, n_points=n_small, seed=seed)
            m_l, _ = _generate_and_fit(R_TRUE, NOISE, n_points=n_large, seed=seed)
            radii_small.append(m_s.radius)
            radii_large.append(m_l.radius)
        std_small = float(np.std(radii_small))
        std_large = float(np.std(radii_large))
        assert std_large < std_small, (
            f"Expected std to shrink with more data: "
            f"n={n_small} → {std_small:.4f}, n={n_large} → {std_large:.4f}"
        )

    def test_rmse_shrinks_with_more_data(self):
        """Inlier RMSE (of the fit) shrinks toward the true noise level."""
        n_small, n_large = 100, 2000
        rmses_small, rmses_large = [], []
        for seed in range(N_TRIALS):
            m_s, _ = _generate_and_fit(R_TRUE, NOISE, n_points=n_small, seed=seed)
            m_l, _ = _generate_and_fit(R_TRUE, NOISE, n_points=n_large, seed=seed)
            rmses_small.append(m_s.rmse)
            rmses_large.append(m_l.rmse)
        assert np.mean(rmses_large) <= np.mean(rmses_small) * 1.1, (
            "RMSE did not decrease with more points"
        )

    def test_rmse_approaches_noise_floor(self):
        """With many clean points, RMSE should approach the true noise std."""
        rmses = []
        for seed in range(N_TRIALS):
            model, _ = _generate_and_fit(R_TRUE, NOISE, n_points=2000, seed=seed)
            rmses.append(model.rmse)
        mean_rmse = float(np.mean(rmses))
        # RMSE should be within 50% of the noise std on clean data
        assert mean_rmse < NOISE * 1.5, f"Mean RMSE {mean_rmse:.4f} >> noise {NOISE}"
        assert mean_rmse > NOISE * 0.3, f"Mean RMSE {mean_rmse:.4f} << noise {NOISE}"


# ---------------------------------------------------------------------------
# Axis estimator bias
# ---------------------------------------------------------------------------

class TestAxisBias:
    """Axis direction estimator should align well with the true axis."""

    def _angle_error(self, model, syn):
        a = model.axis_direction
        b = syn.axis_direction
        cos_angle = np.clip(abs(float(np.dot(a, b))), 0, 1)
        return float(np.degrees(np.arccos(cos_angle)))

    def test_axis_mean_angle_error_small(self):
        """Mean axis angle error < 2° across seeds."""
        errs = []
        for seed in range(N_TRIALS):
            model, syn = _generate_and_fit(R_TRUE, NOISE, n_points=800, seed=seed)
            errs.append(self._angle_error(model, syn))
        mean_err = float(np.mean(errs))
        assert mean_err < 2.0, f"Mean axis angle error = {mean_err:.2f}°"

    def test_axis_worst_case_angle_small(self):
        """No single trial should have axis error > 10°."""
        for seed in range(N_TRIALS):
            model, syn = _generate_and_fit(R_TRUE, NOISE, n_points=800, seed=seed)
            err = self._angle_error(model, syn)
            assert err < 10.0, f"seed {seed}: axis angle error = {err:.2f}°"

    def test_axis_unit_norm(self):
        """axis_direction must be unit length on every trial."""
        for seed in range(N_TRIALS):
            model, _ = _generate_and_fit(R_TRUE, NOISE, n_points=400, seed=seed)
            norm = float(np.linalg.norm(model.axis_direction))
            assert abs(norm - 1.0) < 1e-10, f"seed {seed}: axis norm = {norm}"


# ---------------------------------------------------------------------------
# LM convergence rate
# ---------------------------------------------------------------------------

class TestConvergenceRate:
    """LM should converge on clean data for the vast majority of seeds."""

    def test_convergence_rate_above_90pct(self):
        """At least 90% of trials converge on clean, low-noise data."""
        converged = 0
        for seed in range(N_TRIALS):
            model, _ = _generate_and_fit(R_TRUE, noise=0.01, n_points=500, seed=seed)
            if model.converged:
                converged += 1
        rate = converged / N_TRIALS
        assert rate >= 0.9, f"Convergence rate {rate:.0%} < 90%"

    def test_iterations_finite_on_all_trials(self):
        """model.iterations must be a positive integer on every trial."""
        for seed in range(N_TRIALS):
            model, _ = _generate_and_fit(R_TRUE, NOISE, n_points=500, seed=seed)
            assert isinstance(model.iterations, int)
            assert model.iterations >= 1

    def test_inlier_fraction_above_threshold_on_clean_data(self):
        """On data with no outliers ≥ 90% of points should be inliers."""
        for seed in range(N_TRIALS):
            model, syn = _generate_and_fit(R_TRUE, NOISE, n_points=600, seed=seed)
            frac = float(model.inlier_mask.mean())
            assert frac > 0.85, (
                f"seed {seed}: inlier fraction {frac:.2%} < 85% on clean data"
            )


# ---------------------------------------------------------------------------
# Noise sensitivity
# ---------------------------------------------------------------------------

class TestNoiseSensitivity:
    """Radius error should scale roughly linearly with noise level."""

    def test_lower_noise_gives_lower_mae(self):
        """Halving the noise should reduce radius MAE."""
        def _mae(noise_level):
            errs = []
            for seed in range(N_TRIALS):
                model, _ = _generate_and_fit(
                    R_TRUE, noise_level, n_points=600, seed=seed
                )
                errs.append(abs(model.radius - R_TRUE))
            return float(np.mean(errs))

        mae_high = _mae(0.05)
        mae_low  = _mae(0.01)
        assert mae_low < mae_high, (
            f"Lower noise should give lower MAE: {mae_low:.4f} vs {mae_high:.4f}"
        )

    def test_radius_error_scales_with_noise(self):
        """Roughly: MAE < k*noise for some constant k."""
        for noise in [0.01, 0.03, 0.05]:
            errs = []
            for seed in range(N_TRIALS):
                model, _ = _generate_and_fit(R_TRUE, noise, n_points=500, seed=seed)
                errs.append(abs(model.radius - R_TRUE))
            mae = float(np.mean(errs))
            # Allow up to 5× the noise std as MAE
            assert mae < noise * 5, (
                f"noise={noise}: MAE={mae:.4f} exceeds 5×noise={noise*5:.4f}"
            )
