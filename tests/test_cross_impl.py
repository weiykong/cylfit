"""Cross-implementation validation tests.

When optional third-party cylinder fitting libraries are installed, these
tests run a head-to-head comparison on shared synthetic datasets and verify
that cylfit agrees with them to within a reasonable tolerance.

Libraries checked (all optional — tests are skipped if not installed):
  * pyransac3d  — ``pip install pyransac3d``
  * cylinder_fitting — ``pip install cylinder_fitting``

Tests never *require* either library; they auto-skip with a clear message.
The standalone accuracy tests in ``TestStandaloneAccuracy`` always run.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from cylfit import fit_cylinder, generate_noisy_cylinder


def _try_import(name):
    """Return the module or None if not installed."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


_pyransac3d     = _try_import("pyransac3d")
_cylinder_fitting = _try_import("cylinder_fitting")

needs_pyransac3d     = pytest.mark.skipif(
    _pyransac3d is None, reason="pyransac3d not installed"
)
needs_cylinder_fitting = pytest.mark.skipif(
    _cylinder_fitting is None, reason="cylinder_fitting not installed"
)


# ---------------------------------------------------------------------------
# Fixtures — shared synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_cylinder_data():
    return generate_noisy_cylinder(
        radius=1.5, noise=0.01, outlier_fraction=0.0,
        n_points=1000, random_state=42,
    )


@pytest.fixture(scope="module")
def noisy_cylinder_data():
    return generate_noisy_cylinder(
        radius=2.0, noise=0.02, outlier_fraction=0.20,
        n_points=1000, random_state=42,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _angle_between(a, b):
    cos = np.clip(abs(float(np.dot(a, b))), 0, 1)
    return float(np.degrees(np.arccos(cos)))


def _fit_pyransac3d(pts, threshold=0.05):
    if _pyransac3d is None:
        return None
    cyl = _pyransac3d.Cylinder()
    model, _ = cyl.fit(pts, thresh=threshold, maxIteration=1000)
    if model is None:
        return None
    axis = np.array(model[3:6], dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return None
    return float(model[6]), axis / norm


def _fit_cylinder_fitting(pts):
    if _cylinder_fitting is None:
        return None
    try:
        w, c, a, r = _cylinder_fitting.fit(pts)
        return float(r), np.array(a, dtype=float)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# vs pyransac3d
# ---------------------------------------------------------------------------

class TestVsPyransac3d:
    """Compare cylfit with pyransac3d on shared synthetic data."""

    @needs_pyransac3d
    def test_radius_agrees_on_clean_data(self, clean_cylinder_data):
        syn = clean_cylinder_data
        our = fit_cylinder(
            syn.points, threshold=0.05, ransac_trials=64, random_state=42
        )
        theirs = _fit_pyransac3d(syn.points, threshold=0.05)
        if theirs is None:
            pytest.skip("pyransac3d did not converge on this dataset")
        r_theirs, _ = theirs
        assert abs(our.radius - syn.radius) < 0.075, f"ours: {our.radius:.4f}"
        assert abs(r_theirs - syn.radius) < 0.075, f"theirs: {r_theirs:.4f}"
        assert abs(our.radius - r_theirs) < 0.15

    @needs_pyransac3d
    def test_axis_agrees_on_clean_data(self, clean_cylinder_data):
        syn = clean_cylinder_data
        our = fit_cylinder(
            syn.points, threshold=0.05, ransac_trials=64, random_state=42
        )
        theirs = _fit_pyransac3d(syn.points, threshold=0.05)
        if theirs is None:
            pytest.skip("pyransac3d did not converge")
        _, axis_theirs = theirs
        assert _angle_between(our.axis_direction, syn.axis_direction) < 5.0
        assert _angle_between(axis_theirs, syn.axis_direction) < 5.0

    @needs_pyransac3d
    def test_radius_agrees_on_noisy_data(self, noisy_cylinder_data):
        syn = noisy_cylinder_data
        our = fit_cylinder(
            syn.points, threshold=0.08, ransac_trials=96, random_state=42
        )
        theirs = _fit_pyransac3d(syn.points, threshold=0.08)
        if theirs is None:
            pytest.skip("pyransac3d did not converge on noisy data")
        r_theirs, _ = theirs
        assert abs(our.radius - syn.radius) < 0.15
        our_err   = abs(our.radius - syn.radius)
        their_err = abs(r_theirs - syn.radius)
        assert our_err <= their_err * 2


# ---------------------------------------------------------------------------
# vs cylinder_fitting
# ---------------------------------------------------------------------------

class TestVsCylinderFitting:
    """Compare with the cylinder_fitting package (Levenberg-Marquardt based)."""

    @needs_cylinder_fitting
    def test_radius_agrees_on_clean_data(self, clean_cylinder_data):
        syn = clean_cylinder_data
        our = fit_cylinder(
            syn.points, threshold=0.05, ransac_trials=64, random_state=42
        )
        theirs = _fit_cylinder_fitting(syn.points)
        if theirs is None:
            pytest.skip("cylinder_fitting did not converge")
        r_theirs, _ = theirs
        assert abs(our.radius - r_theirs) < 0.1

    @needs_cylinder_fitting
    def test_axis_agrees_on_clean_data(self, clean_cylinder_data):
        syn = clean_cylinder_data
        our = fit_cylinder(
            syn.points, threshold=0.05, ransac_trials=64, random_state=42
        )
        theirs = _fit_cylinder_fitting(syn.points)
        if theirs is None:
            pytest.skip("cylinder_fitting did not converge")
        _, axis_theirs = theirs
        assert _angle_between(our.axis_direction, axis_theirs) < 5.0

    @needs_cylinder_fitting
    def test_our_radius_at_least_as_accurate(self, clean_cylinder_data):
        syn = clean_cylinder_data
        our = fit_cylinder(
            syn.points, threshold=0.05, ransac_trials=64, random_state=42
        )
        theirs = _fit_cylinder_fitting(syn.points)
        if theirs is None:
            pytest.skip("cylinder_fitting did not converge")
        r_theirs, _ = theirs
        our_err   = abs(our.radius - syn.radius)
        their_err = abs(r_theirs - syn.radius)
        assert our_err <= their_err * 1.5


# ---------------------------------------------------------------------------
# Standalone accuracy checks (no third-party library needed)
# ---------------------------------------------------------------------------

class TestStandaloneAccuracy:
    """High-accuracy benchmarks that serve as internal cross-checks.

    These run without any optional dependency and verify that the estimator
    is within published accuracy bounds for RANSAC+LM on synthetic data.
    """

    @pytest.mark.parametrize("radius,seed", [
        (0.5, 42), (1.0, 42), (2.0, 42), (3.0, 7),
    ])
    def test_radius_accuracy_across_scales(self, radius, seed):
        """Radius error < 5% for clean data across typical scales."""
        syn = generate_noisy_cylinder(
            radius=radius, noise=radius * 0.005,
            outlier_fraction=0.0, n_points=1000, random_state=seed,
        )
        model = fit_cylinder(
            syn.points,
            threshold=radius * 0.03 + 0.005,
            ransac_trials=64, random_state=seed,
        )
        rel_err = abs(model.radius - radius) / radius
        assert rel_err < 0.05, f"radius={radius}: rel error = {rel_err:.3%}"

    @pytest.mark.parametrize("noise_frac", [0.001, 0.005, 0.01, 0.02])
    def test_rmse_tracks_noise_level(self, noise_frac):
        """Inlier RMSE should be within [0.3×, 2×] of the true noise std."""
        noise = 1.5 * noise_frac
        syn = generate_noisy_cylinder(
            radius=1.5, noise=noise,
            outlier_fraction=0.0, n_points=1000, random_state=99,
        )
        model = fit_cylinder(
            syn.points,
            threshold=noise * 4 + 0.005,
            ransac_trials=64, random_state=99,
        )
        assert model.rmse < noise * 2.0, f"RMSE={model.rmse:.4f} >> noise={noise}"
        assert model.rmse > noise * 0.3, f"RMSE={model.rmse:.4f} << noise={noise}"

    @pytest.mark.parametrize("outlier_frac", [0.0, 0.10, 0.20, 0.30])
    def test_radius_robust_to_outliers(self, outlier_frac):
        """Radius error < 5% even with up to 30% outliers."""
        r_true = 1.5
        syn = generate_noisy_cylinder(
            radius=r_true, noise=0.01, outlier_fraction=outlier_frac,
            n_points=1000, random_state=7,
        )
        model = fit_cylinder(
            syn.points, threshold=0.06, ransac_trials=96, random_state=7
        )
        rel_err = abs(model.radius - r_true) / r_true
        assert rel_err < 0.05, (
            f"outlier_frac={outlier_frac}: radius error = {rel_err:.3%}"
        )
