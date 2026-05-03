"""Edge case and adversarial input tests.

Verifies that the fitting functions handle degenerate, unusual, and
malformed inputs gracefully — either by raising clear errors or by
returning a valid (if inaccurate) model without crashing.
"""
from __future__ import annotations

import numpy as np
import pytest

from cylinderfit2026 import (
    fit_cylinder,
    fit_cylinder_known_radius,
    fit_cylinder_with_normals,
    generate_noisy_cylinder,
    residuals_to_cylinder,
)
from cylinderfit2026.core import _auto_threshold, _normalize
from cylinderfit2026.elliptical import _fit_ellipse_algebraic, fit_elliptical_cylinder
from cylinderfit2026.cone import fit_cone


# ---------------------------------------------------------------------------
# Input validation — fit_cylinder
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_too_few_points_raises(self):
        pts = np.random.default_rng(0).standard_normal((5, 3))
        with pytest.raises(ValueError, match="8"):
            fit_cylinder(pts)

    def test_wrong_shape_raises(self):
        pts = np.ones((20, 2))           # N×2 instead of N×3
        with pytest.raises(ValueError):
            fit_cylinder(pts)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            fit_cylinder(np.ones(30))

    def test_nan_in_points_silently_filtered(self):
        """NaN rows are silently dropped — fit_cylinder treats them as missing data."""
        pts = generate_noisy_cylinder(random_state=0).points.copy()
        pts[5, 1] = float("nan")
        # Must not crash; NaN row is excluded from the fit
        model = fit_cylinder(pts, threshold=0.06, ransac_trials=32, random_state=0)
        assert model.radius > 0

    def test_inf_in_points_silently_filtered(self):
        """Inf rows are silently dropped — treated as missing data."""
        pts = generate_noisy_cylinder(random_state=0).points.copy()
        pts[3, 2] = float("inf")
        model = fit_cylinder(pts, threshold=0.06, ransac_trials=32, random_state=0)
        assert model.radius > 0

    def test_mostly_nan_raises(self):
        """When NaN removal leaves < 8 points, must raise."""
        pts = generate_noisy_cylinder(random_state=0).points.copy()
        pts[5:, :] = float("nan")   # keep only 5 rows
        with pytest.raises(ValueError):
            fit_cylinder(pts)

    def test_known_radius_negative_raises(self):
        pts = generate_noisy_cylinder(random_state=0).points
        with pytest.raises(ValueError):
            fit_cylinder_known_radius(pts, radius=-1.0)

    def test_known_radius_zero_raises(self):
        pts = generate_noisy_cylinder(random_state=0).points
        with pytest.raises(ValueError):
            fit_cylinder_known_radius(pts, radius=0.0)

    def test_normals_shape_mismatch_raises(self):
        syn = generate_noisy_cylinder(random_state=0)
        pts = syn.points
        normals = syn.normals[:-2]       # wrong length
        with pytest.raises(ValueError):
            fit_cylinder_with_normals(pts, normals)

    def test_residuals_to_cylinder_nan_raises(self):
        pts = np.array([[1., 0., 0.], [0., 1., 0.], [np.nan, 0., 0.]])
        with pytest.raises(ValueError, match="finite"):
            residuals_to_cylinder(pts, np.zeros(3), np.array([0., 0., 1.]), 1.0)

    def test_residuals_to_cylinder_single_point(self):
        """residuals_to_cylinder must work for n=1 (just a distance query)."""
        pt = np.array([[2., 0., 0.]])
        res = residuals_to_cylinder(pt, np.zeros(3), np.array([0., 0., 1.]), 1.0)
        np.testing.assert_allclose(res, [1.0], atol=1e-12)

    def test_residuals_to_cylinder_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            residuals_to_cylinder(np.ones((10, 2)), np.zeros(3), np.array([0., 0., 1.]), 1.0)


# ---------------------------------------------------------------------------
# Minimum-viable inputs
# ---------------------------------------------------------------------------

class TestMinimumPoints:
    def test_exactly_8_points_does_not_crash(self):
        """fit_cylinder with exactly 8 points must not crash."""
        rng = np.random.default_rng(1)
        # 8 points approximately on a cylinder
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(8)])
        model = fit_cylinder(pts, ransac_trials=4, random_state=1)
        assert model.radius > 0
        assert np.isfinite(model.rmse)

    def test_exactly_8_points_known_radius(self):
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(8)])
        model = fit_cylinder_known_radius(pts, 1.0, ransac_trials=4, random_state=1)
        assert abs(model.radius - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Degenerate geometry
# ---------------------------------------------------------------------------

class TestDegenerateGeometry:
    def test_flat_cylinder_no_crash(self):
        """Points all at z=0 (single ring) — fits a cylinder axis along Z."""
        rng = np.random.default_rng(2)
        theta = rng.uniform(0, 2 * np.pi, 200)
        pts = np.column_stack([
            1.5 * np.cos(theta) + rng.normal(0, 0.01, 200),
            1.5 * np.sin(theta) + rng.normal(0, 0.01, 200),
            np.zeros(200),
        ])
        model = fit_cylinder(pts, threshold=0.05, ransac_trials=32, random_state=2)
        assert model.radius > 0
        assert np.isfinite(model.rmse)

    def test_partial_arc_90deg_no_crash(self):
        """Points only on a 90° arc — algorithm should not crash."""
        rng = np.random.default_rng(3)
        theta = rng.uniform(0, np.pi / 2, 200)
        pts = np.column_stack([
            2.0 * np.cos(theta) + rng.normal(0, 0.02, 200),
            2.0 * np.sin(theta) + rng.normal(0, 0.02, 200),
            rng.uniform(0, 5, 200),
        ])
        model = fit_cylinder(pts, threshold=0.1, ransac_trials=32, random_state=3)
        assert model.radius > 0

    def test_partial_arc_180deg_reasonable_radius(self):
        """Half-cylinder: radius estimate should be within 20% of truth."""
        rng = np.random.default_rng(4)
        theta = rng.uniform(0, np.pi, 500)
        r_true = 2.0
        pts = np.column_stack([
            r_true * np.cos(theta) + rng.normal(0, 0.01, 500),
            r_true * np.sin(theta) + rng.normal(0, 0.01, 500),
            rng.uniform(0, 5, 500),
        ])
        model = fit_cylinder(pts, threshold=0.05, ransac_trials=64, random_state=4)
        assert abs(model.radius - r_true) < 0.4  # within 20%

    def test_very_short_cylinder_no_crash(self):
        """Cylinder with height << radius — should not crash."""
        syn = generate_noisy_cylinder(
            radius=2.0, height=0.05, noise=0.005, n_points=200, random_state=5
        )
        model = fit_cylinder(syn.points, threshold=0.05, ransac_trials=32, random_state=5)
        assert model.radius > 0

    def test_very_large_radius_no_crash(self):
        """Very large radius (nearly flat surface) — should not crash."""
        rng = np.random.default_rng(6)
        theta = rng.uniform(0, np.pi / 8, 500)   # tight arc of huge cylinder
        r = 1000.0
        pts = np.column_stack([
            r * np.cos(theta) + rng.normal(0, 0.1, 500),
            r * np.sin(theta) + rng.normal(0, 0.1, 500),
            rng.uniform(0, 5, 500),
        ])
        model = fit_cylinder(pts, threshold=1.0, ransac_trials=32, random_state=6)
        assert model.radius > 0
        assert np.isfinite(model.rmse)

    def test_nearly_collinear_points_no_crash(self):
        """Points nearly all on a line — axis estimation is ambiguous but must not crash."""
        rng = np.random.default_rng(7)
        pts = np.column_stack([
            rng.normal(0, 0.01, 100),
            rng.normal(0, 0.01, 100),
            np.linspace(0, 10, 100),
        ])
        # This is a degenerate case — just check no exception
        try:
            model = fit_cylinder(pts, threshold=0.1, ransac_trials=16, random_state=7)
            assert model.radius > 0
        except Exception as exc:
            pytest.fail(f"fit_cylinder raised unexpectedly: {exc}")

    def test_duplicate_points_no_crash(self):
        """All points identical — pathological but must not crash."""
        pts = np.tile([1.0, 0.0, 0.0], (50, 1))
        try:
            fit_cylinder(pts, ransac_trials=4, random_state=8)
        except (ValueError, np.linalg.LinAlgError):
            pass  # acceptable to raise on degenerate input
        except Exception as exc:
            pytest.fail(f"Unexpected exception type: {exc}")


# ---------------------------------------------------------------------------
# Output invariants under adversarial noise
# ---------------------------------------------------------------------------

class TestAdversarialNoise:
    def test_heavy_outlier_fraction_model_valid(self):
        """50% outliers — model must still be structurally valid."""
        syn = generate_noisy_cylinder(
            radius=1.5, noise=0.01, outlier_fraction=0.50,
            n_points=1000, random_state=10,
        )
        model = fit_cylinder(
            syn.points, threshold=0.06, ransac_trials=128, random_state=10
        )
        assert model.radius > 0
        assert np.isclose(np.linalg.norm(model.axis_direction), 1.0, atol=1e-10)
        assert np.isfinite(model.rmse)

    def test_extreme_outlier_fraction_model_valid(self):
        """When nearly all points are outliers (99%) the model must not crash."""
        syn = generate_noisy_cylinder(
            radius=1.0, noise=0.0, outlier_fraction=0.99,
            n_points=500, random_state=11,
        )
        model = fit_cylinder(
            syn.points, threshold=0.05, ransac_trials=32, random_state=11
        )
        assert model.radius > 0

    def test_uniform_noise_cloud_model_valid(self):
        """Pure uniform random cloud — no cylinder structure at all."""
        rng = np.random.default_rng(12)
        pts = rng.uniform(-5, 5, (300, 3))
        model = fit_cylinder(pts, threshold=0.5, ransac_trials=32, random_state=12)
        assert model.radius > 0
        assert np.isfinite(model.rmse)

    def test_inlier_mask_always_boolean(self):
        syn = generate_noisy_cylinder(noise=0.02, outlier_fraction=0.3, random_state=13)
        model = fit_cylinder(syn.points, threshold=0.08, ransac_trials=32, random_state=13)
        assert model.inlier_mask.dtype == bool

    def test_inlier_mask_consistent_with_residuals(self):
        """Core contract: inlier ↔ |residual| ≤ threshold."""
        thr = 0.08
        syn = generate_noisy_cylinder(noise=0.02, outlier_fraction=0.2, random_state=14)
        model = fit_cylinder(syn.points, threshold=thr, ransac_trials=32, random_state=14)
        expected = np.abs(model.residuals) <= thr
        np.testing.assert_array_equal(model.inlier_mask, expected)


# ---------------------------------------------------------------------------
# _normalize edge cases
# ---------------------------------------------------------------------------

class TestNormalizeEdgeCases:
    def test_already_unit_vector(self):
        v = np.array([1., 0., 0.])
        u = _normalize(v)
        np.testing.assert_allclose(u, v, atol=1e-15)

    def test_large_magnitude_vector(self):
        v = np.array([1e8, 0., 0.])
        u = _normalize(v)
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-14)

    def test_small_magnitude_vector(self):
        v = np.array([1e-8, 0., 0.])
        u = _normalize(v)
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _auto_threshold edge cases
# ---------------------------------------------------------------------------

class TestAutoThresholdEdgeCases:
    def test_constant_residuals(self):
        r = np.ones(100) * 0.05
        t = _auto_threshold(r)
        assert t > 0 and np.isfinite(t)

    def test_all_zero_residuals(self):
        r = np.zeros(50)
        t = _auto_threshold(r)
        assert t > 0 and np.isfinite(t)

    def test_single_residual(self):
        t = _auto_threshold(np.array([0.1]))
        assert t > 0 and np.isfinite(t)

    def test_mixed_sign_residuals(self):
        rng = np.random.default_rng(99)
        r = rng.normal(0, 0.05, 200)
        t = _auto_threshold(r)
        assert t > 0 and np.isfinite(t)


# ---------------------------------------------------------------------------
# _fit_ellipse_algebraic edge cases
# ---------------------------------------------------------------------------

class TestEllipseAlgebraicEdgeCases:
    def test_too_few_points_returns_none(self):
        u = np.array([1., 2., 3., 4., 0.])
        v = np.array([0., 1., 0., -1., 0.])
        assert _fit_ellipse_algebraic(u, v) is None

    def test_circle_does_not_crash(self):
        """A perfect circle is degenerate for ellipse fitting (all rotations
        are equivalent); the fitter may return None or a near-circular result."""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        u = 2.0 * np.cos(theta)
        v = 2.0 * np.sin(theta)
        result = _fit_ellipse_algebraic(u, v)
        # Either None (degenerate) or a valid ellipse with semi-axes ≈ radius
        if result is not None:
            _, _, sa, sb, _ = result
            fitted = sorted([sa, sb], reverse=True)
            assert abs(fitted[0] - 2.0) < 0.1
            assert abs(fitted[1] - 2.0) < 0.1

    def test_collinear_points_returns_none_or_valid(self):
        """Collinear points cannot form an ellipse."""
        u = np.linspace(0, 5, 20)
        v = np.zeros(20)
        result = _fit_ellipse_algebraic(u, v)
        # Either None (degenerate) or some result — just must not crash
        assert result is None or isinstance(result, tuple)

    def test_high_aspect_ratio_ellipse(self):
        """Very elongated ellipse (a/b = 10) should still fit."""
        a, b = 5.0, 0.5
        theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        u = a * np.cos(theta)
        v = b * np.sin(theta)
        result = _fit_ellipse_algebraic(u, v)
        assert result is not None
        _, _, sa, sb, _ = result
        fitted = sorted([sa, sb], reverse=True)
        assert abs(fitted[0] - a) < 0.1
        assert abs(fitted[1] - b) < 0.05


# ---------------------------------------------------------------------------
# fit_cone edge cases
# ---------------------------------------------------------------------------

class TestConeEdgeCases:
    def test_cone_min_points_no_crash(self):
        """Exactly 8 points on a perfect cone."""
        apex = np.zeros(3)
        axis = np.array([0., 0., 1.])
        alpha = np.deg2rad(30.0)
        t_vals = np.linspace(0.5, 2.0, 8)
        r_vals = t_vals * np.tan(alpha)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pts = np.column_stack([
            r_vals * np.cos(angles),
            r_vals * np.sin(angles),
            t_vals,
        ])
        model = fit_cone(pts, ransac_trials=4, random_state=0)
        assert model.half_angle_deg > 0
        assert np.isfinite(model.rmse)

    def test_cone_nan_silently_filtered(self):
        """NaN rows are silently dropped before cone fitting."""
        pts = generate_noisy_cylinder(random_state=0).points.copy()
        pts[2, 0] = np.nan
        # Must not crash
        model = fit_cone(pts, ransac_trials=16, random_state=0)
        assert model.half_angle_deg > 0


# ---------------------------------------------------------------------------
# fit_elliptical_cylinder edge cases
# ---------------------------------------------------------------------------

class TestEllipticalCylinderEdgeCases:
    def test_circular_input_gives_near_equal_axes(self):
        """A circular cylinder should give semi_major ≈ semi_minor."""
        syn = generate_noisy_cylinder(radius=1.5, noise=0.005, random_state=20)
        model = fit_elliptical_cylinder(
            syn.points, threshold=0.05, ransac_trials=32, random_state=20
        )
        ratio = model.semi_major / max(model.semi_minor, 1e-9)
        assert ratio < 1.3, f"Aspect ratio {ratio:.3f} too large for circular input"

    def test_elliptical_model_structural_validity(self):
        syn = generate_noisy_cylinder(radius=1.0, noise=0.01, random_state=21)
        model = fit_elliptical_cylinder(syn.points, ransac_trials=16, random_state=21)
        assert model.semi_major >= model.semi_minor > 0
        assert np.isclose(np.linalg.norm(model.axis_direction), 1.0, atol=1e-10)
        assert model.height_max >= model.height_min
        assert np.isfinite(model.rmse)
