"""Numerical regression tests — golden-value pins.

These tests lock specific numerical outputs to values computed once from a
known-good version of the algorithm.  If a refactor changes any of them it
will be caught here, prompting a deliberate decision to update the pin.

Values were computed with:
  * analytic Jacobian LM
  * PROSAC + MAGSAC RANSAC scoring
  * numpy default_rng seeding

To regenerate a pin after an intentional algorithmic change, run the
computation in a REPL and update the expected value + the associated comment.
"""
import numpy as np
import pytest

from cylfit import (
    fit_cylinder,
    fit_cylinder_known_radius,
    fit_cylinder_with_normals,
    generate_noisy_cylinder,
    residuals_to_cylinder,
)
from cylfit.cone import _cone_residuals, fit_cone
from cylfit.elliptical import _fit_ellipse_algebraic


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _angle_between(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0, 1))))


# ---------------------------------------------------------------------------
# Golden 1 — clean cylinder (2 000 pts, noise 0.015, radius 2.5)
# ---------------------------------------------------------------------------

class TestGoldenCleanCylinder:
    """Pin the radius, RMSE, and inlier count for a fixed-seed clean fit."""

    @pytest.fixture(scope="class")
    def model(self):
        syn = generate_noisy_cylinder(
            radius=2.5, noise=0.015, random_state=42, n_points=2000
        )
        return fit_cylinder(
            syn.points, threshold=0.06, ransac_trials=64, random_state=42
        )

    def test_radius_pinned(self, model):
        assert abs(model.radius - 2.50106955) < 1e-4

    def test_rmse_pinned(self, model):
        assert abs(model.rmse - 0.01612195) < 5e-4

    def test_inlier_count_pinned(self, model):
        # Allow ±20 due to threshold boundary sensitivity
        assert abs(model.inlier_mask.sum() - 1800) <= 20

    def test_axis_angle_small(self, model):
        true_axis = generate_noisy_cylinder(
            radius=2.5, noise=0.015, random_state=42, n_points=2000
        ).axis_direction
        assert _angle_between(model.axis_direction, true_axis) < 0.5

    def test_converged(self, model):
        assert model.converged

    def test_inlier_mask_consistent_with_residuals(self, model):
        """Core invariant: inlier_mask must match |residuals| ≤ threshold."""
        # Reconstruct threshold from the boundary residual
        r = np.abs(model.residuals)
        inlier_vals = r[model.inlier_mask]
        outlier_vals = r[~model.inlier_mask]
        if inlier_vals.size and outlier_vals.size:
            assert inlier_vals.max() < outlier_vals.min() + 1e-9


# ---------------------------------------------------------------------------
# Golden 2 — 25 % outliers (radius 1.8)
# ---------------------------------------------------------------------------

class TestGoldenOutlierCylinder:
    @pytest.fixture(scope="class")
    def model(self):
        syn = generate_noisy_cylinder(
            radius=1.8, noise=0.02, outlier_fraction=0.25,
            random_state=7, n_points=2000,
        )
        return fit_cylinder(
            syn.points, threshold=0.08, ransac_trials=96, random_state=7
        )

    def test_radius_within_3pct(self, model):
        assert abs(model.radius - 1.8) < 0.054

    def test_rmse_reasonable(self, model):
        assert model.rmse < 0.04

    def test_inlier_recall(self, model):
        # At least 65 % of 2000 points should be inliers
        assert model.inlier_mask.mean() > 0.65


# ---------------------------------------------------------------------------
# Golden 3 — known-radius fit (radius pinned to 3.0)
# ---------------------------------------------------------------------------

class TestGoldenKnownRadius:
    @pytest.fixture(scope="class")
    def model(self):
        syn = generate_noisy_cylinder(
            radius=3.0, noise=0.01, random_state=99, n_points=1500
        )
        return fit_cylinder_known_radius(
            syn.points, 3.0, threshold=0.05, ransac_trials=48, random_state=99
        )

    def test_radius_exactly_pinned(self, model):
        assert abs(model.radius - 3.0) < 1e-12

    def test_rmse_pinned(self, model):
        assert abs(model.rmse - 0.00965366) < 5e-4

    def test_axis_direction_unit(self, model):
        assert np.isclose(np.linalg.norm(model.axis_direction), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Golden 4 — normal-seeded fit
# ---------------------------------------------------------------------------

class TestGoldenNormals:
    @pytest.fixture(scope="class")
    def result(self):
        syn = generate_noisy_cylinder(noise=0.015, random_state=55, n_points=1500)
        model = fit_cylinder_with_normals(
            syn.points, syn.normals,
            threshold=0.06, ransac_trials=32, random_state=55,
        )
        return model, syn

    def test_radius_pinned(self, result):
        model, _ = result
        assert abs(model.radius - 1.50004781) < 5e-4

    def test_rmse_pinned(self, result):
        model, _ = result
        assert abs(model.rmse - 0.01548425) < 5e-4

    def test_axis_angle_small(self, result):
        model, syn = result
        assert _angle_between(model.axis_direction, syn.axis_direction) < 1.0


# ---------------------------------------------------------------------------
# Golden 5 — residuals_to_cylinder exact geometry
# ---------------------------------------------------------------------------

class TestGoldenResiduals:
    def test_unit_cylinder_exact(self):
        """Points at radius r from the unit Z-axis have residuals r − 1."""
        axis = np.array([0., 0., 1.])
        p0 = np.zeros(3)
        pts = np.array([
            [2., 0., 0.],   # r = 2 → residual = 1
            [0., 3., 5.],   # r = 3 → residual = 2
            [0., 0., 7.],   # r = 0 → residual = -1  (on axis)
        ])
        res = residuals_to_cylinder(pts, p0, axis, 1.0)
        np.testing.assert_allclose(res, [1.0, 2.0, -1.0], atol=1e-12)

    def test_tilted_axis_distances_invariant(self):
        """Residuals are invariant to translation along the axis."""
        axis = np.array([1., 1., 1.]) / np.sqrt(3)
        p0 = np.zeros(3)
        # Point 1 unit from axis in the u direction
        u = np.array([1., -1., 0.]) / np.sqrt(2)
        pt = u[None, :] * 2.0  # 2 units from axis
        shift = axis * 5.0
        res_orig = residuals_to_cylinder(pt, p0, axis, 2.0)
        res_shift = residuals_to_cylinder(pt + shift, p0 + shift, axis, 2.0)
        np.testing.assert_allclose(res_orig, res_shift, atol=1e-12)

    def test_cone_residuals_zero_on_exact_cone(self):
        """Points exactly on a 45° cone have residuals ≈ 0."""
        apex = np.array([0., 0., 0.])
        axis = np.array([0., 0., 1.])
        alpha = np.deg2rad(45.0)
        t_vals = np.array([1., 2., 3., 0.5])
        r_vals = t_vals * np.tan(alpha)
        pts = np.column_stack([r_vals, np.zeros_like(r_vals), t_vals])
        res = _cone_residuals(pts, apex, axis, alpha)
        np.testing.assert_allclose(np.abs(res), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Golden 6 — 2-D ellipse algebraic fit on known ellipse
# ---------------------------------------------------------------------------

class TestGoldenEllipseAlgebra:
    def test_known_axis_aligned_ellipse(self):
        """Algebraic ellipse fit on a perfect axis-aligned ellipse."""
        a, b = 3.0, 1.5
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        u = a * np.cos(theta)
        v = b * np.sin(theta)
        result = _fit_ellipse_algebraic(u, v)
        assert result is not None
        cx, cy, semi_a, semi_b, angle = result
        # Centre should be near origin
        assert abs(cx) < 0.01
        assert abs(cy) < 0.01
        # Semi-axes in either order
        fitted = sorted([semi_a, semi_b], reverse=True)
        assert abs(fitted[0] - a) < 0.05
        assert abs(fitted[1] - b) < 0.05

    def test_rotated_ellipse(self):
        """Fit a 45°-rotated ellipse."""
        a, b = 2.0, 1.0
        theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        # Rotate 45°
        c45, s45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
        u_raw = a * np.cos(theta)
        v_raw = b * np.sin(theta)
        u = c45 * u_raw - s45 * v_raw
        v = s45 * u_raw + c45 * v_raw
        result = _fit_ellipse_algebraic(u, v)
        assert result is not None
        _, _, semi_a, semi_b, _ = result
        fitted = sorted([semi_a, semi_b], reverse=True)
        assert abs(fitted[0] - a) < 0.05
        assert abs(fitted[1] - b) < 0.05

    def test_too_few_points_returns_none(self):
        result = _fit_ellipse_algebraic(
            np.array([1., 2., 3., 4., 0.]),
            np.array([0., 1., 0., -1., 0.]),
        )
        # 5 points — below the 6-point minimum
        assert result is None


# ---------------------------------------------------------------------------
# Golden 7 — LM convergence reproducibility
# ---------------------------------------------------------------------------

class TestGoldenReproducibility:
    def test_same_seed_same_output(self):
        """Two calls with the same seed must return bit-identical results."""
        syn = generate_noisy_cylinder(radius=2.0, noise=0.02, random_state=77)
        m1 = fit_cylinder(syn.points, threshold=0.07, ransac_trials=48, random_state=77)
        m2 = fit_cylinder(syn.points, threshold=0.07, ransac_trials=48, random_state=77)
        np.testing.assert_array_equal(m1.axis_direction, m2.axis_direction)
        np.testing.assert_array_equal(m1.axis_point, m2.axis_point)
        assert m1.radius == m2.radius
        assert m1.iterations == m2.iterations

    def test_different_seeds_close_for_easy_case(self):
        """On a clean easy cylinder, different seeds should agree within 1%."""
        syn = generate_noisy_cylinder(
            radius=2.0, noise=0.005, outlier_fraction=0.0,
            n_points=2000, random_state=0,
        )
        radii = []
        for seed in range(5):
            m = fit_cylinder(
                syn.points, threshold=0.02, ransac_trials=48, random_state=seed
            )
            radii.append(m.radius)
        spread = max(radii) - min(radii)
        assert spread < 0.02, f"Radius spread {spread:.4f} > 0.02 across seeds"
