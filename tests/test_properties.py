"""Property-based tests using Hypothesis.

These tests verify geometric invariants that must hold regardless of the
specific cylinder parameters — things that unit tests with fixed examples
cannot exhaustively cover.
"""

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cylfit import fit_cylinder, residuals_to_cylinder
from cylfit.core import (
    _analytic_jacobian,
    _auto_threshold,
    _axis_distances,
    _normalize,
    _numeric_jacobian,
    _weighted_residuals,
)
from cylfit.synthetic import generate_noisy_cylinder


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

reasonable_float = st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
unit_float = st.floats(min_value=1e-3, max_value=1.0, allow_nan=False)
angle_float = st.floats(min_value=0.0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False)

@st.composite
def random_unit_axis(draw):
    """Draw a random 3D unit vector."""
    v = draw(arrays(np.float64, (3,), elements=st.floats(-1, 1, allow_nan=False)))
    norm = np.linalg.norm(v)
    assume(norm > 0.1)
    return v / norm


@st.composite
def cylinder_params(draw):
    """Draw valid synthetic cylinder parameters."""
    radius = draw(st.floats(min_value=0.3, max_value=5.0, allow_nan=False))
    height = draw(st.floats(min_value=1.0, max_value=15.0, allow_nan=False))
    noise = draw(st.floats(min_value=0.0, max_value=radius * 0.05, allow_nan=False))
    seed = draw(st.integers(min_value=0, max_value=9999))
    return radius, height, noise, seed


# ---------------------------------------------------------------------------
# Invariant: residuals_to_cylinder on perfectly clean data
# ---------------------------------------------------------------------------

class TestResidualInvariants:
    @given(cylinder_params())
    @settings(max_examples=25, deadline=5000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
    def test_zero_residuals_on_clean_cylinder(self, params):
        """Points exactly on a cylinder surface have residuals ≈ 0."""
        radius, height, _, seed = params
        syn = generate_noisy_cylinder(
            radius=radius, height=height, noise=0.0,
            outlier_fraction=0.0, n_points=300, random_state=seed,
        )
        res = residuals_to_cylinder(
            syn.clean_points, syn.axis_point, syn.axis_direction, syn.radius
        )
        np.testing.assert_allclose(np.abs(res), 0.0, atol=1e-9)

    @given(cylinder_params())
    @settings(max_examples=25, deadline=5000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
    def test_residuals_scale_with_radius_error(self, params):
        """Adding delta to radius shifts all residuals by -delta."""
        radius, height, _, seed = params
        syn = generate_noisy_cylinder(
            radius=radius, height=height, noise=0.0,
            outlier_fraction=0.0, n_points=300, random_state=seed,
        )
        delta = 0.1
        res_true = residuals_to_cylinder(
            syn.clean_points, syn.axis_point, syn.axis_direction, syn.radius
        )
        res_enlarged = residuals_to_cylinder(
            syn.clean_points, syn.axis_point, syn.axis_direction, syn.radius + delta
        )
        np.testing.assert_allclose(res_enlarged, res_true - delta, atol=1e-9)

    @given(random_unit_axis(), random_unit_axis())
    @settings(max_examples=30, deadline=2000)
    def test_residuals_axis_flip_symmetric(self, axis1, axis2):
        """Flipping the axis direction must not change radial distances."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((100, 3))
        p0 = np.zeros(3)
        dist_pos = _axis_distances(pts, p0, axis1)
        dist_neg = _axis_distances(pts, p0, -axis1)
        np.testing.assert_allclose(dist_pos, dist_neg, rtol=1e-10)


# ---------------------------------------------------------------------------
# Invariant: normalize is idempotent
# ---------------------------------------------------------------------------

class TestNormalizeInvariant:
    @given(arrays(np.float64, (3,), elements=st.floats(0.1, 10.0)))
    @settings(max_examples=50, deadline=1000)
    def test_double_normalize_is_idempotent(self, v):
        assume(np.linalg.norm(v) > 0.01)
        u = _normalize(v)
        uu = _normalize(u)
        np.testing.assert_allclose(u, uu, atol=1e-14)

    @given(arrays(np.float64, (3,), elements=st.floats(0.1, 10.0)))
    @settings(max_examples=50, deadline=1000)
    def test_normalize_unit_length(self, v):
        assume(np.linalg.norm(v) > 0.01)
        u = _normalize(v)
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-13)


# ---------------------------------------------------------------------------
# Invariant: analytic Jacobian matches numeric Jacobian
# ---------------------------------------------------------------------------

class TestAnalyticJacobianCorrectness:
    @given(cylinder_params())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
    def test_analytic_matches_numeric(self, params):
        """Analytic Jacobian must agree with central-difference numeric Jacobian."""
        radius, height, noise, seed = params
        syn = generate_noisy_cylinder(
            radius=radius, height=height, noise=noise,
            outlier_fraction=0.0, n_points=200, random_state=seed,
        )
        pts = syn.clean_points
        centroid = pts.mean(axis=0)
        p0 = syn.axis_point
        axis = syn.axis_direction
        params_vec = np.r_[p0, axis]
        res, _, _ = _weighted_residuals(params_vec, pts, centroid, None, None)

        J_analytic = _analytic_jacobian(params_vec, pts, centroid, None, None)
        J_numeric = _numeric_jacobian(params_vec, pts, centroid, None, None, res)

        np.testing.assert_allclose(J_analytic, J_numeric, atol=1e-5, rtol=1e-4)

    @given(cylinder_params())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
    def test_analytic_known_radius_matches_numeric(self, params):
        """Analytic Jacobian with fixed radius must match numeric version."""
        radius, height, noise, seed = params
        syn = generate_noisy_cylinder(
            radius=radius, height=height, noise=noise,
            outlier_fraction=0.0, n_points=200, random_state=seed,
        )
        pts = syn.clean_points
        centroid = pts.mean(axis=0)
        p0 = syn.axis_point
        axis = syn.axis_direction
        params_vec = np.r_[p0, axis]
        res, _, _ = _weighted_residuals(params_vec, pts, centroid, None, radius)

        J_analytic = _analytic_jacobian(params_vec, pts, centroid, None, radius)
        J_numeric = _numeric_jacobian(params_vec, pts, centroid, None, radius, res)

        np.testing.assert_allclose(J_analytic, J_numeric, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Invariant: fit_cylinder radius + axis accuracy on clean data
# ---------------------------------------------------------------------------

class TestFitInvariants:
    @given(
        st.floats(min_value=0.3, max_value=2.0, allow_nan=False),
        st.integers(min_value=0, max_value=199),
    )
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
    def test_fit_returns_valid_model(self, radius, seed):
        """fit_cylinder must always return a structurally valid CylinderModel."""
        syn = generate_noisy_cylinder(
            radius=radius, noise=0.005 * radius, outlier_fraction=0.0,
            n_points=800, random_state=seed,
        )
        model = fit_cylinder(
            syn.points, threshold=0.05, ransac_trials=48, random_state=seed
        )
        # Structural invariants — always hold regardless of accuracy
        assert model.radius > 0, "radius must be positive"
        assert np.isclose(np.linalg.norm(model.axis_direction), 1.0, atol=1e-10), \
            "axis_direction must be unit length"
        assert model.height_max >= model.height_min, "height_max must be >= height_min"
        assert model.inlier_mask.dtype == bool
        assert model.inlier_mask.shape == (syn.points.shape[0],)
        assert np.isfinite(model.rmse), "RMSE must be finite"

    @given(cylinder_params())
    @settings(max_examples=15, deadline=8000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
    def test_inlier_mask_consistency(self, params):
        """inlier_mask must be consistent with residuals and threshold."""
        radius, height, noise, seed = params
        assume(noise < radius * 0.04)
        syn = generate_noisy_cylinder(
            radius=radius, height=height, noise=noise,
            outlier_fraction=0.0, n_points=600, random_state=seed,
        )
        thr = max(noise * 3 + 0.01, 0.02)
        model = fit_cylinder(
            syn.points, threshold=thr, ransac_trials=32, random_state=seed
        )
        expected = np.abs(model.residuals) <= thr
        np.testing.assert_array_equal(model.inlier_mask, expected)


# ---------------------------------------------------------------------------
# Invariant: auto_threshold is positive and finite
# ---------------------------------------------------------------------------

class TestAutoThreshold:
    @given(arrays(np.float64, st.integers(10, 500),
                  elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=50, deadline=1000)
    def test_auto_threshold_positive_finite(self, residuals):
        t = _auto_threshold(residuals)
        assert np.isfinite(t)
        assert t > 0
