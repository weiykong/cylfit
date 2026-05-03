"""Tests for extended fitting capabilities:
  - Elliptical cylinder
  - Cone
  - Curved cylinder
  - Pipe network / cylinder joints
"""
import numpy as np
import pytest

from cylfit import (
    detect_cylinders,
    find_cylinder_joints,
    build_pipe_network,
    fit_cone,
    fit_curved_cylinder,
    fit_elliptical_cylinder,
    generate_noisy_cylinder,
    ConeModel,
    CurvedCylinderModel,
    EllipticalCylinderModel,
    CylinderJoint,
    PipeNetwork,
)
from cylfit.cone import _cone_residuals
from cylfit.network import _segment_to_segment_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_elliptical_cylinder(
    semi_major=2.0,
    semi_minor=1.2,
    angle_deg=30.0,
    n_points=1500,
    noise=0.01,
    seed=42,
):
    """Generate a synthetic elliptical cylinder point cloud."""
    rng = np.random.default_rng(seed)
    axis = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])

    a_rad = np.deg2rad(angle_deg)
    u_rot = np.cos(a_rad) * u + np.sin(a_rad) * v
    v_rot = -np.sin(a_rad) * u + np.cos(a_rad) * v

    theta = rng.uniform(0, 2 * np.pi, n_points)
    t = rng.uniform(-5.0, 5.0, n_points)
    x = semi_major * np.cos(theta)
    y = semi_minor * np.sin(theta)

    pts = (
        x[:, None] * u_rot
        + y[:, None] * v_rot
        + t[:, None] * axis
    )
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts, axis, semi_major, semi_minor


def _make_cone(
    half_angle_deg=20.0,
    apex=None,
    axis=None,
    t_min=1.0,
    t_max=8.0,
    n_points=1500,
    noise=0.01,
    seed=0,
):
    """Generate a synthetic cone point cloud."""
    rng = np.random.default_rng(seed)
    if apex is None:
        apex = np.array([0.0, 0.0, 0.0])
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)

    helper = np.array([1.0, 0.0, 0.0])
    if abs(axis @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = helper - (helper @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    alpha = np.deg2rad(half_angle_deg)
    t = rng.uniform(t_min, t_max, n_points)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    r = t * np.tan(alpha)

    pts = (
        apex
        + t[:, None] * axis
        + r[:, None] * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    )
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts, apex, axis, half_angle_deg


def _make_bent_cylinder(
    radius=1.0,
    bend_angle_deg=30.0,
    n_points=2000,
    noise=0.01,
    seed=5,
):
    """Generate a bent (L-shaped) cylinder point cloud."""
    rng = np.random.default_rng(seed)
    pts_list = []

    # Segment 1: along Z
    n1 = n_points // 2
    axis1 = np.array([0.0, 0.0, 1.0])
    u1 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.0, 1.0, 0.0])
    t1 = rng.uniform(0, 5.0, n1)
    theta1 = rng.uniform(0, 2 * np.pi, n1)
    p1 = (t1[:, None] * axis1
          + radius * np.cos(theta1)[:, None] * u1
          + radius * np.sin(theta1)[:, None] * v1)
    pts_list.append(p1 + rng.normal(scale=noise, size=p1.shape))

    # Segment 2: bent by bend_angle_deg from Z in the XZ plane
    ba = np.deg2rad(bend_angle_deg)
    axis2 = np.array([np.sin(ba), 0.0, np.cos(ba)])
    u2 = np.array([np.cos(ba), 0.0, -np.sin(ba)])
    v2 = np.array([0.0, 1.0, 0.0])
    junction = np.array([0.0, 0.0, 5.0])
    n2 = n_points - n1
    t2 = rng.uniform(0, 5.0, n2)
    theta2 = rng.uniform(0, 2 * np.pi, n2)
    p2 = (junction
          + t2[:, None] * axis2
          + radius * np.cos(theta2)[:, None] * u2
          + radius * np.sin(theta2)[:, None] * v2)
    pts_list.append(p2 + rng.normal(scale=noise, size=p2.shape))

    return np.vstack(pts_list)


# ===========================================================================
# Elliptical cylinder tests
# ===========================================================================

class TestEllipticalCylinder:
    def test_returns_correct_type(self):
        pts, *_ = _make_elliptical_cylinder()
        model = fit_elliptical_cylinder(pts, threshold=0.08, ransac_trials=32, random_state=0)
        assert isinstance(model, EllipticalCylinderModel)

    def test_semi_axes_ordering(self):
        pts, _, a, b = _make_elliptical_cylinder(semi_major=2.5, semi_minor=1.0)
        model = fit_elliptical_cylinder(pts, threshold=0.1, ransac_trials=32, random_state=0)
        assert model.semi_major >= model.semi_minor > 0

    def test_aspect_ratio_positive(self):
        pts, *_ = _make_elliptical_cylinder()
        model = fit_elliptical_cylinder(pts, threshold=0.08, ransac_trials=32, random_state=0)
        assert model.aspect_ratio >= 1.0

    def test_equivalent_radius_in_range(self):
        a, b = 2.0, 1.2
        pts, *_ = _make_elliptical_cylinder(semi_major=a, semi_minor=b, noise=0.005)
        model = fit_elliptical_cylinder(pts, threshold=0.08, ransac_trials=48, random_state=0)
        eq_r = model.equivalent_radius
        expected = np.sqrt(a * b)
        assert abs(eq_r - expected) < 0.25 * expected

    def test_axis_direction_unit_length(self):
        pts, *_ = _make_elliptical_cylinder()
        model = fit_elliptical_cylinder(pts, threshold=0.08, ransac_trials=32, random_state=0)
        assert np.isclose(np.linalg.norm(model.axis_direction), 1.0, atol=1e-10)

    def test_height_bounds(self):
        pts, *_ = _make_elliptical_cylinder()
        model = fit_elliptical_cylinder(pts, threshold=0.08, ransac_trials=32, random_state=0)
        assert model.height_max >= model.height_min
        assert model.height > 0

    def test_to_dict_json(self):
        pts, *_ = _make_elliptical_cylinder()
        model = fit_elliptical_cylinder(pts, threshold=0.08, ransac_trials=32, random_state=0)
        d = model.to_dict()
        assert "semi_major" in d and "semi_minor" in d
        j = model.to_json()
        assert '"semi_major"' in j

    def test_inlier_mask_shape(self):
        pts, *_ = _make_elliptical_cylinder(n_points=800)
        model = fit_elliptical_cylinder(pts, threshold=0.08, ransac_trials=32, random_state=0)
        assert model.inlier_mask.shape == (800,)
        assert model.inlier_mask.dtype == bool

    def test_rmse_finite(self):
        pts, *_ = _make_elliptical_cylinder()
        model = fit_elliptical_cylinder(pts, threshold=0.1, ransac_trials=32, random_state=0)
        assert np.isfinite(model.rmse)

    def test_circular_cylinder_detected(self):
        """When a = b the fit should still work (degenerate → circular ellipse)."""
        syn = generate_noisy_cylinder(radius=1.5, noise=0.01, random_state=7)
        model = fit_elliptical_cylinder(
            syn.points, threshold=0.06, ransac_trials=32, random_state=7
        )
        assert isinstance(model, EllipticalCylinderModel)
        assert model.semi_major > 0 and model.semi_minor > 0


# ===========================================================================
# Cone tests
# ===========================================================================

class TestCone:
    def test_returns_correct_type(self):
        pts, *_ = _make_cone()
        model = fit_cone(pts, threshold=0.1, ransac_trials=32, random_state=0)
        assert isinstance(model, ConeModel)

    def test_half_angle_within_range(self):
        pts, *_ = _make_cone(half_angle_deg=20.0, noise=0.005)
        model = fit_cone(pts, threshold=0.1, ransac_trials=64, random_state=0)
        assert 0 < model.half_angle_deg < 90

    def test_half_angle_accuracy(self):
        true_angle = 25.0
        pts, *_ = _make_cone(half_angle_deg=true_angle, noise=0.005, n_points=2000)
        model = fit_cone(pts, threshold=0.1, ransac_trials=96, random_state=1)
        assert abs(model.half_angle_deg - true_angle) < 10.0

    def test_known_half_angle_pinned(self):
        true_angle = 20.0
        pts, *_ = _make_cone(half_angle_deg=true_angle, noise=0.005)
        model = fit_cone(
            pts, known_half_angle_deg=true_angle,
            threshold=0.08, ransac_trials=32, random_state=0,
        )
        assert abs(model.half_angle_deg - true_angle) < 1e-8

    def test_axis_unit_length(self):
        pts, *_ = _make_cone()
        model = fit_cone(pts, threshold=0.1, ransac_trials=32, random_state=0)
        assert np.isclose(np.linalg.norm(model.axis_direction), 1.0, atol=1e-10)

    def test_height_bounds(self):
        pts, *_ = _make_cone()
        model = fit_cone(pts, threshold=0.1, ransac_trials=32, random_state=0)
        assert model.height_max >= model.height_min

    def test_inlier_mask_shape(self):
        pts, *_ = _make_cone(n_points=1000)
        model = fit_cone(pts, threshold=0.1, ransac_trials=32, random_state=0)
        assert model.inlier_mask.shape == (1000,)

    def test_residuals_near_zero_clean(self):
        """On near-noise-free cone the RMSE should be small."""
        pts, *_ = _make_cone(noise=0.001, n_points=2000)
        model = fit_cone(pts, threshold=0.05, ransac_trials=64, random_state=2)
        assert model.rmse < 0.05

    def test_to_dict_json(self):
        pts, *_ = _make_cone()
        model = fit_cone(pts, threshold=0.1, ransac_trials=32, random_state=0)
        d = model.to_dict()
        assert "half_angle_deg" in d and "apex" in d
        j = model.to_json()
        assert '"half_angle_deg"' in j

    def test_cone_residuals_zero_on_clean_cone(self):
        """_cone_residuals on exact cone points should be ≈ 0."""
        pts, apex, axis, angle_deg = _make_cone(noise=0.0, n_points=500)
        alpha = np.deg2rad(angle_deg)
        res = _cone_residuals(pts, apex, axis, alpha)
        np.testing.assert_allclose(np.abs(res), 0.0, atol=1e-9)

    def test_invalid_half_angle_raises(self):
        pts, *_ = _make_cone()
        with pytest.raises(ValueError):
            fit_cone(pts, known_half_angle_deg=0.0)
        with pytest.raises(ValueError):
            fit_cone(pts, known_half_angle_deg=90.0)

    def test_radius_at_base_positive(self):
        pts, *_ = _make_cone()
        model = fit_cone(pts, threshold=0.1, ransac_trials=32, random_state=0)
        assert model.radius_at_base > 0


# ===========================================================================
# Curved cylinder tests
# ===========================================================================

class TestCurvedCylinder:
    def test_returns_correct_type(self):
        pts = _make_bent_cylinder()
        model = fit_curved_cylinder(pts, n_segments=6, threshold=0.1, random_state=0)
        assert isinstance(model, CurvedCylinderModel)

    def test_spine_shape(self):
        pts = _make_bent_cylinder()
        n_segs = 6
        model = fit_curved_cylinder(pts, n_segments=n_segs, threshold=0.1, random_state=0)
        assert model.spine.shape == (n_segs + 1, 3)
        assert model.n_segments == n_segs

    def test_radius_positive(self):
        pts = _make_bent_cylinder(radius=1.5)
        model = fit_curved_cylinder(pts, n_segments=6, threshold=0.15, random_state=0)
        assert model.radius > 0

    def test_radius_roughly_correct(self):
        true_r = 1.0
        pts = _make_bent_cylinder(radius=true_r, noise=0.01)
        model = fit_curved_cylinder(pts, n_segments=6, threshold=0.1, random_state=0)
        assert abs(model.radius - true_r) < 0.3 * true_r

    def test_total_length_positive(self):
        pts = _make_bent_cylinder()
        model = fit_curved_cylinder(pts, n_segments=6, threshold=0.1, random_state=0)
        assert model.total_length > 0

    def test_curvature_straight_near_zero(self):
        """A straight cylinder should have near-zero curvature."""
        syn = generate_noisy_cylinder(n_points=2000, noise=0.01, random_state=10)
        model = fit_curved_cylinder(
            syn.points, n_segments=8, threshold=0.08, random_state=10
        )
        assert model.curvature_mean < 0.5

    def test_curvature_is_finite_and_non_negative(self):
        """curvature_mean must always be finite and ≥ 0."""
        pts_bent = _make_bent_cylinder(bend_angle_deg=60, n_points=3000, noise=0.005)
        bent_model = fit_curved_cylinder(
            pts_bent, n_segments=8, threshold=0.1, random_state=10
        )
        assert np.isfinite(bent_model.curvature_mean)
        assert bent_model.curvature_mean >= 0

    def test_inlier_mask_shape(self):
        pts = _make_bent_cylinder(n_points=1000)
        model = fit_curved_cylinder(pts, n_segments=4, threshold=0.15, random_state=0)
        assert model.inlier_mask.shape == (1000,)

    def test_segment_radii_shape(self):
        pts = _make_bent_cylinder()
        n_segs = 5
        model = fit_curved_cylinder(pts, n_segments=n_segs, threshold=0.1, random_state=0)
        assert model.segment_radii.shape == (n_segs,)

    def test_to_dict_json(self):
        pts = _make_bent_cylinder(n_points=500)
        model = fit_curved_cylinder(pts, n_segments=4, threshold=0.15, random_state=0)
        d = model.to_dict()
        assert "spine" in d and "radius" in d and "total_length" in d
        j = model.to_json()
        assert '"spine"' in j

    def test_rmse_finite(self):
        pts = _make_bent_cylinder()
        model = fit_curved_cylinder(pts, n_segments=6, threshold=0.1, random_state=0)
        assert np.isfinite(model.rmse)


# ===========================================================================
# Pipe network tests
# ===========================================================================

class TestSegmentDistance:
    def test_parallel_segments(self):
        p1, p2 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
        q1, q2 = np.array([0., 1., 0.]), np.array([1., 1., 0.])
        d, pa, pb = _segment_to_segment_distance(p1, p2, q1, q2)
        assert abs(d - 1.0) < 1e-10

    def test_perpendicular_intersecting(self):
        # X-axis and Y-axis both through origin
        p1, p2 = np.array([-1., 0., 0.]), np.array([1., 0., 0.])
        q1, q2 = np.array([0., -1., 0.]), np.array([0., 1., 0.])
        d, _, _ = _segment_to_segment_distance(p1, p2, q1, q2)
        assert d < 1e-10

    def test_skew_lines(self):
        p1, p2 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
        q1, q2 = np.array([0., 1., 1.]), np.array([1., 1., 1.])
        d, _, _ = _segment_to_segment_distance(p1, p2, q1, q2)
        assert abs(d - np.sqrt(2)) < 1e-10

    def test_degenerate_points(self):
        p = np.array([0., 0., 0.])
        q = np.array([3., 4., 0.])
        d, _, _ = _segment_to_segment_distance(p, p, q, q)
        assert abs(d - 5.0) < 1e-10


class TestPipeNetwork:
    def _two_touching_cylinders(self):
        """Two cylinders whose endpoints touch at the origin."""
        syn_a = generate_noisy_cylinder(
            n_points=800, radius=0.5, height=3.0,
            axis_point=np.array([0., 0., -1.5]),
            noise=0.005, outlier_fraction=0.0, random_state=20,
        )
        syn_b = generate_noisy_cylinder(
            n_points=800, radius=0.5, height=3.0,
            axis_point=np.array([0., 0., 1.5]),
            noise=0.005, outlier_fraction=0.0, random_state=21,
        )
        return syn_a, syn_b

    def test_find_joints_returns_list(self):
        syn_a, syn_b = self._two_touching_cylinders()
        dets = detect_cylinders(
            np.vstack([syn_a.points, syn_b.points]),
            max_cylinders=2, min_inliers=300,
            threshold=0.04, ransac_trials=32, random_state=0,
        )
        joints = find_cylinder_joints(dets, distance_threshold=2.0)
        assert isinstance(joints, list)

    def test_joint_fields_valid(self):
        syn_a, syn_b = self._two_touching_cylinders()
        dets = detect_cylinders(
            np.vstack([syn_a.points, syn_b.points]),
            max_cylinders=2, min_inliers=300,
            threshold=0.04, ransac_trials=32, random_state=0,
        )
        joints = find_cylinder_joints(dets, distance_threshold=2.0)
        for joint in joints:
            assert isinstance(joint, CylinderJoint)
            assert joint.gap >= 0
            assert 0 <= joint.angle_deg <= 90
            assert joint.point_a.shape == (3,)
            assert joint.point_b.shape == (3,)
            assert joint.midpoint.shape == (3,)

    def test_build_pipe_network_type(self):
        syn_a, syn_b = self._two_touching_cylinders()
        dets = detect_cylinders(
            np.vstack([syn_a.points, syn_b.points]),
            max_cylinders=2, min_inliers=300,
            threshold=0.04, ransac_trials=32, random_state=0,
        )
        net = build_pipe_network(dets, distance_threshold=2.0)
        assert isinstance(net, PipeNetwork)
        assert len(net.cylinders) == len(dets)

    def test_adjacency_symmetric(self):
        syn_a, syn_b = self._two_touching_cylinders()
        dets = detect_cylinders(
            np.vstack([syn_a.points, syn_b.points]),
            max_cylinders=2, min_inliers=300,
            threshold=0.04, ransac_trials=32, random_state=0,
        )
        net = build_pipe_network(dets, distance_threshold=2.0)
        for joint in net.joints:
            assert joint.cylinder_b_idx in net.adjacency[joint.cylinder_a_idx]
            assert joint.cylinder_a_idx in net.adjacency[joint.cylinder_b_idx]

    def test_joint_dict_json(self):
        syn_a, syn_b = self._two_touching_cylinders()
        dets = detect_cylinders(
            np.vstack([syn_a.points, syn_b.points]),
            max_cylinders=2, min_inliers=300,
            threshold=0.04, ransac_trials=32, random_state=0,
        )
        joints = find_cylinder_joints(dets, distance_threshold=2.0)
        if joints:
            d = joints[0].to_dict()
            assert "gap" in d and "angle_deg" in d

    def test_no_joints_far_apart(self):
        """Cylinders far apart should produce no joints."""
        syn_a, syn_b = self._two_touching_cylinders()
        dets = detect_cylinders(
            np.vstack([syn_a.points, syn_b.points]),
            max_cylinders=2, min_inliers=300,
            threshold=0.04, ransac_trials=32, random_state=0,
        )
        joints = find_cylinder_joints(dets, distance_threshold=0.001)
        assert len(joints) == 0

    def test_network_to_dict_json(self):
        syn_a, syn_b = self._two_touching_cylinders()
        dets = detect_cylinders(
            np.vstack([syn_a.points, syn_b.points]),
            max_cylinders=2, min_inliers=300,
            threshold=0.04, ransac_trials=32, random_state=0,
        )
        net = build_pipe_network(dets, distance_threshold=2.0)
        d = net.to_dict()
        assert "n_cylinders" in d
        j = net.to_json()
        assert '"n_cylinders"' in j
