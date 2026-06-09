"""Tests for visualize.py — runs headlessly via the Agg backend."""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # must be set before any other matplotlib import

from cylfit import fit_cylinder, fit_hyperboloid, generate_noisy_cylinder, generate_noisy_hyperboloid  # noqa: E402
from cylfit.visualize import (  # noqa: E402
    _cylinder_surface,
    _hyperboloid_surface,
    _orthonormal_basis,
    _set_axes_equal,
    plot_cylinder_fit,
    plot_hyperboloid_fit,
)


@pytest.fixture(scope="module")
def cylinder_data():
    syn = generate_noisy_cylinder(n_points=500, radius=1.0, height=3.0, noise=0.02, random_state=0)
    pts = syn.points
    model = fit_cylinder(pts, random_state=0)
    return pts, model


@pytest.fixture(scope="module")
def hyperboloid_data():
    syn = generate_noisy_hyperboloid(n_points=500, waist_radius=1.0, curvature=2.0,
                                     height=4.0, noise=0.02, random_state=1)
    pts = syn.points
    model = fit_hyperboloid(pts, random_state=1)
    return pts, model


class TestPlotCylinderFit:
    def test_returns_fig_and_ax(self, cylinder_data):
        pts, model = cylinder_data
        fig, ax = plot_cylinder_fit(pts, model)
        assert fig is not None
        assert ax is not None

    def test_custom_title(self, cylinder_data):
        pts, model = cylinder_data
        fig, ax = plot_cylinder_fit(pts, model, title="test title")
        assert ax.get_title() == "test title"

    def test_no_outliers(self, cylinder_data):
        pts, model = cylinder_data
        fig, ax = plot_cylinder_fit(pts, model, show_outliers=False)
        assert fig is not None

    def test_with_provided_ax(self, cylinder_data):
        import matplotlib.pyplot as plt
        pts, model = cylinder_data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        returned_fig, returned_ax = plot_cylinder_fit(pts, model, ax=ax)
        assert returned_ax is ax

    def test_invalid_points_raises(self, cylinder_data):
        _, model = cylinder_data
        with pytest.raises(ValueError, match="N x 3"):
            plot_cylinder_fit(np.ones((10, 2)), model)

    def test_max_points_downsampling(self, cylinder_data):
        pts, model = cylinder_data
        # max_points less than point count should still work
        fig, ax = plot_cylinder_fit(pts, model, max_points=50)
        assert fig is not None


class TestPlotHyperboloidFit:
    def test_returns_fig_and_ax(self, hyperboloid_data):
        pts, model = hyperboloid_data
        fig, ax = plot_hyperboloid_fit(pts, model)
        assert fig is not None
        assert ax is not None

    def test_custom_title(self, hyperboloid_data):
        pts, model = hyperboloid_data
        fig, ax = plot_hyperboloid_fit(pts, model, title="hyp test")
        assert ax.get_title() == "hyp test"

    def test_invalid_points_raises(self, hyperboloid_data):
        _, model = hyperboloid_data
        with pytest.raises(ValueError, match="N x 3"):
            plot_hyperboloid_fit(np.ones(10), model)


class TestHelpers:
    def test_orthonormal_basis_z_axis(self):
        axis = np.array([0.0, 0.0, 1.0])
        u, v = _orthonormal_basis(axis)
        assert abs(float(np.dot(u, v))) < 1e-12
        assert abs(float(np.dot(u, axis))) < 1e-12
        assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-12
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-12

    def test_orthonormal_basis_x_axis(self):
        axis = np.array([1.0, 0.0, 0.0])
        u, v = _orthonormal_basis(axis)
        assert abs(float(np.dot(u, v))) < 1e-12
        assert abs(float(np.dot(u, axis))) < 1e-12

    def test_orthonormal_basis_arbitrary(self):
        axis = np.array([1.0, 2.0, 3.0])
        axis /= np.linalg.norm(axis)
        u, v = _orthonormal_basis(axis)
        assert abs(float(np.dot(u, axis))) < 1e-12
        assert abs(float(np.dot(v, axis))) < 1e-12
        assert abs(float(np.dot(u, v))) < 1e-12

    def test_cylinder_surface_shape(self, cylinder_data):
        _, model = cylinder_data
        xx, yy, zz = _cylinder_surface(model, (12, 6))
        assert xx.shape == (6, 12)
        assert yy.shape == (6, 12)
        assert zz.shape == (6, 12)

    def test_hyperboloid_surface_shape(self, hyperboloid_data):
        _, model = hyperboloid_data
        xx, yy, zz = _hyperboloid_surface(model, (12, 6))
        assert xx.shape == (6, 12)

    def test_set_axes_equal(self, cylinder_data):
        import matplotlib.pyplot as plt
        pts, _ = cylinder_data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        _set_axes_equal(ax, pts)  # should not raise
