from __future__ import annotations

from typing import Optional

import numpy as np

from .core import CylinderModel


def plot_cylinder_fit(
    points: np.ndarray,
    model: CylinderModel,
    *,
    ax=None,
    max_points: int = 12000,
    show_outliers: bool = True,
    surface_alpha: float = 0.22,
    surface_resolution: tuple[int, int] = (48, 24),
    title: Optional[str] = None,
    random_state: Optional[int] = 0,
):
    """Visualize a point cloud, fitted cylinder surface, and fitted axis.

    Returns ``(fig, ax)``. Matplotlib is imported lazily so fitting remains a
    NumPy-only dependency.
    """

    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    rng = np.random.default_rng(random_state)
    draw_idx = np.arange(pts.shape[0])
    if pts.shape[0] > max_points:
        draw_idx = rng.choice(pts.shape[0], max_points, replace=False)

    shown = pts[draw_idx]
    shown_inliers = model.inlier_mask[draw_idx]
    if show_outliers and (~shown_inliers).any():
        ax.scatter(
            shown[~shown_inliers, 0],
            shown[~shown_inliers, 1],
            shown[~shown_inliers, 2],
            s=4,
            c="#b8b8b8",
            alpha=0.35,
            depthshade=False,
            label="outliers",
        )
    ax.scatter(
        shown[shown_inliers, 0],
        shown[shown_inliers, 1],
        shown[shown_inliers, 2],
        s=5,
        c="#2563eb",
        alpha=0.58,
        depthshade=False,
        label="inliers",
    )

    xx, yy, zz = _cylinder_surface(model, surface_resolution)
    ax.plot_surface(
        xx,
        yy,
        zz,
        color="#f97316",
        alpha=surface_alpha,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    start = model.start_point
    end = model.end_point
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color="#111827",
        linewidth=3,
        label="axis",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title or f"Cylinder fit: radius={model.radius:.4g}, RMSE={model.rmse:.4g}")
    ax.legend(loc="upper left")
    _set_axes_equal(ax, pts)
    fig.tight_layout()
    return fig, ax


def _cylinder_surface(model: CylinderModel, resolution: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_theta, n_height = resolution
    axis = model.axis_direction
    u, v = _orthonormal_basis(axis)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    t = np.linspace(model.height_min, model.height_max, n_height)
    theta_grid, t_grid = np.meshgrid(theta, t)

    center = model.axis_point + t_grid[..., None] * axis
    surface = center
    surface = surface + model.radius * np.cos(theta_grid)[..., None] * u
    surface = surface + model.radius * np.sin(theta_grid)[..., None] * v
    return surface[..., 0], surface[..., 1], surface[..., 2]


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = helper - float(helper @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-9)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
