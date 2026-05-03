from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SyntheticCylinder:
    points: np.ndarray
    clean_points: np.ndarray
    normals: np.ndarray
    axis_point: np.ndarray
    axis_direction: np.ndarray
    radius: float
    height: float
    inlier_mask: np.ndarray


def generate_noisy_cylinder(
    *,
    n_points: int = 5000,
    radius: float = 1.5,
    height: float = 8.0,
    axis_point: Optional[np.ndarray] = None,
    axis_direction: Optional[np.ndarray] = None,
    noise: float = 0.02,
    outlier_fraction: float = 0.1,
    partial_arc: float = 1.0,
    random_state: Optional[int] = 7,
) -> SyntheticCylinder:
    """Generate a noisy synthetic cylinder point cloud.

    ``partial_arc`` is the fraction of the full circumference to sample. Use a
    value such as ``0.35`` to create a partial scan.
    """

    if n_points < 8:
        raise ValueError("n_points must be at least 8")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if height <= 0:
        raise ValueError("height must be positive")
    if not 0 < partial_arc <= 1:
        raise ValueError("partial_arc must be in the interval (0, 1]")
    if not 0 <= outlier_fraction < 1:
        raise ValueError("outlier_fraction must be in the interval [0, 1)")

    rng = np.random.default_rng(random_state)
    p0 = np.zeros(3) if axis_point is None else np.asarray(axis_point, dtype=float).reshape(3)
    axis = np.array([0.35, -0.42, 0.837], dtype=float) if axis_direction is None else np.asarray(axis_direction, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    u, v = _orthonormal_basis(axis)

    n_outliers = int(round(n_points * outlier_fraction))
    n_inliers = n_points - n_outliers
    theta0 = rng.uniform(0.0, 2.0 * np.pi)
    theta = theta0 + rng.uniform(0.0, 2.0 * np.pi * partial_arc, n_inliers)
    t = rng.uniform(-0.5 * height, 0.5 * height, n_inliers)

    clean = p0 + t[:, None] * axis
    normals = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    clean = clean + radius * normals
    noisy = clean + rng.normal(scale=noise, size=clean.shape)

    if n_outliers:
        span = max(radius * 3.5, height * 0.65)
        outliers = _surface_rejecting_outliers(
            rng,
            p0,
            axis,
            radius,
            n_outliers,
            span,
            min_gap=max(4.0 * noise, 0.08 * radius),
        )
        points = np.vstack([noisy, outliers])
        clean_points = np.vstack([clean, outliers])
        outlier_normals = rng.normal(size=(n_outliers, 3))
        outlier_normals /= np.linalg.norm(outlier_normals, axis=1)[:, None]
        normals_all = np.vstack([normals, outlier_normals])
        inlier_mask = np.r_[np.ones(n_inliers, dtype=bool), np.zeros(n_outliers, dtype=bool)]
    else:
        points = noisy
        clean_points = clean
        normals_all = normals
        inlier_mask = np.ones(n_inliers, dtype=bool)

    order = rng.permutation(points.shape[0])
    return SyntheticCylinder(
        points=points[order],
        clean_points=clean_points[order],
        normals=normals_all[order],
        axis_point=p0,
        axis_direction=axis,
        radius=float(radius),
        height=float(height),
        inlier_mask=inlier_mask[order],
    )


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = helper - float(helper @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _surface_rejecting_outliers(
    rng: np.random.Generator,
    p0: np.ndarray,
    axis: np.ndarray,
    radius: float,
    n_outliers: int,
    span: float,
    min_gap: float,
) -> np.ndarray:
    outliers = []
    while sum(chunk.shape[0] for chunk in outliers) < n_outliers:
        candidate = p0 + rng.uniform(-span, span, size=(max(256, n_outliers), 3))
        centered = candidate - p0
        axial = centered @ axis
        radial = centered - axial[:, None] * axis
        radial_distance = np.linalg.norm(radial, axis=1)
        keep = np.abs(radial_distance - radius) >= min_gap
        outliers.append(candidate[keep])

    return np.vstack(outliers)[:n_outliers]
