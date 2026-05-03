from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import CylinderModel, fit_cylinder


@dataclass(frozen=True)
class PreprocessResult:
    points: np.ndarray
    original_indices: np.ndarray
    weights: Optional[np.ndarray]


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    *,
    return_weights: bool = True,
) -> PreprocessResult:
    """Downsample points by voxel centroid while preserving source indices."""

    pts = _validate_array(points)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.size, 3), dtype=float)
    np.add.at(sums, inverse, pts)
    centroids = sums / counts[:, None]
    first_indices = np.full(counts.size, -1, dtype=int)
    for idx, group in enumerate(inverse):
        if first_indices[group] < 0:
            first_indices[group] = idx
    weights = counts.astype(float) if return_weights else None
    return PreprocessResult(points=centroids, original_indices=first_indices, weights=weights)


def robust_spatial_trim(
    points: np.ndarray,
    *,
    zscore: float = 6.0,
) -> PreprocessResult:
    """Remove points far from the robust coordinate-wise median."""

    pts = _validate_array(points)
    if zscore <= 0:
        raise ValueError("zscore must be positive")
    center = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - center), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-12)
    keep = np.all(np.abs((pts - center) / scale) <= zscore, axis=1)
    return PreprocessResult(points=pts[keep], original_indices=np.flatnonzero(keep), weights=None)


def fit_cylinder_auto(
    points: np.ndarray,
    *,
    voxel_size: Optional[float] = None,
    trim_zscore: Optional[float] = 8.0,
    threshold: Optional[float] = None,
    use_voxel_weights: bool = True,
    **fit_kwargs,
) -> CylinderModel:
    """Preprocess then fit, while returning residuals on the original points."""

    pts = _validate_array(points)
    work = PreprocessResult(points=pts, original_indices=np.arange(pts.shape[0]), weights=None)
    if trim_zscore is not None:
        work = robust_spatial_trim(work.points, zscore=trim_zscore)
    if voxel_size is not None:
        work = voxel_downsample(work.points, voxel_size=voxel_size, return_weights=use_voxel_weights)

    kwargs = dict(fit_kwargs)
    if use_voxel_weights and work.weights is not None and "point_weights" not in kwargs:
        kwargs["point_weights"] = work.weights
    model = fit_cylinder(work.points, threshold=threshold, **kwargs)

    residuals = _residuals_original(pts, model)
    inliers = np.abs(residuals) <= (threshold if threshold is not None else max(model.rmse * 3.0, 1e-9))
    t = (pts - model.axis_point) @ model.axis_direction
    t_fit = t[inliers] if inliers.any() else t
    return CylinderModel(
        axis_point=model.axis_point,
        axis_direction=model.axis_direction,
        radius=model.radius,
        height_min=float(np.min(t_fit)),
        height_max=float(np.max(t_fit)),
        inlier_mask=inliers,
        residuals=residuals,
        iterations=model.iterations,
        converged=model.converged,
    )


def _residuals_original(points: np.ndarray, model: CylinderModel) -> np.ndarray:
    centered = points - model.axis_point
    axial = centered @ model.axis_direction
    radial = centered - axial[:, None] * model.axis_direction
    return np.linalg.norm(radial, axis=1) - model.radius


def _validate_array(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 8:
        raise ValueError("at least 8 finite points are required")
    return np.ascontiguousarray(pts)
