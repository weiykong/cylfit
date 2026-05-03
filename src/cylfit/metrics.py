from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import CylinderModel
from .synthetic import SyntheticCylinder


@dataclass(frozen=True)
class FitMetrics:
    radius_error: float
    axis_angle_deg: float
    height_error: float
    rmse: float
    mae: float
    inlier_precision: Optional[float]
    inlier_recall: Optional[float]

    def to_dict(self) -> dict:
        return {
            "radius_error": self.radius_error,
            "axis_angle_deg": self.axis_angle_deg,
            "height_error": self.height_error,
            "rmse": self.rmse,
            "mae": self.mae,
            "inlier_precision": self.inlier_precision,
            "inlier_recall": self.inlier_recall,
        }


@dataclass(frozen=True)
class FitMeasures:
    points: int
    inliers: int
    inlier_fraction: float
    radius: float
    height: float
    rmse: float
    mae: float
    residual_p50: float
    residual_p90: float
    residual_p95: float
    residual_p99: float
    max_abs_residual: float
    angular_coverage_deg: float
    axial_coverage: float
    points_per_area: float
    quality_score: float

    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "inliers": self.inliers,
            "inlier_fraction": self.inlier_fraction,
            "radius": self.radius,
            "height": self.height,
            "rmse": self.rmse,
            "mae": self.mae,
            "residual_p50": self.residual_p50,
            "residual_p90": self.residual_p90,
            "residual_p95": self.residual_p95,
            "residual_p99": self.residual_p99,
            "max_abs_residual": self.max_abs_residual,
            "angular_coverage_deg": self.angular_coverage_deg,
            "axial_coverage": self.axial_coverage,
            "points_per_area": self.points_per_area,
            "quality_score": self.quality_score,
        }


def evaluate_fit(model: CylinderModel, truth: SyntheticCylinder) -> FitMetrics:
    """Evaluate a fitted model against synthetic ground truth."""

    dot = abs(float(np.dot(model.axis_direction, truth.axis_direction)))
    dot = min(1.0, max(-1.0, dot))
    pred = model.inlier_mask
    actual = truth.inlier_mask
    tp = int(np.sum(pred & actual))
    fp = int(np.sum(pred & ~actual))
    fn = int(np.sum(~pred & actual))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return FitMetrics(
        radius_error=abs(model.radius - truth.radius),
        axis_angle_deg=float(np.degrees(np.arccos(dot))),
        height_error=abs(model.height - truth.height),
        rmse=model.rmse,
        mae=model.mae,
        inlier_precision=precision,
        inlier_recall=recall,
    )


def measure_fit(points: np.ndarray, model: CylinderModel) -> FitMeasures:
    """Compute model diagnostics that do not require ground truth."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    mask = model.inlier_mask
    residuals = np.abs(model.residuals[mask])
    if residuals.size == 0:
        residuals = np.array([float("nan")])
    inlier_points = pts[mask] if mask.any() else pts
    angular_coverage = _angular_coverage_deg(inlier_points, model)
    axial_coverage = _axial_coverage(inlier_points, model)
    surface_area = max(2.0 * np.pi * model.radius * model.height, 1e-12)
    density = int(mask.sum()) / surface_area
    normalized_rmse = model.rmse / max(model.radius, 1e-12)
    coverage_factor = min(1.0, angular_coverage / 360.0) * min(1.0, axial_coverage)
    inlier_fraction = float(np.mean(mask))
    quality = 100.0 * coverage_factor * inlier_fraction / (1.0 + 25.0 * normalized_rmse)

    return FitMeasures(
        points=int(mask.size),
        inliers=int(mask.sum()),
        inlier_fraction=inlier_fraction,
        radius=model.radius,
        height=model.height,
        rmse=model.rmse,
        mae=model.mae,
        residual_p50=float(np.percentile(residuals, 50)),
        residual_p90=float(np.percentile(residuals, 90)),
        residual_p95=float(np.percentile(residuals, 95)),
        residual_p99=float(np.percentile(residuals, 99)),
        max_abs_residual=float(np.max(residuals)),
        angular_coverage_deg=angular_coverage,
        axial_coverage=axial_coverage,
        points_per_area=float(density),
        quality_score=float(np.clip(quality, 0.0, 100.0)),
    )


def _angular_coverage_deg(points: np.ndarray, model: CylinderModel) -> float:
    if points.shape[0] < 2:
        return 0.0
    u, v = _orthonormal_basis(model.axis_direction)
    centered = points - model.axis_point
    x = centered @ u
    y = centered @ v
    angles = np.sort(np.mod(np.arctan2(y, x), 2.0 * np.pi))
    if angles.size < 2:
        return 0.0
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    coverage = 2.0 * np.pi - float(np.max(gaps))
    return float(np.degrees(coverage))


def _axial_coverage(points: np.ndarray, model: CylinderModel) -> float:
    if points.shape[0] < 2 or model.height <= 1e-12:
        return 0.0
    t = (points - model.axis_point) @ model.axis_direction
    occupied = float(np.percentile(t, 99) - np.percentile(t, 1))
    return max(0.0, min(1.0, occupied / model.height))


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = helper - float(helper @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v
