from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import CylinderModel, fit_cylinder


@dataclass(frozen=True)
class FitUncertainty:
    samples: int
    radius_mean: float
    radius_std: float
    radius_ci95: tuple[float, float]
    height_mean: float
    height_std: float
    axis_angle_std_deg: float
    axis_angle_ci95_deg: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "radius_mean": self.radius_mean,
            "radius_std": self.radius_std,
            "radius_ci95": list(self.radius_ci95),
            "height_mean": self.height_mean,
            "height_std": self.height_std,
            "axis_angle_std_deg": self.axis_angle_std_deg,
            "axis_angle_ci95_deg": list(self.axis_angle_ci95_deg),
        }


def estimate_uncertainty(
    points: np.ndarray,
    model: CylinderModel,
    *,
    samples: int = 64,
    sample_fraction: float = 0.75,
    threshold: Optional[float] = None,
    random_state: Optional[int] = 0,
    **fit_kwargs,
) -> FitUncertainty:
    """Estimate fit uncertainty by bootstrap refitting inliers."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")

    inlier_points = pts[model.inlier_mask]
    if inlier_points.shape[0] < 8:
        raise ValueError("model has too few inliers for uncertainty estimation")

    rng = np.random.default_rng(random_state)
    n = max(8, int(round(inlier_points.shape[0] * sample_fraction)))
    radii = []
    heights = []
    axis_angles = []
    kwargs = {
        "threshold": threshold if threshold is not None else max(3.0 * model.rmse, 1e-9),
        "ransac_trials": 0,
        "initial_axis": model.axis_direction,
        "random_state": random_state,
        **fit_kwargs,
    }

    for i in range(samples):
        idx = rng.choice(inlier_points.shape[0], n, replace=True)
        try:
            sample_model = fit_cylinder(inlier_points[idx], **{**kwargs, "random_state": int(rng.integers(0, 2**31 - 1))})
        except Exception:
            continue
        radii.append(sample_model.radius)
        heights.append(sample_model.height)
        dot = abs(float(np.dot(model.axis_direction, sample_model.axis_direction)))
        dot = min(1.0, max(-1.0, dot))
        axis_angles.append(float(np.degrees(np.arccos(dot))))

    if len(radii) < 2:
        raise RuntimeError("not enough successful bootstrap fits")

    radii_arr = np.asarray(radii)
    heights_arr = np.asarray(heights)
    angles_arr = np.asarray(axis_angles)
    return FitUncertainty(
        samples=int(radii_arr.size),
        radius_mean=float(np.mean(radii_arr)),
        radius_std=float(np.std(radii_arr, ddof=1)),
        radius_ci95=(float(np.percentile(radii_arr, 2.5)), float(np.percentile(radii_arr, 97.5))),
        height_mean=float(np.mean(heights_arr)),
        height_std=float(np.std(heights_arr, ddof=1)),
        axis_angle_std_deg=float(np.std(angles_arr, ddof=1)),
        axis_angle_ci95_deg=(float(np.percentile(angles_arr, 2.5)), float(np.percentile(angles_arr, 97.5))),
    )
