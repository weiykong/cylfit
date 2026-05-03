from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import CylinderModel, fit_cylinder


@dataclass(frozen=True)
class DetectedCylinder:
    model: CylinderModel
    original_indices: np.ndarray


def detect_cylinders(
    points: np.ndarray,
    *,
    max_cylinders: int = 3,
    threshold: Optional[float] = None,
    min_inliers: int = 80,
    min_inlier_fraction: float = 0.05,
    cluster_eps: Optional[float] = None,
    random_state: Optional[int] = None,
    **fit_kwargs,
) -> list[DetectedCylinder]:
    """Detect multiple cylinders by repeated robust fitting and inlier removal.

    If ``cluster_eps`` is provided, Euclidean components are fitted separately
    first. This is useful for scenes with separated pipes or columns.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    if max_cylinders < 1:
        return []

    if cluster_eps is not None and cluster_eps > 0:
        detections = []
        for component in _connected_components(pts, cluster_eps):
            if component.size < min_inliers:
                continue
            model = fit_cylinder(
                pts[component],
                threshold=threshold,
                random_state=int(np.random.default_rng(random_state).integers(0, 2**31 - 1)),
                **fit_kwargs,
            )
            if int(model.inlier_mask.sum()) >= min_inliers:
                detections.append(DetectedCylinder(model=model, original_indices=component[model.inlier_mask]))
        detections.sort(key=lambda item: int(item.model.inlier_mask.sum()), reverse=True)
        return detections[:max_cylinders]

    rng = np.random.default_rng(random_state)
    remaining = np.arange(pts.shape[0])
    detections: list[DetectedCylinder] = []

    for _ in range(max_cylinders):
        if remaining.size < max(8, min_inliers):
            break
        model = fit_cylinder(
            pts[remaining],
            threshold=threshold,
            random_state=int(rng.integers(0, 2**31 - 1)),
            **fit_kwargs,
        )
        inlier_count = int(model.inlier_mask.sum())
        if inlier_count < min_inliers:
            break
        if inlier_count / pts.shape[0] < min_inlier_fraction:
            break
        detection_indices = remaining[model.inlier_mask]
        detections.append(DetectedCylinder(model=model, original_indices=detection_indices))
        remaining = remaining[~model.inlier_mask]

    return detections


def _connected_components(points: np.ndarray, eps: float) -> list[np.ndarray]:
    n = points.shape[0]
    remaining = np.ones(n, dtype=bool)
    components = []
    eps2 = eps * eps

    while remaining.any():
        start = int(np.flatnonzero(remaining)[0])
        queue = [start]
        remaining[start] = False
        component = []
        while queue:
            idx = queue.pop()
            component.append(idx)
            candidates = np.flatnonzero(remaining)
            if candidates.size == 0:
                continue
            delta = points[candidates] - points[idx]
            neighbors = candidates[np.einsum("ij,ij->i", delta, delta) <= eps2]
            if neighbors.size:
                remaining[neighbors] = False
                queue.extend(neighbors.tolist())
        components.append(np.asarray(component, dtype=int))

    components.sort(key=lambda item: item.size, reverse=True)
    return components
