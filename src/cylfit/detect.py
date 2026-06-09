from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import CylinderModel, fit_cylinder

try:
    from scipy.spatial import KDTree as _KDTree  # type: ignore[import]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


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
    detections = []

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
    """Group points into Euclidean connected components.

    Uses a KD-tree (via scipy) when available for O(N log N) performance.
    Falls back to a vectorised numpy BFS that is O(N²) in memory but avoids
    the O(N²) *Python-loop* overhead of rebuilding the candidate list on every
    BFS pop.
    """
    if _HAS_SCIPY:
        return _connected_components_kdtree(points, eps)
    return _connected_components_numpy(points, eps)


def _connected_components_kdtree(points: np.ndarray, eps: float) -> list[np.ndarray]:
    tree = _KDTree(points)
    # query_ball_tree builds the full neighbor list in one C call
    neighbor_lists = tree.query_ball_tree(tree, r=eps)
    n = points.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            idx = queue.pop()
            component.append(idx)
            for nb in neighbor_lists[idx]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(np.asarray(component, dtype=int))
    components.sort(key=lambda c: c.size, reverse=True)
    return components


def _connected_components_numpy(points: np.ndarray, eps: float) -> list[np.ndarray]:
    """Pure-numpy fallback. Avoids re-scanning all points on each BFS pop by
    maintaining a visited mask and scanning only unvisited points per step."""
    n = points.shape[0]
    eps2 = eps * eps
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            idx = queue.pop()
            component.append(idx)
            unvisited = np.flatnonzero(~visited)
            if unvisited.size == 0:
                continue
            delta = points[unvisited] - points[idx]
            close = unvisited[np.einsum("ij,ij->i", delta, delta) <= eps2]
            visited[close] = True
            queue.extend(close.tolist())
        components.append(np.asarray(component, dtype=int))
    components.sort(key=lambda c: c.size, reverse=True)
    return components
