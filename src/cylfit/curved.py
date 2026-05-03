"""Curved / bent cylinder fitting.

Real-world pipes, hoses, conduits, and tree trunks are not perfectly straight.
This module fits a **curved cylinder** whose axis is a piecewise-linear spline
(the *spine*) rather than a single infinite line.

Algorithm
---------
1. **Global circular fit** — :func:`~cylfit.fit_cylinder` estimates a
   global axis and radius to initialise the spine coordinate system.
2. **Axial partitioning** — points are projected onto the global axis and split
   into *n_segments* overlapping windows.
3. **Per-segment 2-D circle fit** — each window's points are projected onto the
   plane perpendicular to the global axis and a 2-D algebraic circle fit finds
   the local spine centre and radius.
4. **Spine smoothing** — optional Gaussian smoothing of the spine to suppress
   sampling noise between segments.
5. **Residuals** — for each point the signed distance to the nearest spine
   segment surface is computed; points within *threshold* are marked inliers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import (
    _auto_threshold,
    _axis_distances,
    _normalize,
    _orthonormal_basis,
    _pca_initial,
    _validate_points,
    fit_cylinder,
)

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurvedCylinderModel:
    """A cylinder with a piecewise-linear (spine) axis.

    Attributes
    ----------
    spine:
        ``K × 3`` array of spine control points.  The spine axis at segment
        *k* is the unit vector from ``spine[k]`` to ``spine[k+1]``.
    radius:
        Fitted radius (constant along the spine; mean of per-segment radii).
    segment_radii:
        Per-segment radii (length K−1).
    inlier_mask:
        Boolean array marking inlier points.
    residuals:
        Signed radial residuals for all N points (distance to nearest spine
        surface minus radius).
    n_segments:
        Number of segments (= K−1).
    """

    spine: np.ndarray
    radius: float
    segment_radii: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray
    n_segments: int

    @property
    def total_length(self) -> float:
        """Arc length of the spine polyline."""
        diffs = np.diff(self.spine, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def curvature_mean(self) -> float:
        """Mean turning angle per unit length (rad / unit), a proxy for curvature."""
        if self.n_segments < 2:
            return 0.0
        dirs = np.diff(self.spine, axis=0)
        lengths = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.where(lengths > 1e-15, lengths, 1e-15)
        dots = np.einsum("ij,ij->i", dirs[:-1], dirs[1:])
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        arc = self.total_length
        return float(np.sum(angles) / arc) if arc > 1e-15 else 0.0

    @property
    def rmse(self) -> float:
        r = self.residuals[self.inlier_mask]
        return float(np.sqrt(np.mean(r * r))) if r.size else float("nan")

    def to_dict(self) -> dict:
        return {
            "spine": self.spine.tolist(),
            "radius": self.radius,
            "segment_radii": self.segment_radii.tolist(),
            "total_length": self.total_length,
            "curvature_mean": self.curvature_mean,
            "n_segments": self.n_segments,
            "inliers": int(self.inlier_mask.sum()),
            "points": int(self.inlier_mask.size),
            "rmse": self.rmse,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Public fitting function
# ---------------------------------------------------------------------------

def fit_curved_cylinder(
    points: ArrayLike,
    *,
    n_segments: int = 8,
    overlap: float = 0.5,
    smooth_sigma: float = 1.0,
    threshold: Optional[float] = None,
    ransac_trials: int = 64,
    sample_size: int = 64,
    random_state: Optional[int] = None,
) -> CurvedCylinderModel:
    """Fit a curved (bent) cylinder to a 3-D point cloud.

    Parameters
    ----------
    points:
        ``N × 3`` input point cloud.
    n_segments:
        Number of spine segments.  More segments capture tighter bends but
        require more data per segment.
    overlap:
        Fractional overlap between adjacent windows (0 = no overlap, 0.5 = 50%).
        Higher overlap makes the spine smoother.
    smooth_sigma:
        Gaussian smoothing kernel width in *segment units*.  ``0`` disables
        smoothing.  Values around 1–2 are typical.
    threshold:
        Inlier threshold; auto-estimated when ``None``.
    ransac_trials:
        Trials for the initial global circular fit.
    sample_size:
        Points per RANSAC sample.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    CurvedCylinderModel
    """
    pts = _validate_points(points)
    rng = np.random.default_rng(random_state)
    n_segs = max(2, int(n_segments))

    # --- Global circular fit -------------------------------------------------
    global_model = fit_cylinder(
        pts,
        threshold=threshold,
        ransac_trials=ransac_trials,
        sample_size=sample_size,
        random_state=random_state,
    )
    global_axis = global_model.axis_direction
    global_p0 = global_model.axis_point
    if threshold is None:
        threshold = _auto_threshold(global_model.residuals)

    u_vec, v_vec = _orthonormal_basis(global_axis)

    # Project all points onto the global axis
    t_all = (pts - global_p0) @ global_axis

    t_min, t_max = float(t_all.min()), float(t_all.max())
    span = t_max - t_min
    if span < 1e-12:
        raise ValueError("Points are coplanar: cannot compute axial extent.")

    # --- Per-segment window fitting ------------------------------------------
    seg_step = span / n_segs
    half_win = seg_step * (0.5 + overlap * 0.5)  # half-window radius

    spine_pts: list[np.ndarray] = []
    seg_radii: list[float] = []
    seg_t_centres: list[float] = []

    for k in range(n_segs + 1):    # +1 so the last spine point is at t_max
        t_centre = t_min + k * seg_step
        mask = (t_all >= t_centre - half_win) & (t_all <= t_centre + half_win)
        if mask.sum() < 6:
            # Fall back to nearest points
            dists = np.abs(t_all - t_centre)
            mask = np.zeros(len(pts), dtype=bool)
            mask[np.argsort(dists)[:max(6, len(pts) // (n_segs + 1))]] = True

        seg_pts = pts[mask]
        centered = seg_pts - global_p0
        u = centered @ u_vec
        v = centered @ v_vec

        cx, cy, r = _fit_circle_2d(u, v)
        t_mean = float(t_all[mask].mean())
        spine_3d = global_p0 + t_mean * global_axis + cx * u_vec + cy * v_vec
        spine_pts.append(spine_3d)
        seg_radii.append(max(r, 1e-12))
        seg_t_centres.append(t_mean)

    spine = np.array(spine_pts)

    # --- Optional Gaussian smoothing of spine --------------------------------
    if smooth_sigma > 0 and n_segs >= 2:
        spine = _smooth_spine(spine, sigma=float(smooth_sigma))

    seg_radii_arr = np.array(seg_radii[: n_segs], dtype=float)  # K−1 radii
    mean_radius = float(np.median(seg_radii_arr))

    # --- Residuals -----------------------------------------------------------
    res = _curved_cylinder_residuals(pts, spine, mean_radius)
    if threshold is None:
        threshold = _auto_threshold(res)
    inlier_mask = np.abs(res) <= threshold

    return CurvedCylinderModel(
        spine=spine,
        radius=mean_radius,
        segment_radii=seg_radii_arr,
        inlier_mask=inlier_mask,
        residuals=res,
        n_segments=n_segs,
    )


# ---------------------------------------------------------------------------
# 2-D algebraic circle fit
# ---------------------------------------------------------------------------

def _fit_circle_2d(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit: minimise ‖(u−cx)²+(v−cy)²−r²‖.

    Returns (cx, cy, radius).
    """
    # Rewrite (u−cx)²+(v−cy)²=r² as  2cx·u + 2cy·v + c = u²+v²
    # where c = r²−cx²−cy².  Solve the linear system for [cx, cy, c].
    D = u * u + v * v
    A = np.column_stack([2 * u, 2 * v, np.ones(len(u))])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, D, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, max(float(np.std(np.sqrt(u ** 2 + v ** 2))), 1e-12)
    cx, cy, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    r2 = c + cx * cx + cy * cy
    return cx, cy, float(np.sqrt(max(r2, 1e-24)))


# ---------------------------------------------------------------------------
# Spine smoothing
# ---------------------------------------------------------------------------

def _smooth_spine(spine: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to spine control points (interior only)."""
    n = len(spine)
    if n <= 2 or sigma <= 0:
        return spine
    out = spine.copy()
    kernel_radius = int(np.ceil(2.5 * sigma))
    ks = np.arange(-kernel_radius, kernel_radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (ks / sigma) ** 2)
    kernel /= kernel.sum()
    for dim in range(3):
        col = spine[:, dim]
        # Convolve with reflect-padding
        padded = np.pad(col, kernel_radius, mode="reflect")
        conv = np.convolve(padded, kernel, mode="valid")
        out[:, dim] = conv[:n]
    # Preserve endpoints
    out[0] = spine[0]
    out[-1] = spine[-1]
    return out


# ---------------------------------------------------------------------------
# Residuals for curved cylinder
# ---------------------------------------------------------------------------

def _curved_cylinder_residuals(
    points: np.ndarray,
    spine: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Signed residuals: for each point, distance to nearest spine segment − radius.

    The local axis at segment (k, k+1) is the unit tangent between spine[k]
    and spine[k+1].  The radial distance from a point to the segment is computed
    in the plane perpendicular to that tangent.
    """
    n = len(points)
    n_segs = len(spine) - 1
    best_dist = np.full(n, np.inf)

    for k in range(n_segs):
        p_start = spine[k]
        p_end = spine[k + 1]
        seg_vec = p_end - p_start
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-15:
            continue
        tangent = seg_vec / seg_len

        # Project points onto segment
        rel = points - p_start
        t_proj = np.clip(rel @ tangent, 0.0, seg_len)
        closest = p_start + t_proj[:, None] * tangent
        radial = np.linalg.norm(points - closest, axis=1)

        best_dist = np.minimum(best_dist, radial)

    return best_dist - radius
