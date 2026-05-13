"""Elliptical cylinder fitting.

An elliptical cylinder relaxes the circular-cross-section assumption: instead of
a single radius *R* the cross-section is an ellipse with semi-major axis *a* and
semi-minor axis *b* (a ≥ b > 0) rotated by angle *theta* in the plane
perpendicular to the cylinder axis.

The fitting pipeline is:

1. **Axis estimation** — reuse the circular-cylinder RANSAC/PCA initialiser to
   obtain a good axis direction.  The axis is not altered by the ellipse fit.

2. **2-D ellipse fit** — project all inlier points onto the cross-section plane
   and fit a general conic (Fitzgibbon direct method, generalised-eigenvalue
   formulation) subject to the ellipse constraint 4ac − b² = 1.

3. **Parameter extraction** — convert the algebraic conic to the geometric
   (semi-axes, centre, angle) form via eigendecomposition.

4. **LM refinement** — iteratively re-weight to suppress outliers and re-fit the
   2-D ellipse on the inlier set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import (
    _auto_threshold,
    _orthonormal_basis,
    _ransac_or_pca_initial,
    _validate_points,
    residuals_to_cylinder,
)

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EllipticalCylinderModel:
    """Fitted elliptical cylinder.

    The cross-section in the plane perpendicular to *axis_direction* is an
    ellipse with semi-axes *semi_major* ≥ *semi_minor* and orientation
    *cross_section_angle* (radians, measured from the first basis vector of the
    cross-section plane).
    """

    axis_point: np.ndarray
    axis_direction: np.ndarray
    semi_major: float
    semi_minor: float
    cross_section_angle: float
    height_min: float
    height_max: float
    inlier_mask: np.ndarray
    residuals: np.ndarray

    @property
    def height(self) -> float:
        return float(self.height_max - self.height_min)

    @property
    def aspect_ratio(self) -> float:
        """a / b — 1.0 means perfectly circular."""
        return self.semi_major / max(self.semi_minor, 1e-15)

    @property
    def equivalent_radius(self) -> float:
        """Geometric-mean radius √(a·b); equals the circular radius when a = b."""
        return float(np.sqrt(self.semi_major * self.semi_minor))

    @property
    def rmse(self) -> float:
        r = self.residuals[self.inlier_mask]
        return float(np.sqrt(np.mean(r * r))) if r.size else float("nan")

    @property
    def start_point(self) -> np.ndarray:
        return self.axis_point + self.height_min * self.axis_direction

    @property
    def end_point(self) -> np.ndarray:
        return self.axis_point + self.height_max * self.axis_direction

    def to_dict(self) -> dict:
        return {
            "axis_point": self.axis_point.tolist(),
            "axis_direction": self.axis_direction.tolist(),
            "semi_major": self.semi_major,
            "semi_minor": self.semi_minor,
            "aspect_ratio": self.aspect_ratio,
            "equivalent_radius": self.equivalent_radius,
            "cross_section_angle_deg": float(np.degrees(self.cross_section_angle)),
            "height_min": self.height_min,
            "height_max": self.height_max,
            "height": self.height,
            "inliers": int(self.inlier_mask.sum()),
            "points": int(self.inlier_mask.size),
            "rmse": self.rmse,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Public fitting function
# ---------------------------------------------------------------------------

def fit_elliptical_cylinder(
    points: ArrayLike,
    *,
    threshold: Optional[float] = None,
    ransac_trials: int = 128,
    sample_size: int = 64,
    max_iterations: int = 6,
    initial_axis: Optional[ArrayLike] = None,
    ransac_stop_inlier_fraction: Optional[float] = 0.98,
    random_state: Optional[int] = None,
) -> EllipticalCylinderModel:
    """Fit an elliptical cylinder to a 3-D point cloud.

    Parameters
    ----------
    points:
        ``N × 3`` input point cloud.
    threshold:
        Inlier distance threshold (in the same units as the point cloud).
        Auto-estimated from the data when omitted.
    ransac_trials:
        RANSAC trials for the initial circular-axis estimate.
    sample_size:
        Points per RANSAC sample.
    max_iterations:
        Iterative-reweighting passes on the 2-D ellipse fit.
    initial_axis:
        Optional known axis direction to skip RANSAC.
    ransac_stop_inlier_fraction:
        Early-stop threshold for the RANSAC axis search.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    EllipticalCylinderModel
    """
    pts = _validate_points(points)
    rng = np.random.default_rng(random_state)

    # --- Step 1: axis estimation via circular RANSAC -------------------------
    p0, axis, r_circ = _ransac_or_pca_initial(
        pts,
        threshold=threshold,
        trials=max(0, int(ransac_trials)),
        sample_size=max(8, int(sample_size)),
        known_radius=None,
        initial_axis=initial_axis,
        orientation_axis=None,
        min_axis_cos=None,
        stop_inlier_fraction=ransac_stop_inlier_fraction,
        rng=rng,
    )

    circ_res = residuals_to_cylinder(pts, p0, axis, r_circ)
    if threshold is None:
        threshold = _auto_threshold(circ_res)

    inlier_mask = np.abs(circ_res) <= threshold
    if inlier_mask.sum() < 8:
        inlier_mask = np.ones(pts.shape[0], dtype=bool)

    # Cross-section basis vectors perpendicular to axis
    u_vec, v_vec = _orthonormal_basis(axis)

    # Fallback (circular) values in case algebraic fit cannot converge
    semi_a = float(r_circ)
    semi_b = float(r_circ)
    angle = 0.0

    # --- Steps 2–4: iterative 2-D ellipse fit --------------------------------
    for _ in range(max(1, int(max_iterations))):
        inlier_pts = pts[inlier_mask]
        centered = inlier_pts - p0
        u = centered @ u_vec
        v = centered @ v_vec

        result = _fit_ellipse_algebraic(u, v)
        if result is None:
            break
        cx, cy, semi_a, semi_b, angle = result

        # Update axis anchor to include ellipse centre offset
        p0 = p0 + cx * u_vec + cy * v_vec

        # Recompute residuals with elliptic distance
        res = _ellipse_residuals(pts, p0, axis, u_vec, v_vec, semi_a, semi_b, angle)
        if threshold is None:
            threshold = _auto_threshold(res)
        new_mask = np.abs(res) <= threshold
        if new_mask.sum() < 8:
            break
        inlier_mask = new_mask

    # Final residuals on all points
    res = _ellipse_residuals(pts, p0, axis, u_vec, v_vec, semi_a, semi_b, angle)
    inlier_mask = np.abs(res) <= threshold

    t = (pts - p0) @ axis
    t_fit = t[inlier_mask] if inlier_mask.any() else t

    return EllipticalCylinderModel(
        axis_point=p0,
        axis_direction=axis,
        semi_major=float(semi_a),
        semi_minor=float(semi_b),
        cross_section_angle=float(angle),
        height_min=float(np.min(t_fit)),
        height_max=float(np.max(t_fit)),
        inlier_mask=inlier_mask,
        residuals=res,
    )


# ---------------------------------------------------------------------------
# 2-D algebraic ellipse fit  (Fitzgibbon 1999, direct method)
# ---------------------------------------------------------------------------

def _fit_ellipse_algebraic(
    u: np.ndarray,
    v: np.ndarray,
) -> Optional[tuple[float, float, float, float, float]]:
    """Direct algebraic ellipse fit to 2-D points (u, v).

    Returns ``(cx, cy, semi_major, semi_minor, angle_rad)`` or ``None`` when
    the fit is degenerate (too few points or non-elliptic conic).

    Uses the Fitzgibbon / Bookstein constraint 4ac − b² = 1 which guarantees
    the fitted conic is an ellipse rather than a hyperbola or parabola.
    """
    if len(u) < 6:
        return None

    # Design matrix: [u², uv, v², u, v, 1]
    D = np.column_stack([u * u, u * v, v * v, u, v, np.ones(len(u))])

    # Scatter matrix
    S = D.T @ D

    # Constraint matrix enforcing 4ac − b² = 1
    C = np.zeros((6, 6))
    C[0, 2] = 2.0
    C[2, 0] = 2.0
    C[1, 1] = -1.0

    # Solve generalised eigenproblem: S v = λ C v
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(S, C))
    except np.linalg.LinAlgError:
        return None

    # Select the eigenvector satisfying the ellipse constraint 4ac − b² > 0.
    # Work with real parts; use `np.real` throughout to handle small imaginary
    # artefacts from numerical noise.
    eigvals_r = np.real(eigvals)
    valid = np.isfinite(eigvals_r)
    if not valid.any():
        return None
    # Evaluate the constraint 4ac − b² for each candidate eigenvector
    best_idx, best_disc = -1, -np.inf
    for i in range(6):
        if not valid[i]:
            continue
        v = np.real(eigvecs[:, i])
        disc = 4 * v[0] * v[2] - v[1] ** 2
        if disc > 0 and disc > best_disc:
            best_disc = disc
            best_idx = i
    if best_idx < 0:
        return None
    coeffs = np.real(eigvecs[:, best_idx])
    A, B, C_coef, D_coef, E_coef, F_coef = coeffs

    # Normalise sign so A > 0 (the overall scale is arbitrary but sign matters
    # for the val_c < 0 check below).
    if A < 0:
        coeffs = -coeffs
        A, B, C_coef, D_coef, E_coef, F_coef = coeffs

    # Verify ellipse condition (should be guaranteed by selection above)
    if B * B - 4 * A * C_coef >= 0:
        return None

    # Convert to geometric form via centre + eigendecomposition
    # Centre solves: [2A  B][cx]   [-D]
    #                [B  2C][cy] = [-E]
    M = np.array([[2 * A, B], [B, 2 * C_coef]])
    try:
        centre = np.linalg.solve(M, np.array([-D_coef, -E_coef]))
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(centre[0]), float(centre[1])

    # Value of conic at centre (needed to normalise semi-axes)
    val_c = (A * cx * cx + B * cx * cy + C_coef * cy * cy
             + D_coef * cx + E_coef * cy + F_coef)

    Q = np.array([[A, B / 2.0], [B / 2.0, C_coef]])
    eigvals_q, eigvecs_q = np.linalg.eigh(Q)

    # Semi-axes: aᵢ = sqrt(−val_c / λᵢ)
    if val_c >= 0 or np.any(eigvals_q <= 0):
        return None
    semi_sq = -val_c / eigvals_q
    if np.any(semi_sq <= 0):
        return None
    semi_axes = np.sqrt(semi_sq)

    # Largest eigenvalue of Q → smallest semi-axis (tightest curvature)
    order = np.argsort(-semi_axes)          # descending → [major, minor]
    semi_axes = semi_axes[order]
    eigvecs_q = eigvecs_q[:, order]

    # Angle of major axis w.r.t. u basis
    angle = float(np.arctan2(float(eigvecs_q[1, 0]), float(eigvecs_q[0, 0])))

    return cx, cy, float(semi_axes[0]), float(semi_axes[1]), angle


# ---------------------------------------------------------------------------
# Elliptic distance residuals
# ---------------------------------------------------------------------------

def _ellipse_residuals(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    semi_major: float,
    semi_minor: float,
    angle: float,
) -> np.ndarray:
    """Approximate signed radial residuals for an elliptical cylinder.

    The residual is the algebraic distance in the normalised ellipse frame:
    ``sqrt((u'/a)² + (v'/b)²) − 1``, where (u', v') are coordinates aligned
    with the ellipse axes.  This is not the exact geometric point-to-ellipse
    distance but is smooth and differentiable everywhere except at the origin.
    """
    centered = points - axis_point
    u = centered @ u_vec
    v = centered @ v_vec

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    u_rot = cos_a * u + sin_a * v
    v_rot = -sin_a * u + cos_a * v

    a = max(semi_major, 1e-12)
    b = max(semi_minor, 1e-12)
    norm_dist = np.sqrt((u_rot / a) ** 2 + (v_rot / b) ** 2)
    # Scale to geometric-mean radius so residuals are comparable to circular
    geom_r = np.sqrt(a * b)
    return geom_r * (norm_dist - 1.0)
