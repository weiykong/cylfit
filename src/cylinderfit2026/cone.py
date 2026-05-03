"""Cone fitting with RANSAC initialisation and Levenberg-Marquardt refinement.

A cone is parametrised by its **apex** (3-D point), a unit **axis direction**
d̂ (pointing away from the apex into the cone), and a **half-angle** α ∈ (0°, 90°).

The signed radial residual for point pᵢ is::

    tᵢ = (pᵢ − apex) · d̂          (axial distance from apex, must be > 0)
    qᵢ = (pᵢ − apex) − tᵢ d̂       (radial vector, perpendicular to d̂)
    rᵢ = ‖qᵢ‖ − tᵢ tan α

**Analytic Jacobian** (7 parameters [apex; raw_d; α], evaluated at unit d̂)::

    ∂rᵢ/∂apex  =  −q̂ᵢ + tan α · d̂
    ∂rᵢ/∂d̂     =  −qᵢ (tᵢ/‖qᵢ‖ + tan α)        [at unit d̂]
    ∂rᵢ/∂α     =  −tᵢ / cos²α

**Initialisation.** For each RANSAC trial a random subset is drawn, the axis is
estimated by PCA (longest variance direction), radial distances r and axial
projections t are computed, and a least-squares line fit r = t·tan(α) through
the origin gives the half-angle.  The apex is estimated as the point on the axis
where the fitted line intersects r = 0.
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
)

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConeModel:
    """Fitted cone model.

    Attributes
    ----------
    apex:
        3-D apex (tip) of the cone.
    axis_direction:
        Unit vector pointing from the apex into the cone body.
    half_angle_deg:
        Half-opening angle in degrees.
    height_min, height_max:
        Axial extent of the fitted data relative to the apex.
    inlier_mask:
        Boolean array marking inlier points.
    residuals:
        Signed radial residuals ``‖qᵢ‖ − tᵢ tan α`` for all N points.
    iterations, converged:
        LM iteration count and convergence flag.
    """

    apex: np.ndarray
    axis_direction: np.ndarray
    half_angle_deg: float
    height_min: float
    height_max: float
    inlier_mask: np.ndarray
    residuals: np.ndarray
    iterations: int
    converged: bool

    @property
    def half_angle_rad(self) -> float:
        return float(np.deg2rad(self.half_angle_deg))

    @property
    def height(self) -> float:
        return float(self.height_max - self.height_min)

    @property
    def radius_at_base(self) -> float:
        """Radius of the cone cross-section at height_max from the apex."""
        return float(self.height_max * np.tan(self.half_angle_rad))

    @property
    def rmse(self) -> float:
        r = self.residuals[self.inlier_mask]
        return float(np.sqrt(np.mean(r * r))) if r.size else float("nan")

    def to_dict(self) -> dict:
        return {
            "apex": self.apex.tolist(),
            "axis_direction": self.axis_direction.tolist(),
            "half_angle_deg": self.half_angle_deg,
            "height_min": self.height_min,
            "height_max": self.height_max,
            "height": self.height,
            "radius_at_base": self.radius_at_base,
            "inliers": int(self.inlier_mask.sum()),
            "points": int(self.inlier_mask.size),
            "rmse": self.rmse,
            "iterations": self.iterations,
            "converged": self.converged,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Public fitting function
# ---------------------------------------------------------------------------

def fit_cone(
    points: ArrayLike,
    *,
    threshold: Optional[float] = None,
    ransac_trials: int = 128,
    sample_size: int = 64,
    max_iterations: int = 40,
    known_half_angle_deg: Optional[float] = None,
    initial_axis: Optional[ArrayLike] = None,
    ransac_stop_inlier_fraction: Optional[float] = 0.95,
    random_state: Optional[int] = None,
) -> ConeModel:
    """Fit a cone to a 3-D point cloud using RANSAC + LM with analytic Jacobian.

    Parameters
    ----------
    points:
        ``N × 3`` input point cloud.
    threshold:
        Inlier distance threshold; auto-estimated when ``None``.
    ransac_trials:
        Number of RANSAC trials for the initial estimate.
    sample_size:
        Points per RANSAC sample.
    max_iterations:
        Maximum LM iterations.
    known_half_angle_deg:
        Fix the half-angle; only apex and axis are then optimised.
    initial_axis:
        Seed axis direction (skips the PCA initialisation).
    ransac_stop_inlier_fraction:
        Fraction of inliers at which RANSAC stops early.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    ConeModel
    """
    pts = _validate_points(points)
    rng = np.random.default_rng(random_state)

    if known_half_angle_deg is not None:
        if not (0.0 < known_half_angle_deg < 90.0):
            raise ValueError("known_half_angle_deg must be in (0, 90)")

    # --- RANSAC initialisation -----------------------------------------------
    apex, axis, alpha = _cone_ransac_initial(
        pts,
        trials=max(0, int(ransac_trials)),
        sample_size=max(8, int(sample_size)),
        known_alpha=None if known_half_angle_deg is None
                    else float(np.deg2rad(known_half_angle_deg)),
        initial_axis=initial_axis,
        stop_inlier_fraction=ransac_stop_inlier_fraction,
        rng=rng,
    )

    res = _cone_residuals(pts, apex, axis, alpha)
    if threshold is None:
        threshold = _auto_threshold(res)

    inlier_mask = np.abs(res) <= threshold
    if inlier_mask.sum() < 8:
        inlier_mask = np.ones(pts.shape[0], dtype=bool)

    # --- LM refinement -------------------------------------------------------
    apex, axis, alpha, n_iter, converged = _refine_cone(
        pts[inlier_mask],
        apex,
        axis,
        alpha,
        max_iterations=max_iterations,
        known_alpha=None if known_half_angle_deg is None
                    else float(np.deg2rad(known_half_angle_deg)),
    )

    # Final residuals + inliers
    res = _cone_residuals(pts, apex, axis, alpha)
    inlier_mask = np.abs(res) <= threshold

    t = (pts - apex) @ axis
    t_fit = t[inlier_mask] if inlier_mask.any() else t

    return ConeModel(
        apex=apex,
        axis_direction=axis,
        half_angle_deg=float(np.degrees(alpha)),
        height_min=float(np.min(t_fit)),
        height_max=float(np.max(t_fit)),
        inlier_mask=inlier_mask,
        residuals=res,
        iterations=n_iter,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# RANSAC initialisation
# ---------------------------------------------------------------------------

def _cone_initial_from_subset(
    pts: np.ndarray,
    known_alpha: Optional[float],
    initial_axis: Optional[np.ndarray],
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Fit a cone to a small subset: PCA axis + linear regression for α."""
    try:
        if initial_axis is not None:
            axis = _normalize(np.asarray(initial_axis, dtype=float))
        else:
            _, axis, _ = _pca_initial(pts)
    except Exception:
        return None

    centroid = pts.mean(axis=0)
    c = pts - centroid
    t = c @ axis         # axial projections
    r = _axis_distances(pts, centroid, axis)  # radial distances

    if known_alpha is not None:
        alpha = float(known_alpha)
        # Estimate apex along axis: linear regression r = (t - t0) * tan(alpha)
        # => t0 = t - r / tan(alpha)
        ta = np.tan(alpha)
        if abs(ta) < 1e-12:
            return None
        t0_vals = t - r / ta
        t0 = float(np.median(t0_vals))
    else:
        # Least-squares: r ≈ t * tan(α) through origin, with apex offset
        # Fit r = a * t + b  →  tan(α) = a,  apex at t = -b/a
        A = np.column_stack([t, np.ones(len(t))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
        except Exception:
            return None
        a_slope, b_intercept = coeffs
        if a_slope <= 0:
            axis = -axis      # flip so cone opens in positive t direction
            t = -t
            a_slope = -a_slope
            b_intercept = -b_intercept
        alpha = float(np.arctan(max(a_slope, 1e-6)))
        t0 = float(-b_intercept / max(a_slope, 1e-12))

    apex = centroid + t0 * axis
    # Sanity: alpha must be in (0, π/2)
    alpha = float(np.clip(alpha, 1e-4, np.pi / 2 - 1e-4))
    return apex, axis, alpha


def _cone_ransac_initial(
    points: np.ndarray,
    *,
    trials: int,
    sample_size: int,
    known_alpha: Optional[float],
    initial_axis: Optional[ArrayLike],
    stop_inlier_fraction: Optional[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = points.shape[0]
    k = min(sample_size, n)

    init_axis = (
        _normalize(np.asarray(initial_axis, dtype=float).reshape(3))
        if initial_axis is not None else None
    )

    # PCA as baseline
    result = _cone_initial_from_subset(points, known_alpha, init_axis)
    if result is None:
        centroid = points.mean(axis=0)
        _, best_axis, _ = _pca_initial(points)
        result = (centroid, best_axis, float(np.deg2rad(30.0)))
    best_apex, best_axis, best_alpha = result
    best_res = _cone_residuals(points, best_apex, best_axis, best_alpha)
    best_threshold = _auto_threshold(best_res)
    best_score = _cone_score(best_res, best_threshold)

    for _ in range(trials):
        subset = points[rng.choice(n, k, replace=False)]
        result = _cone_initial_from_subset(subset, known_alpha, init_axis)
        if result is None:
            continue
        apex, axis, alpha = result
        res = _cone_residuals(points, apex, axis, alpha)
        thr = _auto_threshold(res)
        score = _cone_score(res, thr)
        if score > best_score:
            best_apex, best_axis, best_alpha = apex, axis, alpha
            best_score = score
            if stop_inlier_fraction is not None:
                sigma = thr / 3.0
                soft = float(np.sum(np.exp(-0.5 * (res / sigma) ** 2)))
                if soft / n >= stop_inlier_fraction * n:
                    break

    return best_apex, best_axis, best_alpha


def _cone_score(residuals: np.ndarray, threshold: float) -> tuple:
    sigma = threshold / 3.0
    soft = float(np.sum(np.exp(-0.5 * (residuals / sigma) ** 2)))
    hard = int((np.abs(residuals) <= threshold).sum())
    return soft, hard


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def _cone_residuals(
    points: np.ndarray,
    apex: np.ndarray,
    axis: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Signed radial residuals: ‖qᵢ‖ − tᵢ tan α."""
    c = points - apex
    t = c @ axis
    q = c - t[:, None] * axis
    dist = np.linalg.norm(q, axis=1)
    return dist - t * np.tan(alpha)


# ---------------------------------------------------------------------------
# Analytic Jacobian
# ---------------------------------------------------------------------------

def _cone_analytic_jacobian(
    points: np.ndarray,
    apex: np.ndarray,
    axis: np.ndarray,
    alpha: float,
    known_alpha: Optional[float],
) -> np.ndarray:
    """Analytic Jacobian of cone residuals w.r.t. [apex; raw_axis; alpha].

    At the evaluation point (unit axis):

        ∂rᵢ/∂apex  = −q̂ᵢ + tan α · d̂
        ∂rᵢ/∂d̂     = −qᵢ (tᵢ/‖qᵢ‖ + tan α)
        ∂rᵢ/∂α     = −tᵢ / cos²α
    """
    c = points - apex            # N×3
    t = c @ axis                 # N
    q = c - t[:, None] * axis   # N×3
    dist = np.linalg.norm(q, axis=1)  # N

    safe = np.where(dist > 1e-15, dist, 1e-15)
    unit_q = q / safe[:, None]  # N×3
    ta = float(np.tan(alpha))
    cos2a = float(np.cos(alpha) ** 2)

    n_params = 6 if known_alpha is not None else 7
    jac = np.empty((points.shape[0], n_params), dtype=float)

    # d(r)/d(apex)
    jac[:, 0:3] = -unit_q + ta * axis[None, :]

    # d(r)/d(raw_axis)  →  −qᵢ (tᵢ/‖qᵢ‖ + tan α)
    jac[:, 3:6] = -q * (t / safe + ta)[:, None]

    # d(r)/d(alpha)
    if known_alpha is None:
        jac[:, 6] = -t / max(cos2a, 1e-15)

    return jac


# ---------------------------------------------------------------------------
# LM refinement
# ---------------------------------------------------------------------------

def _refine_cone(
    points: np.ndarray,
    apex: np.ndarray,
    axis: np.ndarray,
    alpha: float,
    *,
    max_iterations: int,
    known_alpha: Optional[float],
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    n_params = 6 if known_alpha is not None else 7
    params = np.r_[apex, axis, alpha] if known_alpha is None else np.r_[apex, axis]
    damping = 1e-3
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        cur_axis = _normalize(params[3:6])
        cur_alpha = float(known_alpha) if known_alpha is not None else float(params[6])
        cur_apex = params[:3].copy()

        res = _cone_residuals(points, cur_apex, cur_axis, cur_alpha)
        cost = float(res @ res)
        jac = _cone_analytic_jacobian(points, cur_apex, cur_axis, cur_alpha, known_alpha)

        jtj = jac.T @ jac
        jtr = jac.T @ res
        diag = np.maximum(np.diag(jtj), 1.0)

        try:
            step = np.linalg.solve(jtj + damping * np.diag(diag), -jtr)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jtj + damping * np.eye(n_params), -jtr, rcond=None)[0]

        if np.linalg.norm(step) < 1e-10:
            converged = True
            break

        cand = params + step
        cand[3:6] = _normalize(cand[3:6])
        if known_alpha is None:
            cand[6] = float(np.clip(cand[6], 1e-4, np.pi / 2 - 1e-4))

        cand_axis = _normalize(cand[3:6])
        cand_alpha = float(known_alpha) if known_alpha is not None else float(cand[6])
        cand_res = _cone_residuals(points, cand[:3], cand_axis, cand_alpha)

        if float(cand_res @ cand_res) < cost:
            params = cand
            damping = max(damping * 0.4, 1e-9)
        else:
            damping = min(damping * 10.0, 1e9)

    final_axis = _normalize(params[3:6])
    final_alpha = float(known_alpha) if known_alpha is not None else float(params[6])
    return params[:3].copy(), final_axis, final_alpha, iteration, converged
