from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class CylinderModel:
    """Fitted cylinder model."""

    axis_point: np.ndarray
    axis_direction: np.ndarray
    radius: float
    height_min: float
    height_max: float
    inlier_mask: np.ndarray
    residuals: np.ndarray
    iterations: int
    converged: bool

    @property
    def height(self) -> float:
        return float(self.height_max - self.height_min)

    @property
    def rmse(self) -> float:
        r = self.residuals[self.inlier_mask]
        return float(np.sqrt(np.mean(r * r))) if r.size else float("nan")

    @property
    def mae(self) -> float:
        r = self.residuals[self.inlier_mask]
        return float(np.mean(np.abs(r))) if r.size else float("nan")

    @property
    def start_point(self) -> np.ndarray:
        return self.axis_point + self.height_min * self.axis_direction

    @property
    def end_point(self) -> np.ndarray:
        return self.axis_point + self.height_max * self.axis_direction

    def to_dict(self) -> dict:
        """Return a JSON-serializable model summary."""

        return {
            "axis_point": self.axis_point.tolist(),
            "axis_direction": self.axis_direction.tolist(),
            "radius": self.radius,
            "height_min": self.height_min,
            "height_max": self.height_max,
            "height": self.height,
            "start_point": self.start_point.tolist(),
            "end_point": self.end_point.tolist(),
            "inliers": int(self.inlier_mask.sum()),
            "points": int(self.inlier_mask.size),
            "rmse": self.rmse,
            "mae": self.mae,
            "iterations": self.iterations,
            "converged": self.converged,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Return a JSON model summary."""

        import json

        return json.dumps(self.to_dict(), indent=indent)

    def to_mesh(self, *, n_theta: int = 64, n_height: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Return cylinder surface vertices and triangular faces."""

        u, v = _orthonormal_basis(self.axis_direction)
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        t = np.linspace(self.height_min, self.height_max, n_height)
        vertices = []
        for z in t:
            center = self.axis_point + z * self.axis_direction
            ring = center + self.radius * np.cos(theta)[:, None] * u + self.radius * np.sin(theta)[:, None] * v
            vertices.append(ring)
        vertices_array = np.vstack(vertices)
        faces = []
        for j in range(n_height - 1):
            row = j * n_theta
            next_row = (j + 1) * n_theta
            for i in range(n_theta):
                a = row + i
                b = row + (i + 1) % n_theta
                c = next_row + i
                d = next_row + (i + 1) % n_theta
                faces.append((a, c, b))
                faces.append((b, c, d))
        return vertices_array, np.asarray(faces, dtype=int)

    def plot(self, points: ArrayLike, **kwargs):
        """Plot this model with ``plot_cylinder_fit``."""

        from .visualize import plot_cylinder_fit

        return plot_cylinder_fit(points, self, **kwargs)


def fit_cylinder(
    points: ArrayLike,
    *,
    threshold: Optional[float] = None,
    ransac_trials: int = 128,
    sample_size: int = 64,
    max_iterations: int = 40,
    robust: bool = True,
    max_refine_points: int = 20000,
    known_radius: Optional[float] = None,
    initial_axis: Optional[ArrayLike] = None,
    point_weights: Optional[ArrayLike] = None,
    orientation_axis: Optional[ArrayLike] = None,
    max_axis_angle_deg: Optional[float] = None,
    ransac_stop_inlier_fraction: Optional[float] = 0.98,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
) -> CylinderModel:
    """Fit an arbitrary oriented cylinder to an ``N x 3`` point cloud.

    Parameters
    ----------
    points:
        Input point cloud as an ``N x 3`` array.
    threshold:
        Inlier distance threshold. If omitted, a robust scale estimate from the
        initial model is used.
    ransac_trials:
        Number of random bootstrap trials. Set to ``0`` for deterministic PCA
        initialization only.
    sample_size:
        Number of points sampled per RANSAC candidate.
    max_iterations:
        Maximum robust nonlinear refinement iterations.
    robust:
        Use Huber-style iteratively reweighted refinement.
    max_refine_points:
        Optimization sample cap for large clouds. The final residuals are still
        evaluated on all points.
    known_radius:
        Optional fixed radius. When provided, only axis position and direction
        are optimized.
    initial_axis:
        Optional initial axis direction, useful for constrained or normal-aware
        workflows.
    point_weights:
        Optional non-negative per-point weights for uncertainty-aware fitting.
    orientation_axis:
        Optional preferred axis direction for orientation-constrained RANSAC.
    max_axis_angle_deg:
        Maximum allowed angle from ``orientation_axis``. Sign is ignored, so
        parallel and anti-parallel axes are equivalent.
    ransac_stop_inlier_fraction:
        Stop RANSAC early after a candidate reaches this inlier fraction. Use
        ``None`` to disable adaptive stopping.
    random_state:
        Seed for deterministic RANSAC and refinement sampling.
    n_jobs:
        Number of parallel workers for RANSAC trials. ``1`` (default) runs
        serially. ``-1`` uses all logical CPU cores. Parallelism is implemented
        via :class:`concurrent.futures.ThreadPoolExecutor`; numpy SVD releases
        the GIL so threads provide real concurrency on multi-core hardware.
    """

    pts = _validate_points(points)
    rng = np.random.default_rng(random_state)

    if known_radius is not None and known_radius <= 0:
        raise ValueError("known_radius must be positive")
    weights = _validate_weights(point_weights, pts.shape[0])
    orient = _normalize(np.asarray(orientation_axis, dtype=float).reshape(3)) if orientation_axis is not None else None
    if max_axis_angle_deg is not None and orient is None:
        raise ValueError("orientation_axis is required when max_axis_angle_deg is set")
    min_axis_cos = None
    if max_axis_angle_deg is not None:
        if not 0 <= max_axis_angle_deg <= 90:
            raise ValueError("max_axis_angle_deg must be in [0, 90]")
        min_axis_cos = float(np.cos(np.deg2rad(max_axis_angle_deg)))

    p0, axis, radius = _ransac_or_pca_initial(
        pts,
        threshold=threshold,
        trials=max(0, int(ransac_trials)),
        sample_size=max(8, int(sample_size)),
        known_radius=known_radius,
        initial_axis=initial_axis if initial_axis is not None else orient,
        orientation_axis=orient,
        min_axis_cos=min_axis_cos,
        stop_inlier_fraction=ransac_stop_inlier_fraction,
        rng=rng,
        n_jobs=int(n_jobs),
    )

    seed_p0, seed_axis, seed_radius = p0.copy(), axis.copy(), float(radius)
    residuals = residuals_to_cylinder(pts, p0, axis, radius)
    if threshold is None:
        threshold = _auto_threshold(residuals)
    seed_residuals = residuals.copy()
    seed_inliers = np.abs(seed_residuals) <= threshold

    inlier_mask = np.abs(residuals) <= threshold
    if inlier_mask.sum() < 8:
        inlier_mask = np.ones(pts.shape[0], dtype=bool)

    refine_pts = pts[inlier_mask]
    refine_weights = weights[inlier_mask] if weights is not None else None
    if refine_pts.shape[0] > max_refine_points:
        idx = rng.choice(refine_pts.shape[0], max_refine_points, replace=False)
        refine_pts = refine_pts[idx]
        refine_weights = refine_weights[idx] if refine_weights is not None else None

    p0, axis, radius, iterations, converged = _refine_axis(
        refine_pts,
        p0,
        axis,
        max_iterations=max_iterations,
        robust=robust,
        known_radius=known_radius,
        point_weights=refine_weights,
        orientation_axis=orient,
        min_axis_cos=min_axis_cos,
    )

    residuals = residuals_to_cylinder(pts, p0, axis, radius)
    inlier_mask = np.abs(residuals) <= threshold
    if inlier_mask.sum() >= 8:
        p0, axis, radius, extra_iterations, extra_converged = _refine_axis(
            pts[inlier_mask],
            p0,
            axis,
            max_iterations=max(8, max_iterations // 2),
            robust=robust,
            known_radius=known_radius,
            point_weights=weights[inlier_mask] if weights is not None else None,
            orientation_axis=orient,
            min_axis_cos=min_axis_cos,
        )
        iterations += extra_iterations
        converged = converged or extra_converged
        residuals = residuals_to_cylinder(pts, p0, axis, radius)
        inlier_mask = np.abs(residuals) <= threshold

    t = (pts - p0) @ axis
    if inlier_mask.any():
        t_fit = t[inlier_mask]
    else:
        t_fit = t

    if int(inlier_mask.sum()) < 0.8 * int(seed_inliers.sum()) and int(seed_inliers.sum()) >= 8:
        t_seed = (pts - seed_p0) @ seed_axis
        t_seed_fit = t_seed[seed_inliers]
        final_model = CylinderModel(
            axis_point=seed_p0,
            axis_direction=seed_axis,
            radius=seed_radius,
            height_min=float(np.min(t_seed_fit)),
            height_max=float(np.max(t_seed_fit)),
            inlier_mask=seed_inliers,
            residuals=seed_residuals,
            iterations=0,
            converged=False,
        )
        return _choose_pca_fallback(
            final_model,
            pts,
            threshold,
            ransac_trials,
            sample_size,
            max_iterations,
            robust,
            max_refine_points,
            known_radius,
            initial_axis,
            weights,
            orient,
            max_axis_angle_deg,
            ransac_stop_inlier_fraction,
            random_state,
        )

    final_model = CylinderModel(
        axis_point=p0,
        axis_direction=axis,
        radius=float(radius),
        height_min=float(np.min(t_fit)),
        height_max=float(np.max(t_fit)),
        inlier_mask=inlier_mask,
        residuals=residuals,
        iterations=iterations,
        converged=converged,
    )
    return _choose_pca_fallback(
        final_model,
        pts,
        threshold,
        ransac_trials,
        sample_size,
        max_iterations,
        robust,
        max_refine_points,
        known_radius,
        initial_axis,
        weights,
        orient,
        max_axis_angle_deg,
        ransac_stop_inlier_fraction,
        random_state,
    )


def _choose_pca_fallback(
    final_model: CylinderModel,
    points: np.ndarray,
    threshold: float,
    ransac_trials: int,
    sample_size: int,
    max_iterations: int,
    robust: bool,
    max_refine_points: int,
    known_radius: Optional[float],
    initial_axis: Optional[ArrayLike],
    point_weights: Optional[np.ndarray],
    orientation_axis: Optional[np.ndarray],
    max_axis_angle_deg: Optional[float],
    ransac_stop_inlier_fraction: Optional[float],
    random_state: Optional[int],
) -> CylinderModel:
    if ransac_trials <= 0 or initial_axis is not None or orientation_axis is not None:
        return final_model
    pca_model = fit_cylinder(
        points,
        threshold=threshold,
        ransac_trials=0,
        sample_size=sample_size,
        max_iterations=max_iterations,
        robust=robust,
        max_refine_points=max_refine_points,
        known_radius=known_radius,
        initial_axis=None,
        point_weights=point_weights,
        orientation_axis=None,
        max_axis_angle_deg=max_axis_angle_deg,
        ransac_stop_inlier_fraction=ransac_stop_inlier_fraction,
        random_state=random_state,
    )
    final_inliers = int(final_model.inlier_mask.sum())
    pca_inliers = int(pca_model.inlier_mask.sum())
    if pca_inliers > final_inliers * 1.05:
        return pca_model
    if pca_inliers >= final_inliers * 0.98 and pca_model.rmse < final_model.rmse:
        return pca_model
    return final_model


def fit_cylinder_known_radius(
    points: ArrayLike,
    radius: float,
    **kwargs,
) -> CylinderModel:
    """Fit a cylinder while keeping radius fixed."""

    return fit_cylinder(points, known_radius=radius, **kwargs)


def fit_cylinder_with_normals(
    points: ArrayLike,
    normals: ArrayLike,
    *,
    normal_weight: float = 0.25,
    **kwargs,
) -> CylinderModel:
    """Fit a cylinder using normals to seed the axis direction.

    Cylinder surface normals are approximately perpendicular to the cylinder
    axis, so the smallest singular vector of the normals is a strong axis
    initializer. ``normal_weight`` blends this initializer with point geometry by
    controlling whether the normal-derived axis is trusted.
    """

    pts = _validate_points(points)
    n = np.asarray(normals, dtype=float)
    if n.shape != pts.shape:
        raise ValueError("normals must have the same shape as points")
    valid = np.isfinite(n).all(axis=1)
    n = n[valid]
    if n.shape[0] < 8:
        raise ValueError("at least 8 finite normals are required")
    lengths = np.linalg.norm(n, axis=1)
    n = n[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    if n.shape[0] < 8:
        raise ValueError("at least 8 non-zero normals are required")

    _, _, vh = np.linalg.svd(n, full_matrices=False)
    normal_axis = _normalize(vh[-1])
    if normal_weight <= 0:
        normal_axis = None
    return fit_cylinder(pts, initial_axis=normal_axis, **kwargs)


def fit_cylinder_fixed_axis(
    points: ArrayLike,
    axis: ArrayLike,
    *,
    threshold: Optional[float] = None,
    radius: Optional[float] = None,
    point_weights: Optional[ArrayLike] = None,
) -> CylinderModel:
    """Fit radius and axis point while keeping the axis direction fixed."""

    pts = _validate_points(points)
    axis = _normalize(np.asarray(axis, dtype=float).reshape(3))
    weights = _validate_weights(point_weights, pts.shape[0])
    centroid = _weighted_centroid(pts, weights)
    p0 = _project_axis_point_to_centroid(centroid, axis, centroid)
    distances = _axis_distances(pts, p0, axis)
    fit_radius = float(radius) if radius is not None else float(np.median(distances))
    if fit_radius <= 0:
        raise ValueError("radius must be positive")
    residuals = residuals_to_cylinder(pts, p0, axis, fit_radius)
    if threshold is None:
        threshold = _auto_threshold(residuals)
    inliers = np.abs(residuals) <= threshold
    if radius is None and inliers.sum() >= 8:
        fit_radius = _fit_radius(pts[inliers], p0, axis, weights[inliers] if weights is not None else None)
        residuals = residuals_to_cylinder(pts, p0, axis, fit_radius)
        inliers = np.abs(residuals) <= threshold
    t = (pts - p0) @ axis
    t_fit = t[inliers] if inliers.any() else t
    return CylinderModel(
        axis_point=p0,
        axis_direction=axis,
        radius=fit_radius,
        height_min=float(np.min(t_fit)),
        height_max=float(np.max(t_fit)),
        inlier_mask=inliers,
        residuals=residuals,
        iterations=0,
        converged=True,
    )


def fit_cylinder_constrained_axis(
    points: ArrayLike,
    axis: ArrayLike,
    *,
    max_axis_angle_deg: float = 10.0,
    **kwargs,
) -> CylinderModel:
    """Fit a cylinder whose axis must stay within an angular cone."""

    return fit_cylinder(
        points,
        initial_axis=axis,
        orientation_axis=axis,
        max_axis_angle_deg=max_axis_angle_deg,
        **kwargs,
    )


def residuals_to_cylinder(
    points: ArrayLike,
    axis_point: ArrayLike,
    axis_direction: ArrayLike,
    radius: float,
) -> np.ndarray:
    """Signed radial point-to-cylinder residuals."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    if pts.shape[0] < 1:
        raise ValueError("at least 1 point is required")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite")
    p0 = np.asarray(axis_point, dtype=float).reshape(3)
    axis = _normalize(np.asarray(axis_direction, dtype=float).reshape(3))
    radial = _axis_distances(pts, p0, axis)
    return radial - float(radius)


def _validate_points(points: ArrayLike) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    if pts.shape[0] < 8:
        raise ValueError("at least 8 points are required")
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 8:
        raise ValueError("at least 8 finite points are required")
    return np.ascontiguousarray(pts)


def _validate_weights(weights: Optional[ArrayLike], n_points: int) -> Optional[np.ndarray]:
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n_points:
        raise ValueError("point_weights must have one value per point")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("point_weights must be finite and non-negative")
    if float(np.sum(w)) <= 1e-15:
        raise ValueError("point_weights must contain at least one positive value")
    return w


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("zero-length cylinder axis")
    return v / norm


def _axis_distances(points: np.ndarray, p0: np.ndarray, axis: np.ndarray) -> np.ndarray:
    centered = points - p0
    axial = centered @ axis
    radial_vec = centered - axial[:, None] * axis
    return np.linalg.norm(radial_vec, axis=1)


def _project_axis_point_to_centroid(p0: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return p0 + np.dot(centroid - p0, axis) * axis


def _weighted_centroid(points: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        return points.mean(axis=0)
    return np.average(points, axis=0, weights=weights)


def _pca_initial(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = _normalize(vh[0])
    radius = float(np.median(_axis_distances(points, centroid, axis)))
    return centroid, axis, max(radius, 1e-12)


def _axis_initial(points: np.ndarray, axis: ArrayLike, known_radius: Optional[float]) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    axis = _normalize(np.asarray(axis, dtype=float).reshape(3))
    radius = float(known_radius) if known_radius is not None else float(np.median(_axis_distances(points, centroid, axis)))
    return centroid, axis, max(radius, 1e-12)


def _ransac_or_pca_initial(
    points: np.ndarray,
    *,
    threshold: Optional[float],
    trials: int,
    sample_size: int,
    known_radius: Optional[float],
    initial_axis: Optional[ArrayLike],
    orientation_axis: Optional[np.ndarray],
    min_axis_cos: Optional[float],
    stop_inlier_fraction: Optional[float],
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """RANSAC bootstrap with PROSAC quality-ranked sampling and MAGSAC soft scoring.

    First half of trials use PROSAC: points are ranked by their residual under the
    initial PCA model, and sampling draws preferentially from the best-fitting
    points with a pool that expands each trial. The second half falls back to
    uniform sampling so that outlier-heavy regions can still be explored.
    MAGSAC soft scoring replaces the hard inlier count, making the search
    insensitive to the exact threshold value.

    When *n_jobs* != 1 the trials are split across threads via
    :class:`concurrent.futures.ThreadPoolExecutor`.  NumPy SVD and residual
    evaluation release the GIL, so threading provides real parallelism on
    multi-core hardware without pickling overhead.
    """
    best = _axis_initial(points, initial_axis, known_radius) if initial_axis is not None else _pca_initial(points)
    if known_radius is not None:
        best = (best[0], best[1], float(known_radius))
    best_res = residuals_to_cylinder(points, *best)
    if threshold is None:
        threshold = _auto_threshold(best_res)
    best_score = _candidate_score(best_res, threshold)

    if trials == 0:
        return best

    quality_order = np.argsort(np.abs(best_res))

    # Resolve worker count
    if n_jobs == -1:
        import os
        n_workers = max(1, os.cpu_count() or 1)
    else:
        n_workers = max(1, n_jobs)

    if n_workers == 1 or trials < 16:
        # Serial path (default)
        best, best_score = _ransac_trials_batch(
            points, best, best_score, threshold, trials, sample_size,
            quality_order, known_radius, orientation_axis, min_axis_cos,
            stop_inlier_fraction, rng,
        )
    else:
        # Parallel path: split trials across threads
        import concurrent.futures
        trials_per_worker = max(1, trials // n_workers)
        seeds = rng.integers(0, 2**31, size=n_workers).tolist()
        batch_sizes = [trials_per_worker] * n_workers
        # distribute remainder to last worker
        batch_sizes[-1] += trials - sum(batch_sizes)

        def _run_batch(batch_trials: int, seed: int):
            child_rng = np.random.default_rng(seed)
            return _ransac_trials_batch(
                points, best, best_score, threshold, batch_trials, sample_size,
                quality_order, known_radius, orientation_axis, min_axis_cos,
                stop_inlier_fraction, child_rng,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_batch, bs, sd)
                       for bs, sd in zip(batch_sizes, seeds)]
            for fut in concurrent.futures.as_completed(futures):
                candidate, score = fut.result()
                if score > best_score:
                    best, best_score = candidate, score

    return best


def _ransac_trials_batch(
    points: np.ndarray,
    best: tuple,
    best_score: tuple,
    threshold: float,
    trials: int,
    sample_size: int,
    quality_order: np.ndarray,
    known_radius: Optional[float],
    orientation_axis: Optional[np.ndarray],
    min_axis_cos: Optional[float],
    stop_inlier_fraction: Optional[float],
    rng: np.random.Generator,
) -> tuple:
    """Run a batch of RANSAC trials (PROSAC + uniform) and return the best candidate."""
    n = points.shape[0]
    k = min(sample_size, n)
    prosac_trials = trials // 2
    prosac_pool = min(k, n)
    pool_growth = max(1, (n - prosac_pool) // max(1, prosac_trials))

    for trial_idx in range(trials):
        if trial_idx < prosac_trials:
            prosac_pool = min(prosac_pool + pool_growth, n)
            pool = quality_order[:prosac_pool]
            subset_idx = rng.choice(prosac_pool, min(k, prosac_pool), replace=False)
            subset = points[pool[subset_idx]]
        else:
            subset = points[rng.choice(n, k, replace=False)]

        try:
            candidate = _pca_initial(subset)
        except np.linalg.LinAlgError:
            continue
        if not _axis_allowed(candidate[1], orientation_axis, min_axis_cos):
            continue
        if known_radius is not None:
            candidate = (candidate[0], candidate[1], float(known_radius))
        res = residuals_to_cylinder(points, *candidate)
        score = _candidate_score(res, threshold)
        if score > best_score:
            best = candidate
            best_score = score
            if stop_inlier_fraction is not None and score[1] / n >= stop_inlier_fraction:
                break

    return best, best_score


def _axis_allowed(axis: np.ndarray, orientation_axis: Optional[np.ndarray], min_axis_cos: Optional[float]) -> bool:
    if orientation_axis is None or min_axis_cos is None:
        return True
    return abs(float(np.dot(_normalize(axis), orientation_axis))) >= min_axis_cos


def _candidate_score(residuals: np.ndarray, threshold: float) -> tuple[float, float]:
    """MAGSAC-style soft score: Gaussian-weighted residuals + hard inlier tiebreak.

    Unlike binary RANSAC (count inliers at threshold), we integrate over the
    noise model using a Gaussian with sigma = threshold/3. This marginalizes the
    threshold and makes scoring robust to its exact choice — the core idea of MAGSAC.
    """
    sigma = threshold / 3.0
    x2 = (residuals / sigma) ** 2
    soft_count = float(np.sum(np.exp(-0.5 * x2)))
    # Secondary: penalize large residuals for tiebreaking
    hard_inliers = int((np.abs(residuals) <= threshold).sum())
    return soft_count, float(hard_inliers)


def _auto_threshold(residuals: np.ndarray) -> float:
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    sigma = 1.4826 * mad
    fallback = float(np.percentile(np.abs(residuals), 70)) * 0.2
    return max(3.0 * sigma, fallback, 1e-9)


def _fit_radius(points: np.ndarray, p0: np.ndarray, axis: np.ndarray, weights: Optional[np.ndarray]) -> float:
    d = _axis_distances(points, p0, axis)
    if weights is None:
        return max(float(np.mean(d)), 1e-12)
    wsum = float(np.sum(weights))
    if wsum <= 1e-15:
        return max(float(np.mean(d)), 1e-12)
    return max(float(np.sum(weights * d) / wsum), 1e-12)


def _weighted_residuals(
    params: np.ndarray,
    points: np.ndarray,
    centroid: np.ndarray,
    weights: Optional[np.ndarray],
    known_radius: Optional[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    p0 = params[:3]
    axis = _normalize(params[3:6])
    p0 = _project_axis_point_to_centroid(p0, axis, centroid)
    radius = float(known_radius) if known_radius is not None else _fit_radius(points, p0, axis, weights)
    res = residuals_to_cylinder(points, p0, axis, radius)
    if weights is not None:
        res = np.sqrt(weights) * res
    return res, p0, radius


def _huber_weights(residuals: np.ndarray) -> np.ndarray:
    scale = _auto_threshold(residuals) / 3.0
    scale = max(scale, 1e-12)
    x = np.abs(residuals) / (1.345 * scale)
    # Avoid divide-by-zero: near-zero residuals get weight 1.0
    safe_x = np.where(x > 1e-15, x, 1.0)
    return np.where(x <= 1.0, 1.0, 1.0 / safe_x)


def _refine_axis(
    points: np.ndarray,
    p0: np.ndarray,
    axis: np.ndarray,
    *,
    max_iterations: int,
    robust: bool,
    known_radius: Optional[float],
    point_weights: Optional[np.ndarray],
    orientation_axis: Optional[np.ndarray],
    min_axis_cos: Optional[float],
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    centroid = _weighted_centroid(points, point_weights)
    axis = _normalize(axis)
    p0 = _project_axis_point_to_centroid(np.asarray(p0, dtype=float), axis, centroid)
    params = np.r_[p0, axis]
    robust_weights = None
    damping = 1e-3
    converged = False

    for iteration in range(1, max_iterations + 1):
        weights = _combine_weights(point_weights, robust_weights)
        res, p0_eval, radius = _weighted_residuals(params, points, centroid, weights, known_radius)
        cost = float(res @ res)
        jac = _analytic_jacobian(params, points, centroid, weights, known_radius)
        normal = jac.T @ jac
        gradient = jac.T @ res
        diag = np.maximum(np.diag(normal), 1.0)

        try:
            step = np.linalg.solve(normal + damping * np.diag(diag), -gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(normal + damping * np.eye(6), -gradient, rcond=None)[0]

        if np.linalg.norm(step[:3]) + np.linalg.norm(step[3:]) < 1e-10:
            converged = True
            break

        candidate = params + step
        candidate[3:6] = _normalize(candidate[3:6])
        if not _axis_allowed(candidate[3:6], orientation_axis, min_axis_cos):
            damping = min(damping * 10.0, 1e9)
            continue
        cand_res, _, _ = _weighted_residuals(candidate, points, centroid, weights, known_radius)
        cand_cost = float(cand_res @ cand_res)

        if cand_cost < cost:
            params = candidate
            damping = max(damping * 0.4, 1e-9)
            if robust:
                raw_res, _, _ = _weighted_residuals(params, points, centroid, None, known_radius)
                robust_weights = _huber_weights(raw_res)
        else:
            damping = min(damping * 10.0, 1e9)

    axis = _normalize(params[3:6])
    p0 = _project_axis_point_to_centroid(params[:3], axis, centroid)
    radius = float(known_radius) if known_radius is not None else _fit_radius(points, p0, axis, point_weights)
    return p0, axis, radius, iteration, converged


def _combine_weights(base: Optional[np.ndarray], robust: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if base is None:
        return robust
    if robust is None:
        return base
    return base * robust


def _analytic_jacobian(
    params: np.ndarray,
    points: np.ndarray,
    centroid: np.ndarray,
    weights: Optional[np.ndarray],
    known_radius: Optional[float],
) -> np.ndarray:
    """Closed-form Jacobian of cylinder residuals w.r.t. [p0_x,p0_y,p0_z, d_x,d_y,d_z].

    Derivation: with p0_eff = p0_raw + alpha*d (projected onto centroid plane),
    b_i = p_i - p0_eff, t_i = b_i·d, q_i = b_i - t_i*d (radial vector):

        d(r_i)/d(p0_raw)   = -q_i / ||q_i||
        d(r_i)/d(raw_axis) = -(t_i + alpha) * q_i / ||q_i||   [at unit axis]

    where alpha = (centroid - p0_raw)·d, so (t_i + alpha) = (p_i - p0_raw)·d.
    For fitted radius an additional correction subtracts the (weighted) mean column
    so that the radius degree of freedom is correctly marginalized out.
    """
    p0_raw = params[:3]
    axis = _normalize(params[3:6])

    alpha = float(np.dot(centroid - p0_raw, axis))
    p0_eff = p0_raw + alpha * axis

    b = points - p0_eff                          # N x 3
    t = b @ axis                                  # N
    q = b - t[:, None] * axis                    # N x 3  (radial vectors, ⊥ axis)
    dist = np.linalg.norm(q, axis=1)             # N

    safe = np.where(dist > 1e-15, dist, 1e-15)
    unit_q = q / safe[:, None]                   # N x 3  (unit radial normals)

    # s_i = (p_i - p0_raw)·d = t_i + alpha
    s = t + alpha                                 # N

    jac = np.empty((points.shape[0], 6), dtype=float)
    jac[:, 0:3] = -unit_q
    jac[:, 3:6] = -s[:, None] * unit_q

    # Marginalize fitted radius: subtract (weighted) mean column so that a
    # uniform shift in all dist_i (i.e. a radius change) gives zero gradient.
    if known_radius is None:
        if weights is None:
            jac -= jac.mean(axis=0)
        else:
            w_norm = weights / weights.sum()
            jac -= (w_norm[:, None] * jac).sum(axis=0)

    if weights is not None:
        jac *= np.sqrt(weights)[:, None]

    return jac


def _numeric_jacobian(
    params: np.ndarray,
    points: np.ndarray,
    centroid: np.ndarray,
    weights: Optional[np.ndarray],
    known_radius: Optional[float],
    residuals: np.ndarray,
) -> np.ndarray:
    """Central-difference fallback Jacobian (kept for validation)."""
    jac = np.empty((points.shape[0], 6), dtype=float)
    base_norm = max(float(np.linalg.norm(params)), 1.0)
    eps_base = np.sqrt(np.finfo(float).eps) * base_norm

    for j in range(6):
        step = eps_base * max(abs(float(params[j])), 1.0)
        p_plus = params.copy()
        p_plus[j] += step
        p_minus = params.copy()
        p_minus[j] -= step
        if j >= 3:
            p_plus[3:6] = _normalize(p_plus[3:6])
            p_minus[3:6] = _normalize(p_minus[3:6])
        plus, _, _ = _weighted_residuals(p_plus, points, centroid, weights, known_radius)
        minus, _, _ = _weighted_residuals(p_minus, points, centroid, weights, known_radius)
        jac[:, j] = (plus - minus) / (2.0 * step)

    return jac


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = helper - float(helper @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v
