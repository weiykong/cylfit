from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

import numpy as np

from .core import CylinderModel, fit_cylinder
from .metrics import evaluate_fit
from .synthetic import SyntheticCylinder, generate_noisy_cylinder


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    truth: SyntheticCylinder
    threshold: float


@dataclass(frozen=True)
class BenchmarkResult:
    case: str
    method: str
    ok: bool
    seconds: float
    metrics: Optional[dict]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "case": self.case,
            "method": self.method,
            "ok": self.ok,
            "seconds": self.seconds,
            "metrics": self.metrics,
            "error": self.error,
        }


def default_benchmark_cases(seed: int = 100) -> list[BenchmarkCase]:
    """Return synthetic cases that stress common cylinder-fitting failures."""

    return [
        BenchmarkCase(
            "clean_full_scan",
            generate_noisy_cylinder(n_points=4000, noise=0.01, outlier_fraction=0.0, random_state=seed),
            0.04,
        ),
        BenchmarkCase(
            "noisy_with_outliers",
            generate_noisy_cylinder(n_points=5500, noise=0.035, outlier_fraction=0.2, random_state=seed + 1),
            0.12,
        ),
        BenchmarkCase(
            "partial_arc",
            generate_noisy_cylinder(n_points=5500, noise=0.025, outlier_fraction=0.12, partial_arc=0.38, random_state=seed + 2),
            0.10,
        ),
        BenchmarkCase(
            "short_wide",
            generate_noisy_cylinder(n_points=4500, radius=2.5, height=3.2, noise=0.025, outlier_fraction=0.1, random_state=seed + 3),
            0.10,
        ),
        BenchmarkCase(
            "long_thin",
            generate_noisy_cylinder(n_points=5500, radius=0.45, height=12.0, noise=0.012, outlier_fraction=0.12, random_state=seed + 4),
            0.045,
        ),
    ]


def available_methods() -> dict[str, Callable[[np.ndarray, float, int], CylinderModel]]:
    """Return benchmark methods available in the current Python environment."""

    methods: dict[str, Callable[[np.ndarray, float, int], CylinderModel]] = {
        "cylinderfit2026": _run_ours,
        "pca_baseline": _run_pca_baseline,
    }
    try:
        import cylinder_fitting  # noqa: F401

        methods["cylinder_fitting_pypi"] = _run_cylinder_fitting_pypi
    except Exception:
        pass
    try:
        import pyransac3d  # noqa: F401

        methods["pyransac3d"] = _run_pyransac3d
    except Exception:
        pass
    return methods


def run_benchmark(
    *,
    cases: Optional[list[BenchmarkCase]] = None,
    methods: Optional[dict[str, Callable[[np.ndarray, float, int], CylinderModel]]] = None,
    random_state: int = 0,
) -> list[BenchmarkResult]:
    """Run all available benchmark methods on synthetic scenarios."""

    cases = default_benchmark_cases() if cases is None else cases
    methods = available_methods() if methods is None else methods
    results: list[BenchmarkResult] = []

    for case in cases:
        for name, method in methods.items():
            start = perf_counter()
            try:
                model = method(case.truth.points, case.threshold, random_state)
                seconds = perf_counter() - start
                results.append(
                    BenchmarkResult(
                        case=case.name,
                        method=name,
                        ok=True,
                        seconds=seconds,
                        metrics=evaluate_fit(model, case.truth).to_dict(),
                    )
                )
            except Exception as exc:
                results.append(
                    BenchmarkResult(
                        case=case.name,
                        method=name,
                        ok=False,
                        seconds=perf_counter() - start,
                        metrics=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return results


def summarize_benchmark(results: list[BenchmarkResult]) -> dict:
    """Aggregate benchmark results by method."""

    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.method, []).append(result)

    summary = {}
    for method, rows in grouped.items():
        ok_rows = [row for row in rows if row.ok and row.metrics is not None]
        if not ok_rows:
            summary[method] = {"ok_cases": 0, "total_cases": len(rows), "score": 0.0}
            continue
        radius = np.mean([row.metrics["radius_error"] for row in ok_rows])
        angle = np.mean([row.metrics["axis_angle_deg"] for row in ok_rows])
        height = np.mean([row.metrics["height_error"] for row in ok_rows])
        rmse = np.mean([row.metrics["rmse"] for row in ok_rows])
        seconds = np.mean([row.seconds for row in ok_rows])
        recall_values = [row.metrics["inlier_recall"] for row in ok_rows if row.metrics["inlier_recall"] is not None]
        recall = float(np.mean(recall_values)) if recall_values else 0.0
        score = 100.0 / (1.0 + 10.0 * radius + 0.25 * angle + height + 10.0 * rmse + seconds)
        score *= recall
        summary[method] = {
            "ok_cases": len(ok_rows),
            "total_cases": len(rows),
            "mean_radius_error": float(radius),
            "mean_axis_angle_deg": float(angle),
            "mean_height_error": float(height),
            "mean_rmse": float(rmse),
            "mean_seconds": float(seconds),
            "mean_inlier_recall": recall,
            "score": float(score),
        }
    return dict(sorted(summary.items(), key=lambda item: item[1]["score"], reverse=True))


def benchmark_markdown_report(results: list[BenchmarkResult]) -> str:
    """Return a compact Markdown benchmark report."""

    summary = summarize_benchmark(results)
    lines = [
        "# CylinderFit Benchmark Report",
        "",
        "## Summary",
        "",
        "| method | score | ok cases | radius err | axis deg | height err | rmse | seconds | recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary.items():
        lines.append(
            f"| {method} | {row['score']:.2f} | {row['ok_cases']}/{row['total_cases']} | "
            f"{row.get('mean_radius_error', 0):.5f} | {row.get('mean_axis_angle_deg', 0):.3f} | "
            f"{row.get('mean_height_error', 0):.5f} | {row.get('mean_rmse', 0):.5f} | "
            f"{row.get('mean_seconds', 0):.4f} | {row.get('mean_inlier_recall', 0):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| case | method | ok | seconds | radius err | axis deg | height err | rmse | recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        if result.metrics is None:
            lines.append(f"| {result.case} | {result.method} | no | {result.seconds:.4f} |  |  |  |  |  |")
            continue
        m = result.metrics
        lines.append(
            f"| {result.case} | {result.method} | yes | {result.seconds:.4f} | "
            f"{m['radius_error']:.5f} | {m['axis_angle_deg']:.3f} | {m['height_error']:.5f} | "
            f"{m['rmse']:.5f} | {m['inlier_recall'] if m['inlier_recall'] is not None else 0:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _run_ours(points: np.ndarray, threshold: float, random_state: int) -> CylinderModel:
    return fit_cylinder(points, threshold=threshold, ransac_trials=128, random_state=random_state)


def _run_pca_baseline(points: np.ndarray, threshold: float, random_state: int) -> CylinderModel:
    return fit_cylinder(points, threshold=threshold, ransac_trials=0, robust=False, max_iterations=12, random_state=random_state)


def _run_cylinder_fitting_pypi(points: np.ndarray, threshold: float, random_state: int) -> CylinderModel:
    from cylinder_fitting import fit

    axis, center, radius, _ = fit(points)
    return _model_from_external(points, center, axis, radius, threshold)


def _run_pyransac3d(points: np.ndarray, threshold: float, random_state: int) -> CylinderModel:
    from pyransac3d import Cylinder

    cylinder = Cylinder()
    center, axis, radius, _ = cylinder.fit(points, thresh=threshold, maxIteration=128)
    return _model_from_external(points, center, axis, radius, threshold)


def _model_from_external(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    radius: float,
    threshold: float,
) -> CylinderModel:
    axis_point = np.asarray(axis_point, dtype=float).reshape(3)
    axis_direction = np.asarray(axis_direction, dtype=float).reshape(3)
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    centered = points - axis_point
    t = centered @ axis_direction
    radial = centered - t[:, None] * axis_direction
    residuals = np.linalg.norm(radial, axis=1) - float(radius)
    inlier_mask = np.abs(residuals) <= threshold
    t_fit = t[inlier_mask] if inlier_mask.any() else t
    return CylinderModel(
        axis_point=axis_point,
        axis_direction=axis_direction,
        radius=float(radius),
        height_min=float(np.min(t_fit)),
        height_max=float(np.max(t_fit)),
        inlier_mask=inlier_mask,
        residuals=residuals,
        iterations=0,
        converged=True,
    )
