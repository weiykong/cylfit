# Benchmarks

Run:

```bash
PYTHONPATH=src python examples/run_benchmark.py
```

The runner prints JSON and writes `examples/benchmark_report.md`.

Install optional Python competitors:

```bash
python -m pip install -e ".[competitors]"
```

## Included Methods

- `cylinderfit2026`: robust RANSAC/PCA guarded fitter with nonlinear refinement.
- `pca_baseline`: deterministic PCA initialization and least-squares refinement.
- `cylinder_fitting_pypi`: enabled when `cylinder_fitting` is installed.
- `pyransac3d`: enabled when `pyransac3d` is installed.

Open3D is intentionally not listed as a direct cylinder competitor because its Python API exposes built-in plane segmentation, but not a one-call cylinder fitter comparable to this package. PCL cylinder segmentation is important prior art, but reliable Python packaging varies by platform, so it should be benchmarked through a separate adapter when available locally.

## Default Cases

- clean full scan
- noisy scan with outliers
- partial arc
- short wide cylinder
- long thin cylinder

Each case reports radius error, axis angle error, height error, RMSE, MAE, inlier precision, inlier recall, and runtime.

`summarize_benchmark(results)` ranks methods with a combined score based on radius error, axis error, height error, RMSE, runtime, and recall. `benchmark_markdown_report(results)` emits a portable report.

## Additional Measures

Use `measure_fit(points, model)` for ground-truth-free diagnostics:

- residual percentiles: P50/P90/P95/P99
- maximum absolute residual
- inlier fraction
- angular coverage around the cylinder
- axial coverage along the fitted height
- point density per cylinder surface area
- 0-100 quality score

These measures are intended for real point clouds where radius and axis ground truth are unknown.
