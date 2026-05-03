# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- GPU-accelerated RANSAC via CuPy / CUDA backend
- Batch fitting API for processing large point-cloud archives in parallel
- ROS 2 integration package (`cylinderfit2026_ros`)
- Interactive 3-D visualiser (Plotly Dash)
- STEP/IGES CAD export for fitted primitives

## [0.1.0] - 2026-05-03

### Added

#### Robust estimation
- **MAGSAC+PROSAC RANSAC** — combines MAGSAC+ adaptive noise-scale scoring with
  PROSAC progressive sampling for fast, statistically rigorous inlier selection
  without a hard inlier threshold
- **Parallel RANSAC** — `ThreadPoolExecutor`-based multi-trial parallelism with a
  configurable worker count; best model is selected across all threads

#### Core geometry fitting
- **Circular cylinder fitter** — direct algebraic initialisation followed by
  full 7-parameter (axis point, axis direction, radius) Levenberg–Marquardt
  refinement with **analytic closed-form Jacobian** for maximum convergence speed
- **Elliptical cylinder fitter** — Fitzgibbon algebraic ellipse fit projected onto
  cylinder cross-sections; produces full 9-parameter elliptical cylinder model
- **Cone fitter** — 7-parameter (apex, axis, half-angle) nonlinear LM solver with
  analytic Jacobian; handles near-degenerate apex configurations
- **Curved cylinder fitter** — windowed spine estimation using a sliding-window
  local cylinder fit to recover a smooth polynomial spine for bent pipes and
  curved structural members
- **Pipe network fitter** — automatic segment extraction via Euclidean clustering
  followed by per-segment cylinder fitting and branch-point detection through
  pairwise axis intersection

#### Numerical methods
- Analytic, closed-form Jacobian matrices for all LM solvers, eliminating
  finite-difference approximation and giving 3–10× speed-up on typical inputs
- Numerically stable axis-direction parameterisation using spherical coordinates
  to avoid gimbal lock during optimisation
- Robust centring and scaling pre-conditioning before every nonlinear solve

#### File I/O
- PLY, PCD, LAS/LAZ, XYZ, and CSV reader/writer with automatic format detection
- Thin Open3D adapter (`cylinderfit2026.io.open3d_adapter`) for zero-copy
  interchange with `open3d.geometry.PointCloud`

#### Quality assurance
- **GitHub Actions CI** — matrix across ubuntu-latest / macos-latest /
  windows-latest × Python 3.9, 3.10, 3.11, 3.12; runs on every push and pull
  request
- **Property-based tests** (Hypothesis) — randomly generated cylinders are fit and
  the recovered parameters are checked against ground truth within tolerance
- **Golden-value regression tests** — fixed random seeds produce byte-identical
  parameter arrays; any algorithmic change that shifts values fails the suite
- **Statistical bias tests** — Monte-Carlo sweeps over noise levels verify that
  parameter estimates are unbiased and that standard deviations scale as expected
- **Cross-implementation validation** — results compared against a pure-NumPy
  reference implementation for every geometry type

#### Documentation
- Sphinx docs with `autodoc`, `napoleon`, and `viewcode` extensions
- API reference auto-generated from NumPy-style docstrings
- Four worked example notebooks in `examples/`

[Unreleased]: https://github.com/YOUR_USERNAME/cylinderfit2026/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/YOUR_USERNAME/cylinderfit2026/releases/tag/v0.1.0
