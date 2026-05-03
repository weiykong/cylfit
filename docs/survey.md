# 2026 Cylinder Fitting Survey

Survey date: 2026-05-02.

## MathWorks / MATLAB

- [`pcfitcylinder`](https://www.mathworks.com/help/vision/ref/pcfitcylinder.html): official MATLAB function for fitting a cylinder to a 3-D point cloud. It uses MSAC, supports orientation constraints, sample indices, confidence, and trial limits. Its fitting algorithm requires point-cloud normals; if normals are absent, MATLAB fills them internally.
- [`cylinderModel`](https://www.mathworks.com/help/vision/ref/cylindermodel.html): MATLAB geometric model object returned by `pcfitcylinder`.
- [`Least-squares adjustment of Points to Cylindrical Surfaces`](https://www.mathworks.com/matlabcentral/fileexchange/116905-least-squares-adjustment-of-points-to-cylindrical-surfaces): MATLAB Central File Exchange package by Hao-En Lin. Version 1.0.5 was published 2025-01-29. It fits seven cylinder parameters by least squares and converts the result to endpoints. The author notes it is suitable for cylinders whose height exceeds radius.
- MATLAB Answers discussions show repeated pain points around passing plain `N x 3` data instead of `pointCloud` objects, bad fits without orientation constraints, and users wanting known-radius or toolbox-free variants.

## GitHub / Open Source

- [`leomariga/pyRANSAC-3D`](https://github.com/leomariga/pyRANSAC-3D): popular Python RANSAC primitive package supporting cylinders among other shapes. It is NumPy-based and easy to install, but issue discussions report cylinder instability and normal-related limitations.
- [`cylinder_fitting`](https://pypi.org/project/cylinder_fitting/): small PyPI package last released in 2017. It exposes a simple `fit(data)` function and depends on NumPy/SciPy.
- [`Open3D`](https://www.open3d.org/): important Python 3-D processing library with point-cloud I/O, normal estimation, clustering, visualization, and plane segmentation. It is useful around cylinder fitting workflows, but does not provide a direct `fit_cylinder` equivalent in the Python point-cloud API.
- [`InverseTampere/Cylinder_Fitting`](https://github.com/InverseTampere/Cylinder_Fitting): MATLAB code associated with 2025 work on fitting geometric shapes to fuzzy point-cloud data using expected Mahalanobis distance.
- [`cserteGT3/RANSAC.jl`](https://csertegt3.github.io/RANSAC.jl/dev/): Julia implementation of efficient RANSAC for point-cloud primitive detection, including cylinders.
- [`mikacuy/point2cyl`](https://github.com/mikacuy/point2cyl): CVPR 2022 research code for decomposing point clouds into extrusion cylinders using supervised learning. It targets object decomposition rather than a lightweight single-cylinder fitter.
- PCL examples and bindings expose RANSAC cylinder segmentation, usually with normal estimation and C++/PCL setup.

## Improvement Opportunities

1. Plain array API: accept `N x 3` directly and return simple model fields.
2. No mandatory normals: use point-to-axis radial distances as the primary residual.
3. Robust defaults: combine a RANSAC bootstrap with robust refinement instead of exposing only one brittle method.
4. Deterministic reproducibility: provide `random_state`.
5. Efficient implementation: keep distance calculations vectorized and cap expensive optimization samples for very large clouds.
6. Better diagnostics: return residuals, inlier mask, RMSE/MAE, and axis extent.
7. Extensible constraints: leave room for known radius, orientation priors, and uncertainty-weighted residuals in later versions.

## Implemented in This Package

- Robust default fitter with RANSAC/PCA guarded initialization.
- Known-radius fitting.
- Normal-seeded fitting.
- Point-weighted fitting for uncertainty-aware scans.
- Fixed-axis and orientation-constrained fitting.
- Adaptive RANSAC early stopping.
- Robust trimming, voxel downsampling, and auto-fit preprocessing.
- Bootstrap uncertainty estimation.
- Multi-cylinder detection with optional Euclidean pre-clustering.
- Synthetic noisy/outlier data generator.
- Visualization, JSON export, triangle mesh export, and fit quality measures.
- Benchmark runner against internal baselines and optional Python packages.
