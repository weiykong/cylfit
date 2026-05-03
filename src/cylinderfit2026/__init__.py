"""CylinderFit 2026 public API."""

from .io import from_open3d, load_points
from .elliptical import EllipticalCylinderModel, fit_elliptical_cylinder
from .cone import ConeModel, fit_cone
from .curved import CurvedCylinderModel, fit_curved_cylinder
from .network import CylinderJoint, PipeNetwork, build_pipe_network, find_cylinder_joints
from .core import (
    CylinderModel,
    fit_cylinder,
    fit_cylinder_constrained_axis,
    fit_cylinder_fixed_axis,
    fit_cylinder_known_radius,
    fit_cylinder_with_normals,
    residuals_to_cylinder,
)
from .benchmarks import (
    BenchmarkCase,
    BenchmarkResult,
    available_methods,
    benchmark_markdown_report,
    default_benchmark_cases,
    run_benchmark,
    summarize_benchmark,
)
from .detect import DetectedCylinder, detect_cylinders
from .metrics import FitMeasures, FitMetrics, evaluate_fit, measure_fit
from .preprocess import PreprocessResult, fit_cylinder_auto, robust_spatial_trim, voxel_downsample
from .synthetic import generate_noisy_cylinder
from .uncertainty import FitUncertainty, estimate_uncertainty
from .visualize import plot_cylinder_fit

__all__ = [
    "CylinderModel",
    "BenchmarkCase",
    "BenchmarkResult",
    "DetectedCylinder",
    "FitMetrics",
    "FitMeasures",
    "FitUncertainty",
    "PreprocessResult",
    "available_methods",
    "benchmark_markdown_report",
    "detect_cylinders",
    "default_benchmark_cases",
    "evaluate_fit",
    "fit_cylinder",
    "fit_cylinder_auto",
    "fit_cylinder_constrained_axis",
    "fit_cylinder_fixed_axis",
    "fit_cylinder_known_radius",
    "fit_cylinder_with_normals",
    "generate_noisy_cylinder",
    "measure_fit",
    "plot_cylinder_fit",
    "residuals_to_cylinder",
    "robust_spatial_trim",
    "run_benchmark",
    "summarize_benchmark",
    "estimate_uncertainty",
    "voxel_downsample",
    # I/O
    "load_points",
    "from_open3d",
    # Extended fitting
    "EllipticalCylinderModel",
    "fit_elliptical_cylinder",
    "ConeModel",
    "fit_cone",
    "CurvedCylinderModel",
    "fit_curved_cylinder",
    # Pipe network
    "CylinderJoint",
    "PipeNetwork",
    "find_cylinder_joints",
    "build_pipe_network",
]
