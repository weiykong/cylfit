import unittest

import numpy as np

from cylfit import (
    detect_cylinders,
    estimate_uncertainty,
    evaluate_fit,
    fit_cylinder,
    fit_cylinder_auto,
    fit_cylinder_constrained_axis,
    fit_cylinder_fixed_axis,
    fit_cylinder_known_radius,
    fit_cylinder_with_normals,
    generate_noisy_cylinder,
    measure_fit,
    run_benchmark,
    summarize_benchmark,
    benchmark_markdown_report,
    voxel_downsample,
)


def make_cylinder(n=2500, radius=2.5, height=10.0, noise=0.015, outliers=0.0, seed=3):
    rng = np.random.default_rng(seed)
    axis = np.array([0.35, -0.4, 0.847])
    axis = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(axis @ helper) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = helper - (helper @ axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    center = np.array([1.0, -2.0, 0.5])

    theta = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(-height / 2, height / 2, n)
    pts = center + t[:, None] * axis
    pts += radius * np.cos(theta)[:, None] * u
    pts += radius * np.sin(theta)[:, None] * v
    pts += rng.normal(scale=noise, size=pts.shape)

    if outliers:
        m = int(n * outliers)
        extra = rng.uniform(-7, 7, size=(m, 3))
        pts = np.vstack([pts, extra])
    return pts, axis, radius


class CylinderFitTests(unittest.TestCase):
    def test_fit_clean_cylinder(self):
        points, axis, radius = make_cylinder()
        model = fit_cylinder(points, threshold=0.08, ransac_trials=32, random_state=9)

        self.assertLess(
            min(np.linalg.norm(model.axis_direction - axis), np.linalg.norm(model.axis_direction + axis)),
            0.03,
        )
        self.assertLess(abs(model.radius - radius), 0.04)
        self.assertLess(model.rmse, 0.04)

    def test_fit_with_outliers(self):
        points, axis, radius = make_cylinder(outliers=0.25)
        model = fit_cylinder(points, threshold=0.09, ransac_trials=96, random_state=12)

        self.assertLess(
            min(np.linalg.norm(model.axis_direction - axis), np.linalg.norm(model.axis_direction + axis)),
            0.06,
        )
        self.assertLess(abs(model.radius - radius), 0.08)
        self.assertGreater(model.inlier_mask.mean(), 0.65)

    def test_known_radius_fit_keeps_radius_fixed(self):
        synthetic = generate_noisy_cylinder(radius=1.75, noise=0.02, outlier_fraction=0.08, random_state=5)
        model = fit_cylinder_known_radius(
            synthetic.points,
            1.75,
            threshold=0.08,
            ransac_trials=48,
            random_state=5,
        )

        self.assertAlmostEqual(model.radius, 1.75)
        self.assertLess(evaluate_fit(model, synthetic).axis_angle_deg, 2.5)

    def test_normal_aware_fit_uses_normals(self):
        synthetic = generate_noisy_cylinder(noise=0.015, outlier_fraction=0.05, random_state=8)
        model = fit_cylinder_with_normals(
            synthetic.points,
            synthetic.normals,
            threshold=0.06,
            ransac_trials=16,
            random_state=8,
        )

        metrics = evaluate_fit(model, synthetic)
        self.assertLess(metrics.axis_angle_deg, 2.0)
        self.assertLess(metrics.radius_error, 0.05)

    def test_model_exports_dict_json_and_mesh(self):
        synthetic = generate_noisy_cylinder(n_points=1200, random_state=2)
        model = fit_cylinder(synthetic.points, threshold=0.08, ransac_trials=24, random_state=2)
        vertices, faces = model.to_mesh(n_theta=12, n_height=4)

        self.assertIn("radius", model.to_dict())
        self.assertIn('"radius"', model.to_json())
        self.assertEqual(vertices.shape, (48, 3))
        self.assertEqual(faces.shape, (72, 3))

    def test_fixed_axis_and_measures(self):
        synthetic = generate_noisy_cylinder(noise=0.01, outlier_fraction=0.05, random_state=13)
        model = fit_cylinder_fixed_axis(
            synthetic.points,
            synthetic.axis_direction,
            threshold=0.05,
        )
        measures = measure_fit(synthetic.points, model)

        self.assertLess(abs(model.radius - synthetic.radius), 0.04)
        self.assertGreater(measures.angular_coverage_deg, 300.0)
        self.assertGreater(measures.quality_score, 60.0)

    def test_constrained_axis_and_weighted_fit(self):
        synthetic = generate_noisy_cylinder(noise=0.02, outlier_fraction=0.08, random_state=14)
        weights = np.where(synthetic.inlier_mask, 1.0, 0.15)
        model = fit_cylinder_constrained_axis(
            synthetic.points,
            synthetic.axis_direction,
            max_axis_angle_deg=4.0,
            point_weights=weights,
            threshold=0.08,
            ransac_trials=48,
            ransac_stop_inlier_fraction=0.95,
            random_state=14,
        )

        metrics = evaluate_fit(model, synthetic)
        self.assertLess(metrics.axis_angle_deg, 4.0)
        self.assertLess(metrics.radius_error, 0.05)

    def test_detect_two_cylinders(self):
        a = generate_noisy_cylinder(
            n_points=1200,
            radius=0.8,
            height=4.0,
            axis_point=np.array([-3.0, 0.0, 0.0]),
            noise=0.01,
            outlier_fraction=0.0,
            random_state=11,
        )
        b = generate_noisy_cylinder(
            n_points=1200,
            radius=1.1,
            height=5.0,
            axis_point=np.array([3.0, 0.0, 0.0]),
            noise=0.01,
            outlier_fraction=0.0,
            random_state=12,
        )
        points = np.vstack([a.points, b.points])
        detections = detect_cylinders(
            points,
            max_cylinders=2,
            threshold=0.05,
            min_inliers=500,
            cluster_eps=0.45,
            ransac_trials=64,
            random_state=3,
        )

        self.assertEqual(len(detections), 2)

    def test_preprocess_auto_and_uncertainty(self):
        synthetic = generate_noisy_cylinder(n_points=1800, noise=0.02, outlier_fraction=0.08, random_state=22)
        downsampled = voxel_downsample(synthetic.points, voxel_size=0.08)
        model = fit_cylinder_auto(
            synthetic.points,
            voxel_size=0.08,
            threshold=0.08,
            ransac_trials=24,
            random_state=22,
        )
        uncertainty = estimate_uncertainty(
            synthetic.points,
            model,
            samples=8,
            sample_fraction=0.65,
            random_state=22,
        )

        self.assertLess(downsampled.points.shape[0], synthetic.points.shape[0])
        self.assertLess(abs(model.radius - synthetic.radius), 0.06)
        self.assertEqual(uncertainty.samples, 8)
        self.assertGreaterEqual(uncertainty.radius_ci95[1], uncertainty.radius_ci95[0])

    def test_benchmark_summary_and_report(self):
        synthetic = generate_noisy_cylinder(n_points=900, noise=0.01, outlier_fraction=0.02, random_state=23)
        from cylfit.benchmarks import BenchmarkCase

        results = run_benchmark(
            cases=[BenchmarkCase("small", synthetic, 0.05)],
            methods=None,
            random_state=23,
        )
        summary = summarize_benchmark(results)
        report = benchmark_markdown_report(results)

        self.assertIn("cylfit", summary)
        self.assertIn("CylinderFit Benchmark Report", report)


if __name__ == "__main__":
    unittest.main()
