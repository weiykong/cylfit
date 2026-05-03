from pathlib import Path

from cylinderfit2026 import fit_cylinder, generate_noisy_cylinder, plot_cylinder_fit


def main() -> None:
    synthetic = generate_noisy_cylinder(
        n_points=7000,
        radius=1.35,
        height=7.5,
        noise=0.035,
        outlier_fraction=0.18,
        partial_arc=0.82,
        random_state=21,
    )
    model = fit_cylinder(
        synthetic.points,
        threshold=0.12,
        ransac_trials=128,
        random_state=21,
    )
    fig, _ = plot_cylinder_fit(
        synthetic.points,
        model,
        title="CylinderFit 2026 synthetic noisy point cloud",
        random_state=21,
    )
    out = Path(__file__).with_name("synthetic_visualization.png")
    fig.savefig(out, dpi=180)
    print(out)
    print(f"radius={model.radius:.5f}, height={model.height:.5f}, rmse={model.rmse:.5f}")


if __name__ == "__main__":
    main()
