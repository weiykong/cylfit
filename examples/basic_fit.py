import numpy as np

from cylfit import fit_cylinder


def main() -> None:
    rng = np.random.default_rng(42)
    axis = np.array([0.2, 0.6, 0.77])
    axis /= np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    u = helper - (helper @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    theta = rng.uniform(0.0, 2.0 * np.pi, 4000)
    t = rng.uniform(-4.0, 4.0, 4000)
    radius = 1.25
    points = t[:, None] * axis + radius * np.cos(theta)[:, None] * u + radius * np.sin(theta)[:, None] * v
    points += rng.normal(scale=0.01, size=points.shape)

    model = fit_cylinder(points, threshold=0.04, ransac_trials=64, random_state=42)
    print("axis_point:", model.axis_point)
    print("axis_direction:", model.axis_direction)
    print("radius:", model.radius)
    print("height:", model.height)
    print("rmse:", model.rmse)


if __name__ == "__main__":
    main()
