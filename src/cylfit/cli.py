from __future__ import annotations

import argparse
import json

import numpy as np

from .core import fit_cylinder
from .metrics import measure_fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a cylinder to an N x 3 point cloud.")
    parser.add_argument("points", help="CSV/TXT file with x,y,z columns")
    parser.add_argument("--delimiter", default=None, help="Input delimiter. Defaults to NumPy auto parsing.")
    parser.add_argument("--threshold", type=float, default=None, help="Inlier residual threshold")
    parser.add_argument("--ransac-trials", type=int, default=128, help="RANSAC bootstrap trials")
    parser.add_argument("--known-radius", type=float, default=None, help="Fix the cylinder radius")
    parser.add_argument("--measures", action="store_true", help="Include fit quality measures")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    points = np.loadtxt(args.points, delimiter=args.delimiter)
    model = fit_cylinder(
        points,
        threshold=args.threshold,
        ransac_trials=args.ransac_trials,
        known_radius=args.known_radius,
        random_state=args.random_state,
    )

    payload = model.to_dict()
    if args.measures:
        payload["measures"] = measure_fit(points, model).to_dict()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
