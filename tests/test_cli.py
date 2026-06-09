"""Tests for the cylfit CLI entry point (cylfit.cli:main)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cylfit import generate_noisy_cylinder
from cylfit.cli import main


@pytest.fixture(scope="module")
def csv_file(tmp_path_factory):
    """Write a deterministic noisy cylinder to a temp CSV file."""
    syn = generate_noisy_cylinder(n_points=300, radius=1.0, height=3.0, noise=0.02, random_state=42)
    p = tmp_path_factory.mktemp("cli") / "points.txt"
    np.savetxt(str(p), syn.points)  # space-delimited (CLI default)
    return p


def _invoke(monkeypatch, capsys, *args: str):
    """Run main() with patched sys.argv; return (stdout, exit_code)."""
    monkeypatch.setattr("sys.argv", ["cylfit", *args])
    try:
        main()
        exit_code = 0
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    return capsys.readouterr().out, exit_code


class TestCLIBasic:
    def test_fit_returns_zero_exit_code(self, monkeypatch, capsys, csv_file):
        _, code = _invoke(monkeypatch, capsys, str(csv_file), "--random-state", "0")
        assert code == 0

    def test_output_is_valid_json(self, monkeypatch, capsys, csv_file):
        out, _ = _invoke(monkeypatch, capsys, str(csv_file), "--random-state", "0")
        payload = json.loads(out)
        assert isinstance(payload, dict)

    def test_output_has_required_keys(self, monkeypatch, capsys, csv_file):
        out, _ = _invoke(monkeypatch, capsys, str(csv_file), "--random-state", "0")
        payload = json.loads(out)
        for key in ("radius", "rmse", "axis_direction", "axis_point"):
            assert key in payload, f"missing key: {key}"

    def test_radius_close_to_truth(self, monkeypatch, capsys, csv_file):
        out, _ = _invoke(monkeypatch, capsys, str(csv_file), "--random-state", "0")
        payload = json.loads(out)
        assert abs(payload["radius"] - 1.0) < 0.15

    def test_measures_flag(self, monkeypatch, capsys, csv_file):
        out, code = _invoke(monkeypatch, capsys, str(csv_file), "--random-state", "0", "--measures")
        assert code == 0
        payload = json.loads(out)
        assert "measures" in payload

    def test_known_radius_flag(self, monkeypatch, capsys, csv_file):
        out, code = _invoke(monkeypatch, capsys, str(csv_file), "--known-radius", "1.0",
                            "--random-state", "0")
        assert code == 0
        payload = json.loads(out)
        assert abs(payload["radius"] - 1.0) < 1e-9

    def test_delimiter_flag(self, monkeypatch, capsys, tmp_path):
        syn = generate_noisy_cylinder(n_points=100, radius=1.0, height=2.0, noise=0.01, random_state=7)
        p = tmp_path / "points_tab.txt"
        np.savetxt(str(p), syn.points, delimiter="\t")
        out, code = _invoke(monkeypatch, capsys, str(p), "--delimiter", "\t", "--random-state", "0")
        assert code == 0

    def test_missing_file_exits_nonzero(self, monkeypatch, capsys):
        with pytest.raises((SystemExit, Exception)):
            _invoke(monkeypatch, capsys, "/nonexistent/path/points.csv")

    def test_ransac_trials_flag(self, monkeypatch, capsys, csv_file):
        out, code = _invoke(monkeypatch, capsys, str(csv_file), "--ransac-trials", "32",
                            "--random-state", "0")
        assert code == 0
        payload = json.loads(out)
        assert "radius" in payload
