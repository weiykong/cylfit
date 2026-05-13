"""Tests for file I/O and Open3D adapter."""
import struct

import numpy as np
import pytest

from cylfit import fit_cylinder, from_open3d, load_points
from cylfit.synthetic import generate_noisy_cylinder


def _write_ply_ascii(path, pts):
    n = len(pts)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for row in pts:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")


def _write_ply_binary(path, pts):
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for row in pts:
            f.write(struct.pack("<fff", *row))


def _write_pcd_ascii(path, pts):
    n = len(pts)
    with open(path, "w") as f:
        f.write("VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n")
        f.write(f"COUNT 1 1 1\nWIDTH {n}\nHEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\nDATA ascii\n")
        for row in pts:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")


class MockO3DCloud:
    """Duck-typed Open3D PointCloud for testing without installing open3d."""
    def __init__(self, pts):
        self.points = pts


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

class TestLoadPLY:
    def test_ascii_roundtrip(self, tmp_path):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((200, 3))
        p = str(tmp_path / "cloud.ply")
        _write_ply_ascii(p, pts)
        loaded = load_points(p)
        assert loaded.shape == (200, 3)
        np.testing.assert_allclose(loaded, pts, atol=1e-5)

    def test_binary_roundtrip(self, tmp_path):
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((300, 3)).astype(np.float32)
        p = str(tmp_path / "cloud.ply")
        _write_ply_binary(p, pts)
        loaded = load_points(p)
        assert loaded.shape == (300, 3)
        np.testing.assert_allclose(loaded, pts, atol=1e-5)

    def test_missing_xyz_raises(self, tmp_path):
        p = str(tmp_path / "bad.ply")
        with open(p, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\n"
                    "property float r\nend_header\n1.0\n")
        with pytest.raises(ValueError, match="x/y/z"):
            load_points(p)


# ---------------------------------------------------------------------------
# PCD
# ---------------------------------------------------------------------------

class TestLoadPCD:
    def test_ascii_roundtrip(self, tmp_path):
        rng = np.random.default_rng(2)
        pts = rng.standard_normal((150, 3))
        p = str(tmp_path / "cloud.pcd")
        _write_pcd_ascii(p, pts)
        loaded = load_points(p)
        assert loaded.shape == (150, 3)
        np.testing.assert_allclose(loaded, pts, atol=1e-6)

    def test_missing_xyz_raises(self, tmp_path):
        p = str(tmp_path / "bad.pcd")
        with open(p, "w") as f:
            f.write("VERSION .7\nFIELDS r g b\nSIZE 1 1 1\nTYPE U U U\n"
                    "COUNT 1 1 1\nWIDTH 1\nHEIGHT 1\n"
                    "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA ascii\n255 0 0\n")
        with pytest.raises(ValueError, match="x/y/z"):
            load_points(p)


# ---------------------------------------------------------------------------
# XYZ / CSV
# ---------------------------------------------------------------------------

class TestLoadXYZ:
    def test_space_delimited(self, tmp_path):
        rng = np.random.default_rng(3)
        pts = rng.standard_normal((100, 3))
        p = str(tmp_path / "cloud.xyz")
        np.savetxt(p, pts)
        loaded = load_points(p)
        assert loaded.shape == (100, 3)
        np.testing.assert_allclose(loaded, pts, atol=1e-6)

    def test_csv(self, tmp_path):
        rng = np.random.default_rng(4)
        pts = rng.standard_normal((80, 3))
        p = str(tmp_path / "cloud.csv")
        np.savetxt(p, pts, delimiter=",")
        loaded = load_points(p)
        assert loaded.shape == (80, 3)


# ---------------------------------------------------------------------------
# Unknown extension
# ---------------------------------------------------------------------------

class TestLoadUnknownExt:
    def test_raises(self, tmp_path):
        p = str(tmp_path / "cloud.e57")
        open(p, "w").close()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_points(p)


# ---------------------------------------------------------------------------
# from_open3d adapter
# ---------------------------------------------------------------------------

class TestFromOpen3D:
    def test_duck_type_mock(self):
        rng = np.random.default_rng(5)
        pts = rng.standard_normal((400, 3))
        cloud = MockO3DCloud(pts)
        arr = from_open3d(cloud)
        assert arr.shape == (400, 3)
        assert arr.dtype == float
        np.testing.assert_allclose(arr, pts)

    def test_float32_input_promoted(self):
        pts = np.ones((50, 3), dtype=np.float32)
        arr = from_open3d(MockO3DCloud(pts))
        assert arr.dtype == float

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            from_open3d("not a cloud")

    def test_empty_cloud_raises(self):
        with pytest.raises(ValueError):
            from_open3d(MockO3DCloud(np.empty((0, 3))))

    def test_end_to_end_fit_from_ply(self, tmp_path):
        """load_points → fit_cylinder round-trip using synthetic data."""
        synthetic = generate_noisy_cylinder(n_points=1000, noise=0.01, random_state=7)
        pts = synthetic.points
        p = str(tmp_path / "cylinder.ply")
        _write_ply_ascii(p, pts)
        loaded = load_points(p)
        model = fit_cylinder(loaded, threshold=0.05, ransac_trials=32, random_state=7)
        assert abs(model.radius - synthetic.radius) < 0.1
