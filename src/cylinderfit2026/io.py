"""Point cloud I/O helpers.

Supports PLY (ASCII + binary little-endian), PCD (ASCII + binary), and LAS/LAZ
(requires ``laspy``).  An Open3D adapter converts ``o3d.geometry.PointCloud``
objects without any additional dependencies beyond open3d itself.

All loaders return an ``N x 3`` float64 NumPy array ready to pass straight into
``fit_cylinder`` and friends.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import numpy as np


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_points(path: PathLike) -> np.ndarray:
    """Load a point cloud from *path* and return an ``N x 3`` float64 array.

    Format is inferred from the file extension:

    * ``.ply``  — ASCII or binary-little-endian PLY
    * ``.pcd``  — ASCII or binary PCD (PCL format)
    * ``.las`` / ``.laz`` — LAS/LAZ (requires ``laspy``)
    * ``.xyz`` / ``.txt`` / ``.csv`` — whitespace- or comma-delimited XYZ
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".ply":
        return _load_ply(p)
    if suffix == ".pcd":
        return _load_pcd(p)
    if suffix in (".las", ".laz"):
        return _load_las(p)
    if suffix in (".xyz", ".txt", ".csv"):
        return _load_xyz(p)
    raise ValueError(f"Unsupported file extension '{suffix}'. "
                     "Supported: .ply, .pcd, .las, .laz, .xyz, .txt, .csv")


# ---------------------------------------------------------------------------
# Open3D adapter
# ---------------------------------------------------------------------------

def from_open3d(cloud) -> np.ndarray:
    """Convert an ``open3d.geometry.PointCloud`` to an ``N x 3`` float64 array.

    Parameters
    ----------
    cloud:
        An ``open3d.geometry.PointCloud`` instance.

    Returns
    -------
    np.ndarray
        ``N x 3`` float64 array of XYZ coordinates.

    Raises
    ------
    TypeError
        If *cloud* is not an Open3D PointCloud.
    ValueError
        If the cloud contains no points.
    """
    try:
        pts = np.asarray(cloud.points, dtype=float)
    except AttributeError as exc:
        raise TypeError(
            "Expected an open3d.geometry.PointCloud; "
            f"got {type(cloud).__name__}"
        ) from exc
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("open3d cloud has unexpected point array shape")
    if pts.shape[0] == 0:
        raise ValueError("open3d cloud is empty")
    return np.ascontiguousarray(pts)


# ---------------------------------------------------------------------------
# PLY loader  (ASCII + binary-little-endian, xyz only)
# ---------------------------------------------------------------------------

def _load_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()

    # Parse header (always ASCII)
    header_end = raw.find(b"end_header")
    if header_end == -1:
        raise ValueError(f"No 'end_header' found in PLY file: {path}")
    header_bytes = raw[:header_end]
    data_start = header_end + len(b"end_header")
    # skip the newline after end_header
    while data_start < len(raw) and raw[data_start:data_start+1] in (b"\r", b"\n"):
        data_start += 1

    header = header_bytes.decode("ascii", errors="replace")
    lines = [l.strip() for l in header.splitlines()]

    fmt = "ascii"
    n_vertices = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False

    for line in lines:
        if line.startswith("format"):
            parts = line.split()
            fmt = parts[1] if len(parts) > 1 else "ascii"
        elif line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element") and not line.startswith("element vertex"):
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            parts = line.split()
            if len(parts) >= 3:
                properties.append((parts[1], parts[2]))

    prop_names = [p[1] for p in properties]
    # Find x, y, z column indices
    try:
        xi, yi, zi = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
    except ValueError as exc:
        raise ValueError(f"PLY file does not contain x/y/z properties: {path}") from exc

    if fmt == "ascii":
        data = raw[data_start:].decode("ascii", errors="replace")
        rows = []
        for line in data.splitlines():
            vals = line.split()
            if len(vals) >= len(properties):
                rows.append((float(vals[xi]), float(vals[yi]), float(vals[zi])))
                if len(rows) == n_vertices:
                    break
        return np.array(rows, dtype=float)

    # Binary little-endian
    _ply_type_map = {
        "char": ("b", 1), "uchar": ("B", 1), "short": ("h", 2), "ushort": ("H", 2),
        "int": ("i", 4), "uint": ("I", 4), "float": ("f", 4), "double": ("d", 8),
        "int8": ("b", 1), "uint8": ("B", 1), "int16": ("h", 2), "uint16": ("H", 2),
        "int32": ("i", 4), "uint32": ("I", 4), "float32": ("f", 4), "float64": ("d", 8),
    }
    fmts = [_ply_type_map.get(t, ("f", 4)) for t, _ in properties]
    row_size = sum(s for _, s in fmts)
    struct_fmt = "<" + "".join(c for c, _ in fmts)
    pts = np.empty((n_vertices, 3), dtype=float)
    offset = data_start
    for i in range(n_vertices):
        vals = struct.unpack_from(struct_fmt, raw, offset)
        pts[i, 0] = vals[xi]
        pts[i, 1] = vals[yi]
        pts[i, 2] = vals[zi]
        offset += row_size
    return pts


# ---------------------------------------------------------------------------
# PCD loader  (ASCII + binary, PCL format)
# ---------------------------------------------------------------------------

def _load_pcd(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()

    header_lines = []
    pos = 0
    data_type = "ascii"
    width = height = points = 0
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []

    while pos < len(raw):
        end = raw.find(b"\n", pos)
        if end == -1:
            end = len(raw)
        line = raw[pos:end].decode("ascii", errors="replace").strip()
        pos = end + 1
        header_lines.append(line)
        if line.startswith("DATA"):
            data_type = line.split()[-1].lower()
            break
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("SIZE"):
            sizes = [int(s) for s in line.split()[1:]]
        elif line.startswith("TYPE"):
            types = line.split()[1:]
        elif line.startswith("WIDTH"):
            width = int(line.split()[-1])
        elif line.startswith("HEIGHT"):
            height = int(line.split()[-1])
        elif line.startswith("POINTS"):
            points = int(line.split()[-1])

    n = points or width * height
    try:
        xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError as exc:
        raise ValueError(f"PCD file does not contain x/y/z fields: {path}") from exc

    if data_type == "ascii":
        data = raw[pos:].decode("ascii", errors="replace")
        rows = []
        for line in data.splitlines():
            vals = line.split()
            if len(vals) >= len(fields):
                rows.append((float(vals[xi]), float(vals[yi]), float(vals[zi])))
                if len(rows) == n:
                    break
        return np.array(rows, dtype=float)

    # Binary PCD
    _pcd_type_map = {
        ("F", 4): "f", ("F", 8): "d",
        ("I", 1): "b", ("I", 2): "h", ("I", 4): "i",
        ("U", 1): "B", ("U", 2): "H", ("U", 4): "I",
    }
    fmt_chars = [_pcd_type_map.get((t, s), "f") for t, s in zip(types, sizes)]
    struct_fmt = "<" + "".join(fmt_chars)
    row_bytes = sum(sizes)
    pts = np.empty((n, 3), dtype=float)
    for i in range(n):
        vals = struct.unpack_from(struct_fmt, raw, pos)
        pts[i, 0] = vals[xi]
        pts[i, 1] = vals[yi]
        pts[i, 2] = vals[zi]
        pos += row_bytes
    return pts


# ---------------------------------------------------------------------------
# LAS/LAZ loader  (requires laspy)
# ---------------------------------------------------------------------------

def _load_las(path: Path) -> np.ndarray:
    try:
        import laspy  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "laspy is required to load .las/.laz files. "
            "Install it with: pip install laspy[lazrs]"
        ) from exc
    las = laspy.read(str(path))
    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)
    z = np.asarray(las.z, dtype=float)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Plain XYZ / CSV loader
# ---------------------------------------------------------------------------

def _load_xyz(path: Path) -> np.ndarray:
    # Sniff delimiter from the first non-comment data line
    delim = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "//")):
                if "," in stripped:
                    delim = ","
                break
    pts = np.loadtxt(str(path), comments=["#", "//"], delimiter=delim, usecols=(0, 1, 2))
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return pts.astype(float)
