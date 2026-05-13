"""Cylinder network / pipe graph analysis.

After detecting a set of cylinders with :func:`~cylfit.detect_cylinders`,
this module infers which cylinders are connected, the connection geometry, and
builds a graph description of the pipe network.

A **joint** is formed between two cylinders when the minimum distance between
their axis line segments falls below a distance threshold.  The joint records:

* which two cylinders are involved
* the closest point on each cylinder axis (the *handshake points*)
* the gap between them (0 if they actually intersect)
* the angle between their axes

The resulting :class:`PipeNetwork` exposes the joints as a list and the
cylinders as nodes.  An optional adjacency list gives a simple graph structure
for downstream traversal.

Usage::

    from cylfit import detect_cylinders
    from cylfit.network import find_cylinder_joints, build_pipe_network

    detections  = detect_cylinders(points, max_cylinders=6)
    joints      = find_cylinder_joints(detections)
    network     = build_pipe_network(detections, joints)

    for joint in network.joints:
        print(joint.cylinder_a_idx, "↔", joint.cylinder_b_idx,
              f"  gap={joint.gap:.3f}  angle={joint.angle_deg:.1f}°")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .detect import DetectedCylinder

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CylinderJoint:
    """A connection between two cylinders.

    Attributes
    ----------
    cylinder_a_idx, cylinder_b_idx:
        Indices into the list of cylinders passed to :func:`find_cylinder_joints`.
    point_a:
        Closest point on cylinder A's axis line segment.
    point_b:
        Closest point on cylinder B's axis line segment.
    gap:
        Distance between *point_a* and *point_b* (zero if axes cross).
    midpoint:
        Geometric midpoint of the joint, useful as the junction location.
    angle_deg:
        Angle between the two cylinder axes in degrees (always in [0°, 90°]).
    """

    cylinder_a_idx: int
    cylinder_b_idx: int
    point_a: np.ndarray
    point_b: np.ndarray
    gap: float
    midpoint: np.ndarray
    angle_deg: float

    def to_dict(self) -> dict:
        return {
            "cylinder_a": self.cylinder_a_idx,
            "cylinder_b": self.cylinder_b_idx,
            "point_a": self.point_a.tolist(),
            "point_b": self.point_b.tolist(),
            "gap": self.gap,
            "midpoint": self.midpoint.tolist(),
            "angle_deg": self.angle_deg,
        }


@dataclass
class PipeNetwork:
    """A graph representation of a set of connected cylinders.

    Attributes
    ----------
    cylinders:
        List of detected cylinders (nodes).
    joints:
        List of joints between cylinders (edges).
    adjacency:
        ``adjacency[i]`` is the list of cylinder indices connected to cylinder ``i``.
    """

    cylinders: List[DetectedCylinder]
    joints: List[CylinderJoint]
    adjacency: List[List[int]]

    def to_dict(self) -> dict:
        return {
            "n_cylinders": len(self.cylinders),
            "n_joints": len(self.joints),
            "joints": [j.to_dict() for j in self.joints],
            "adjacency": self.adjacency,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def neighbors(self, idx: int) -> List[int]:
        """Return the indices of cylinders connected to cylinder *idx*."""
        return list(self.adjacency[idx])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_cylinder_joints(
    cylinders: Sequence[DetectedCylinder],
    *,
    distance_threshold: Optional[float] = None,
    angle_threshold_deg: float = 160.0,
) -> List[CylinderJoint]:
    """Find joints (connections) between cylinders based on axis proximity.

    Two cylinders are considered connected if the minimum distance between
    their axis line segments is below *distance_threshold*.  If the threshold
    is ``None`` it is auto-estimated as ``max(r_a + r_b)`` over all pairs —
    i.e., cylinders are connected when their surfaces touch or overlap.

    Parameters
    ----------
    cylinders:
        List of :class:`~cylfit.DetectedCylinder` objects, typically
        the output of :func:`~cylfit.detect_cylinders`.
    distance_threshold:
        Maximum axis-to-axis gap to consider two cylinders connected.
        ``None`` uses the sum of the two radii as the threshold (surface contact).
    angle_threshold_deg:
        Maximum axis-to-axis angle (in degrees) for a valid joint.  Pairs with
        a larger angle are filtered out (near-parallel reversed axes are still
        kept because angle is measured as min(θ, 180°−θ)).

    Returns
    -------
    list of CylinderJoint
        Sorted by gap (smallest first).
    """
    n = len(cylinders)
    joints: List[CylinderJoint] = []

    for i in range(n):
        for j in range(i + 1, n):
            mod_a = cylinders[i].model
            mod_b = cylinders[j].model

            thr = (
                float(distance_threshold)
                if distance_threshold is not None
                else mod_a.radius + mod_b.radius
            )

            gap, pa, pb = _segment_to_segment_distance(
                mod_a.start_point, mod_a.end_point,
                mod_b.start_point, mod_b.end_point,
            )

            if gap > thr:
                continue

            dot = float(np.dot(mod_a.axis_direction, mod_b.axis_direction))
            angle_deg = float(np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0))))
            if angle_deg > float(angle_threshold_deg):
                continue

            joints.append(CylinderJoint(
                cylinder_a_idx=i,
                cylinder_b_idx=j,
                point_a=pa,
                point_b=pb,
                gap=float(gap),
                midpoint=0.5 * (pa + pb),
                angle_deg=angle_deg,
            ))

    joints.sort(key=lambda j: j.gap)
    return joints


def build_pipe_network(
    cylinders: Sequence[DetectedCylinder],
    joints: Optional[List[CylinderJoint]] = None,
    *,
    distance_threshold: Optional[float] = None,
    angle_threshold_deg: float = 160.0,
) -> PipeNetwork:
    """Build a :class:`PipeNetwork` from cylinders, computing joints if needed.

    Parameters
    ----------
    cylinders:
        Detected cylinders (nodes).
    joints:
        Pre-computed joints.  If ``None``, :func:`find_cylinder_joints` is
        called with the remaining keyword arguments.
    distance_threshold, angle_threshold_deg:
        Forwarded to :func:`find_cylinder_joints` when *joints* is ``None``.

    Returns
    -------
    PipeNetwork
    """
    cylinders = list(cylinders)
    if joints is None:
        joints = find_cylinder_joints(
            cylinders,
            distance_threshold=distance_threshold,
            angle_threshold_deg=angle_threshold_deg,
        )

    adjacency: List[List[int]] = [[] for _ in cylinders]
    for j in joints:
        adjacency[j.cylinder_a_idx].append(j.cylinder_b_idx)
        adjacency[j.cylinder_b_idx].append(j.cylinder_a_idx)

    return PipeNetwork(cylinders=cylinders, joints=joints, adjacency=adjacency)


# ---------------------------------------------------------------------------
# Geometry: segment-to-segment minimum distance
# ---------------------------------------------------------------------------

def _segment_to_segment_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimum distance between two line segments (P1→P2) and (Q1→Q2).

    Returns ``(distance, closest_on_P, closest_on_Q)``.

    Adapted from the algorithm by Dan Sunday, "Distance Between Lines and
    Segments", Geometry Algorithms.
    """
    d1 = p2 - p1    # direction of segment P
    d2 = q2 - q1    # direction of segment Q
    r = p1 - q1

    a = float(np.dot(d1, d1))   # squared length of P
    e = float(np.dot(d2, d2))   # squared length of Q
    f = float(np.dot(d2, r))

    if a <= 1e-15 and e <= 1e-15:
        # Both degenerate (points)
        return float(np.linalg.norm(r)), p1.copy(), q1.copy()

    if a <= 1e-15:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= 1e-15:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b

            if abs(denom) > 1e-15:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0  # arbitrary for parallel segments

            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    closest_p = p1 + s * d1
    closest_q = q1 + t * d2
    dist = float(np.linalg.norm(closest_p - closest_q))
    return dist, closest_p, closest_q
