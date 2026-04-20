"""Fast arithmetic helpers for 3-element vectors.

``numpy.cross`` and ``numpy.linalg.norm`` carry significant Python-level
dispatch overhead (axis normalisation, broadcasting, ``moveaxis``) that
dominates their cost when operating on tiny fixed-length vectors.  The
helpers here perform the same computations with plain scalar arithmetic
and a single ``numpy.array`` allocation, which is roughly an order of
magnitude faster for the 3-vector case that pervades this simulation.
"""

from __future__ import annotations

import math

import numpy as np


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cross product of two 3-element vectors."""
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    return np.array(
        [
            a1 * b2 - a2 * b1,
            a2 * b0 - a0 * b2,
            a0 * b1 - a1 * b0,
        ]
    )


def norm3(v: np.ndarray) -> float:
    """Euclidean norm of a 3-element vector as a Python float."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def norm3_sq(v: np.ndarray) -> float:
    """Squared Euclidean norm of a 3-element vector."""
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def dot3(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 3-element vectors as a Python float."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm4(q: np.ndarray) -> float:
    """Euclidean norm of a 4-element vector as a Python float."""
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
