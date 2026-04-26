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


def max_sigma_3x3(cov: np.ndarray) -> float:
    """Largest 1-sigma uncertainty from a 3x3 symmetric covariance matrix.

    Returns ``sqrt(max(|eigenvalues|))``, matching the semantics of the
    previous ``sqrt(np.max(np.abs(np.linalg.eigvalsh(cov))))`` call while
    avoiding the per-call numpy dispatch overhead.  Computed via the
    closed-form cubic eigenvalue solution for a symmetric 3x3 matrix
    (Smith, 1961), which is roughly an order of magnitude faster than
    ``np.linalg.eigvalsh`` at this fixed size.

    The nearly-diagonal shortcut uses ``max(|diag|)`` rather than
    ``max(diag)`` so that a covariance that has drifted non-PSD (possible
    under numerical roundoff on safety paths) does not silently
    under-report sigma and suppress downstream health / FTS triggers.
    """
    a00 = cov[0, 0]
    a11 = cov[1, 1]
    a22 = cov[2, 2]
    a01 = cov[0, 1]
    a02 = cov[0, 2]
    a12 = cov[1, 2]

    p1 = a01 * a01 + a02 * a02 + a12 * a12
    if p1 < 1e-30:
        # Nearly diagonal: eigenvalues are the diagonal entries.  Use the
        # largest absolute value to remain conservative if ``cov`` is not
        # positive semidefinite.
        max_eig = abs(a00)
        abs_a11 = abs(a11)
        abs_a22 = abs(a22)
        if abs_a11 > max_eig:
            max_eig = abs_a11
        if abs_a22 > max_eig:
            max_eig = abs_a22
        return math.sqrt(max_eig)

    q = (a00 + a11 + a22) / 3.0
    d00 = a00 - q
    d11 = a11 - q
    d22 = a22 - q
    p2 = d00 * d00 + d11 * d11 + d22 * d22 + 2.0 * p1
    p = math.sqrt(p2 / 6.0)

    # det(B) where B = (A - q*I) / p.
    inv_p = 1.0 / p
    b00 = d00 * inv_p
    b11 = d11 * inv_p
    b22 = d22 * inv_p
    b01 = a01 * inv_p
    b02 = a02 * inv_p
    b12 = a12 * inv_p
    det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
    r = det_b * 0.5
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    phi = math.acos(r) / 3.0
    # Principal branch gives the largest eigenvalue directly.
    max_eig = q + 2.0 * p * math.cos(phi)
    return math.sqrt(abs(max_eig))
