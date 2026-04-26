"""Reference frame transformations: ECI, ECEF, NED, Body."""

from __future__ import annotations

import math

import numpy as np

from sim import config

#: Convergence tolerance for Bowring's iterative latitude solution (rad).
_LAT_CONVERGENCE_TOL: float = 1e-12

# Pre-computed WGS84 ellipsoid constants.  Hoisted out of ecef_to_lla to avoid
# recomputing `b`, `e2`, etc. on every call (the function is evaluated hundreds
# of thousands of times per simulation).
_WGS84_A: float = config.EARTH_RADIUS_M
_WGS84_F: float = config.EARTH_FLATTENING
_WGS84_B: float = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2: float = 2.0 * _WGS84_F - _WGS84_F * _WGS84_F
_WGS84_ONE_MINUS_E2: float = 1.0 - _WGS84_E2


def eci_to_ecef(pos_eci: np.ndarray, time_s: float) -> np.ndarray:
    """Rotate ECI position to ECEF using Earth rotation angle.

    Args:
        pos_eci: Position in ECI frame (m).
        time_s: Seconds since epoch (mission elapsed time).

    Returns:
        Position in ECEF frame (m).
    """
    theta = config.EARTH_OMEGA * time_s
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = pos_eci[0], pos_eci[1], pos_eci[2]
    return np.array([c * x + s * y, -s * x + c * y, z])


def ecef_to_eci(pos_ecef: np.ndarray, time_s: float) -> np.ndarray:
    """Rotate ECEF position to ECI.

    Args:
        pos_ecef: Position in ECEF frame (m).
        time_s: Seconds since epoch.

    Returns:
        Position in ECI frame (m).
    """
    theta = config.EARTH_OMEGA * time_s
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = pos_ecef[0], pos_ecef[1], pos_ecef[2]
    return np.array([c * x - s * y, s * x + c * y, z])


def ecef_to_lla(pos_ecef: np.ndarray) -> tuple[float, float, float]:
    """Convert ECEF position to geodetic latitude, longitude, altitude.

    Uses iterative method for WGS84 ellipsoid.

    Args:
        pos_ecef: ECEF position (m).

    Returns:
        (latitude_rad, longitude_rad, altitude_m).
    """
    x, y, z = pos_ecef[0], pos_ecef[1], pos_ecef[2]
    a = _WGS84_A
    e2 = _WGS84_E2

    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)

    # Iterative solution (Bowring's method) with early termination
    lat = math.atan2(z, p * _WGS84_ONE_MINUS_E2)
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat_new = math.atan2(z + e2 * N * sin_lat, p)
        if abs(lat_new - lat) < _LAT_CONVERGENCE_TOL:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - _WGS84_B

    return lat, lon, alt


def lla_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    """Convert geodetic LLA to ECEF position.

    Args:
        lat_rad: Geodetic latitude (rad).
        lon_rad: Longitude (rad).
        alt_m: Altitude above WGS84 (m).

    Returns:
        ECEF position (m).
    """
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)

    x = (N + alt_m) * cos_lat * math.cos(lon_rad)
    y = (N + alt_m) * cos_lat * math.sin(lon_rad)
    z = (N * _WGS84_ONE_MINUS_E2 + alt_m) * sin_lat
    return np.array([x, y, z])


def eci_to_ned(pos_eci: np.ndarray, vel_eci: np.ndarray, lat_rad: float, lon_rad: float) -> np.ndarray:
    """Rotate ECI velocity to NED frame at given geodetic location.

    Args:
        pos_eci: ECI position (not used directly, lat/lon extracted externally).
        vel_eci: ECI velocity to rotate (m/s).
        lat_rad: Geodetic latitude (rad).
        lon_rad: Longitude (rad).

    Returns:
        Velocity in NED frame (m/s).
    """
    # ECI to ECEF rotation is identity for velocity direction (ignoring transport term)
    # NED rotation from ECEF
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    R_ecef_ned = np.array(
        [
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0.0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ]
    )
    return R_ecef_ned @ vel_eci


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to direction cosine matrix (body from ECI).

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        3x3 DCM rotating ECI vectors to body frame.
    """
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def body_to_eci(vec_body: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector to ECI using attitude quaternion.

    Uses direct quaternion rotation (v' = q * v * q^-1 expanded) to avoid
    materialising the 3x3 direction cosine matrix for a single rotation.

    Args:
        vec_body: Vector in body frame.
        quaternion: Attitude quaternion [x, y, z, w].

    Returns:
        Vector in ECI frame.
    """
    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    vx, vy, vz = vec_body[0], vec_body[1], vec_body[2]

    # t = 2 * (q_vec x v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # v_rot = v + w * t + q_vec x t   (body-to-inertial: +w*t + q×t)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]
    )


def eci_to_body(vec_eci: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Rotate an ECI vector to body frame using attitude quaternion.

    Inverse of :func:`body_to_eci`; uses conjugate quaternion (negated
    vector part) without materialising the DCM.

    Args:
        vec_eci: Vector in ECI frame.
        quaternion: Attitude quaternion [x, y, z, w].

    Returns:
        Vector in body frame.
    """
    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    vx, vy, vz = vec_eci[0], vec_eci[1], vec_eci[2]

    # t = 2 * (q_vec x v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # v_rot = v - w * t + q_vec x t   (inertial-to-body: -w*t + q×t)
    return np.array(
        [
            vx - w * tx + (y * tz - z * ty),
            vy - w * ty + (z * tx - x * tz),
            vz - w * tz + (x * ty - y * tx),
        ]
    )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w].

    Args:
        q1: First quaternion.
        q2: Second quaternion.

    Returns:
        Product quaternion q1 * q2.
    """
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate (inverse for unit quaternion) of q = [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_error(q_desired: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Compute error quaternion: q_err = q_desired * conj(q_current).

    Args:
        q_desired: Target attitude quaternion.
        q_current: Current attitude quaternion.

    Returns:
        Error quaternion [x, y, z, w].
    """
    return quaternion_multiply(q_desired, quaternion_conjugate(q_current))


def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Create quaternion from axis-angle representation.

    Args:
        axis: Unit rotation axis.
        angle_rad: Rotation angle (rad).

    Returns:
        Quaternion [x, y, z, w].
    """
    half = angle_rad * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)])


def quaternion_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Compute quaternion time derivative from body angular velocity.

    Args:
        q: Current quaternion [x, y, z, w].
        omega_body: Angular velocity in body frame (rad/s).

    Returns:
        Quaternion derivative [dx, dy, dz, dw]/dt.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    ox, oy, oz = omega_body[0], omega_body[1], omega_body[2]
    # 0.5 * q * [omega, 0], expanded directly
    return np.array(
        [
            0.5 * (w * ox + y * oz - z * oy),
            0.5 * (w * oy - x * oz + z * ox),
            0.5 * (w * oz + x * oy - y * ox),
            -0.5 * (x * ox + y * oy + z * oz),
        ]
    )
