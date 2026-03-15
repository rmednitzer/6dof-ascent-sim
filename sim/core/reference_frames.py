"""Reference frame transformations: ECI, ECEF, NED, Body."""

from __future__ import annotations

import math

import numpy as np

from sim import config


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
    R = np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return R @ pos_eci


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
    R = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return R @ pos_ecef


def ecef_to_lla(pos_ecef: np.ndarray) -> tuple[float, float, float]:
    """Convert ECEF position to geodetic latitude, longitude, altitude.

    Uses iterative method for WGS84 ellipsoid.

    Args:
        pos_ecef: ECEF position (m).

    Returns:
        (latitude_rad, longitude_rad, altitude_m).
    """
    x, y, z = pos_ecef
    a = config.EARTH_RADIUS_M
    f = config.EARTH_FLATTENING
    b = a * (1.0 - f)
    e2 = 2 * f - f**2

    lon = math.atan2(y, x)
    p = math.sqrt(x**2 + y**2)

    # Iterative solution (Bowring's method)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat**2)
        lat = math.atan2(z + e2 * N * sin_lat, p)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat**2)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - b

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
    a = config.EARTH_RADIUS_M
    f = config.EARTH_FLATTENING
    e2 = 2 * f - f**2
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    N = a / math.sqrt(1.0 - e2 * sin_lat**2)

    x = (N + alt_m) * cos_lat * math.cos(lon_rad)
    y = (N + alt_m) * cos_lat * math.sin(lon_rad)
    z = (N * (1.0 - e2) + alt_m) * sin_lat
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

    Args:
        vec_body: Vector in body frame.
        quaternion: Attitude quaternion [x, y, z, w].

    Returns:
        Vector in ECI frame.
    """
    dcm = quat_to_dcm(quaternion)
    return dcm.T @ vec_body


def eci_to_body(vec_eci: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Rotate an ECI vector to body frame using attitude quaternion.

    Args:
        vec_eci: Vector in ECI frame.
        quaternion: Attitude quaternion [x, y, z, w].

    Returns:
        Vector in body frame.
    """
    dcm = quat_to_dcm(quaternion)
    return dcm @ vec_eci


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w].

    Args:
        q1: First quaternion.
        q2: Second quaternion.

    Returns:
        Product quaternion q1 * q2.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
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
    omega_quat = np.array([omega_body[0], omega_body[1], omega_body[2], 0.0])
    return 0.5 * quaternion_multiply(q, omega_quat)
