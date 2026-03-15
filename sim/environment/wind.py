"""Simple altitude-dependent wind profile with stochastic gusts.

The model produces a deterministic mean-wind component (surface wind that
decays to zero above ~30 km) plus a Gaussian gust perturbation.  Wind is
first constructed in the local NED frame and then transformed to ECI so
it can be subtracted from the vehicle's inertial velocity to obtain airspeed.

Configuration parameters (from ``sim.config``):
    - ``WIND_SPEED_MS``: Mean surface wind speed (m/s).
    - ``WIND_DIRECTION_DEG``: Meteorological convention — direction the wind
      is *coming from*, measured clockwise from true north (deg).
    - ``WIND_GUST_SIGMA_MS``: 1-sigma gust magnitude (m/s).
"""

from __future__ import annotations

import math

import numpy as np

from sim import config
from sim.core.reference_frames import (
    ecef_to_eci,
    ecef_to_lla,
    eci_to_ecef,
)

#: Altitude above which mean wind is zero (m).
_WIND_CEILING_M: float = 30_000.0

#: Altitude at which mean wind reaches peak strength (m).
_WIND_PEAK_ALT_M: float = 12_000.0

#: Peak wind multiplier relative to surface speed (jet-stream scaling).
_WIND_PEAK_FACTOR: float = 2.0


def _mean_wind_fraction(altitude_m: float) -> float:
    """Return a dimensionless wind-speed multiplier as a function of altitude.

    The profile ramps linearly from 1.0 at the surface to
    ``_WIND_PEAK_FACTOR`` at ``_WIND_PEAK_ALT_M`` (crude jet-stream), then
    decays linearly to zero at ``_WIND_CEILING_M``.

    Args:
        altitude_m: Geodetic altitude (m).

    Returns:
        Multiplier in [0, ``_WIND_PEAK_FACTOR``].
    """
    if altitude_m <= 0.0:
        return 1.0
    if altitude_m <= _WIND_PEAK_ALT_M:
        # Linear ramp from 1.0 at surface to peak factor
        return 1.0 + (_WIND_PEAK_FACTOR - 1.0) * (altitude_m / _WIND_PEAK_ALT_M)
    if altitude_m <= _WIND_CEILING_M:
        # Linear decay from peak to zero
        return _WIND_PEAK_FACTOR * (
            1.0 - (altitude_m - _WIND_PEAK_ALT_M) / (_WIND_CEILING_M - _WIND_PEAK_ALT_M)
        )
    return 0.0


def _ned_to_ecef_rotation(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Build the 3x3 rotation matrix from NED to ECEF.

    Args:
        lat_rad: Geodetic latitude (rad).
        lon_rad: Longitude (rad).

    Returns:
        3x3 rotation matrix R such that ``v_ecef = R @ v_ned``.
    """
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # Transpose of the ECEF->NED matrix in reference_frames.py
    return np.array([
        [-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon],
        [-sin_lat * sin_lon, cos_lon, -cos_lat * sin_lon],
        [cos_lat, 0.0, -sin_lat],
    ])


def wind_velocity_eci(
    position_eci: np.ndarray,
    time_s: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute wind velocity in the ECI frame at a given position and time.

    The wind model is:
        1. Build a deterministic mean-wind vector in NED from config speed,
           direction, and an altitude-dependent profile.
        2. Add a zero-mean Gaussian gust independently to each NED axis.
        3. Rotate the result from NED -> ECEF -> ECI.

    Args:
        position_eci: 3-element ECI position vector (m).
        time_s: Mission elapsed time (s), used for ECI/ECEF rotation.
        rng: NumPy random generator for reproducible gusts.  If ``None``
             the gust component is omitted (deterministic mode).

    Returns:
        3-element wind velocity vector in ECI (m/s).
    """
    # ---- Convert position to geodetic coordinates -----------------------
    pos_ecef = eci_to_ecef(position_eci, time_s)
    lat_rad, lon_rad, alt_m = ecef_to_lla(pos_ecef)

    # ---- Mean wind in NED -----------------------------------------------
    wind_speed = config.WIND_SPEED_MS
    wind_dir_rad = math.radians(config.WIND_DIRECTION_DEG)

    # Meteorological convention: WIND_DIRECTION_DEG is the direction the
    # wind is *coming from*.  The velocity vector points in the opposite
    # direction.
    wind_heading_rad = wind_dir_rad + math.pi  # direction wind is going *to*

    fraction = _mean_wind_fraction(alt_m)
    v_north = fraction * wind_speed * math.cos(wind_heading_rad)
    v_east = fraction * wind_speed * math.sin(wind_heading_rad)
    v_down = 0.0  # no mean vertical wind component

    wind_ned = np.array([v_north, v_east, v_down])

    # ---- Stochastic gust ------------------------------------------------
    if rng is not None:
        gust_sigma = config.WIND_GUST_SIGMA_MS
        gust_ned = rng.normal(0.0, gust_sigma, size=3)
        # Scale gusts with altitude the same way as mean wind so they vanish
        # in the upper atmosphere where density is negligible.
        gust_ned *= fraction
        wind_ned = wind_ned + gust_ned

    # ---- Rotate NED -> ECEF -> ECI --------------------------------------
    R_ned_to_ecef = _ned_to_ecef_rotation(lat_rad, lon_rad)
    wind_ecef = R_ned_to_ecef @ wind_ned

    # The atmosphere co-rotates with the Earth, so the atmospheric velocity
    # in ECI includes Earth-rotation.  This ensures that a vehicle at rest
    # on the launch pad experiences near-zero airspeed.
    omega_earth = np.array([0.0, 0.0, config.EARTH_OMEGA])
    atmo_corotation_eci = np.cross(omega_earth, position_eci)

    wind_eci = ecef_to_eci(wind_ecef, time_s) + atmo_corotation_eci

    return wind_eci
