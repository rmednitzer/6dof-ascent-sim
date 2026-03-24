"""WGS84 gravity model with J2-J6 zonal harmonic perturbations.

Provides gravitational acceleration in the ECI frame accounting for Earth's
oblateness via zonal harmonics J2 through J6.  J2 alone introduces ~1 km-scale
trajectory deviations over a typical ascent; J3-J6 add corrections of order
10-100 m, significant for precision orbit insertion.

References:
    Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
    Section 8.6, Eqs. 8-20 through 8-25.
    Montenbruck & Gill, *Satellite Orbits*, 3rd ed., Section 3.2.
"""

from __future__ import annotations

import numpy as np

from sim import config


def gravitational_acceleration(position_eci: np.ndarray) -> np.ndarray:
    """Compute gravitational acceleration at an ECI position using J2-J6 gravity.

    The zonal harmonic perturbations account for Earth's non-spherical mass
    distribution.  J2 (oblateness) is the dominant term; J3 (pear-shaped
    asymmetry) through J6 provide progressively smaller corrections that
    improve trajectory accuracy by 10-100 m over a typical ascent.

    Args:
        position_eci: 3-element ECI position vector [x, y, z] in metres.

    Returns:
        3-element gravitational acceleration vector in ECI (m/s^2).

    Raises:
        ValueError: If the position vector has zero magnitude.
    """
    mu = config.EARTH_MU
    r_e = config.EARTH_RADIUS_M

    x, y, z = position_eci
    r_sq = x * x + y * y + z * z
    r = np.sqrt(r_sq)

    if r < 1.0:
        raise ValueError(f"Position magnitude {r:.3f} m is too small for gravity computation.")

    r_inv = 1.0 / r
    mu_over_r2 = mu / r_sq
    re_over_r = r_e * r_inv
    z_over_r = z * r_inv

    # Precompute powers of (R_e/r) and (z/r)
    re_r2 = re_over_r * re_over_r
    z_r2 = z_over_r * z_over_r
    z_r3 = z_r2 * z_over_r
    z_r4 = z_r2 * z_r2
    z_r5 = z_r4 * z_over_r
    z_r6 = z_r4 * z_r2

    re_r3 = re_r2 * re_over_r
    re_r4 = re_r2 * re_r2
    re_r5 = re_r4 * re_over_r
    re_r6 = re_r4 * re_r2

    # --- J2 contribution (Vallado Eq. 8-20) ---
    j2 = config.EARTH_J2
    j2_fac = 1.5 * j2 * re_r2
    c_xy_j2 = 1.0 - j2_fac * (5.0 * z_r2 - 1.0)
    c_z_j2 = 1.0 - j2_fac * (5.0 * z_r2 - 3.0)

    # --- J3 contribution (Vallado Eq. 8-21) ---
    j3 = config.EARTH_J3
    j3_fac = 0.5 * j3 * re_r3
    c_xy_j3 = j3_fac * (35.0 * z_r3 - 15.0 * z_over_r) * r_inv
    # J3 z-component has different structure: asymmetric in z
    c_z_j3 = j3_fac * (35.0 * z_r3 - 21.0 * z_over_r) * r_inv

    # --- J4 contribution (Vallado Eq. 8-22) ---
    j4 = config.EARTH_J4
    j4_fac = -0.625 * j4 * re_r4
    c_xy_j4 = j4_fac * (63.0 * z_r4 - 42.0 * z_r2 + 3.0)
    c_z_j4 = j4_fac * (63.0 * z_r4 - 70.0 * z_r2 + 15.0)

    # --- J5 contribution ---
    j5 = config.EARTH_J5
    j5_fac = 0.125 * j5 * re_r5
    c_xy_j5 = j5_fac * (693.0 * z_r5 - 630.0 * z_r3 + 105.0 * z_over_r) * r_inv
    c_z_j5 = j5_fac * (693.0 * z_r5 - 945.0 * z_r3 + 315.0 * z_over_r) * r_inv

    # --- J6 contribution ---
    j6 = config.EARTH_J6
    j6_fac = -1.0 / 16.0 * j6 * re_r6
    c_xy_j6 = j6_fac * (3003.0 * z_r6 - 3465.0 * z_r4 + 945.0 * z_r2 - 35.0)
    c_z_j6 = j6_fac * (3003.0 * z_r6 - 5005.0 * z_r4 + 2205.0 * z_r2 - 245.0)

    # Sum all perturbation terms
    factor_xy = c_xy_j2 + c_xy_j3 + c_xy_j4 + c_xy_j5 + c_xy_j6
    factor_z = c_z_j2 + c_z_j3 + c_z_j4 + c_z_j5 + c_z_j6

    ax = -mu_over_r2 * r_inv * x * factor_xy
    ay = -mu_over_r2 * r_inv * y * factor_xy
    az = -mu_over_r2 * r_inv * z * factor_z

    return np.array([ax, ay, az])
