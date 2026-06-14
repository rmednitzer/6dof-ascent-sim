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

    # --- Point-mass (monopole) term: a = -mu * r_vec / r^3 ---
    inv_r3 = 1.0 / (r * r_sq)
    ax = -mu * x * inv_r3
    ay = -mu * y * inv_r3
    az = -mu * z * inv_r3

    # --- Zonal harmonic perturbations J2..J6 ---
    # Computed as the exact analytic gradient of the perturbing geopotential
    #     U_n = -(mu/r) * J_n * (R_e/r)^n * P_n(sinφ),   sinφ = z/r
    # giving, with the chain rule through r and s = z/r,
    #     a_x,y = (mu J_n (R_e/r)^n / r^2) * (x,y / r) * [(n+1) P_n(s) + s P_n'(s)]
    #     a_z   = (mu J_n (R_e/r)^n / r^2) * [(n+1) s P_n(s) - (1 - s^2) P_n'(s)]
    # This handles every zonal term (even *and* odd) with one consistent form
    # and is non-singular at the poles. Verified against a finite-difference
    # gradient of the geopotential to ~1e-10 relative at latitudes 0-89 deg.
    # The previous hand-coded J3/J5 terms carried an extra 1/r factor and a sign
    # error, which silently zeroed (and flipped) the odd zonals — undetectable in
    # a total-acceleration check because J3/J5 are ~1e-6 of the signal (AD-03/11).
    # Reference: Vallado, *Fundamentals of Astrodynamics and Applications*,
    # 4th ed., §8.6; Montenbruck & Gill, *Satellite Orbits*, §3.2.
    s = z / r  # sin(geocentric latitude)
    s2 = s * s
    re_r = r_e / r

    # Exact Legendre polynomials P_n(s) and their derivatives P_n'(s).
    legendre_p = (
        0.5 * (3.0 * s2 - 1.0),  # P2
        0.5 * s * (5.0 * s2 - 3.0),  # P3
        0.125 * (35.0 * s2 * s2 - 30.0 * s2 + 3.0),  # P4
        0.125 * s * (63.0 * s2 * s2 - 70.0 * s2 + 15.0),  # P5
        (231.0 * s2 * s2 * s2 - 315.0 * s2 * s2 + 105.0 * s2 - 5.0) / 16.0,  # P6
    )
    legendre_dp = (
        3.0 * s,  # P2'
        0.5 * (15.0 * s2 - 3.0),  # P3'
        0.5 * s * (35.0 * s2 - 15.0),  # P4'
        0.125 * (315.0 * s2 * s2 - 210.0 * s2 + 15.0),  # P5'
        s * (1386.0 * s2 * s2 - 1260.0 * s2 + 210.0) / 16.0,  # P6'
    )
    j_coeffs = (config.EARTH_J2, config.EARTH_J3, config.EARTH_J4, config.EARTH_J5, config.EARTH_J6)

    inv_r2 = 1.0 / r_sq
    re_rn = re_r * re_r  # (R_e/r)^n, starting at n=2
    x_r = x / r
    y_r = y / r
    for i, n in enumerate(range(2, 7)):
        p_n = legendre_p[i]
        dp_n = legendre_dp[i]
        pref = mu * j_coeffs[i] * re_rn * inv_r2  # mu J_n (R_e/r)^n / r^2
        bracket_xy = (n + 1) * p_n + s * dp_n
        ax += pref * x_r * bracket_xy
        ay += pref * y_r * bracket_xy
        az += pref * ((n + 1) * s * p_n - (1.0 - s2) * dp_n)
        re_rn *= re_r  # advance to (R_e/r)^(n+1)

    return np.array([ax, ay, az])
