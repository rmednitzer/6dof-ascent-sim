"""WGS84 gravity model with J2 zonal harmonic perturbation.

Provides gravitational acceleration in the ECI frame accounting for Earth's
oblateness via the J2 term.  This is significantly more accurate than a
simple spherical (1/r^2) model for low-Earth orbit trajectories where the
equatorial bulge introduces ~1 km-scale trajectory deviations over a
typical ascent.
"""

from __future__ import annotations

import numpy as np

from sim import config


def gravitational_acceleration(position_eci: np.ndarray) -> np.ndarray:
    """Compute gravitational acceleration at an ECI position using J2 gravity.

    The J2 perturbation accounts for Earth's oblate shape.  The resulting
    acceleration is expressed in the ECI frame and includes both the central
    body term and the dominant zonal harmonic correction.

    Reference:
        Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
        Eq. 8-20.

    Args:
        position_eci: 3-element ECI position vector [x, y, z] in metres.

    Returns:
        3-element gravitational acceleration vector in ECI (m/s^2).

    Raises:
        ValueError: If the position vector has zero magnitude.
    """
    mu = config.EARTH_MU
    r_earth = config.EARTH_RADIUS_M
    j2 = config.EARTH_J2

    x, y, z = position_eci
    r_sq = x * x + y * y + z * z
    r = np.sqrt(r_sq)

    if r < 1.0:
        raise ValueError(f"Position magnitude {r:.3f} m is too small for gravity computation.")

    r_inv = 1.0 / r
    r_sq_inv = r_inv * r_inv
    mu_over_r3 = mu * r_inv * r_sq_inv  # mu / r^3

    # Ratio (R_earth / r)^2 used in J2 expressions
    re_over_r_sq = (r_earth * r_inv) ** 2

    # z/r ratio squared — measures how "polar" the position is
    z_over_r_sq = (z * r_inv) ** 2

    # J2 perturbation prefactor: (3/2) * J2 * (R_e / r)^2
    j2_factor = 1.5 * j2 * re_over_r_sq

    # Acceleration components (Vallado Eq. 8-20)
    ax = -mu_over_r3 * x * (1.0 - j2_factor * (5.0 * z_over_r_sq - 1.0))
    ay = -mu_over_r3 * y * (1.0 - j2_factor * (5.0 * z_over_r_sq - 1.0))
    az = -mu_over_r3 * z * (1.0 - j2_factor * (5.0 * z_over_r_sq - 3.0))

    return np.array([ax, ay, az])
