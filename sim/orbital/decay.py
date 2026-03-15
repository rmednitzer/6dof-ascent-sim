"""Atmospheric drag decay estimation using King-Hele theory.

Estimates orbital lifetime due to atmospheric drag at periapsis, using an
exponential atmosphere model and the ballistic coefficient of the vehicle.
"""

from __future__ import annotations

import math

import numpy as np

from sim.config import EARTH_MU, EARTH_RADIUS_M
from sim.core.state import VehicleState
from sim.orbital.propagator import OrbitalElements


# ---------------------------------------------------------------------------
# Exponential atmosphere model
# ---------------------------------------------------------------------------

# (base_altitude_km, density_kg_m3, scale_height_km)
# Piecewise exponential fit for the upper atmosphere
_ATMOSPHERE_BANDS = [
    (0.0, 1.225, 7.249),
    (100.0, 5.297e-7, 5.877),
    (150.0, 2.070e-9, 26.32),
    (200.0, 2.789e-10, 37.11),
    (250.0, 7.248e-11, 45.55),
    (300.0, 2.418e-11, 53.63),
    (350.0, 9.518e-12, 53.30),
    (400.0, 3.725e-12, 58.52),
    (450.0, 1.585e-12, 60.83),
    (500.0, 6.967e-13, 63.82),
    (600.0, 1.454e-13, 71.84),
    (700.0, 3.614e-14, 88.67),
    (800.0, 1.170e-14, 124.6),
    (900.0, 5.245e-15, 181.1),
    (1000.0, 3.019e-15, 268.0),
]


def _atmosphere_density(altitude_m: float) -> float:
    """Return approximate atmospheric density at the given altitude.

    Uses a piecewise exponential model suitable for lifetime estimation.

    Parameters
    ----------
    altitude_m : float
        Altitude above the Earth surface (m).

    Returns
    -------
    float
        Atmospheric density (kg/m^3).
    """
    alt_km = altitude_m / 1000.0

    if alt_km < 0.0:
        return _ATMOSPHERE_BANDS[0][1]

    # Select the appropriate band
    selected = _ATMOSPHERE_BANDS[0]
    for band in _ATMOSPHERE_BANDS:
        if alt_km >= band[0]:
            selected = band
        else:
            break

    base_alt_km, rho_base, scale_height_km = selected
    rho = rho_base * math.exp(-(alt_km - base_alt_km) / scale_height_km)
    return rho


def _scale_height_at_altitude(altitude_m: float) -> float:
    """Return the atmospheric scale height at the given altitude.

    Parameters
    ----------
    altitude_m : float
        Altitude above the Earth surface (m).

    Returns
    -------
    float
        Scale height (m).
    """
    alt_km = altitude_m / 1000.0

    selected = _ATMOSPHERE_BANDS[0]
    for band in _ATMOSPHERE_BANDS:
        if alt_km >= band[0]:
            selected = band
        else:
            break

    return selected[2] * 1000.0  # convert km -> m


# ---------------------------------------------------------------------------
# Ballistic coefficient
# ---------------------------------------------------------------------------

def ballistic_coefficient(
    mass_kg: float,
    cd: float = 2.2,
    area_m2: float = 10.52,
) -> float:
    """Compute the ballistic coefficient BC = m / (Cd * A).

    Parameters
    ----------
    mass_kg : float
        Spacecraft dry mass (kg).
    cd : float, optional
        Drag coefficient. Default is 2.2 (typical for LEO).
    area_m2 : float, optional
        Cross-sectional area (m^2). Default matches the vehicle reference area.

    Returns
    -------
    float
        Ballistic coefficient (kg/m^2).
    """
    if cd * area_m2 < 1e-12:
        return math.inf
    return mass_kg / (cd * area_m2)


# ---------------------------------------------------------------------------
# King-Hele lifetime estimation
# ---------------------------------------------------------------------------

def estimate_lifetime(
    elements: OrbitalElements,
    dry_mass_kg: float,
    cd: float = 2.2,
    area_m2: float = 10.52,
    deorbit_alt_km: float = 80.0,
) -> dict:
    """Estimate orbital lifetime due to atmospheric drag using King-Hele theory.

    The King-Hele method estimates the secular decay of the semi-major axis
    by evaluating drag at periapsis with an exponential atmosphere model.
    For nearly circular orbits the decay rate is approximately:

        da/dt = -rho_p * a^2 * V_p / BC

    where rho_p is the atmospheric density at periapsis, V_p is the
    periapsis velocity, and BC is the ballistic coefficient.

    The orbit is propagated analytically in discrete steps until the
    periapsis drops below the deorbit threshold.

    Parameters
    ----------
    elements : OrbitalElements
        Orbital elements at the start of the lifetime estimation.
    dry_mass_kg : float
        Spacecraft dry mass (kg).
    cd : float, optional
        Drag coefficient. Default is 2.2.
    area_m2 : float, optional
        Cross-sectional area (m^2). Default is 10.52.
    deorbit_alt_km : float, optional
        Altitude (km) below which the orbit is considered to have decayed.
        Default is 80 km.

    Returns
    -------
    dict
        Lifetime estimate containing:
        - ``days_to_deorbit``: Estimated days until deorbit.
        - ``initial_periapsis_alt_km``: Starting periapsis altitude (km).
        - ``initial_apoapsis_alt_km``: Starting apoapsis altitude (km).
        - ``final_periapsis_alt_km``: Periapsis altitude at end of estimation (km).
        - ``ballistic_coefficient_kg_m2``: Ballistic coefficient used.
        - ``periapsis_density_kg_m3``: Atmospheric density at initial periapsis.
        - ``revolutions``: Approximate number of orbits until deorbit.
    """
    bc = ballistic_coefficient(dry_mass_kg, cd, area_m2)
    mu = EARTH_MU
    Re = EARTH_RADIUS_M

    a = elements.semi_major_axis_m
    e = elements.eccentricity

    initial_peri_alt_km = elements.periapsis_alt_km
    initial_apo_alt_km = elements.apoapsis_alt_km
    initial_peri_density = _atmosphere_density(initial_peri_alt_km * 1000.0)

    deorbit_radius = Re + deorbit_alt_km * 1000.0
    total_time_s = 0.0
    total_revolutions = 0.0

    # Step through the decay one orbit (or fraction) at a time
    max_iterations = 10_000_000
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        r_periapsis = a * (1.0 - e)
        periapsis_alt_m = r_periapsis - Re

        if r_periapsis <= deorbit_radius:
            break

        # Atmospheric density and scale height at periapsis
        rho_p = _atmosphere_density(periapsis_alt_m)
        H = _scale_height_at_altitude(periapsis_alt_m)

        if rho_p < 1e-20:
            # Density negligible — orbit will not decay in practical time
            total_time_s = math.inf
            break

        # Orbital period
        period = 2.0 * math.pi * math.sqrt(a ** 3 / mu)

        # Periapsis velocity (vis-viva)
        v_p = math.sqrt(mu * (2.0 / r_periapsis - 1.0 / a))

        # King-Hele: change in semi-major axis per orbit
        # For an elliptical orbit: da/rev = -2*pi * rho_p * (a^2 / BC) * I0(e*a/H)
        # For moderate eccentricity use the approximation:
        #   da/rev ~ -2*pi * a * rho_p * a / BC * exp(...)
        # Simplified decay rate per orbit (valid for low-to-moderate eccentricity):
        eaH = e * a / H
        if eaH < 50.0:
            # Modified Bessel function I0 approximation for the drag integral
            # For small arguments: I0(x) ~ 1 + x^2/4 + ...
            # For larger arguments: I0(x) ~ exp(x) / sqrt(2*pi*x)
            if eaH < 2.0:
                bessel_factor = 1.0 + eaH ** 2 / 4.0 + eaH ** 4 / 64.0
            else:
                bessel_factor = math.exp(eaH) / math.sqrt(2.0 * math.pi * eaH)

            da_per_rev = -2.0 * math.pi * (a ** 2 / bc) * rho_p * bessel_factor * math.exp(-eaH)
        else:
            # Very high eccentricity — the exponential terms dominate
            da_per_rev = -2.0 * math.pi * (a ** 2 / bc) * rho_p / math.sqrt(2.0 * math.pi * eaH)

        # Eccentricity change per orbit (King-Hele, first order)
        if eaH < 50.0 and eaH > 0.01:
            if eaH < 2.0:
                bessel_i1 = eaH / 2.0 * (1.0 + eaH ** 2 / 8.0 + eaH ** 4 / 192.0)
            else:
                bessel_i1 = math.exp(eaH) / math.sqrt(2.0 * math.pi * eaH)

            de_per_rev = -(1.0 - e) * da_per_rev / a - (2.0 * math.pi * rho_p * a / bc
                          * math.exp(-eaH) * (bessel_i1 - bessel_factor))
        else:
            # Approximate: eccentricity decays proportionally to a
            de_per_rev = -e * abs(da_per_rev) / a if a > 0 else 0.0

        # Adaptive stepping: take multiple orbits at once when decay is slow
        if abs(da_per_rev) > 1e-12:
            # Limit step so semi-major axis changes by at most 1%
            max_revs = max(1.0, 0.01 * a / abs(da_per_rev))
        else:
            max_revs = 1000.0

        n_revs = min(max_revs, 1000.0)
        n_revs = max(n_revs, 1.0)

        a_new = a + da_per_rev * n_revs
        e_new = e + de_per_rev * n_revs

        # Clamp eccentricity
        e_new = max(0.0, min(e_new, 0.999))

        # Ensure a doesn't go negative
        if a_new < Re:
            # Interpolate to find remaining time
            if abs(da_per_rev) > 1e-12:
                frac = (Re - a) / (da_per_rev * n_revs)
                total_time_s += frac * n_revs * period
                total_revolutions += frac * n_revs
            break

        total_time_s += n_revs * period
        total_revolutions += n_revs
        a = a_new
        e = e_new

        if total_time_s == math.inf:
            break

    r_periapsis_final = a * (1.0 - e) if math.isfinite(a) else 0.0
    final_peri_alt_km = (r_periapsis_final - Re) / 1000.0

    days_to_deorbit = total_time_s / 86400.0 if math.isfinite(total_time_s) else math.inf

    return {
        "days_to_deorbit": days_to_deorbit,
        "initial_periapsis_alt_km": initial_peri_alt_km,
        "initial_apoapsis_alt_km": initial_apo_alt_km,
        "final_periapsis_alt_km": final_peri_alt_km,
        "ballistic_coefficient_kg_m2": bc,
        "periapsis_density_kg_m3": initial_peri_density,
        "revolutions": total_revolutions,
    }
