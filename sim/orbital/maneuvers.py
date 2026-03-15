"""Orbital maneuver calculator.

Provides impulsive delta-v calculations for common orbital maneuvers:
Hohmann transfers, circularisation burns, and plane changes.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from sim.config import EARTH_MU, EARTH_RADIUS_M
from sim.core.state import VehicleState
from sim.orbital.propagator import OrbitalElements


def hohmann_transfer(r1: float, r2: float) -> Tuple[float, float, float]:
    """Compute the delta-v budget for a coplanar Hohmann transfer.

    Parameters
    ----------
    r1 : float
        Radius of the initial circular orbit (m).
    r2 : float
        Radius of the target circular orbit (m).

    Returns
    -------
    tuple[float, float, float]
        (dv1, dv2, transfer_time) where dv1 and dv2 are the magnitudes of
        the two impulsive burns (m/s) and transfer_time is the half-period
        of the transfer ellipse (s).
    """
    mu = EARTH_MU

    # Circular velocities
    v_circ_1 = math.sqrt(mu / r1)
    v_circ_2 = math.sqrt(mu / r2)

    # Transfer ellipse semi-major axis
    a_transfer = (r1 + r2) / 2.0

    # Velocities on the transfer ellipse at departure and arrival radii
    v_transfer_1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_transfer_2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

    dv1 = abs(v_transfer_1 - v_circ_1)
    dv2 = abs(v_circ_2 - v_transfer_2)

    # Transfer time is half the period of the transfer ellipse
    transfer_time = math.pi * math.sqrt(a_transfer ** 3 / mu)

    return dv1, dv2, transfer_time


def circularization_dv(state: VehicleState) -> float:
    """Compute the delta-v required to circularise the orbit at the current position.

    The burn is applied tangentially at the current radius to achieve
    circular velocity.

    Parameters
    ----------
    state : VehicleState
        Current vehicle state in ECI coordinates.

    Returns
    -------
    float
        Magnitude of the required circularisation burn (m/s).
    """
    r = np.linalg.norm(state.position_eci)
    v = np.linalg.norm(state.velocity_eci)

    if r < 1.0:
        return 0.0

    v_circular = math.sqrt(EARTH_MU / r)

    # Radial component of velocity
    r_hat = state.position_eci / r
    v_radial = np.dot(state.velocity_eci, r_hat)
    v_tangential = math.sqrt(max(v ** 2 - v_radial ** 2, 0.0))

    # Delta-v is the vector difference to achieve purely tangential circular velocity.
    # dv^2 = v_radial^2 + (v_circular - v_tangential)^2
    dv = math.sqrt(v_radial ** 2 + (v_circular - v_tangential) ** 2)
    return dv


def plane_change_dv(state: VehicleState, target_inclination_deg: float) -> float:
    """Compute the delta-v for a simple plane change maneuver.

    Assumes the burn is performed at an optimal location (node crossing)
    and applies the cosine rule for an impulsive inclination change.

    Parameters
    ----------
    state : VehicleState
        Current vehicle state in ECI coordinates.
    target_inclination_deg : float
        Desired orbital inclination (degrees).

    Returns
    -------
    float
        Magnitude of the plane-change burn (m/s).
    """
    r_vec = state.position_eci.astype(np.float64)
    v_vec = state.velocity_eci.astype(np.float64)

    # Current angular momentum -> current inclination
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    if h < 1e-12:
        return 0.0

    current_inc = math.acos(np.clip(h_vec[2] / h, -1.0, 1.0))
    target_inc = math.radians(target_inclination_deg)

    delta_inc = abs(target_inc - current_inc)

    if delta_inc < 1e-12:
        return 0.0

    # Velocity at the node crossing (use current speed as approximation)
    v = np.linalg.norm(v_vec)

    # Cosine rule: dv = 2 * v * sin(delta_i / 2)
    dv = 2.0 * v * math.sin(delta_inc / 2.0)
    return dv


def total_correction_budget(
    achieved: OrbitalElements,
    target_alt_m: float,
    target_inc_deg: float,
) -> float:
    """Estimate the total delta-v budget to correct from achieved orbit to target.

    Combines a Hohmann transfer to adjust altitude and a plane change to
    adjust inclination. This is a first-order budget; combined maneuvers
    would be more efficient in practice.

    Parameters
    ----------
    achieved : OrbitalElements
        Orbital elements of the achieved (actual) orbit.
    target_alt_m : float
        Target circular orbit altitude above Earth surface (m).
    target_inc_deg : float
        Target orbital inclination (degrees).

    Returns
    -------
    float
        Total estimated delta-v correction budget (m/s).
    """
    # Current semi-major axis as reference radius for the achieved orbit
    r_achieved = achieved.semi_major_axis_m
    r_target = EARTH_RADIUS_M + target_alt_m

    # Altitude correction via Hohmann transfer
    if abs(r_achieved - r_target) > 1.0:
        dv1, dv2, _ = hohmann_transfer(r_achieved, r_target)
        dv_altitude = dv1 + dv2
    else:
        dv_altitude = 0.0

    # Inclination correction
    delta_inc = abs(achieved.inclination_deg - target_inc_deg)
    if delta_inc > 1e-6:
        # Use circular velocity at target altitude for plane change cost
        v_circ = math.sqrt(EARTH_MU / r_target)
        dv_inclination = 2.0 * v_circ * math.sin(math.radians(delta_inc) / 2.0)
    else:
        dv_inclination = 0.0

    # Eccentricity correction (circularisation at target altitude)
    if achieved.eccentricity > 1e-4:
        r_peri = achieved.semi_major_axis_m * (1.0 - achieved.eccentricity)
        v_at_peri = math.sqrt(EARTH_MU * (2.0 / r_peri - 1.0 / achieved.semi_major_axis_m))
        v_circ_peri = math.sqrt(EARTH_MU / r_peri)
        dv_circ = abs(v_at_peri - v_circ_peri)
    else:
        dv_circ = 0.0

    return dv_altitude + dv_inclination + dv_circ
