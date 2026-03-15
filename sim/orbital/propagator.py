"""Post-insertion orbit propagator with J2 perturbation.

Converts the Cartesian state at orbital insertion into Keplerian elements
and propagates the orbit forward using J2-perturbed equations of motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from sim.config import EARTH_MU, EARTH_RADIUS_M, EARTH_J2
from sim.core.state import VehicleState


@dataclass
class OrbitalElements:
    """Classical Keplerian orbital elements.

    Attributes:
        semi_major_axis_m: Semi-major axis (m).
        eccentricity: Orbital eccentricity (dimensionless).
        inclination_deg: Inclination (degrees).
        raan_deg: Right ascension of the ascending node (degrees).
        arg_periapsis_deg: Argument of periapsis (degrees).
        true_anomaly_deg: True anomaly (degrees).
        period_s: Orbital period (s).
        apoapsis_alt_km: Apoapsis altitude above Earth surface (km).
        periapsis_alt_km: Periapsis altitude above Earth surface (km).
    """

    semi_major_axis_m: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_periapsis_deg: float
    true_anomaly_deg: float
    period_s: float
    apoapsis_alt_km: float
    periapsis_alt_km: float


class OrbitPropagator:
    """Propagates a post-insertion orbit with optional J2 perturbation.

    Parameters
    ----------
    state_at_insertion : VehicleState
        Vehicle state at the moment of orbital insertion (ECI frame).
    """

    def __init__(self, state_at_insertion: VehicleState) -> None:
        self._insertion_state = state_at_insertion.copy()
        self._elements: OrbitalElements | None = None

    # ------------------------------------------------------------------
    # Cartesian -> Keplerian conversion
    # ------------------------------------------------------------------

    def state_to_elements(self) -> OrbitalElements:
        """Convert the insertion Cartesian state to Keplerian orbital elements.

        Uses standard orbital mechanics formulas to compute a, e, i, RAAN,
        argument of periapsis, and true anomaly from the ECI position and
        velocity vectors.

        Returns
        -------
        OrbitalElements
            Classical Keplerian elements derived from the insertion state.
        """
        r_vec = self._insertion_state.position_eci.astype(np.float64)
        v_vec = self._insertion_state.velocity_eci.astype(np.float64)

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        mu = EARTH_MU

        # Specific angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        # Node vector (K x h)
        k_hat = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(k_hat, h_vec)
        n = np.linalg.norm(n_vec)

        # Eccentricity vector
        e_vec = ((v ** 2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
        e = np.linalg.norm(e_vec)

        # Specific mechanical energy -> semi-major axis
        energy = 0.5 * v ** 2 - mu / r
        if abs(energy) < 1e-12:
            # Parabolic edge case — treat as very large a
            a = 1e15
        else:
            a = -mu / (2.0 * energy)

        # Inclination
        inc = math.acos(np.clip(h_vec[2] / h, -1.0, 1.0))

        # Right ascension of ascending node (RAAN / Omega)
        if n > 1e-12:
            raan = math.acos(np.clip(n_vec[0] / n, -1.0, 1.0))
            if n_vec[1] < 0.0:
                raan = 2.0 * math.pi - raan
        else:
            raan = 0.0

        # Argument of periapsis (omega)
        if n > 1e-12 and e > 1e-12:
            arg_pe = math.acos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
            if e_vec[2] < 0.0:
                arg_pe = 2.0 * math.pi - arg_pe
        else:
            arg_pe = 0.0

        # True anomaly (nu)
        if e > 1e-12:
            nu = math.acos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
            if np.dot(r_vec, v_vec) < 0.0:
                nu = 2.0 * math.pi - nu
        else:
            # Circular orbit — measure from ascending node
            if n > 1e-12:
                nu = math.acos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1.0, 1.0))
                if r_vec[2] < 0.0:
                    nu = 2.0 * math.pi - nu
            else:
                nu = 0.0

        # Derived quantities
        period = 2.0 * math.pi * math.sqrt(abs(a ** 3) / mu) if a > 0 else math.inf
        r_apoapsis = a * (1.0 + e)
        r_periapsis = a * (1.0 - e)
        apoapsis_alt_km = (r_apoapsis - EARTH_RADIUS_M) / 1000.0
        periapsis_alt_km = (r_periapsis - EARTH_RADIUS_M) / 1000.0

        self._elements = OrbitalElements(
            semi_major_axis_m=a,
            eccentricity=e,
            inclination_deg=math.degrees(inc),
            raan_deg=math.degrees(raan),
            arg_periapsis_deg=math.degrees(arg_pe),
            true_anomaly_deg=math.degrees(nu),
            period_s=period,
            apoapsis_alt_km=apoapsis_alt_km,
            periapsis_alt_km=periapsis_alt_km,
        )
        return self._elements

    # ------------------------------------------------------------------
    # J2 perturbed propagation
    # ------------------------------------------------------------------

    def propagate(self, duration_s: float, dt_s: float = 10.0) -> List[VehicleState]:
        """Propagate the orbit forward using J2-perturbed Cowell's method.

        Integrates the equations of motion in Cartesian coordinates with a
        4th-order Runge-Kutta scheme, including the J2 gravitational
        perturbation acceleration.

        Parameters
        ----------
        duration_s : float
            Total propagation duration (seconds).
        dt_s : float, optional
            Integration time step (seconds). Default is 10.0.

        Returns
        -------
        list[VehicleState]
            Sequence of vehicle states sampled at each time step.
        """
        r_vec = self._insertion_state.position_eci.astype(np.float64).copy()
        v_vec = self._insertion_state.velocity_eci.astype(np.float64).copy()
        mass = self._insertion_state.mass_kg
        t0 = self._insertion_state.time_s

        states: List[VehicleState] = []
        t = 0.0

        while t <= duration_s + 1e-9:
            state = VehicleState(
                position_eci=r_vec.copy(),
                velocity_eci=v_vec.copy(),
                quaternion=self._insertion_state.quaternion.copy(),
                angular_velocity_body=self._insertion_state.angular_velocity_body.copy(),
                mass_kg=mass,
                time_s=t0 + t,
            )
            states.append(state)

            # RK4 step
            dt = min(dt_s, duration_s - t) if t < duration_s else 0.0
            if dt <= 0.0:
                break

            k1v = self._accel_j2(r_vec) * dt
            k1r = v_vec * dt

            k2v = self._accel_j2(r_vec + 0.5 * k1r) * dt
            k2r = (v_vec + 0.5 * k1v) * dt

            k3v = self._accel_j2(r_vec + 0.5 * k2r) * dt
            k3r = (v_vec + 0.5 * k2v) * dt

            k4v = self._accel_j2(r_vec + k3r) * dt
            k4r = (v_vec + k3v) * dt

            v_vec = v_vec + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
            r_vec = r_vec + (k1r + 2.0 * k2r + 2.0 * k3r + k4r) / 6.0
            t += dt

        return states

    @staticmethod
    def _accel_j2(r_vec: np.ndarray) -> np.ndarray:
        """Compute gravitational acceleration with J2 perturbation.

        Parameters
        ----------
        r_vec : np.ndarray
            Position vector in ECI frame (m).

        Returns
        -------
        np.ndarray
            Acceleration vector in ECI frame (m/s^2).
        """
        r = np.linalg.norm(r_vec)
        if r < 1.0:
            return np.zeros(3)

        x, y, z = r_vec
        r2 = r * r
        r5 = r2 * r2 * r
        mu = EARTH_MU
        Re = EARTH_RADIUS_M
        J2 = EARTH_J2

        # Two-body acceleration
        a_two_body = -mu / (r2 * r) * r_vec

        # J2 perturbation
        coeff = -1.5 * J2 * mu * Re ** 2 / r5
        z2_over_r2 = (z / r) ** 2

        a_j2 = coeff * np.array([
            x * (1.0 - 5.0 * z2_over_r2),
            y * (1.0 - 5.0 * z2_over_r2),
            z * (3.0 - 5.0 * z2_over_r2),
        ])

        return a_two_body + a_j2

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def orbit_summary(self) -> dict:
        """Return a human-readable dictionary summarising the insertion orbit.

        Calls ``state_to_elements`` if elements have not been computed yet.

        Returns
        -------
        dict
            Orbital element values and derived parameters.
        """
        if self._elements is None:
            self.state_to_elements()
        el = self._elements

        return {
            "semi_major_axis_km": el.semi_major_axis_m / 1000.0,
            "eccentricity": el.eccentricity,
            "inclination_deg": el.inclination_deg,
            "raan_deg": el.raan_deg,
            "arg_periapsis_deg": el.arg_periapsis_deg,
            "true_anomaly_deg": el.true_anomaly_deg,
            "period_min": el.period_s / 60.0,
            "apoapsis_alt_km": el.apoapsis_alt_km,
            "periapsis_alt_km": el.periapsis_alt_km,
        }
