"""Propellant slosh model using the mechanical pendulum analogy.

Each propellant tank is represented as a simple pendulum whose bob
carries the *slosh mass* — a configurable fraction of the remaining
propellant.  The pendulum equation of motion is:

    θ̈ + 2 * ζ * ω * θ̇ + ω² * θ = a_lat / L

where
    ω    = natural slosh frequency (interpolated with fill level),
    ζ    = damping ratio (baffled tank),
    a_lat = lateral acceleration at the tank CG,
    L    = effective pendulum arm length.

The model outputs the lateral slosh force and the torque about the
vehicle CG that result from pendulum motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from sim import config


@dataclass
class SloshTankState:
    """Dynamic state of a single pendulum-analogy slosh mass.

    Attributes
    ----------
    theta : float
        Pendulum angle from vertical (rad).  Positive = displacement in
        the lateral direction corresponding to the modelled plane.
    theta_dot : float
        Pendulum angular rate (rad/s).
    """

    theta: float = 0.0
    theta_dot: float = 0.0


class SloshModel:
    """Pendulum-analogy propellant slosh model.

    Supports an arbitrary number of tanks; the default constructor
    creates a single-tank model using the global config parameters.

    Parameters
    ----------
    n_tanks : int
        Number of independent slosh tanks.
    tank_cg_offsets_m : np.ndarray, shape (n_tanks,)
        Axial distance from vehicle CG to each tank CG (m).  Positive
        forward (toward nose).  Used to convert slosh forces to torques.
    """

    def __init__(
        self,
        n_tanks: int = 1,
        tank_cg_offsets_m: np.ndarray | None = None,
    ) -> None:
        self._n: int = n_tanks

        # Tank-CG offsets default to zeros (force-only, no torque arm).
        if tank_cg_offsets_m is not None:
            self._tank_offsets: np.ndarray = np.asarray(
                tank_cg_offsets_m, dtype=float
            )
        else:
            self._tank_offsets = np.zeros(n_tanks, dtype=float)

        if self._tank_offsets.shape[0] != n_tanks:
            raise ValueError(
                f"tank_cg_offsets_m length ({self._tank_offsets.shape[0]}) "
                f"does not match n_tanks ({n_tanks})"
            )

        # Config scalars (same for every tank unless extended later).
        self._mass_fraction: float = float(config.SLOSH_MASS_FRACTION)
        self._freq_full_hz: float = float(config.SLOSH_FREQ_FULL_HZ)
        self._freq_empty_hz: float = float(config.SLOSH_FREQ_EMPTY_HZ)
        self._zeta: float = float(config.SLOSH_DAMPING_RATIO)
        self._arm_m: float = float(config.SLOSH_ARM_LENGTH_M)

        # Per-tank pendulum state.
        self.tanks: List[SloshTankState] = [
            SloshTankState() for _ in range(n_tanks)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _omega(self, propellant_fraction: float) -> float:
        """Natural frequency (rad/s) interpolated by fill level."""
        frac = float(np.clip(propellant_fraction, 0.0, 1.0))
        freq_hz = self._freq_full_hz * frac + self._freq_empty_hz * (1.0 - frac)
        return 2.0 * np.pi * freq_hz

    def _slosh_mass(self, propellant_mass_kg: float) -> float:
        """Participating slosh mass (kg) for one tank."""
        return self._mass_fraction * propellant_mass_kg / self._n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_tanks(self) -> int:
        """Number of slosh tanks."""
        return self._n

    def reset(self) -> None:
        """Zero all pendulum states."""
        for t in self.tanks:
            t.theta = 0.0
            t.theta_dot = 0.0

    def update(
        self,
        dt: float,
        lateral_accel_mps2: float,
        propellant_mass_kg: float,
        propellant_fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance the slosh model by one timestep.

        Parameters
        ----------
        dt : float
            Integration timestep (s).
        lateral_accel_mps2 : float
            Lateral acceleration at the tank CG (m/s^2).  This includes
            contributions from vehicle rotation, TVC side-force, and
            aerodynamic loads.
        propellant_mass_kg : float
            Total remaining propellant mass across all tanks (kg).
        propellant_fraction : float
            Fraction of propellant remaining [0, 1].

        Returns
        -------
        forces : np.ndarray, shape (n_tanks,)
            Lateral slosh force from each tank on the vehicle (N).
            Positive direction matches the lateral-acceleration sign
            convention.
        torques : np.ndarray, shape (n_tanks,)
            Torque about the vehicle CG from each tank (N-m).  Positive
            follows the right-hand rule for the pitch/yaw axis implied by
            the lateral direction.
        """
        omega = self._omega(propellant_fraction)
        m_slosh = self._slosh_mass(propellant_mass_kg)
        arm = self._arm_m  # could be fill-dependent in a future extension

        forces = np.empty(self._n, dtype=float)
        torques = np.empty(self._n, dtype=float)

        for i, tank in enumerate(self.tanks):
            # Pendulum EOM:
            #   θ̈ + 2ζωθ̇ + ω²θ = a_lat / L
            theta_ddot = (
                -2.0 * self._zeta * omega * tank.theta_dot
                - omega * omega * tank.theta
                + lateral_accel_mps2 / arm
            )

            # Semi-implicit Euler integration.
            tank.theta_dot += theta_ddot * dt
            tank.theta += tank.theta_dot * dt

            # Force exerted by the slosh mass on the vehicle.
            # The restoring spring pulls the vehicle toward the slosh mass:
            #   F = -m_slosh * L * θ̈  (reaction on vehicle, lateral)
            # Equivalently, from the EOM rearranged:
            #   F = m_slosh * (ω²*L*θ + 2ζωL*θ̇ )
            # which avoids recomputing θ̈ after integration.
            f = m_slosh * arm * (
                omega * omega * tank.theta
                + 2.0 * self._zeta * omega * tank.theta_dot
            )
            forces[i] = f
            torques[i] = f * self._tank_offsets[i]

        return forces, torques

    def total_force_n(self) -> float:
        """Sum of current slosh forces across all tanks (N).

        This is a convenience accessor; for correct results, query the
        values returned by :meth:`update` directly (they use post-step
        state).  This method reconstructs forces from the *current* modal
        state, which is the post-step state if called after ``update``.
        """
        # Re-derive from stored state — note: requires propellant info,
        # so we store last-computed values instead.
        raise NotImplementedError(
            "Use the force array returned by update() instead."
        )

    def total_torque_nm(self) -> float:
        """Sum of current slosh torques across all tanks (N-m)."""
        raise NotImplementedError(
            "Use the torque array returned by update() instead."
        )

    def pendulum_angles(self) -> np.ndarray:
        """Current pendulum angles for all tanks (rad)."""
        return np.array([t.theta for t in self.tanks], dtype=float)

    def pendulum_rates(self) -> np.ndarray:
        """Current pendulum angular rates for all tanks (rad/s)."""
        return np.array([t.theta_dot for t in self.tanks], dtype=float)

    def kinetic_energy(
        self, propellant_mass_kg: float
    ) -> float:
        """Total slosh kinetic energy across all tanks (J)."""
        m = self._slosh_mass(propellant_mass_kg)
        arm = self._arm_m
        return 0.5 * m * arm ** 2 * float(
            np.sum(self.pendulum_rates() ** 2)
        )

    def potential_energy(
        self,
        propellant_mass_kg: float,
        propellant_fraction: float,
    ) -> float:
        """Total slosh potential energy across all tanks (J)."""
        m = self._slosh_mass(propellant_mass_kg)
        arm = self._arm_m
        omega = self._omega(propellant_fraction)
        return 0.5 * m * arm ** 2 * omega ** 2 * float(
            np.sum(self.pendulum_angles() ** 2)
        )
