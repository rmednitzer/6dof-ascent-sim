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

import math

import numpy as np

from sim import config


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
            self._tank_offsets: np.ndarray = np.asarray(tank_cg_offsets_m, dtype=float)
        else:
            self._tank_offsets = np.zeros(n_tanks, dtype=float)

        if self._tank_offsets.shape[0] != n_tanks:
            raise ValueError(
                f"tank_cg_offsets_m length ({self._tank_offsets.shape[0]}) does not match n_tanks ({n_tanks})"
            )

        # Config scalars (same for every tank unless extended later).
        self._mass_fraction: float = float(config.SLOSH_MASS_FRACTION)
        self._freq_full_hz: float = float(config.SLOSH_FREQ_FULL_HZ)
        self._freq_empty_hz: float = float(config.SLOSH_FREQ_EMPTY_HZ)
        self._zeta: float = float(config.SLOSH_DAMPING_RATIO)
        self._arm_m: float = float(config.SLOSH_ARM_LENGTH_M)

        # Pre-computed scalar derivatives — save two multiplies per update.
        two_pi = 2.0 * math.pi
        self._omega_full: float = two_pi * self._freq_full_hz
        self._omega_empty: float = two_pi * self._freq_empty_hz
        self._two_zeta: float = 2.0 * self._zeta
        self._mass_per_tank: float = self._mass_fraction / self._n

        # Per-tank pendulum state (vectorised).
        self._theta: np.ndarray = np.zeros(n_tanks, dtype=float)
        self._theta_dot: np.ndarray = np.zeros(n_tanks, dtype=float)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _omega(self, propellant_fraction: float) -> float:
        """Natural frequency (rad/s) interpolated by fill level."""
        frac = propellant_fraction
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        return self._omega_full * frac + self._omega_empty * (1.0 - frac)

    def _slosh_mass(self, propellant_mass_kg: float) -> float:
        """Participating slosh mass (kg) for one tank."""
        return self._mass_per_tank * propellant_mass_kg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_tanks(self) -> int:
        """Number of slosh tanks."""
        return self._n

    def reset(self) -> None:
        """Zero all pendulum states."""
        self._theta.fill(0.0)
        self._theta_dot.fill(0.0)

    def update(
        self,
        dt: float,
        lateral_accel_mps2: float,
        propellant_mass_kg: float,
        propellant_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        # Pendulum EOM:
        #   θ̈ + 2ζωθ̇ + ω²θ = a_lat / L
        omega_sq = omega * omega
        damping = self._two_zeta * omega
        theta_ddot = lateral_accel_mps2 / arm - damping * self._theta_dot - omega_sq * self._theta

        # Semi-implicit Euler integration.
        self._theta_dot += theta_ddot * dt
        self._theta += self._theta_dot * dt

        # Force exerted by the slosh mass on the vehicle.
        #   F = m_slosh * L * (ω²*θ + 2ζω*θ̇ )
        forces = (m_slosh * arm) * (omega_sq * self._theta + damping * self._theta_dot)
        torques = forces * self._tank_offsets

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
        raise NotImplementedError("Use the force array returned by update() instead.")

    def total_torque_nm(self) -> float:
        """Sum of current slosh torques across all tanks (N-m)."""
        raise NotImplementedError("Use the torque array returned by update() instead.")

    def pendulum_angles(self) -> np.ndarray:
        """Current pendulum angles for all tanks (rad)."""
        return self._theta.copy()

    def pendulum_rates(self) -> np.ndarray:
        """Current pendulum angular rates for all tanks (rad/s)."""
        return self._theta_dot.copy()

    def kinetic_energy(self, propellant_mass_kg: float) -> float:
        """Total slosh kinetic energy across all tanks (J)."""
        m = self._slosh_mass(propellant_mass_kg)
        arm = self._arm_m
        return 0.5 * m * arm**2 * float(np.sum(self._theta_dot**2))

    def potential_energy(
        self,
        propellant_mass_kg: float,
        propellant_fraction: float,
    ) -> float:
        """Total slosh potential energy across all tanks (J)."""
        m = self._slosh_mass(propellant_mass_kg)
        arm = self._arm_m
        omega = self._omega(propellant_fraction)
        return 0.5 * m * arm**2 * omega**2 * float(np.sum(self._theta**2))
