"""Structural bending modes for a flexible launch vehicle.

Models the first N lateral bending modes as damped harmonic oscillators.
Each mode is governed by:

    q̈_i + 2 * ζ_i * ω_i * q̇_i + ω_i² * q_i = F_modal_i / m_modal_i

Modal frequencies shift with propellant depletion (interpolated linearly
between full- and empty-stage values).  TVC gimbal deflection is projected
onto each mode shape at the engine station to compute generalised forcing.
The resulting bending rates are projected onto the IMU station to yield an
angular-rate contribution that corrupts the gyro measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from sim import config


@dataclass
class _ModalState:
    """Internal state for a single bending mode."""

    q: float = 0.0      # Generalised displacement (rad)
    q_dot: float = 0.0   # Generalised velocity (rad/s)


class FlexBody:
    """First-N lateral bending mode model.

    Parameters
    ----------
    n_modes : int, optional
        Number of bending modes to model (default: uses length of
        ``config.FLEX_MODE_FREQS_HZ``).

    Attributes
    ----------
    modes : list[_ModalState]
        Per-mode generalised coordinate and rate.
    """

    def __init__(self, n_modes: int | None = None) -> None:
        # Mode count — clamp to available config entries.
        max_modes = len(config.FLEX_MODE_FREQS_HZ)
        self._n: int = min(n_modes, max_modes) if n_modes is not None else max_modes

        # Config arrays (converted to numpy for vectorised math).
        self._freq_full_hz: np.ndarray = np.array(
            config.FLEX_MODE_FREQS_HZ[: self._n], dtype=float
        )
        self._freq_empty_hz: np.ndarray = np.array(
            config.FLEX_MODE_FREQS_EMPTY_HZ[: self._n], dtype=float
        )
        self._zeta: np.ndarray = np.array(
            config.FLEX_DAMPING_RATIOS[: self._n], dtype=float
        )
        self._slope_imu: np.ndarray = np.array(
            config.FLEX_MODE_SLOPES_AT_IMU[: self._n], dtype=float
        )
        self._slope_engine: np.ndarray = np.array(
            config.FLEX_MODE_SLOPES_AT_ENGINE[: self._n], dtype=float
        )

        # Per-mode state.
        self.modes: List[_ModalState] = [_ModalState() for _ in range(self._n)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _omega(self, propellant_fraction: float) -> np.ndarray:
        """Return current natural frequencies (rad/s) for each mode.

        Parameters
        ----------
        propellant_fraction : float
            Fraction of propellant remaining, in [0, 1].
        """
        frac = float(np.clip(propellant_fraction, 0.0, 1.0))
        freq_hz = self._freq_full_hz * frac + self._freq_empty_hz * (1.0 - frac)
        return 2.0 * np.pi * freq_hz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        """Number of active bending modes."""
        return self._n

    def reset(self) -> None:
        """Zero all modal states."""
        for m in self.modes:
            m.q = 0.0
            m.q_dot = 0.0

    def update(
        self,
        dt: float,
        tvc_force_n: float,
        propellant_fraction: float,
        modal_mass_kg: float = 1.0,
    ) -> np.ndarray:
        """Advance the bending modes by one timestep.

        Parameters
        ----------
        dt : float
            Integration timestep (s).
        tvc_force_n : float
            Lateral component of TVC thrust at the engine gimbal point (N).
            Positive = sideways force that would excite bending.
        propellant_fraction : float
            Fraction of propellant remaining [0, 1].  Used to interpolate
            natural frequencies.
        modal_mass_kg : float, optional
            Generalised (modal) mass common to all modes (kg).  Defaults
            to 1.0 (i.e. forcing is already normalised).

        Returns
        -------
        bending_rate_at_imu : np.ndarray, shape (n_modes,)
            Angular-rate contribution of each mode at the IMU location
            (rad/s).  Sum these and add to the body-rate measurement to
            model gyro corruption.
        """
        omega = self._omega(propellant_fraction)  # (n,)

        bending_rate_at_imu = np.empty(self._n, dtype=float)

        for i, mode in enumerate(self.modes):
            # Generalised force: TVC projected onto mode shape at engine.
            f_modal = tvc_force_n * self._slope_engine[i]
            f_over_m = f_modal / modal_mass_kg

            w = omega[i]
            z = self._zeta[i]

            # q̈ = -2ζωq̇ - ω²q + F/m
            q_ddot = -2.0 * z * w * mode.q_dot - w * w * mode.q + f_over_m

            # Semi-implicit Euler (symplectic — conserves energy better
            # than explicit Euler for oscillators).
            mode.q_dot += q_ddot * dt
            mode.q += mode.q_dot * dt

            # Bending angular rate sensed at IMU = q̇_i * (mode slope at IMU).
            bending_rate_at_imu[i] = mode.q_dot * self._slope_imu[i]

        return bending_rate_at_imu

    def total_bending_rate_at_imu(self) -> float:
        """Return the summed bending angular rate at the IMU (rad/s).

        Call *after* :meth:`update` within the same timestep.
        """
        total = 0.0
        for i, mode in enumerate(self.modes):
            total += mode.q_dot * self._slope_imu[i]
        return total

    def modal_displacements(self) -> np.ndarray:
        """Return current generalised displacements for all modes."""
        return np.array([m.q for m in self.modes], dtype=float)

    def modal_velocities(self) -> np.ndarray:
        """Return current generalised velocities for all modes."""
        return np.array([m.q_dot for m in self.modes], dtype=float)

    def kinetic_energy(self, modal_mass_kg: float = 1.0) -> float:
        """Total modal kinetic energy across all modes (J)."""
        return 0.5 * modal_mass_kg * float(
            np.sum(self.modal_velocities() ** 2)
        )

    def potential_energy(
        self, propellant_fraction: float, modal_mass_kg: float = 1.0
    ) -> float:
        """Total modal potential energy across all modes (J)."""
        omega = self._omega(propellant_fraction)
        return 0.5 * modal_mass_kg * float(
            np.sum((omega ** 2) * (self.modal_displacements() ** 2))
        )
