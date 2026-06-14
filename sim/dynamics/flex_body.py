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

import numpy as np

from sim import config


class FlexBody:
    """First-N lateral bending mode model.

    Parameters
    ----------
    n_modes : int, optional
        Number of bending modes to model (default: uses length of
        ``config.FLEX_MODE_FREQS_HZ``).

    Attributes
    ----------
    n_modes : int
        Number of active bending modes.
    """

    def __init__(self, n_modes: int | None = None) -> None:
        # Mode count — clamp to available config entries.
        max_modes = len(config.FLEX_MODE_FREQS_HZ)
        self._n: int = min(n_modes, max_modes) if n_modes is not None else max_modes

        # Config arrays (converted to numpy for vectorised math).
        self._freq_full_hz: np.ndarray = np.array(config.FLEX_MODE_FREQS_HZ[: self._n], dtype=float)
        self._freq_empty_hz: np.ndarray = np.array(config.FLEX_MODE_FREQS_EMPTY_HZ[: self._n], dtype=float)
        self._zeta: np.ndarray = np.array(config.FLEX_DAMPING_RATIOS[: self._n], dtype=float)
        self._slope_imu: np.ndarray = np.array(config.FLEX_MODE_SLOPES_AT_IMU[: self._n], dtype=float)
        self._slope_engine: np.ndarray = np.array(config.FLEX_MODE_SLOPES_AT_ENGINE[: self._n], dtype=float)

        # Pre-scaled natural frequency arrays (rad/s) — avoids repeating the
        # `2 * pi * f` multiplication on every update() call.
        two_pi = 2.0 * np.pi
        self._omega_full: np.ndarray = two_pi * self._freq_full_hz
        self._omega_empty: np.ndarray = two_pi * self._freq_empty_hz
        self._two_zeta: np.ndarray = 2.0 * self._zeta

        # Per-mode state (vectorised).
        self._q: np.ndarray = np.zeros(self._n, dtype=float)
        self._q_dot: np.ndarray = np.zeros(self._n, dtype=float)

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
        frac = propellant_fraction
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        return self._omega_full * frac + self._omega_empty * (1.0 - frac)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        """Number of active bending modes."""
        return self._n

    def modal_frequencies_hz(self, propellant_fraction: float) -> np.ndarray:
        """Current modal natural frequencies (Hz) for each mode.

        Uses the same full-to-empty interpolation as the internal dynamics, so a
        structural notch filter scheduled on these frequencies tracks exactly the
        modes this model excites (AD-04).
        """
        return self._omega(propellant_fraction) / (2.0 * np.pi)

    def reset(self) -> None:
        """Zero all modal states."""
        self._q.fill(0.0)
        self._q_dot.fill(0.0)

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

        # Generalised force: TVC projected onto mode shape at engine.
        force_scale = tvc_force_n / modal_mass_kg

        # q̈ = F/m - 2ζω*q̇ - ω²*q   (evaluated in-place on scratch buffer)
        damping = self._two_zeta * omega  # 2ζω
        q_ddot = force_scale * self._slope_engine
        q_ddot -= damping * self._q_dot
        q_ddot -= (omega * omega) * self._q

        # Semi-implicit Euler (symplectic — conserves energy better
        # than explicit Euler for oscillators).
        self._q_dot += q_ddot * dt
        self._q += self._q_dot * dt

        # Bending angular rate sensed at IMU = q̇_i * (mode slope at IMU).
        return self._q_dot * self._slope_imu

    def total_bending_rate_at_imu(self) -> float:
        """Return the summed bending angular rate at the IMU (rad/s).

        Call *after* :meth:`update` within the same timestep.
        """
        return float(np.sum(self._q_dot * self._slope_imu))

    def modal_displacements(self) -> np.ndarray:
        """Return current generalised displacements for all modes."""
        return self._q.copy()

    def modal_velocities(self) -> np.ndarray:
        """Return current generalised velocities for all modes."""
        return self._q_dot.copy()

    def kinetic_energy(self, modal_mass_kg: float = 1.0) -> float:
        """Total modal kinetic energy across all modes (J)."""
        return 0.5 * modal_mass_kg * float(np.sum(self._q_dot**2))

    def potential_energy(self, propellant_fraction: float, modal_mass_kg: float = 1.0) -> float:
        """Total modal potential energy across all modes (J)."""
        omega = self._omega(propellant_fraction)
        return 0.5 * modal_mass_kg * float(np.sum((omega**2) * (self._q**2)))
