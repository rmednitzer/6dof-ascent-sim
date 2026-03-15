"""System health monitor — continuous assessment of vehicle subsystem health.

Provides a unified :class:`HealthVector` that downstream systems (guidance,
telemetry, FTS) can inspect each timestep.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from sim.config import (
    FTS_COVARIANCE_LIMIT_M,
    MAX_Q_PA,
)


# ------------------------------------------------------------------
# Health status enum
# ------------------------------------------------------------------
class HealthStatus(enum.IntEnum):
    """Ordered severity levels — higher numeric value means worse health."""

    NOMINAL = 0
    WARNING = 1
    ALERT = 2
    CRITICAL = 3


# ------------------------------------------------------------------
# Per-channel health dataclass
# ------------------------------------------------------------------
@dataclass
class HealthVector:
    """Snapshot of all monitored health channels.

    Each channel is a :class:`HealthStatus` value.  Use
    :meth:`HealthMonitor.overall_status` to obtain the worst across all
    channels.

    Attributes:
        ekf_covariance:  Health of EKF position uncertainty.
        dynamic_pressure: Health relative to structural q limit.
        propellant_margin: Propellant remaining status.
        sensor_status:    Aggregated sensor-degradation status.
        engine_health:    Engine thrust-deviation status.
    """

    ekf_covariance: HealthStatus = HealthStatus.NOMINAL
    dynamic_pressure: HealthStatus = HealthStatus.NOMINAL
    propellant_margin: HealthStatus = HealthStatus.NOMINAL
    sensor_status: HealthStatus = HealthStatus.NOMINAL
    engine_health: HealthStatus = HealthStatus.NOMINAL


# ------------------------------------------------------------------
# Monitor
# ------------------------------------------------------------------
class HealthMonitor:
    """Evaluates vehicle subsystem health every timestep.

    Thresholds
    ----------
    - **EKF covariance**: WARNING at 50 % of FTS limit, ALERT at 80 %.
    - **Dynamic pressure**: tracks peak q; WARNING at 80 % of structural
      limit, ALERT at 95 %, CRITICAL if exceeded.
    - **Propellant margin**: WARNING when remaining fraction drops below 5 %.
    - **Sensor status**: passthrough for externally-supplied degradation flags.
    - **Engine health**: WARNING if thrust deviates > 5 % from commanded,
      ALERT > 10 %, CRITICAL > 20 %.
    """

    # EKF thresholds (fractions of FTS_COVARIANCE_LIMIT_M)
    _EKF_WARN_FRAC: float = 0.50
    _EKF_ALERT_FRAC: float = 0.80

    # Dynamic-pressure thresholds (fractions of MAX_Q_PA)
    _Q_WARN_FRAC: float = 0.80
    _Q_ALERT_FRAC: float = 0.95

    # Propellant margin threshold (fraction of initial load)
    _PROP_WARN_FRAC: float = 0.05

    # Engine thrust deviation thresholds (fraction of commanded)
    _ENGINE_WARN_FRAC: float = 0.05
    _ENGINE_ALERT_FRAC: float = 0.10
    _ENGINE_CRITICAL_FRAC: float = 0.20

    def __init__(self) -> None:
        self.health = HealthVector()
        self.peak_dynamic_pressure_pa: float = 0.0
        self._sensor_flags: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        ekf_pos_covariance: np.ndarray,
        dynamic_pressure_pa: float,
        propellant_remaining_kg: float,
        propellant_initial_kg: float,
        sensor_degradation_flags: dict[str, bool] | None = None,
        commanded_thrust_n: float = 0.0,
        actual_thrust_n: float = 0.0,
    ) -> HealthVector:
        """Evaluate all health channels and return an updated :class:`HealthVector`.

        Args:
            ekf_pos_covariance:       3x3 EKF position covariance (m²).
            dynamic_pressure_pa:      Current dynamic pressure (Pa).
            propellant_remaining_kg:  Remaining propellant mass (kg).
            propellant_initial_kg:    Initial propellant mass (kg).
            sensor_degradation_flags: ``{sensor_name: is_degraded}``.
            commanded_thrust_n:       Commanded thrust (N).
            actual_thrust_n:          Measured / estimated actual thrust (N).

        Returns:
            The updated :class:`HealthVector`.
        """
        self.health.ekf_covariance = self._assess_ekf(ekf_pos_covariance)
        self.health.dynamic_pressure = self._assess_dynamic_pressure(dynamic_pressure_pa)
        self.health.propellant_margin = self._assess_propellant(propellant_remaining_kg, propellant_initial_kg)
        self.health.sensor_status = self._assess_sensors(sensor_degradation_flags or {})
        self.health.engine_health = self._assess_engine(commanded_thrust_n, actual_thrust_n)
        return self.health

    def overall_status(self) -> HealthStatus:
        """Return the worst (highest severity) status across all channels."""
        return HealthStatus(
            max(
                self.health.ekf_covariance,
                self.health.dynamic_pressure,
                self.health.propellant_margin,
                self.health.sensor_status,
                self.health.engine_health,
            )
        )

    # ------------------------------------------------------------------
    # Channel assessors
    # ------------------------------------------------------------------
    def _assess_ekf(self, cov: np.ndarray) -> HealthStatus:
        """EKF position covariance health.

        Largest 1-sigma position uncertainty compared against FTS limit.
        """
        cov = np.asarray(cov, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(cov)
        sigma = float(np.sqrt(np.max(np.abs(eigenvalues))))
        frac = sigma / FTS_COVARIANCE_LIMIT_M if FTS_COVARIANCE_LIMIT_M else 0.0

        if frac >= 1.0:
            return HealthStatus.CRITICAL
        if frac >= self._EKF_ALERT_FRAC:
            return HealthStatus.ALERT
        if frac >= self._EKF_WARN_FRAC:
            return HealthStatus.WARNING
        return HealthStatus.NOMINAL

    def _assess_dynamic_pressure(self, q_pa: float) -> HealthStatus:
        """Dynamic-pressure health, also tracking peak q."""
        self.peak_dynamic_pressure_pa = max(self.peak_dynamic_pressure_pa, q_pa)
        frac = q_pa / MAX_Q_PA if MAX_Q_PA else 0.0

        if frac > 1.0:
            return HealthStatus.CRITICAL
        if frac >= self._Q_ALERT_FRAC:
            return HealthStatus.ALERT
        if frac >= self._Q_WARN_FRAC:
            return HealthStatus.WARNING
        return HealthStatus.NOMINAL

    @staticmethod
    def _assess_propellant(
        remaining_kg: float,
        initial_kg: float,
    ) -> HealthStatus:
        """Propellant margin health — warn below 5 % remaining."""
        if initial_kg <= 0.0:
            return HealthStatus.CRITICAL
        frac = remaining_kg / initial_kg
        if frac <= 0.0:
            return HealthStatus.CRITICAL
        if frac <= HealthMonitor._PROP_WARN_FRAC:
            return HealthStatus.WARNING
        return HealthStatus.NOMINAL

    def _assess_sensors(
        self,
        flags: dict[str, bool],
    ) -> HealthStatus:
        """Aggregate sensor degradation flags.

        Any single degraded sensor -> WARNING; two or more -> ALERT.
        """
        self._sensor_flags = dict(flags)
        degraded_count = sum(1 for v in flags.values() if v)
        if degraded_count >= 2:
            return HealthStatus.ALERT
        if degraded_count == 1:
            return HealthStatus.WARNING
        return HealthStatus.NOMINAL

    @staticmethod
    def _assess_engine(
        commanded_n: float,
        actual_n: float,
    ) -> HealthStatus:
        """Engine health based on thrust deviation from commanded value."""
        if commanded_n <= 0.0:
            # Engine is off — nothing to compare.
            return HealthStatus.NOMINAL
        deviation_frac = abs(actual_n - commanded_n) / commanded_n
        if deviation_frac >= HealthMonitor._ENGINE_CRITICAL_FRAC:
            return HealthStatus.CRITICAL
        if deviation_frac >= HealthMonitor._ENGINE_ALERT_FRAC:
            return HealthStatus.ALERT
        if deviation_frac >= HealthMonitor._ENGINE_WARN_FRAC:
            return HealthStatus.WARNING
        return HealthStatus.NOMINAL
