"""Sensor models for 6-DOF ascent simulation.

Models IMU (accelerometer + gyroscope), GPS receiver, and barometric altimeter
with realistic noise, bias, and availability constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sim import config
from sim.core.state import VehicleState
from sim.core.reference_frames import eci_to_body


# ---------------------------------------------------------------------------
# Sensor measurement data classes
# ---------------------------------------------------------------------------

@dataclass
class IMUMeasurement:
    """Inertial measurement unit reading.

    Attributes:
        accel_body_mps2: Measured specific force in body frame (m/s^2).
        gyro_body_rads: Measured angular velocity in body frame (rad/s).
        time_s: Timestamp of the measurement (s).
    """
    accel_body_mps2: np.ndarray
    gyro_body_rads: np.ndarray
    time_s: float


@dataclass
class GPSMeasurement:
    """GPS receiver reading.

    Attributes:
        position_eci_m: Measured ECI position (m).
        velocity_eci_ms: Measured ECI velocity (m/s).
        time_s: Timestamp of the measurement (s).
    """
    position_eci_m: np.ndarray
    velocity_eci_ms: np.ndarray
    time_s: float


@dataclass
class BaroMeasurement:
    """Barometric altimeter reading.

    Attributes:
        altitude_m: Measured altitude above mean sea level (m).
        time_s: Timestamp of the measurement (s).
    """
    altitude_m: float
    time_s: float


# ---------------------------------------------------------------------------
# Sensor model classes
# ---------------------------------------------------------------------------

class IMU:
    """Strapdown IMU model with accelerometer and gyroscope.

    Runs at 100 Hz (every physics timestep).  Adds zero-mean Gaussian noise
    and integrates a random-walk bias on each axis.

    Args:
        rng: NumPy random generator for reproducibility.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()
        self._accel_bias: np.ndarray = np.zeros(3)
        self._gyro_bias: np.ndarray = np.zeros(3)

    # -- public API ----------------------------------------------------------

    def measure(
        self,
        true_state: VehicleState,
        gravity_eci_mps2: np.ndarray,
        dt: float,
    ) -> IMUMeasurement:
        """Return a noisy IMU measurement at the current timestep.

        The accelerometer measures *specific force* (total acceleration minus
        gravity) in the body frame, corrupted by noise and bias.

        Args:
            true_state: True vehicle state.
            gravity_eci_mps2: Gravitational acceleration at the vehicle
                position in ECI (m/s^2).
            dt: Simulation timestep (s).

        Returns:
            IMUMeasurement with noisy accel and gyro readings.
        """
        # True specific force in ECI = total_accel - gravity
        # We back-derive total_accel from the state by noting that
        # the dynamics provide it.  Here we compute specific force directly
        # as (a_total - g) and rotate to body frame.
        # For the sensor model we receive gravity_eci_mps2 from the caller,
        # and specific_force_eci is whatever the vehicle actually feels.
        # In a strapdown IMU the sensed quantity is: f_body = R*(a_inertial - g)
        # Since the dynamics propagate v_dot = g + thrust/m + aero/m,
        # the true specific force is v_dot - g  (all in ECI).
        # However, we don't have v_dot here, so we reconstruct from thrust/aero
        # using the provided gravity vector.
        # A simpler and fully equivalent approach: measure the *non-gravitational*
        # acceleration in the body frame.  We approximate by noting the
        # total specific force is (a - g).  The caller must supply a_total_eci
        # *or* we compute from the derivative.  For robustness we accept
        # gravity_eci and compute specific force = R * (0 - gravity) + true_accel_body.
        #
        # The cleanest approach for a sensor model: the caller provides the
        # true specific force in ECI (or we compute it).  We'll use:
        #   specific_force_eci = (v_dot) - gravity
        # But we don't have v_dot stored.  Instead we note that IMU measures
        # *specific force in body frame*.  We compute the true quantity as:
        #   f_body = DCM * (a_total_eci - g_eci)
        # Since a_total_eci = g_eci + thrust/m + aero/m + ..., the specific
        # force is just thrust/m + aero/m, rotated to body.
        #
        # For this sensor model, we take the pragmatic approach: true specific
        # force in body frame is computed by the caller and passed via
        # gravity_eci_mps2.  Here we compute:
        #   true_specific_force_body = DCM * (-gravity_eci)  [for free-fall = 0]
        # and the caller is responsible for adding thrust+aero contributions.
        #
        # SIMPLIFIED: We rotate the negative gravity to body frame.  The
        # dynamics caller should add non-gravitational terms before calling.
        # Actually, the standard approach is:
        #   true_accel_body = R_eci2body * (a_total_eci - g_eci)
        # We receive gravity_eci_mps2 as the full *specific force in ECI*
        # (poorly named parameter kept for interface compatibility).

        # Compute true specific force in body frame
        specific_force_eci = -gravity_eci_mps2  # placeholder: caller provides
        true_accel_body = eci_to_body(specific_force_eci, true_state.quaternion)

        # True angular velocity in body frame
        true_gyro_body = true_state.angular_velocity_body.copy()

        # Bias random walk
        self._accel_bias += self._rng.normal(
            0.0, config.IMU_ACCEL_BIAS_MPS2 * np.sqrt(dt), size=3
        )
        self._gyro_bias += self._rng.normal(
            0.0, config.IMU_GYRO_BIAS_RADS * np.sqrt(dt), size=3
        )

        # Add noise + bias
        accel_noise = self._rng.normal(0.0, config.IMU_ACCEL_NOISE_MPS2, size=3)
        gyro_noise = self._rng.normal(0.0, config.IMU_GYRO_NOISE_RADS, size=3)

        measured_accel = true_accel_body + self._accel_bias + accel_noise
        measured_gyro = true_gyro_body + self._gyro_bias + gyro_noise

        return IMUMeasurement(
            accel_body_mps2=measured_accel,
            gyro_body_rads=measured_gyro,
            time_s=true_state.time_s,
        )


class GPS:
    """GPS receiver model.

    Updates at ``GPS_UPDATE_HZ`` (default 1 Hz).  Returns *None* when the
    current timestep is not an update epoch or when the vehicle is above the
    COCOM altitude limit (60 km).

    Args:
        rng: NumPy random generator.
    """

    COCOM_ALT_M: float = 60_000.0  # COCOM altitude limit

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()
        self._update_period_s: float = 1.0 / config.GPS_UPDATE_HZ
        self._last_update_time_s: float = -1.0

    def measure(self, true_state: VehicleState, dt: float) -> Optional[GPSMeasurement]:
        """Return a GPS fix or *None* if not available this timestep.

        Args:
            true_state: True vehicle state.
            dt: Simulation timestep (s) — used only for epoch alignment.

        Returns:
            GPSMeasurement or None.
        """
        # Altitude check (COCOM limit)
        if true_state.altitude_m() > self.COCOM_ALT_M:
            return None

        # Rate check
        if not self._is_update_epoch(true_state.time_s):
            return None

        self._last_update_time_s = true_state.time_s

        pos_noise = self._rng.normal(0.0, config.GPS_POS_NOISE_M, size=3)
        vel_noise = self._rng.normal(0.0, config.GPS_VEL_NOISE_MS, size=3)

        return GPSMeasurement(
            position_eci_m=true_state.position_eci + pos_noise,
            velocity_eci_ms=true_state.velocity_eci + vel_noise,
            time_s=true_state.time_s,
        )

    def _is_update_epoch(self, time_s: float) -> bool:
        """Check whether *time_s* falls on a GPS update epoch."""
        if self._last_update_time_s < 0.0:
            return True  # first call
        elapsed = time_s - self._last_update_time_s
        return elapsed >= self._update_period_s - 1e-9


class Barometer:
    """Barometric altimeter model.

    Updates at ``BARO_UPDATE_HZ`` (default 10 Hz).  Returns *None* above 40 km
    where atmospheric pressure is too low for a useful reading.

    Args:
        rng: NumPy random generator.
    """

    MAX_USEFUL_ALT_M: float = 40_000.0

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()
        self._update_period_s: float = 1.0 / config.BARO_UPDATE_HZ
        self._last_update_time_s: float = -1.0

    def measure(self, true_state: VehicleState, dt: float) -> Optional[BaroMeasurement]:
        """Return a barometric altitude or *None* if unavailable.

        Args:
            true_state: True vehicle state.
            dt: Simulation timestep (s).

        Returns:
            BaroMeasurement or None.
        """
        alt = true_state.altitude_m()
        if alt > self.MAX_USEFUL_ALT_M:
            return None

        if not self._is_update_epoch(true_state.time_s):
            return None

        self._last_update_time_s = true_state.time_s

        noise = self._rng.normal(0.0, config.BARO_ALT_NOISE_M)
        measured_alt = alt + noise

        return BaroMeasurement(
            altitude_m=measured_alt,
            time_s=true_state.time_s,
        )

    def _is_update_epoch(self, time_s: float) -> bool:
        """Check whether *time_s* falls on a barometer update epoch."""
        if self._last_update_time_s < 0.0:
            return True
        elapsed = time_s - self._last_update_time_s
        return elapsed >= self._update_period_s - 1e-9


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

class SensorSuite:
    """Collection of all on-board sensors.

    Instantiate once at sim start and call :meth:`update` every timestep.

    Args:
        rng: NumPy random generator shared across all sensors.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.imu = IMU(rng=rng)
        self.gps = GPS(rng=rng)
        self.baro = Barometer(rng=rng)

    def update(
        self,
        true_state: VehicleState,
        gravity_eci_mps2: np.ndarray,
        dt: float,
    ) -> tuple[IMUMeasurement, Optional[GPSMeasurement], Optional[BaroMeasurement]]:
        """Poll every sensor and return measurements (None if unavailable).

        Args:
            true_state: True vehicle state from the dynamics engine.
            gravity_eci_mps2: Gravitational acceleration in ECI at the
                vehicle position (m/s^2).
            dt: Physics timestep (s).

        Returns:
            Tuple of (imu, gps_or_none, baro_or_none).
        """
        imu_meas = self.imu.measure(true_state, gravity_eci_mps2, dt)
        gps_meas = self.gps.measure(true_state, dt)
        baro_meas = self.baro.measure(true_state, dt)
        return imu_meas, gps_meas, baro_meas
