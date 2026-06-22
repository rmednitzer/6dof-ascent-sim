"""Sensor models for 6-DOF ascent simulation.

Models IMU (accelerometer + gyroscope), GPS receiver, and barometric altimeter
with realistic noise, bias, and availability constraints.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass

import numpy as np

from sim import config
from sim.core.reference_frames import (
    ecef_to_eci,
    eci_to_body,
    eci_to_ecef,
    lla_to_ecef,
    quaternion_from_axis_angle,
    quaternion_multiply,
)
from sim.core.state import VehicleState

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


@dataclass
class StarTrackerMeasurement:
    """Star-tracker attitude reading.

    Attributes:
        quaternion: Measured inertial (body->ECI) attitude, scalar-last
            ``[x, y, z, w]``.
        time_s: Timestamp of the measurement (s).
    """

    quaternion: np.ndarray
    time_s: float


@dataclass
class GroundRangeMeasurement:
    """Slant-range reading from one ground tracking station.

    Attributes:
        station_position_eci: ECI position of the station at the measurement
            epoch (m) — the filter recomputes the predicted range from this.
        range_m: Measured slant range to the vehicle (m).
        station_name: Station identifier (for telemetry/debug).
        time_s: Timestamp of the measurement (s).
    """

    station_position_eci: np.ndarray
    range_m: float
    station_name: str
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
        # Use a cryptographically secure seed if no generator is provided
        self._rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
        self._accel_bias: np.ndarray = np.zeros(3)
        self._gyro_bias: np.ndarray = np.zeros(3)

    # -- public API ----------------------------------------------------------

    def measure(
        self,
        true_state: VehicleState,
        specific_force_eci_mps2: np.ndarray,
        dt: float,
    ) -> IMUMeasurement:
        """Return a noisy IMU measurement at the current timestep.

        A strapdown accelerometer senses *specific force* — the
        non-gravitational acceleration (thrust + aerodynamic + structural
        contact forces divided by mass). It does not sense gravity: in
        free fall the accelerometer reads ~0. The sensed quantity is
        ``f_body = R_eci->body * f_eci`` where ``f_eci = a_total - g``.

        Args:
            true_state: True vehicle state.
            specific_force_eci_mps2: True non-gravitational (specific
                force) acceleration in ECI (m/s^2) — i.e. ``(thrust +
                aero + ...) / mass``, excluding gravity.
            dt: Simulation timestep (s).

        Returns:
            IMUMeasurement with noisy accel and gyro readings.
        """
        true_accel_body = eci_to_body(specific_force_eci_mps2, true_state.quaternion)

        # True angular velocity in body frame
        true_gyro_body = true_state.angular_velocity_body.copy()

        # Bias random walk
        self._accel_bias += self._rng.normal(0.0, config.IMU_ACCEL_BIAS_MPS2 * np.sqrt(dt), size=3)
        self._gyro_bias += self._rng.normal(0.0, config.IMU_GYRO_BIAS_RADS * np.sqrt(dt), size=3)

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
    GPS availability ceiling (``config.GPS_AVAILABILITY_CEILING_M``, default
    60 km — the COCOM export limit). Above it the upper stage is GPS-denied;
    attitude observability there is provided by the star tracker (ADR 0020).
    Set the ceiling to +inf to model a cleared (SAASM / M-code) receiver.

    Args:
        rng: NumPy random generator.
    """

    STALE_GRACE_S: float = 3.0  # dropout grace (~3× the 1 Hz update period)

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        # Use a cryptographically secure seed if no generator is provided
        self._rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
        self._update_period_s: float = 1.0 / config.GPS_UPDATE_HZ
        self._last_update_time_s: float = -1.0

    def measure(self, true_state: VehicleState, dt: float) -> GPSMeasurement | None:
        """Return a GPS fix or *None* if not available this timestep.

        Args:
            true_state: True vehicle state.
            dt: Simulation timestep (s) — used only for epoch alignment.

        Returns:
            GPSMeasurement or None.
        """
        # Availability ceiling (COCOM for a COTS receiver; +inf for a cleared
        # launch receiver — GPS through ascent, ADR 0020).
        if true_state.altitude_m() > config.GPS_AVAILABILITY_CEILING_M:
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

    def degraded(self, true_state: VehicleState) -> bool:
        """True if a fix is *expected* (below the ceiling) but none has arrived
        within the staleness grace — a real dropout (Q-03). The expected loss of
        fix above the COCOM ceiling is not a fault and is not flagged."""
        if true_state.altitude_m() > config.GPS_AVAILABILITY_CEILING_M:
            return False  # out of envelope — expected, not degraded
        if self._last_update_time_s < 0.0:
            return False  # no fix yet at startup
        return (true_state.time_s - self._last_update_time_s) > self.STALE_GRACE_S


class Barometer:
    """Barometric altimeter model.

    Updates at ``BARO_UPDATE_HZ`` (default 10 Hz).  Returns *None* above 40 km
    where atmospheric pressure is too low for a useful reading.

    Args:
        rng: NumPy random generator.
    """

    MAX_USEFUL_ALT_M: float = 40_000.0
    STALE_GRACE_S: float = 1.0  # dropout grace (~10× the 10 Hz update period)

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        # Use a cryptographically secure seed if no generator is provided
        self._rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
        self._update_period_s: float = 1.0 / config.BARO_UPDATE_HZ
        self._last_update_time_s: float = -1.0

    def measure(self, true_state: VehicleState, dt: float) -> BaroMeasurement | None:
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

    def degraded(self, true_state: VehicleState) -> bool:
        """True if a reading is *expected* (below MAX_USEFUL_ALT_M) but none has
        arrived within the staleness grace — a real dropout (Q-03). The expected
        loss of signal at high altitude is not a fault and is not flagged."""
        if true_state.altitude_m() > self.MAX_USEFUL_ALT_M:
            return False  # out of envelope — expected, not degraded
        if self._last_update_time_s < 0.0:
            return False
        return (true_state.time_s - self._last_update_time_s) > self.STALE_GRACE_S


class StarTracker:
    """Star-tracker inertial-attitude sensor (ADR 0020).

    Measures the full inertial attitude quaternion directly (arcsecond-class
    noise) by imaging star fields. It is the attitude aid that keeps the
    error-state EKF observable while the vehicle is GPS-denied above the COCOM
    ceiling. Like a real unit it is usable only above the sensible atmosphere
    (clear sky) and below a slew rate that would smear the star image — i.e. the
    upper-stage coast/burn regime — and updates at ``STAR_TRACKER_UPDATE_HZ``.
    Returns *None* when unavailable or off-epoch.

    Args:
        rng: NumPy random generator.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        # Use a cryptographically secure seed if no generator is provided
        self._rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
        self._update_period_s: float = 1.0 / config.STAR_TRACKER_UPDATE_HZ
        self._last_update_time_s: float = -1.0

    def measure(self, true_state: VehicleState, dt: float) -> StarTrackerMeasurement | None:
        """Return a noisy inertial-attitude fix or *None* if unavailable.

        Args:
            true_state: True vehicle state.
            dt: Simulation timestep (s) — used only for epoch alignment.

        Returns:
            StarTrackerMeasurement or None.
        """
        # Availability: above the sensible atmosphere and below the slew limit.
        if true_state.altitude_m() < config.STAR_TRACKER_MIN_ALT_M:
            return None
        if float(np.linalg.norm(true_state.angular_velocity_body)) > config.STAR_TRACKER_MAX_RATE_RADS:
            return None
        if not self._is_update_epoch(true_state.time_s):
            return None

        self._last_update_time_s = true_state.time_s

        # Small-angle Gaussian attitude noise applied as an ECI-frame rotation.
        noise = self._rng.normal(0.0, config.STAR_TRACKER_NOISE_RAD, size=3)
        angle = float(np.linalg.norm(noise))
        if angle > 1e-15:
            dq = quaternion_from_axis_angle(noise / angle, angle)
            q_meas = quaternion_multiply(dq, true_state.quaternion)
            q_meas = q_meas / np.linalg.norm(q_meas)
        else:
            q_meas = true_state.quaternion.copy()

        return StarTrackerMeasurement(quaternion=q_meas, time_s=true_state.time_s)

    def _is_update_epoch(self, time_s: float) -> bool:
        """Check whether *time_s* falls on a star-tracker update epoch."""
        if self._last_update_time_s < 0.0:
            return True
        elapsed = time_s - self._last_update_time_s
        return elapsed >= self._update_period_s - 1e-9


class GroundStation:
    """Ground tracking station that ranges the vehicle (ADR 0023).

    A launch-range radar / transponder that measures slant range to the vehicle
    while it is above the station's elevation mask. It is independent of GPS — the
    vehicle is a tracked target, not a self-locating receiver — so it is not bound
    by the COCOM ceiling and keeps aiding EKF position through the GPS-denied coast.
    Updates at ``GROUND_TRACK_UPDATE_HZ``; returns *None* when the vehicle is below
    the elevation mask or off-epoch.

    Args:
        name: Station identifier.
        lat_deg, lon_deg, alt_m: Geodetic station location.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        name: str,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.name = name
        self._ecef = lla_to_ecef(math.radians(lat_deg), math.radians(lon_deg), alt_m)
        self._up = self._ecef / np.linalg.norm(self._ecef)  # geocentric local vertical
        self._rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
        self._update_period_s = 1.0 / config.GROUND_TRACK_UPDATE_HZ
        self._last_update_time_s = -1.0

    def measure(self, true_state: VehicleState, dt: float) -> GroundRangeMeasurement | None:
        """Return a noisy slant-range fix or *None* if the vehicle is not visible."""
        if not self._is_update_epoch(true_state.time_s):
            return None

        # Line of sight and elevation in ECEF (station is fixed in ECEF).
        veh_ecef = eci_to_ecef(true_state.position_eci, true_state.time_s)
        los = veh_ecef - self._ecef
        slant = float(np.linalg.norm(los))
        if slant < 1.0:
            return None
        elevation = math.asin(float(np.clip(np.dot(los, self._up) / slant, -1.0, 1.0)))
        if elevation < math.radians(config.GROUND_TRACK_ELEV_MASK_DEG):
            return None  # below the horizon mask — no track

        self._last_update_time_s = true_state.time_s
        noise = float(self._rng.normal(0.0, config.GROUND_RANGE_NOISE_M))
        return GroundRangeMeasurement(
            station_position_eci=ecef_to_eci(self._ecef, true_state.time_s),
            range_m=slant + noise,
            station_name=self.name,
            time_s=true_state.time_s,
        )

    def _is_update_epoch(self, time_s: float) -> bool:
        """Check whether *time_s* falls on a ranging epoch."""
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
            # Use a cryptographically secure seed if no generator is provided
            rng = np.random.default_rng(secrets.randbits(128))
        self.imu = IMU(rng=rng)
        self.gps = GPS(rng=rng)
        self.baro = Barometer(rng=rng)
        self.star_tracker = StarTracker(rng=rng)
        self.ground_stations = [
            GroundStation(name, lat, lon, alt, rng=rng) for (name, lat, lon, alt) in config.GROUND_STATIONS
        ]

    def update(
        self,
        true_state: VehicleState,
        specific_force_eci_mps2: np.ndarray,
        dt: float,
    ) -> tuple[
        IMUMeasurement,
        GPSMeasurement | None,
        BaroMeasurement | None,
        StarTrackerMeasurement | None,
        list[GroundRangeMeasurement],
    ]:
        """Poll every sensor and return measurements (None / empty if unavailable).

        Args:
            true_state: True vehicle state from the dynamics engine.
            specific_force_eci_mps2: True non-gravitational (specific
                force) acceleration in ECI at the vehicle (m/s^2).
            dt: Physics timestep (s).

        Returns:
            Tuple of (imu, gps_or_none, baro_or_none, star_tracker_or_none,
            ground_ranges) where ``ground_ranges`` is a list with one entry per
            station currently tracking the vehicle (possibly empty).
        """
        imu_meas = self.imu.measure(true_state, specific_force_eci_mps2, dt)
        gps_meas = self.gps.measure(true_state, dt)
        baro_meas = self.baro.measure(true_state, dt)
        star_meas = self.star_tracker.measure(true_state, dt)
        ground_meas = [m for st in self.ground_stations if (m := st.measure(true_state, dt)) is not None]
        return imu_meas, gps_meas, baro_meas, star_meas, ground_meas

    def degradation_flags(self, true_state: VehicleState) -> dict[str, bool]:
        """Per-sensor degradation flags for the health monitor (Q-03).

        A sensor is *degraded* only when it is within its operating envelope yet
        has gone stale (a real dropout) — expected loss of aiding outside the
        envelope (GPS above the COCOM ceiling, baro at high altitude) is not
        flagged. Call after :meth:`update`. Covers the position aids with simple
        altitude envelopes; star-tracker / ground-network health is future work.
        """
        return {
            "gps": self.gps.degraded(true_state),
            "baro": self.baro.degraded(true_state),
        }
