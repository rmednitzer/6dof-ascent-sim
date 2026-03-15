"""Extended Kalman Filter for launch vehicle navigation.

12-state EKF estimating position, velocity, accelerometer bias, and gyro bias
using IMU, GPS, and barometric altimeter measurements.

State vector: [px, py, pz, vx, vy, vz, bax, bay, baz, bgx, bgy, bgz]
    - position (ECI, m)
    - velocity (ECI, m/s)
    - accelerometer bias (body frame, m/s^2)
    - gyro bias (body frame, rad/s)
"""

from __future__ import annotations

import numpy as np

from sim import config
from sim.core.reference_frames import quat_to_dcm
from sim.core.state import VehicleState
from sim.gnc.sensors import BaroMeasurement, GPSMeasurement, IMUMeasurement


class NavigationEKF:
    """12-state Extended Kalman Filter for ascent navigation.

    Predict at 100 Hz using strapdown IMU mechanisation.  Update with GPS
    (1 Hz, below 60 km) and barometer (10 Hz, below 40 km).  An innovation
    gate rejects measurements whose residual exceeds
    ``EKF_RESIDUAL_SIGMA_THRESHOLD`` standard deviations.

    Args:
        initial_state: Vehicle state used to seed the filter.
    """

    N_STATES: int = 12

    def __init__(self, initial_state: VehicleState) -> None:
        # State vector
        self._x = np.zeros(self.N_STATES)
        self._x[0:3] = initial_state.position_eci.copy()
        self._x[3:6] = initial_state.velocity_eci.copy()
        # biases initialised to zero

        # Covariance
        self._P = np.diag(
            [
                100.0,
                100.0,
                100.0,  # position (m^2)
                1.0,
                1.0,
                1.0,  # velocity (m/s)^2
                1e-4,
                1e-4,
                1e-4,  # accel bias
                1e-6,
                1e-6,
                1e-6,  # gyro bias
            ]
        )

        # Store latest quaternion estimate (propagated externally or from
        # the attitude controller; the EKF does not estimate attitude).
        self._quaternion = initial_state.quaternion.copy()
        self._angular_velocity_body = initial_state.angular_velocity_body.copy()
        self._mass_kg = initial_state.mass_kg
        self._time_s = initial_state.time_s

    # -- public properties ---------------------------------------------------

    @property
    def state_vector(self) -> np.ndarray:
        """Current 12-element state estimate."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current 12x12 covariance matrix."""
        return self._P.copy()

    def position_uncertainty_m(self) -> float:
        """1-sigma position uncertainty: sqrt(trace(P[0:3, 0:3]))."""
        return float(np.sqrt(np.trace(self._P[0:3, 0:3])))

    def velocity_uncertainty_ms(self) -> float:
        """1-sigma velocity uncertainty: sqrt(trace(P[3:6, 3:6]))."""
        return float(np.sqrt(np.trace(self._P[3:6, 3:6])))

    # -- predict step --------------------------------------------------------

    def predict(
        self,
        imu: IMUMeasurement,
        gravity_eci_mps2: np.ndarray,
        dt: float,
    ) -> None:
        """Propagate state and covariance using IMU measurement.

        Strapdown mechanisation:
            p_new = p + v*dt + 0.5*a_eci*dt^2
            v_new = v + a_eci*dt
        where a_eci = DCM^T * (accel_body - bias_accel) + gravity.

        Biases are modelled as random walks (constant + process noise).

        Args:
            imu: IMU measurement for this timestep.
            gravity_eci_mps2: Gravitational acceleration in ECI (m/s^2).
            dt: Timestep (s).
        """
        self._time_s = imu.time_s

        # Current estimates
        pos = self._x[0:3]
        vel = self._x[3:6]
        ba = self._x[6:9]  # accel bias estimate (body)
        bg = self._x[9:12]  # gyro bias estimate (body)

        # Corrected IMU readings
        accel_body_corrected = imu.accel_body_mps2 - ba
        gyro_body_corrected = imu.gyro_body_rads - bg

        # Body-to-ECI rotation (DCM^T since quat_to_dcm gives ECI-to-body)
        R_body2eci = quat_to_dcm(self._quaternion).T

        # Specific force in ECI
        accel_eci = R_body2eci @ accel_body_corrected

        # Total acceleration in ECI (specific force + gravity)
        a_total = accel_eci + gravity_eci_mps2

        # State propagation
        self._x[0:3] = pos + vel * dt + 0.5 * a_total * dt**2
        self._x[3:6] = vel + a_total * dt
        # Biases: random walk, no deterministic drift
        # self._x[6:9] unchanged
        # self._x[9:12] unchanged

        # Update stored angular velocity
        self._angular_velocity_body = gyro_body_corrected.copy()

        # -- Covariance propagation via linearised dynamics --
        # State transition Jacobian F (12x12)
        F = np.eye(self.N_STATES)
        F[0:3, 3:6] = np.eye(3) * dt  # dp/dv
        F[0:3, 6:9] = -0.5 * R_body2eci * dt**2  # dp/dba
        F[3:6, 6:9] = -R_body2eci * dt  # dv/dba

        # Process noise covariance Q
        Q = np.zeros((self.N_STATES, self.N_STATES))
        # Position and velocity noise from accelerometer noise
        accel_var = config.IMU_ACCEL_NOISE_MPS2**2
        _gyro_var = config.IMU_GYRO_NOISE_RADS**2  # noqa: F841 (reserved for gyro process noise)
        Q[0:3, 0:3] = np.eye(3) * accel_var * dt**4 / 4.0
        Q[3:6, 3:6] = np.eye(3) * accel_var * dt**2
        Q[0:3, 3:6] = np.eye(3) * accel_var * dt**3 / 2.0
        Q[3:6, 0:3] = Q[0:3, 3:6].T
        # Bias random walk
        Q[6:9, 6:9] = np.eye(3) * config.IMU_ACCEL_BIAS_MPS2**2 * dt
        Q[9:12, 9:12] = np.eye(3) * config.IMU_GYRO_BIAS_RADS**2 * dt

        self._P = F @ self._P @ F.T + Q

        # Symmetrise to avoid numerical drift
        self._P = 0.5 * (self._P + self._P.T)

    # -- update steps --------------------------------------------------------

    def update_gps(self, gps: GPSMeasurement) -> None:
        """Fuse a GPS measurement (position + velocity).

        Measurement model: z = [pos; vel], H picks out states 0..5.

        Args:
            gps: GPS measurement.
        """
        # Measurement vector
        z = np.concatenate([gps.position_eci_m, gps.velocity_eci_ms])

        # Predicted measurement
        z_pred = self._x[0:6]

        # Innovation
        y = z - z_pred

        # Observation matrix (6x12)
        H = np.zeros((6, self.N_STATES))
        H[0:6, 0:6] = np.eye(6)

        # Measurement noise
        R = np.diag(
            [
                config.GPS_POS_NOISE_M**2,
                config.GPS_POS_NOISE_M**2,
                config.GPS_POS_NOISE_M**2,
                config.GPS_VEL_NOISE_MS**2,
                config.GPS_VEL_NOISE_MS**2,
                config.GPS_VEL_NOISE_MS**2,
            ]
        )

        self._apply_update(y, H, R)

    def update_baro(self, baro: BaroMeasurement) -> None:
        """Fuse a barometric altitude measurement.

        The barometer measures altitude = |position| - R_earth.
        Linearised about the current position estimate.

        Args:
            baro: Barometer measurement.
        """
        pos = self._x[0:3]
        r = np.linalg.norm(pos)
        if r < 1.0:
            return  # degenerate

        # Predicted altitude
        alt_pred = r - config.EARTH_RADIUS_M

        # Innovation
        y = np.array([baro.altitude_m - alt_pred])

        # Jacobian of altitude w.r.t. position: d|r|/dr = r_hat
        r_hat = pos / r
        H = np.zeros((1, self.N_STATES))
        H[0, 0:3] = r_hat

        # Measurement noise
        R = np.array([[config.BARO_ALT_NOISE_M**2]])

        self._apply_update(y, H, R)

    # -- private helpers -----------------------------------------------------

    def _apply_update(
        self,
        y: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """Run the Kalman update with innovation gating.

        Rejects the measurement if any component of the normalised residual
        exceeds ``EKF_RESIDUAL_SIGMA_THRESHOLD``.

        Args:
            y: Innovation (residual) vector.
            H: Observation Jacobian.
            R: Measurement noise covariance.
        """
        S = H @ self._P @ H.T + R  # innovation covariance

        # Innovation gate
        threshold = config.EKF_RESIDUAL_SIGMA_THRESHOLD
        for i in range(len(y)):
            sigma_i = np.sqrt(S[i, i])
            if sigma_i > 0.0 and abs(y[i]) > threshold * sigma_i:
                return  # reject entire measurement

        # Kalman gain
        S_inv = np.linalg.inv(S)
        K = self._P @ H.T @ S_inv

        # State update
        self._x = self._x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.N_STATES) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

        # Symmetrise
        self._P = 0.5 * (self._P + self._P.T)

    # -- estimated state output ----------------------------------------------

    def set_attitude(
        self,
        quaternion: np.ndarray,
        angular_velocity_body: np.ndarray,
    ) -> None:
        """Update the attitude estimate used for IMU mechanisation.

        The EKF does not estimate attitude directly; it must be fed from
        the gyro-propagated or controller-estimated quaternion.

        Args:
            quaternion: Attitude quaternion [x, y, z, w].
            angular_velocity_body: Angular velocity in body frame (rad/s).
        """
        self._quaternion = quaternion.copy()
        self._angular_velocity_body = angular_velocity_body.copy()

    def set_mass(self, mass_kg: float) -> None:
        """Update mass estimate (used only for state output)."""
        self._mass_kg = mass_kg

    def estimated_state(self) -> VehicleState:
        """Build a VehicleState from the current EKF estimate.

        Returns:
            VehicleState with estimated position, velocity, and the
            externally-provided attitude and mass.
        """
        return VehicleState(
            position_eci=self._x[0:3].copy(),
            velocity_eci=self._x[3:6].copy(),
            quaternion=self._quaternion.copy(),
            angular_velocity_body=self._angular_velocity_body.copy(),
            mass_kg=self._mass_kg,
            time_s=self._time_s,
        )
