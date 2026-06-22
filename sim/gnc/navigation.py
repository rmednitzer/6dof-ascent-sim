"""Error-state Extended Kalman Filter for launch-vehicle navigation.

15-error-state EKF estimating attitude, position, velocity, accelerometer bias,
and gyro bias from IMU, GPS, and barometric-altimeter measurements. Attitude is
carried as a nominal quaternion (propagated from the *measured* gyro) plus a
3-DOF multiplicative error in the covariance — the standard error-state /
multiplicative-EKF formulation. This replaces the earlier 12-state filter that
did not estimate attitude (it was handed the true quaternion); see ADR 0020.

Nominal state (stored in ``_x``, 12 elements) + nominal quaternion (``_quat``):
    [px, py, pz, vx, vy, vz, bax, bay, baz, bgx, bgy, bgz]  (ECI pos/vel; body bias)
Error state (tracked by the 15x15 covariance ``_P``):
    [δpx..δpz, δvx..δvz, δbax..δbaz, δbgx..δbgz, δθx, δθy, δθz]
where δθ is a small ECI-frame attitude error: R_true_b2n = exp([δθ]_x) R_est_b2n.

Continuous error dynamics (ECI nav frame; f_n = R_b2n f_b is specific force):
    δṗ = δv
    δv̇ = -[f_n]_x δθ - R_b2n δb_a
    δθ̇ = -R_b2n δb_g
    δḃ_a = δḃ_g = 0   (random walks, driven by process noise)

References:
    Sola, J. (2017), "Quaternion kinematics for the error-state Kalman filter".
    Groves, P. (2013), "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems", 2nd ed., Ch. 14 (error dynamics).
    Savage, P.G. (1998), AIAA JGCD (coning/sculling).
    Titterton & Weston, *Strapdown Inertial Navigation Technology*, 2nd ed.
    Cross-checked against the PX4 ECL/EKF2 error-state formulation.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.stats import chi2

from sim import config
from sim.core.fast_math import cross3
from sim.core.reference_frames import (
    quat_to_dcm,
    quaternion_conjugate,
    quaternion_from_axis_angle,
    quaternion_multiply,
)
from sim.core.state import VehicleState
from sim.gnc.sensors import (
    BaroMeasurement,
    GPSMeasurement,
    GroundRangeMeasurement,
    IMUMeasurement,
    StarTrackerMeasurement,
)

# Error-state indices.
_P_IDX, _V_IDX, _BA_IDX, _BG_IDX, _TH_IDX = 0, 3, 6, 9, 12
N_ERR = 15


@lru_cache(maxsize=16)
def _nis_gate_threshold(dim: int, gate_p: float) -> float:
    """Chi-square quantile used by the EKF innovation-consistency (NIS) gate.

    The normalised innovation squared ``yᵀ S⁻¹ y`` is chi-square distributed
    with ``dim`` degrees of freedom under the filter-consistency hypothesis; a
    measurement is rejected when it exceeds the ``gate_p`` quantile.
    """
    return float(chi2.ppf(gate_p, dim))


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix [v]_x."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _coning_sculling_dv(
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
    prev_accel: np.ndarray,
    prev_gyro: np.ndarray,
    prev_valid: bool,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Savage two-sample coning/sculling-compensated angle and velocity increments.

    Returns ``(theta_corrected, dv_body)`` — the attitude increment (rad) and the
    body-frame velocity increment (m/s) for this sample.
    """
    theta_curr = gyro_body * dt
    dv_curr = accel_body * dt
    if not prev_valid:
        return theta_curr, dv_curr
    theta_prev = prev_gyro * dt
    dv_prev = prev_accel * dt
    theta_corrected = theta_curr + (2.0 / 3.0) * cross3(theta_prev, theta_curr)
    dv_corrected = dv_curr + (2.0 / 3.0) * (cross3(theta_prev, dv_curr) + cross3(dv_prev, theta_curr))
    dv_body = dv_corrected + 0.5 * cross3(theta_corrected, dv_corrected)
    return theta_corrected, dv_body


def _propagate_nominal(
    pos: np.ndarray,
    vel: np.ndarray,
    quat: np.ndarray,
    theta_corrected: np.ndarray,
    dv_body: np.ndarray,
    gravity_eci: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate the nominal (pos, vel, quaternion) one step. Pure (no state).

    Attitude integrates the (coning-corrected) body angular increment via the
    quaternion exponential map; velocity/position integrate the body specific
    force rotated into ECI by the *current* attitude estimate.
    """
    R_b2e = quat_to_dcm(quat).T
    dv_eci = R_b2e @ dv_body
    dv_total = dv_eci + gravity_eci * dt
    pos_n = pos + vel * dt + 0.5 * dv_total * dt
    vel_n = vel + dv_total
    # Quaternion update: body-frame increment -> right-multiply.
    angle = float(np.linalg.norm(theta_corrected))
    if angle > 1e-12:
        dq = quaternion_from_axis_angle(theta_corrected / angle, angle)
        quat_n = quaternion_multiply(quat, dq)
    else:
        quat_n = quat.copy()
    n = float(np.linalg.norm(quat_n))
    if n > 1e-12:
        quat_n = quat_n / n
    return pos_n, vel_n, quat_n


def _error_state_transition(R_b2e: np.ndarray, f_n: np.ndarray, dt: float) -> np.ndarray:
    """Discrete error-state transition F (15x15), F = I + A·dt to first order.

    Encodes the INS error dynamics (see module docstring): position integrates
    velocity error; velocity error is driven by the attitude error through the
    specific-force coupling ``-[f_n]_x`` and by accel-bias error through
    ``-R_b2e``; attitude error is driven by gyro-bias error through ``-R_b2e``.
    The second-order ``δp <- δθ, δb_a`` terms (½·(…)·dt²) are retained so a
    single 100 Hz step matches the finite-difference Jacobian of the nominal
    propagation closely (validated in ``tests/test_ekf.py``).
    """
    F = np.eye(N_ERR)
    dt_sq = dt * dt
    F[_P_IDX : _P_IDX + 3, _V_IDX : _V_IDX + 3] = np.eye(3) * dt  # δp <- δv
    F[_V_IDX : _V_IDX + 3, _BA_IDX : _BA_IDX + 3] = -R_b2e * dt  # δv <- δb_a
    F[_P_IDX : _P_IDX + 3, _BA_IDX : _BA_IDX + 3] = -0.5 * R_b2e * dt_sq
    skew_fn = _skew(f_n)
    F[_V_IDX : _V_IDX + 3, _TH_IDX : _TH_IDX + 3] = -skew_fn * dt  # δv <- δθ (attitude -> vel)
    F[_P_IDX : _P_IDX + 3, _TH_IDX : _TH_IDX + 3] = -0.5 * skew_fn * dt_sq
    F[_TH_IDX : _TH_IDX + 3, _BG_IDX : _BG_IDX + 3] = -R_b2e * dt  # δθ <- δb_g (gyro bias -> attitude)
    return F


def _process_noise(dt: float) -> np.ndarray:
    """Discrete process-noise covariance Q (15x15).

    Accelerometer white noise drives the position/velocity block (with the exact
    integrated cross term); gyro white noise drives the attitude block; the two
    bias-instability random walks drive their own blocks.
    """
    Q = np.zeros((N_ERR, N_ERR))
    dt_sq = dt * dt
    accel_var = config.IMU_ACCEL_NOISE_MPS2**2
    gyro_var = config.IMU_GYRO_NOISE_RADS**2
    q_pos = accel_var * dt_sq * dt_sq / 4.0
    q_vel = accel_var * dt_sq
    q_cross = accel_var * dt_sq * dt / 2.0
    q_bias_a = config.IMU_ACCEL_BIAS_MPS2**2 * dt
    q_bias_g = config.IMU_GYRO_BIAS_RADS**2 * dt
    q_att = gyro_var * dt_sq
    for i in range(3):
        Q[_P_IDX + i, _P_IDX + i] = q_pos
        Q[_V_IDX + i, _V_IDX + i] = q_vel
        Q[_BA_IDX + i, _BA_IDX + i] = q_bias_a
        Q[_BG_IDX + i, _BG_IDX + i] = q_bias_g
        Q[_TH_IDX + i, _TH_IDX + i] = q_att
        Q[_P_IDX + i, _V_IDX + i] = q_cross
        Q[_V_IDX + i, _P_IDX + i] = q_cross
    return Q


class NavigationEKF:
    """15-error-state multiplicative EKF (attitude + pos/vel + IMU biases).

    Predict at 100 Hz from the measured IMU (Savage coning/sculling); attitude
    propagates from the bias-corrected gyro. Update with GPS (pos+vel) and
    barometer; attitude is observed through the attitude-velocity coupling in the
    error dynamics. Corrections are applied multiplicatively to the quaternion
    with an error-state reset; covariance uses the Joseph form and a chi-square
    innovation gate (ADR 0013).

    Args:
        initial_state: Vehicle state used to seed the filter (its quaternion is
            the known initial attitude, e.g. the launch-pad orientation).
    """

    N_STATES: int = N_ERR

    def __init__(self, initial_state: VehicleState) -> None:
        self._x = np.zeros(12)  # nominal [pos, vel, ba, bg]
        self._x[0:3] = initial_state.position_eci.copy()
        self._x[3:6] = initial_state.velocity_eci.copy()
        self._quat = initial_state.quaternion.copy()  # nominal attitude

        # 15x15 error-state covariance. Attitude error seeded small but non-zero
        # (the launch attitude is known but not perfect).
        self._P = np.diag(
            np.array(
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
                    3e-4,
                    3e-4,
                    3e-4,  # attitude error (rad^2) ~ 1 deg 1-sigma
                ]
            )
        )

        self._angular_velocity_body = initial_state.angular_velocity_body.copy()
        self._mass_kg = initial_state.mass_kg
        self._time_s = initial_state.time_s

        self._prev_gyro = np.zeros(3)
        self._prev_accel = np.zeros(3)
        self._prev_imu_valid = False

        self._I15 = np.eye(N_ERR)

        # Diagnostics.
        self.measurement_rejections: int = 0
        self.last_nis: float = 0.0

    # -- public properties ---------------------------------------------------

    @property
    def state_vector(self) -> np.ndarray:
        """Current 12-element nominal state [pos, vel, ba, bg]."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current 15x15 error-state covariance."""
        return self._P.copy()

    @property
    def quaternion(self) -> np.ndarray:
        """Current attitude estimate (quaternion [x, y, z, w])."""
        return self._quat.copy()

    def position_uncertainty_m(self) -> float:
        """1-sigma position uncertainty: sqrt(trace(P_pos))."""
        P = self._P
        return math.sqrt(P[0, 0] + P[1, 1] + P[2, 2])

    def velocity_uncertainty_ms(self) -> float:
        """1-sigma velocity uncertainty: sqrt(trace(P_vel))."""
        P = self._P
        return math.sqrt(P[3, 3] + P[4, 4] + P[5, 5])

    def attitude_uncertainty_rad(self) -> float:
        """1-sigma attitude uncertainty: sqrt(trace(P_att))."""
        P = self._P
        return math.sqrt(P[_TH_IDX, _TH_IDX] + P[_TH_IDX + 1, _TH_IDX + 1] + P[_TH_IDX + 2, _TH_IDX + 2])

    # -- predict step --------------------------------------------------------

    def predict(self, imu: IMUMeasurement, gravity_eci_mps2: np.ndarray, dt: float) -> None:
        """Propagate nominal state and error covariance from an IMU sample."""
        self._time_s = imu.time_s
        ba = self._x[_BA_IDX : _BA_IDX + 3]
        bg = self._x[_BG_IDX : _BG_IDX + 3]

        accel_body = imu.accel_body_mps2 - ba
        gyro_body = imu.gyro_body_rads - bg

        theta_corrected, dv_body = _coning_sculling_dv(
            accel_body, gyro_body, self._prev_accel, self._prev_gyro, self._prev_imu_valid, dt
        )
        self._prev_gyro = gyro_body.copy()
        self._prev_accel = accel_body.copy()
        self._prev_imu_valid = True

        R_b2e = quat_to_dcm(self._quat).T
        f_n = R_b2e @ accel_body  # specific force in ECI

        pos_n, vel_n, quat_n = _propagate_nominal(
            self._x[_P_IDX : _P_IDX + 3],
            self._x[_V_IDX : _V_IDX + 3],
            self._quat,
            theta_corrected,
            dv_body,
            gravity_eci_mps2,
            dt,
        )
        self._x[_P_IDX : _P_IDX + 3] = pos_n
        self._x[_V_IDX : _V_IDX + 3] = vel_n
        self._quat = quat_n
        self._angular_velocity_body = gyro_body.copy()

        # --- Error-state transition F and process noise Q (15x15) ---
        F = _error_state_transition(R_b2e, f_n, dt)
        Q = _process_noise(dt)
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)

    # -- update steps --------------------------------------------------------

    def update_gps(self, gps: GPSMeasurement) -> None:
        """Fuse a GPS position+velocity measurement."""
        z = np.concatenate([gps.position_eci_m, gps.velocity_eci_ms])
        z_pred = self._x[0:6]
        y = z - z_pred
        H = np.zeros((6, N_ERR))
        H[0:6, 0:6] = np.eye(6)
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
        """Fuse a barometric-altitude measurement (altitude = |pos| - R_earth)."""
        pos = self._x[0:3]
        r = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        if r < 1.0:
            return
        y = np.array([baro.altitude_m - (r - config.EARTH_RADIUS_M)])
        H = np.zeros((1, N_ERR))
        H[0, 0:3] = pos / r
        R = np.array([[config.BARO_ALT_NOISE_M**2]])
        self._apply_update(y, H, R)

    def update_star_tracker(self, star: StarTrackerMeasurement) -> None:
        """Fuse a star-tracker inertial-attitude measurement (3-DOF δθ update).

        The star tracker observes the full attitude directly, so the innovation
        is the ECI-frame small-angle error between the measured and estimated
        quaternion — ``q_meas = exp([δθ]_x) ⊗ q_est`` ⇒ ``δθ ≈ 2·vec(q_meas ⊗
        q_est⁻¹)`` — and H is the identity on the attitude-error block. This
        directly observes all three axes (including roll about the thrust axis,
        which the GPS specific-force/velocity coupling cannot).
        """
        q_err = quaternion_multiply(star.quaternion, quaternion_conjugate(self._quat))
        if q_err[3] < 0.0:
            q_err = -q_err  # shortest rotation
        y = 2.0 * q_err[0:3]
        H = np.zeros((3, N_ERR))
        H[0:3, _TH_IDX : _TH_IDX + 3] = np.eye(3)
        R = np.eye(3) * config.STAR_TRACKER_NOISE_RAD**2
        self._apply_update(y, H, R)

    def update_ground_range(self, ground: GroundRangeMeasurement) -> None:
        """Fuse a slant-range measurement from one ground tracking station.

        Range is a nonlinear function of position, ``ρ = |r − r_station|``, so the
        measurement Jacobian is the line-of-sight unit vector on the position
        error block: ``∂ρ/∂δp = (r̂ − r_station)/|r̂ − r_station|``. A single range
        constrains only the line-of-sight component; the station network's
        differing look-angles (and the line-of-sight sweep as the vehicle moves)
        multilaterate the full position, bounding the covariance through the
        GPS-denied coast (ADR 0023).
        """
        r_est = self._x[_P_IDX : _P_IDX + 3]
        los = r_est - ground.station_position_eci
        rng = math.sqrt(los[0] ** 2 + los[1] ** 2 + los[2] ** 2)
        if rng < 1.0:
            return
        y = np.array([ground.range_m - rng])
        H = np.zeros((1, N_ERR))
        H[0, _P_IDX : _P_IDX + 3] = los / rng
        R = np.array([[config.GROUND_RANGE_NOISE_M**2]])
        self._apply_update(y, H, R)

    # -- private helpers -----------------------------------------------------

    def _apply_update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Kalman update with NIS gating and a multiplicative attitude reset."""
        S = H @ self._P @ H.T + R
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(S))):
            self.measurement_rejections += 1
            return
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        if nis > _nis_gate_threshold(len(y), config.EKF_INNOVATION_GATE_P):
            self.measurement_rejections += 1
            return

        K = np.linalg.solve(S.T, H @ self._P).T
        dx = K @ y  # 15-vector error correction

        # Apply additive corrections to the nominal pos/vel/biases ...
        self._x = self._x + dx[0:12]
        # ... and the attitude error multiplicatively to the quaternion (reset).
        dtheta = dx[_TH_IDX : _TH_IDX + 3]
        angle = float(np.linalg.norm(dtheta))
        if angle > 1e-12:
            dq = quaternion_from_axis_angle(dtheta / angle, angle)
            # ECI-frame error: left-multiply (R_new_b2e = exp([dθ]) R_b2e).
            self._quat = quaternion_multiply(dq, self._quat)
            n = float(np.linalg.norm(self._quat))
            if n > 1e-12:
                self._quat = self._quat / n

        I_KH = self._I15 - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    # -- estimated state output ----------------------------------------------

    def set_mass(self, mass_kg: float) -> None:
        """Update mass estimate (used only for state output)."""
        self._mass_kg = mass_kg

    def estimated_state(self) -> VehicleState:
        """Build a VehicleState from the current estimate (incl. estimated attitude)."""
        return VehicleState(
            position_eci=self._x[0:3].copy(),
            velocity_eci=self._x[3:6].copy(),
            quaternion=self._quat.copy(),
            angular_velocity_body=self._angular_velocity_body.copy(),
            mass_kg=self._mass_kg,
            time_s=self._time_s,
        )
