"""Tests for the Navigation EKF (sim.gnc.navigation).

Covers the 15-error-state multiplicative EKF (ADR 0020): initialization, predict
stability, GPS/baro updates, the chi-square innovation gate (ADR 0013), and —
new for the attitude-estimating filter — three validations of correctness:

* **Error-state transition Jacobian** checked against a finite-difference
  Jacobian of the nominal one-step propagation (the defining linearization of an
  error-state EKF; Sola 2017 §5).
* **Attitude observability/convergence** — an injected attitude error is driven
  out by GPS through the specific-force/velocity coupling (in-motion alignment).
* **Filter consistency** — the GPS normalised innovation squared (NIS) sits near
  the measurement dimension over many updates (Bar-Shalom consistency test).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
from scipy.stats import chi2

from sim import config
from sim.core.reference_frames import (
    ecef_to_eci,
    lla_to_ecef,
    quat_to_dcm,
    quaternion_from_axis_angle,
    quaternion_multiply,
)
from sim.core.state import VehicleState
from sim.gnc.navigation import (
    N_ERR,
    NavigationEKF,
    _error_state_transition,
    _nis_gate_threshold,
    _propagate_nominal,
)
from sim.gnc.sensors import (
    BaroMeasurement,
    GPSMeasurement,
    GroundRangeMeasurement,
    GroundStation,
    IMUMeasurement,
    StarTracker,
    StarTrackerMeasurement,
)


def _make_initial_state() -> VehicleState:
    """Create a vehicle state at the Earth's surface for testing."""
    return VehicleState(
        position_eci=np.array([config.EARTH_RADIUS_M, 0.0, 0.0]),
        velocity_eci=np.array([0.0, 0.0, 0.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity_body=np.zeros(3),
        mass_kg=500_000.0,
        time_s=0.0,
    )


def _att_err_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic attitude error between two quaternions, in degrees."""
    d = abs(float(np.dot(q1, q2)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a scalar-last quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_boxminus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """ECI-frame small-angle error θ such that q1 = exp([θ]_x) ⊗ q2."""
    dq = quaternion_multiply(q1, _quat_conj(q2))
    if dq[3] < 0.0:
        dq = -dq
    return 2.0 * dq[0:3]


class TestEKFInitialization:
    """Verify EKF construction and initial state."""

    def test_state_vector_matches_initial(self):
        """State vector position/velocity should match the seed state."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)

        sv = ekf.state_vector
        npt.assert_allclose(sv[0:3], state.position_eci)
        npt.assert_allclose(sv[3:6], state.velocity_eci)
        # Biases initialised to zero
        npt.assert_allclose(sv[6:12], np.zeros(6))

    def test_quaternion_matches_initial(self):
        """The nominal attitude is seeded from the initial state."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        npt.assert_allclose(ekf.quaternion, state.quaternion)

    def test_covariance_shape(self):
        """Covariance matrix should be 15x15 (12 nominal + 3 attitude error)."""
        ekf = NavigationEKF(_make_initial_state())
        assert ekf.covariance.shape == (N_ERR, N_ERR) == (15, 15)

    def test_covariance_is_symmetric(self):
        """Initial covariance should be symmetric."""
        ekf = NavigationEKF(_make_initial_state())
        P = ekf.covariance
        npt.assert_allclose(P, P.T, atol=1e-15)

    def test_covariance_is_positive_definite(self):
        """Initial covariance should be positive definite."""
        ekf = NavigationEKF(_make_initial_state())
        eigenvalues = np.linalg.eigvalsh(ekf.covariance)
        assert np.all(eigenvalues > 0), "Covariance is not positive definite"

    def test_position_uncertainty_initial(self):
        """Initial position uncertainty should match the diagonal."""
        ekf = NavigationEKF(_make_initial_state())
        # sqrt(100 + 100 + 100) = sqrt(300) ~ 17.32
        expected = np.sqrt(300.0)
        npt.assert_allclose(ekf.position_uncertainty_m(), expected, rtol=1e-10)

    def test_attitude_uncertainty_initial(self):
        """Initial attitude uncertainty ~ 1 deg 1-sigma (seeded 3e-4 rad^2/axis)."""
        ekf = NavigationEKF(_make_initial_state())
        expected = math.sqrt(3.0 * 3e-4)
        npt.assert_allclose(ekf.attitude_uncertainty_rad(), expected, rtol=1e-10)

    def test_estimated_state_matches_initial(self):
        """estimated_state() should reproduce the initial state."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        est = ekf.estimated_state()

        npt.assert_allclose(est.position_eci, state.position_eci)
        npt.assert_allclose(est.velocity_eci, state.velocity_eci)
        npt.assert_allclose(est.quaternion, state.quaternion)
        assert est.mass_kg == state.mass_kg


class TestEKFPredict:
    """Verify predict step behaviour."""

    def test_predict_does_not_crash(self):
        """A basic predict step should complete without error."""
        ekf = NavigationEKF(_make_initial_state())

        imu = IMUMeasurement(
            accel_body_mps2=np.array([0.0, 0.0, 9.81]),
            gyro_body_rads=np.zeros(3),
            time_s=0.01,
        )
        gravity_eci = np.array([-9.81, 0.0, 0.0])
        ekf.predict(imu, gravity_eci, dt=0.01)

    def test_predict_grows_uncertainty(self):
        """Predict step (without updates) should increase position uncertainty."""
        ekf = NavigationEKF(_make_initial_state())
        unc_before = ekf.position_uncertainty_m()

        gravity_eci = np.array([-9.81, 0.0, 0.0])

        # Run several predict steps
        for i in range(100):
            imu_step = IMUMeasurement(
                accel_body_mps2=np.array([0.0, 0.0, 9.81]),
                gyro_body_rads=np.zeros(3),
                time_s=(i + 1) * 0.01,
            )
            ekf.predict(imu_step, gravity_eci, dt=0.01)

        unc_after = ekf.position_uncertainty_m()
        assert unc_after > unc_before, f"Uncertainty should grow: before={unc_before:.4f}, after={unc_after:.4f}"

    def test_predict_covariance_stays_symmetric(self):
        """Covariance should remain symmetric after predict steps."""
        ekf = NavigationEKF(_make_initial_state())

        imu = IMUMeasurement(
            accel_body_mps2=np.array([1.0, 0.5, 9.81]),
            gyro_body_rads=np.array([0.01, -0.005, 0.002]),
            time_s=0.01,
        )
        gravity_eci = np.array([-9.81, 0.0, 0.0])

        for i in range(50):
            imu_step = IMUMeasurement(
                accel_body_mps2=imu.accel_body_mps2,
                gyro_body_rads=imu.gyro_body_rads,
                time_s=(i + 1) * 0.01,
            )
            ekf.predict(imu_step, gravity_eci, dt=0.01)

        P = ekf.covariance
        npt.assert_allclose(P, P.T, atol=1e-10)

    def test_attitude_propagates_from_gyro(self):
        """A constant body-rate gyro rotates the nominal quaternion accordingly."""
        ekf = NavigationEKF(_make_initial_state())
        rate = 0.05  # rad/s about body z
        dt = 0.01
        steps = 200  # 2 s -> 0.1 rad
        for i in range(steps):
            imu = IMUMeasurement(
                accel_body_mps2=np.zeros(3),
                gyro_body_rads=np.array([0.0, 0.0, rate]),
                time_s=(i + 1) * dt,
            )
            ekf.predict(imu, np.zeros(3), dt)
        expected_angle = rate * steps * dt
        got = _att_err_deg(ekf.quaternion, np.array([0.0, 0.0, 0.0, 1.0]))
        npt.assert_allclose(math.radians(got), expected_angle, rtol=1e-3)


class TestErrorStateTransition:
    """Validate the analytic transition F against a finite-difference Jacobian.

    For an error-state EKF the transition matrix is, by definition, the Jacobian
    of the nominal one-step propagation with respect to the (boxplus) error
    state. A central-difference Jacobian of ``_propagate_nominal`` must therefore
    reproduce ``_error_state_transition`` — this is the strongest single check
    that the attitude/velocity/bias couplings are correct (signs included).
    """

    @staticmethod
    def _nominal_step(pos, vel, quat, ba, bg, imu_accel, imu_gyro, gravity, dt):
        # One nominal step WITHOUT coning/sculling (prev_valid=False), matching
        # the first-order derivation of F. Biases are constant across the step.
        accel_body = imu_accel - ba
        gyro_body = imu_gyro - bg
        theta = gyro_body * dt
        dv = accel_body * dt
        return _propagate_nominal(pos, vel, quat, theta, dv, gravity, dt)

    @staticmethod
    def _perturb(pos, vel, ba, bg, quat, delta):
        p = pos + delta[0:3]
        v = vel + delta[3:6]
        a = ba + delta[6:9]
        g = bg + delta[9:12]
        dtheta = delta[12:15]
        ang = float(np.linalg.norm(dtheta))
        if ang > 0.0:
            dq = quaternion_from_axis_angle(dtheta / ang, ang)
            q = quaternion_multiply(dq, quat)  # ECI-frame error: left-multiply
            q = q / np.linalg.norm(q)
        else:
            q = quat.copy()
        return p, v, a, g, q

    def test_F_matches_finite_difference(self):
        rng = np.random.default_rng(7)
        dt = 0.01
        # Small position magnitude avoids catastrophic cancellation in the FD
        # (F is independent of position anyway).
        pos = rng.normal(0.0, 100.0, 3)
        vel = rng.normal(0.0, 50.0, 3)
        axis = rng.normal(0.0, 1.0, 3)
        axis /= np.linalg.norm(axis)
        quat = quaternion_from_axis_angle(axis, 0.4)
        ba = rng.normal(0.0, 0.01, 3)
        bg = rng.normal(0.0, 0.001, 3)
        imu_accel = np.array([30.0, 0.0, 0.0]) + rng.normal(0.0, 1.0, 3)
        # Small angular increment: F[θ,b_g] = -R·dt is the *first-order* coupling
        # (standard error-state form; PX4 ECL / Sola 2017), which omits the O(θ)
        # SO(3) right-Jacobian term. Keeping θ = gyro·dt tiny isolates the
        # first-order Jacobian that F represents, so the FD check stays exact.
        imu_gyro = rng.normal(0.0, 0.005, 3)
        gravity = np.array([-9.8, 0.1, -0.2])

        R_b2e = quat_to_dcm(quat).T
        f_n = R_b2e @ (imu_accel - ba)
        F = _error_state_transition(R_b2e, f_n, dt)

        eps = 1e-6
        F_num = np.zeros((N_ERR, N_ERR))
        for i in range(N_ERR):
            dp = np.zeros(N_ERR)
            dp[i] = eps
            pp, vp, ap, gp, qp = self._perturb(pos, vel, ba, bg, quat, dp)
            pm, vm, am, gm, qm = self._perturb(pos, vel, ba, bg, quat, -dp)
            p_pn, v_pn, q_pn = self._nominal_step(pp, vp, qp, ap, gp, imu_accel, imu_gyro, gravity, dt)
            p_mn, v_mn, q_mn = self._nominal_step(pm, vm, qm, am, gm, imu_accel, imu_gyro, gravity, dt)
            col = np.concatenate(
                [
                    p_pn - p_mn,
                    v_pn - v_mn,
                    ap - am,  # biases are constant -> identity block
                    gp - gm,
                    _quat_boxminus(q_pn, q_mn),
                ]
            ) / (2.0 * eps)
            F_num[:, i] = col

        npt.assert_allclose(F, F_num, atol=1e-6, rtol=1e-4)

    def test_process_noise_psd_and_diagonal_blocks(self):
        """Q is symmetric PSD and scales the attitude block by gyro noise·dt²."""
        from sim.gnc.navigation import _TH_IDX, _process_noise

        dt = 0.01
        Q = _process_noise(dt)
        assert Q.shape == (N_ERR, N_ERR)
        npt.assert_allclose(Q, Q.T, atol=0.0)
        assert np.all(np.linalg.eigvalsh(Q) >= -1e-18)
        expected_att = config.IMU_GYRO_NOISE_RADS**2 * dt * dt
        npt.assert_allclose(Q[_TH_IDX, _TH_IDX], expected_att, rtol=1e-12)


class TestEKFGPSUpdate:
    """Verify GPS measurement update works."""

    def test_gps_update_does_not_crash(self):
        """A GPS update should complete without error."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)

        gps = GPSMeasurement(
            position_eci_m=state.position_eci.copy(),
            velocity_eci_ms=state.velocity_eci.copy(),
            time_s=0.0,
        )
        ekf.update_gps(gps)
        # Should not raise

    def test_gps_update_at_correct_position_maintains_low_uncertainty(self):
        """GPS at estimated position should keep uncertainty low."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)

        unc_initial = ekf.position_uncertainty_m()

        # GPS matches EKF position exactly
        gps = GPSMeasurement(
            position_eci_m=ekf.state_vector[0:3].copy(),
            velocity_eci_ms=ekf.state_vector[3:6].copy(),
            time_s=0.0,
        )
        ekf.update_gps(gps)

        unc_after = ekf.position_uncertainty_m()
        # Uncertainty should stay the same or decrease
        assert unc_after <= unc_initial + 1e-6


class TestEKFBaroUpdate:
    """Verify barometric altitude update."""

    def test_baro_update_does_not_crash(self):
        """A baro update should complete without error."""
        ekf = NavigationEKF(_make_initial_state())

        baro = BaroMeasurement(altitude_m=100.0, time_s=1.0)
        ekf.update_baro(baro)


class TestAttitudeEstimation:
    """Attitude is observable in motion and the filter drives out an error."""

    def test_injected_attitude_error_converges_with_gps(self):
        """An injected ~10° attitude error is driven out by GPS aiding.

        With sustained specific force along body-x, a yaw (about z) attitude
        error rotates the sensed force into a y-velocity error that GPS observes;
        the V–θ cross-covariance built up in predict then corrects attitude
        (classic in-motion / GPS-aided alignment).
        """
        rng = np.random.default_rng(42)
        dt = 0.01
        f_body = np.array([20.0, 0.0, 0.0])  # sustained specific force
        q_true = np.array([0.0, 0.0, 0.0, 1.0])
        R_true = quat_to_dcm(q_true).T
        gravity = np.zeros(3)

        pos = np.zeros(3)
        vel = np.zeros(3)
        seed = VehicleState(
            position_eci=pos.copy(),
            velocity_eci=vel.copy(),
            quaternion=q_true.copy(),
            angular_velocity_body=np.zeros(3),
            mass_kg=1000.0,
            time_s=0.0,
        )
        ekf = NavigationEKF(seed)
        # Inject a 10° yaw error and tell the filter it is uncertain about it.
        ang0 = math.radians(10.0)
        dq0 = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), ang0)
        ekf._quat = quaternion_multiply(dq0, ekf._quat)
        ekf._P[12:15, 12:15] = np.eye(3) * ang0**2

        err0 = _att_err_deg(ekf.quaternion, q_true)
        att_unc0 = ekf.attitude_uncertainty_rad()

        t = 0.0
        for k in range(6000):  # 60 s
            t += dt
            f_n = R_true @ f_body
            vel = vel + f_n * dt
            pos = pos + vel * dt
            imu = IMUMeasurement(
                accel_body_mps2=f_body + rng.normal(0.0, config.IMU_ACCEL_NOISE_MPS2, 3),
                gyro_body_rads=rng.normal(0.0, config.IMU_GYRO_NOISE_RADS, 3),
                time_s=t,
            )
            ekf.predict(imu, gravity, dt)
            if k % 100 == 99:  # 1 Hz GPS
                gps = GPSMeasurement(
                    position_eci_m=pos + rng.normal(0.0, config.GPS_POS_NOISE_M, 3),
                    velocity_eci_ms=vel + rng.normal(0.0, config.GPS_VEL_NOISE_MS, 3),
                    time_s=t,
                )
                ekf.update_gps(gps)

        err1 = _att_err_deg(ekf.quaternion, q_true)
        att_unc1 = ekf.attitude_uncertainty_rad()

        assert err1 < 0.3 * err0, f"attitude did not converge: {err0:.2f}° -> {err1:.2f}°"
        assert err1 < 2.0, f"residual attitude error too large: {err1:.2f}°"
        # The covariance also collapses (observability), and the error is broadly
        # consistent with it (not wildly overconfident).
        assert att_unc1 < att_unc0
        assert math.radians(err1) < 5.0 * att_unc1 + math.radians(0.5)


class TestFilterConsistency:
    """GPS innovation consistency (Bar-Shalom NIS test)."""

    def test_gps_nis_near_measurement_dimension(self):
        """Mean NIS over many GPS updates sits near the 6-DOF measurement size.

        Truth and filter share the configured noise statistics and the filter is
        initialised at the true attitude, so the normalised innovation squared
        ``yᵀ S⁻¹ y`` should average near its expectation (the 6 GPS components).
        A filter that is ~2× over- or under-confident fails this.
        """
        rng = np.random.default_rng(2024)
        dt = 0.01
        f_body = np.array([20.0, 0.0, 0.0])
        q_true = np.array([0.0, 0.0, 0.0, 1.0])
        R_true = quat_to_dcm(q_true).T
        gravity = np.zeros(3)

        pos = np.zeros(3)
        vel = np.zeros(3)
        seed = VehicleState(
            position_eci=pos.copy(),
            velocity_eci=vel.copy(),
            quaternion=q_true.copy(),
            angular_velocity_body=np.zeros(3),
            mass_kg=1000.0,
            time_s=0.0,
        )
        ekf = NavigationEKF(seed)

        nis_samples: list[float] = []
        burn_in = 20
        n_updates = 0
        rejos = 0
        t = 0.0
        k = 0
        while len(nis_samples) < 220:
            t += dt
            f_n = R_true @ f_body
            vel = vel + f_n * dt
            pos = pos + vel * dt
            imu = IMUMeasurement(
                accel_body_mps2=f_body + rng.normal(0.0, config.IMU_ACCEL_NOISE_MPS2, 3),
                gyro_body_rads=rng.normal(0.0, config.IMU_GYRO_NOISE_RADS, 3),
                time_s=t,
            )
            ekf.predict(imu, gravity, dt)
            if k % 100 == 99:
                before = ekf.measurement_rejections
                gps = GPSMeasurement(
                    position_eci_m=pos + rng.normal(0.0, config.GPS_POS_NOISE_M, 3),
                    velocity_eci_ms=vel + rng.normal(0.0, config.GPS_VEL_NOISE_MS, 3),
                    time_s=t,
                )
                ekf.update_gps(gps)
                n_updates += 1
                if ekf.measurement_rejections > before:
                    rejos += 1
                elif n_updates > burn_in:
                    nis_samples.append(ekf.last_nis)
            k += 1

        samples = np.array(nis_samples)
        mean_nis = float(samples.mean())
        dim = 6
        # Two-sided 99.9% bound on the average of K chi-square(dim) variables.
        k_n = len(samples)
        lo = chi2.ppf(0.0005, dim * k_n) / k_n
        hi = chi2.ppf(0.9995, dim * k_n) / k_n
        assert lo < mean_nis < hi, f"mean NIS {mean_nis:.2f} outside [{lo:.2f}, {hi:.2f}]"
        # Consistent measurements are almost never gated out.
        assert rejos / n_updates < 0.05


class TestStarTrackerUpdate:
    """The star-tracker attitude update corrects attitude on all three axes."""

    @staticmethod
    def _ekf_with_attitude_error(axis, deg):
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        ang = math.radians(deg)
        ax = np.array(axis, dtype=float)
        ax /= np.linalg.norm(ax)
        dq = quaternion_from_axis_angle(ax, ang)
        ekf._quat = quaternion_multiply(dq, ekf._quat)  # ECI-frame error
        ekf._P[12:15, 12:15] = np.eye(3) * ang**2  # filter is uncertain about it
        return ekf, state.quaternion.copy()

    def test_update_reduces_attitude_uncertainty(self):
        ekf, q_true = self._ekf_with_attitude_error([0.0, 0.0, 1.0], 5.0)
        unc0 = ekf.attitude_uncertainty_rad()
        ekf.update_star_tracker(StarTrackerMeasurement(quaternion=q_true.copy(), time_s=0.0))
        assert ekf.attitude_uncertainty_rad() < unc0

    def test_update_corrects_yaw_error(self):
        ekf, q_true = self._ekf_with_attitude_error([0.0, 0.0, 1.0], 5.0)
        err0 = _att_err_deg(ekf.quaternion, q_true)
        for _ in range(5):
            ekf.update_star_tracker(StarTrackerMeasurement(quaternion=q_true.copy(), time_s=0.0))
        assert _att_err_deg(ekf.quaternion, q_true) < 0.1 * err0

    def test_observes_roll_about_thrust_axis(self):
        """Roll about body-x is unobservable to the GPS specific-force coupling
        but is directly observed by the star tracker."""
        ekf, q_true = self._ekf_with_attitude_error([1.0, 0.0, 0.0], 5.0)
        err0 = _att_err_deg(ekf.quaternion, q_true)
        for _ in range(5):
            ekf.update_star_tracker(StarTrackerMeasurement(quaternion=q_true.copy(), time_s=0.0))
        assert _att_err_deg(ekf.quaternion, q_true) < 0.1 * err0


class TestStarTrackerSensor:
    """Star-tracker availability and noise model (ADR 0020)."""

    @staticmethod
    def _state(alt_m, rate=0.0, t=0.0):
        return VehicleState(
            position_eci=np.array([config.EARTH_RADIUS_M + alt_m, 0.0, 0.0]),
            velocity_eci=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.array([rate, 0.0, 0.0]),
            mass_kg=1000.0,
            time_s=t,
        )

    def test_unavailable_in_atmosphere(self):
        st = StarTracker(rng=np.random.default_rng(0))
        assert st.measure(self._state(50_000.0), 0.01) is None

    def test_unavailable_at_high_slew_rate(self):
        st = StarTracker(rng=np.random.default_rng(0))
        s = self._state(config.STAR_TRACKER_MIN_ALT_M + 10_000.0, rate=2.0 * config.STAR_TRACKER_MAX_RATE_RADS)
        assert st.measure(s, 0.01) is None

    def test_available_above_atmosphere_at_low_rate(self):
        st = StarTracker(rng=np.random.default_rng(0))
        s = self._state(config.STAR_TRACKER_MIN_ALT_M + 10_000.0, rate=0.0)
        m = st.measure(s, 0.01)
        assert m is not None
        assert _att_err_deg(m.quaternion, s.quaternion) < 0.05  # arcsec-class

    def test_noise_magnitude_matches_config(self):
        st = StarTracker(rng=np.random.default_rng(1))
        period = 1.0 / config.STAR_TRACKER_UPDATE_HZ
        errs = []
        for k in range(2000):
            s = self._state(config.STAR_TRACKER_MIN_ALT_M + 10_000.0, rate=0.0, t=k * period)
            m = st.measure(s, 0.01)
            if m is not None:
                errs.append(math.radians(_att_err_deg(m.quaternion, s.quaternion)))
        rms = math.sqrt(float(np.mean(np.square(errs))))
        # A 3-axis small rotation with per-axis sigma has geodesic RMS sqrt(3)*sigma.
        expected = math.sqrt(3.0) * config.STAR_TRACKER_NOISE_RAD
        npt.assert_allclose(rms, expected, rtol=0.15)


class TestGroundRangeUpdate:
    """The ground-station slant-range update bounds position (ADR 0023)."""

    @staticmethod
    def _ekf_with_position_uncertainty(var_m2=1.0e6):
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        ekf._P[0:3, 0:3] = np.eye(3) * var_m2
        return ekf, state.position_eci.copy()

    def test_update_reduces_position_covariance(self):
        ekf, r = self._ekf_with_position_uncertainty()
        station = r - np.array([1.0e6, 0.0, 0.0])  # 1000 km along -x
        true_range = float(np.linalg.norm(r - station))
        tr0 = float(np.trace(ekf.covariance[0:3, 0:3]))
        ekf.update_ground_range(GroundRangeMeasurement(station, true_range, "T", 0.0))
        assert float(np.trace(ekf.covariance[0:3, 0:3])) < tr0

    def test_update_drives_range_to_measurement(self):
        ekf, r = self._ekf_with_position_uncertainty()
        station = r - np.array([1.0e6, 0.0, 0.0])
        true_range = float(np.linalg.norm(r - station))
        ekf._x[0] += 500.0  # perturb the estimate along the line of sight
        err0 = abs(float(np.linalg.norm(ekf._x[0:3] - station)) - true_range)
        for _ in range(10):
            ekf.update_ground_range(GroundRangeMeasurement(station, true_range, "T", 0.0))
        err1 = abs(float(np.linalg.norm(ekf._x[0:3] - station)) - true_range)
        assert err1 < 0.1 * err0


class TestGroundStationSensor:
    """Ground-station visibility and ranging (ADR 0023)."""

    @staticmethod
    def _state_above_station(station_lla, alt_m, *, antipode=False):
        s_ecef = lla_to_ecef(math.radians(station_lla[0]), math.radians(station_lla[1]), station_lla[2])
        radial = s_ecef / np.linalg.norm(s_ecef)
        veh_ecef = (-1.0 if antipode else 1.0) * (s_ecef + radial * alt_m)
        return VehicleState(
            position_eci=ecef_to_eci(veh_ecef, 0.0),
            velocity_eci=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=1000.0,
            time_s=0.0,
        )

    def test_tracks_overhead_vehicle(self):
        st = GroundStation("T", 0.0, 0.0, 0.0, rng=np.random.default_rng(0))
        m = st.measure(self._state_above_station((0.0, 0.0, 0.0), 300_000.0), 0.01)
        assert m is not None
        assert abs(m.range_m - 300_000.0) < 250.0  # overhead slant ≈ altitude, within noise

    def test_no_track_below_horizon(self):
        st = GroundStation("T", 0.0, 0.0, 0.0, rng=np.random.default_rng(0))
        # Vehicle above the antipode — well below the station's local horizon.
        assert st.measure(self._state_above_station((0.0, 0.0, 0.0), 300_000.0, antipode=True), 0.01) is None


class TestEKFSetters:
    """Verify set_mass (set_attitude was removed with the 12-state filter)."""

    def test_set_mass(self):
        """set_mass should update the mass in estimated_state."""
        ekf = NavigationEKF(_make_initial_state())
        ekf.set_mass(12345.0)
        assert ekf.estimated_state().mass_kg == 12345.0

    def test_no_set_attitude_method(self):
        """The error-state filter estimates attitude; set_attitude is gone."""
        ekf = NavigationEKF(_make_initial_state())
        assert not hasattr(ekf, "set_attitude")


class TestEKFInnovationGate:
    """Verify the chi-square (NIS) innovation-consistency gate (ADR 0013)."""

    def test_zero_innovation_accepted_nis_zero(self):
        """A measurement at the estimate has NIS 0 and is accepted."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        gps = GPSMeasurement(
            position_eci_m=ekf.state_vector[0:3].copy(),
            velocity_eci_ms=ekf.state_vector[3:6].copy(),
            time_s=0.0,
        )
        ekf.update_gps(gps)
        assert ekf.last_nis == 0.0
        assert ekf.measurement_rejections == 0

    def test_in_family_measurement_accepted_and_moves_state(self):
        """A small, consistent offset passes the gate and nudges the state."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        before = ekf.state_vector.copy()
        gps = GPSMeasurement(
            position_eci_m=state.position_eci + np.array([1.0, 0.0, 0.0]),
            velocity_eci_ms=state.velocity_eci.copy(),
            time_s=0.0,
        )
        ekf.update_gps(gps)
        assert ekf.measurement_rejections == 0
        # State pulled toward the measurement along +x.
        assert ekf.state_vector[0] > before[0]

    def test_gross_outlier_rejected_and_counted(self):
        """A 10 km position outlier is rejected; the state is untouched."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        before = ekf.state_vector.copy()
        q_before = ekf.quaternion.copy()
        gps = GPSMeasurement(
            position_eci_m=state.position_eci + np.array([1.0e4, 1.0e4, 1.0e4]),
            velocity_eci_ms=state.velocity_eci.copy(),
            time_s=0.0,
        )
        ekf.update_gps(gps)
        assert ekf.measurement_rejections == 1
        npt.assert_array_equal(ekf.state_vector, before)
        npt.assert_array_equal(ekf.quaternion, q_before)

    def test_non_finite_measurement_rejected(self):
        """A NaN measurement is rejected as a counted fault, not ingested."""
        state = _make_initial_state()
        ekf = NavigationEKF(state)
        before = ekf.state_vector.copy()
        gps = GPSMeasurement(
            position_eci_m=state.position_eci + np.array([np.nan, 0.0, 0.0]),
            velocity_eci_ms=state.velocity_eci.copy(),
            time_s=0.0,
        )
        ekf.update_gps(gps)
        assert ekf.measurement_rejections == 1
        npt.assert_array_equal(ekf.state_vector, before)
        assert np.all(np.isfinite(ekf.covariance))

    def test_gate_uses_full_covariance_threshold(self):
        """The 1-DOF gate threshold reproduces the old ~3-sigma intent."""
        # chi2.ppf(0.9973, 1) == 3.0**2 (the previous per-component 3-sigma gate).
        npt.assert_allclose(_nis_gate_threshold(1, config.EKF_INNOVATION_GATE_P), 3.0**2, atol=0.02)
        # Higher dimension -> larger joint threshold.
        p = config.EKF_INNOVATION_GATE_P
        assert _nis_gate_threshold(6, p) > _nis_gate_threshold(1, p)
