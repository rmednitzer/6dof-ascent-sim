"""Tests for the Navigation EKF (sim.gnc.navigation).

Validates initialization, predict step stability, GPS update uncertainty
reduction, and state estimation consistency.
"""

import numpy as np
import numpy.testing as npt

from sim import config
from sim.core.state import VehicleState
from sim.gnc.navigation import NavigationEKF
from sim.gnc.sensors import BaroMeasurement, GPSMeasurement, IMUMeasurement


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

    def test_covariance_shape(self):
        """Covariance matrix should be 12x12."""
        ekf = NavigationEKF(_make_initial_state())
        assert ekf.covariance.shape == (12, 12)

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

        _imu = IMUMeasurement(  # noqa: F841
            accel_body_mps2=np.array([0.0, 0.0, 9.81]),
            gyro_body_rads=np.zeros(3),
            time_s=0.01,
        )
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


class TestEKFSetters:
    """Verify set_attitude and set_mass."""

    def test_set_attitude(self):
        """set_attitude should update the quaternion in estimated_state."""
        ekf = NavigationEKF(_make_initial_state())

        new_q = np.array([0.1, 0.2, 0.3, 0.9])
        new_q /= np.linalg.norm(new_q)
        new_omega = np.array([0.01, 0.02, 0.03])

        ekf.set_attitude(new_q, new_omega)
        est = ekf.estimated_state()

        npt.assert_allclose(est.quaternion, new_q)
        npt.assert_allclose(est.angular_velocity_body, new_omega)

    def test_set_mass(self):
        """set_mass should update the mass in estimated_state."""
        ekf = NavigationEKF(_make_initial_state())
        ekf.set_mass(12345.0)
        assert ekf.estimated_state().mass_kg == 12345.0
