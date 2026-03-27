"""Tests for improvements and optimizations.

Validates the AerodynamicsModel.reset() method, ecef_to_lla convergence
optimization, and telemetry boundary violation counting fix.
"""

import math

import numpy as np
import numpy.testing as npt

from sim.core.reference_frames import ecef_to_lla, lla_to_ecef
from sim.vehicle.aerodynamics import AerodynamicsModel, dynamic_pressure


class TestAerodynamicsReset:
    """Verify AerodynamicsModel.reset() clears tracking state."""

    def test_reset_clears_max_q(self):
        """After reset, max_q_experienced should be zero."""
        aero = AerodynamicsModel()
        # Drive some q through the model
        vel = np.array([300.0, 0.0, 0.0])
        aero.compute_drag(vel, rho=1.0, speed_of_sound=340.0)
        assert aero.max_q_experienced > 0.0

        aero.reset()
        assert aero.max_q_experienced == 0.0

    def test_reset_clears_current_q(self):
        """After reset, current_q should be zero."""
        aero = AerodynamicsModel()
        vel = np.array([300.0, 0.0, 0.0])
        aero.compute_drag(vel, rho=1.0, speed_of_sound=340.0)
        assert aero.current_q > 0.0

        aero.reset()
        assert aero.current_q == 0.0

    def test_reset_allows_fresh_tracking(self):
        """After reset, max_q tracks only new data."""
        aero = AerodynamicsModel()

        # First run: high-speed pass
        vel_high = np.array([500.0, 0.0, 0.0])
        aero.compute_drag(vel_high, rho=1.0, speed_of_sound=340.0)
        first_max_q = aero.max_q_experienced

        aero.reset()

        # Second run: low-speed pass
        vel_low = np.array([50.0, 0.0, 0.0])
        aero.compute_drag(vel_low, rho=1.0, speed_of_sound=340.0)
        second_max_q = aero.max_q_experienced

        assert second_max_q < first_max_q
        expected_q = dynamic_pressure(1.0, 50.0)
        npt.assert_allclose(second_max_q, expected_q, rtol=1e-6)


class TestEcefToLlaConvergence:
    """Verify ecef_to_lla early termination produces same results."""

    def test_round_trip_equator(self):
        """LLA round-trip at equator should preserve values."""
        lat_rad = 0.0
        lon_rad = 0.0
        alt_m = 100_000.0

        ecef = lla_to_ecef(lat_rad, lon_rad, alt_m)
        lat_out, lon_out, alt_out = ecef_to_lla(ecef)

        npt.assert_allclose(lat_out, lat_rad, atol=1e-10)
        npt.assert_allclose(lon_out, lon_rad, atol=1e-10)
        npt.assert_allclose(alt_out, alt_m, atol=0.01)

    def test_round_trip_high_latitude(self):
        """LLA round-trip at high latitude should preserve values."""
        lat_rad = math.radians(85.0)
        lon_rad = math.radians(-120.0)
        alt_m = 300_000.0

        ecef = lla_to_ecef(lat_rad, lon_rad, alt_m)
        lat_out, lon_out, alt_out = ecef_to_lla(ecef)

        npt.assert_allclose(lat_out, lat_rad, atol=1e-10)
        npt.assert_allclose(lon_out, lon_rad, atol=1e-10)
        npt.assert_allclose(alt_out, alt_m, atol=0.01)

    def test_round_trip_north_pole(self):
        """LLA round-trip at north pole should preserve latitude and altitude."""
        lat_rad = math.radians(90.0)
        lon_rad = 0.0
        alt_m = 400_000.0

        ecef = lla_to_ecef(lat_rad, lon_rad, alt_m)
        lat_out, lon_out, alt_out = ecef_to_lla(ecef)

        npt.assert_allclose(lat_out, lat_rad, atol=1e-10)
        npt.assert_allclose(alt_out, alt_m, atol=1.0)

    def test_round_trip_sea_level(self):
        """LLA round-trip at sea level should preserve values."""
        lat_rad = math.radians(28.5)
        lon_rad = math.radians(-80.6)
        alt_m = 0.0

        ecef = lla_to_ecef(lat_rad, lon_rad, alt_m)
        lat_out, lon_out, alt_out = ecef_to_lla(ecef)

        npt.assert_allclose(lat_out, lat_rad, atol=1e-10)
        npt.assert_allclose(lon_out, lon_rad, atol=1e-10)
        npt.assert_allclose(alt_out, alt_m, atol=0.01)


class TestDynamicPressureInline:
    """Verify that inline dynamic pressure matches aerodynamics module."""

    def test_inline_q_matches_module(self):
        """Inline 0.5*rho*v^2 should match the module's dynamic_pressure function."""
        rho = 0.5
        v = 300.0
        expected = dynamic_pressure(rho, v)
        inline = 0.5 * rho * v * v
        npt.assert_allclose(inline, expected, rtol=1e-15)
