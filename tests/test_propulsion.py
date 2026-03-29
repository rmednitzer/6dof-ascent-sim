"""Tests for propulsion functions in sim.vehicle.propulsion."""

import unittest
from sim.vehicle.propulsion import G0, P_SL, isp_at_pressure, mass_flow_rate, thrust_at_pressure
from sim.vehicle.vehicle import StageConfig

class TestPropulsion(unittest.TestCase):
    def setUp(self):
        self.test_stage = StageConfig(
            dry_mass=1000.0,
            propellant=5000.0,
            thrust_vac=100000.0,
            thrust_sl=80000.0,
            isp_vac=300.0,
            isp_sl=250.0,
            burn_time=100.0,
            throttle_min=0.2,
        )

    def test_thrust_sea_level(self):
        """At sea-level pressure, thrust should match thrust_sl."""
        thrust = thrust_at_pressure(self.test_stage, P_SL)
        self.assertAlmostEqual(thrust, self.test_stage.thrust_sl)

    def test_thrust_vacuum(self):
        """At zero pressure, thrust should match thrust_vac."""
        thrust = thrust_at_pressure(self.test_stage, 0.0)
        self.assertAlmostEqual(thrust, self.test_stage.thrust_vac)

    def test_thrust_mid_altitude(self):
        """At half sea-level pressure, thrust should be halfway between SL and vacuum."""
        thrust = thrust_at_pressure(self.test_stage, P_SL * 0.5)
        expected = (self.test_stage.thrust_vac + self.test_stage.thrust_sl) / 2.0
        self.assertAlmostEqual(thrust, expected)

    def test_thrust_clamped_high_pressure(self):
        """Pressure above sea-level should be clamped, returning thrust_sl."""
        thrust = thrust_at_pressure(self.test_stage, P_SL * 1.5)
        self.assertAlmostEqual(thrust, self.test_stage.thrust_sl)

    def test_thrust_clamped_negative_pressure(self):
        """Negative pressure should be clamped to zero, returning thrust_vac."""
        thrust = thrust_at_pressure(self.test_stage, -1000.0)
        self.assertAlmostEqual(thrust, self.test_stage.thrust_vac)

    def test_isp_sea_level(self):
        """At sea-level pressure, Isp should match isp_sl."""
        isp = isp_at_pressure(self.test_stage, P_SL)
        self.assertAlmostEqual(isp, self.test_stage.isp_sl)

    def test_isp_vacuum(self):
        """At zero pressure, Isp should match isp_vac."""
        isp = isp_at_pressure(self.test_stage, 0.0)
        self.assertAlmostEqual(isp, self.test_stage.isp_vac)

    def test_isp_mid_altitude(self):
        """At half sea-level pressure, Isp should be halfway between SL and vacuum."""
        isp = isp_at_pressure(self.test_stage, P_SL * 0.5)
        expected = (self.test_stage.isp_vac + self.test_stage.isp_sl) / 2.0
        self.assertAlmostEqual(isp, expected)

    def test_mass_flow_normal(self):
        """Test mass flow calculation under normal conditions."""
        thrust = 100000.0
        isp = 300.0
        expected = thrust / (isp * G0)
        self.assertAlmostEqual(mass_flow_rate(thrust, isp), expected)

    def test_mass_flow_zero_isp(self):
        """Mass flow should be 0.0 if Isp is zero."""
        self.assertEqual(mass_flow_rate(100000.0, 0.0), 0.0)

    def test_mass_flow_negative_isp(self):
        """Mass flow should be 0.0 if Isp is negative."""
        self.assertEqual(mass_flow_rate(100000.0, -10.0), 0.0)

    def test_mass_flow_zero_thrust(self):
        """Mass flow should be 0.0 if thrust is zero."""
        self.assertEqual(mass_flow_rate(0.0, 300.0), 0.0)

if __name__ == "__main__":
    unittest.main()
