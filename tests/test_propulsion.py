"""Tests for propulsion functions in sim.vehicle.propulsion."""

import pytest
from sim.vehicle.propulsion import G0, P_SL, isp_at_pressure, mass_flow_rate, thrust_at_pressure
from sim.vehicle.vehicle import StageConfig


@pytest.fixture
def test_stage() -> StageConfig:
    return StageConfig(
        dry_mass=1000.0,
        propellant=5000.0,
        thrust_vac=100000.0,
        thrust_sl=80000.0,
        isp_vac=300.0,
        isp_sl=250.0,
        burn_time=100.0,
        throttle_min=0.2,
    )


def test_thrust_sea_level(test_stage: StageConfig) -> None:
    """At sea-level pressure, thrust should match thrust_sl."""
    thrust = thrust_at_pressure(test_stage, P_SL)
    assert thrust == pytest.approx(test_stage.thrust_sl)


def test_thrust_vacuum(test_stage: StageConfig) -> None:
    """At zero pressure, thrust should match thrust_vac."""
    thrust = thrust_at_pressure(test_stage, 0.0)
    assert thrust == pytest.approx(test_stage.thrust_vac)


def test_thrust_mid_altitude(test_stage: StageConfig) -> None:
    """At half sea-level pressure, thrust should be halfway between SL and vacuum."""
    thrust = thrust_at_pressure(test_stage, P_SL * 0.5)
    expected = (test_stage.thrust_vac + test_stage.thrust_sl) / 2.0
    assert thrust == pytest.approx(expected)


def test_thrust_clamped_high_pressure(test_stage: StageConfig) -> None:
    """Pressure above sea-level should be clamped, returning thrust_sl."""
    thrust = thrust_at_pressure(test_stage, P_SL * 1.5)
    assert thrust == pytest.approx(test_stage.thrust_sl)


def test_thrust_clamped_negative_pressure(test_stage: StageConfig) -> None:
    """Negative pressure should be clamped to zero, returning thrust_vac."""
    thrust = thrust_at_pressure(test_stage, -1000.0)
    assert thrust == pytest.approx(test_stage.thrust_vac)


def test_isp_sea_level(test_stage: StageConfig) -> None:
    """At sea-level pressure, Isp should match isp_sl."""
    isp = isp_at_pressure(test_stage, P_SL)
    assert isp == pytest.approx(test_stage.isp_sl)


def test_isp_vacuum(test_stage: StageConfig) -> None:
    """At zero pressure, Isp should match isp_vac."""
    isp = isp_at_pressure(test_stage, 0.0)
    assert isp == pytest.approx(test_stage.isp_vac)


def test_isp_mid_altitude(test_stage: StageConfig) -> None:
    """At half sea-level pressure, Isp should be halfway between SL and vacuum."""
    isp = isp_at_pressure(test_stage, P_SL * 0.5)
    expected = (test_stage.isp_vac + test_stage.isp_sl) / 2.0
    assert isp == pytest.approx(expected)


def test_isp_clamped_high_pressure(test_stage: StageConfig) -> None:
    """Pressure above sea-level should be clamped, returning isp_sl."""
    isp = isp_at_pressure(test_stage, P_SL * 1.5)
    assert isp == pytest.approx(test_stage.isp_sl)


def test_isp_clamped_negative_pressure(test_stage: StageConfig) -> None:
    """Negative pressure should be clamped to zero, returning isp_vac."""
    isp = isp_at_pressure(test_stage, -1000.0)
    assert isp == pytest.approx(test_stage.isp_vac)


def test_mass_flow_normal() -> None:
    """Test mass flow calculation under normal conditions."""
    thrust = 100000.0
    isp = 300.0
    expected = thrust / (isp * G0)
    assert mass_flow_rate(thrust, isp) == pytest.approx(expected)


def test_mass_flow_zero_isp() -> None:
    """Mass flow should be 0.0 if Isp is zero."""
    assert mass_flow_rate(100000.0, 0.0) == 0.0


def test_mass_flow_negative_isp() -> None:
    """Mass flow should be 0.0 if Isp is negative."""
    assert mass_flow_rate(100000.0, -10.0) == 0.0


def test_mass_flow_zero_thrust() -> None:
    """Mass flow should be 0.0 if thrust is zero."""
    assert mass_flow_rate(0.0, 300.0) == 0.0
