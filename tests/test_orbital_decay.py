"""Tests for orbital decay estimation (sim.orbital.decay)."""

import math
import sys
from unittest.mock import MagicMock

# Mock numpy before it's imported by any simulation modules
mock_np = MagicMock()
mock_np.bool_ = bool
mock_np.ndarray = MagicMock
mock_np.array = lambda x, **kwargs: x
sys.modules["numpy"] = mock_np

import pytest  # noqa: E402

from sim.orbital.decay import (  # noqa: E402
    _atmosphere_density,
    _scale_height_at_altitude,
    ballistic_coefficient,
    estimate_lifetime,
)
from sim.orbital.propagator import OrbitalElements  # noqa: E402


def test_ballistic_coefficient():
    """Verify ballistic coefficient calculation BC = m / (Cd * A)."""
    # Default: cd=2.2, area=10.52
    mass = 1000.0
    expected_bc = mass / (2.2 * 10.52)
    bc = ballistic_coefficient(mass)
    assert bc == pytest.approx(expected_bc)

    # Custom values
    bc2 = ballistic_coefficient(5000.0, cd=2.0, area_m2=10.0)
    assert bc2 == 250.0

    # Near zero Cd * Area
    bc_inf = ballistic_coefficient(1000.0, cd=0.0)
    assert bc_inf == math.inf


def test_atmosphere_density():
    """Verify piecewise exponential atmosphere model density."""
    # Sea level: (0.0, 1.225, 7.249)
    rho_0 = _atmosphere_density(0.0)
    assert rho_0 == pytest.approx(1.225)

    # Below sea level should return sea level density
    rho_neg = _atmosphere_density(-1000.0)
    assert rho_neg == 1.225

    # 100 km (base of a band): (100.0, 5.297e-7, 5.877)
    rho_100 = _atmosphere_density(100_000.0)
    assert rho_100 == pytest.approx(5.297e-7)

    # Middle of a band: 50km (uses 0km base)
    # rho = rho_base * exp(-(alt_km - base_alt_km) / scale_height_km)
    expected_50 = 1.225 * math.exp(-(50.0 - 0.0) / 7.249)
    assert _atmosphere_density(50_000.0) == pytest.approx(expected_50)


def test_scale_height_at_altitude():
    """Verify scale height selection from bands."""
    # Sea level band
    assert _scale_height_at_altitude(0.0) == 7.249 * 1000.0

    # 100 km band
    assert _scale_height_at_altitude(100_000.0) == 5.877 * 1000.0

    # 1000 km band
    assert _scale_height_at_altitude(1000_000.0) == 268.0 * 1000.0


def test_estimate_lifetime_output_format():
    """Verify the structure of the lifetime estimate dictionary."""
    elements = OrbitalElements(
        semi_major_axis_m=6578137.0,  # 200km alt
        eccentricity=0.001,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_periapsis_deg=0.0,
        true_anomaly_deg=0.0,
        period_s=5300.0,
        apoapsis_alt_km=206.5,
        periapsis_alt_km=193.5,
    )

    result = estimate_lifetime(elements, dry_mass_kg=5000.0)

    assert isinstance(result, dict)
    expected_keys = {
        "days_to_deorbit",
        "initial_periapsis_alt_km",
        "initial_apoapsis_alt_km",
        "final_periapsis_alt_km",
        "ballistic_coefficient_kg_m2",
        "periapsis_density_kg_m3",
        "revolutions",
    }
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], (float, int))


def test_estimate_lifetime_low_circular():
    """Verify that a low orbit decays within a reasonable timeframe."""
    # 200 km circular orbit
    elements = OrbitalElements(
        semi_major_axis_m=6378137.0 + 200_000.0,
        eccentricity=0.0,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_periapsis_deg=0.0,
        true_anomaly_deg=0.0,
        period_s=5300.0,
        apoapsis_alt_km=200.0,
        periapsis_alt_km=200.0,
    )

    # Use a small satellite
    result = estimate_lifetime(elements, dry_mass_kg=500.0, area_m2=1.0)

    # 200km orbit should decay in days or weeks, not years or seconds
    assert 0.1 < result["days_to_deorbit"] < 100.0
    assert result["revolutions"] > 10


def test_estimate_lifetime_high_orbit():
    """Verify that a high orbit has infinite lifetime."""
    # 5000 km orbit — at this altitude density < 1e-20 kg/m^3
    elements = OrbitalElements(
        semi_major_axis_m=6378137.0 + 5000_000.0,
        eccentricity=0.0,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_periapsis_deg=0.0,
        true_anomaly_deg=0.0,
        period_s=12000.0,
        apoapsis_alt_km=5000.0,
        periapsis_alt_km=5000.0,
    )

    result = estimate_lifetime(elements, dry_mass_kg=5000.0)
    assert result["days_to_deorbit"] == math.inf


def test_estimate_lifetime_elliptical():
    """Verify that an elliptical orbit's periapsis eventually drops."""
    # 200 x 600 km orbit
    # a = Re + (200+600)/2 = Re + 400
    # e = (600-200) / (2*Re + 800)
    Re = 6378137.0
    a = Re + 400_000.0
    e = 200_000.0 / a

    elements = OrbitalElements(
        semi_major_axis_m=a,
        eccentricity=e,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_periapsis_deg=0.0,
        true_anomaly_deg=0.0,
        period_s=5500.0,
        apoapsis_alt_km=600.0,
        periapsis_alt_km=200.0,
    )

    result = estimate_lifetime(elements, dry_mass_kg=1000.0)

    assert result["days_to_deorbit"] > 0
    assert result["days_to_deorbit"] != math.inf
    # It should finish at the deorbit threshold (80km) or slightly below
    assert result["final_periapsis_alt_km"] <= 80.1
