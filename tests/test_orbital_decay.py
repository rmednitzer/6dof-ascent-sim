"""Tests for orbital decay estimation (sim.orbital.decay)."""

import math
import sys
from unittest.mock import MagicMock

# Mock numpy before importing simulation modules to allow tests to run in restricted environments.
# This is necessary because the simulation modules depend on numpy which might not be installed.
if 'numpy' not in sys.modules:
    mock_np = MagicMock()
    mock_np.bool_ = bool
    mock_np.ndarray = MagicMock
    mock_np.array = lambda x, **kwargs: x
    sys.modules['numpy'] = mock_np
    sys.modules['numpy.testing'] = MagicMock()

import pytest  # noqa: E402

from sim.orbital.decay import (  # noqa: E402
    _atmosphere_density,
    _scale_height_at_altitude,
    ballistic_coefficient,
    estimate_lifetime,
)
from sim.orbital.propagator import OrbitalElements  # noqa: E402


class TestBallisticCoefficient:
    """Verify ballistic coefficient calculations."""

    def test_ballistic_coefficient_default(self):
        """Verify calculation with default parameters (Cd=2.2, Area=10.52)."""
        # Note: Default values Cd=2.2 and Area=10.52 are from sim.orbital.decay implementation
        mass = 5000.0
        expected = mass / (2.2 * 10.52)
        result = ballistic_coefficient(mass)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ballistic_coefficient_custom(self):
        """Verify calculation with custom provided parameters."""
        mass = 1000.0
        cd = 2.0
        area = 10.0
        expected = 1000.0 / (2.0 * 10.0)  # 50.0
        result = ballistic_coefficient(mass, cd=cd, area_m2=area)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ballistic_coefficient_zero_divisor(self):
        """Verify it returns math.inf when cd * area_m2 is zero."""
        assert ballistic_coefficient(1000.0, cd=0.0) == math.inf
        assert ballistic_coefficient(1000.0, area_m2=0.0) == math.inf
        assert ballistic_coefficient(1000.0, cd=0.0, area_m2=0.0) == math.inf

    def test_ballistic_coefficient_small_divisor(self):
        """Verify it returns math.inf when cd * area_m2 is below the 1e-12 threshold."""
        # Threshold is 1e-12 in sim.orbital.decay implementation
        assert ballistic_coefficient(1000.0, cd=1e-7, area_m2=1e-6) == math.inf

        # Just above threshold
        mass = 1000.0
        cd = 1.1e-6
        area = 1e-6
        # cd * area = 1.1e-12 > 1e-12
        expected = mass / (cd * area)
        result = ballistic_coefficient(mass, cd=cd, area_m2=area)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ballistic_coefficient_zero_mass(self):
        """Verify it returns 0.0 when mass is zero and divisor is valid."""
        assert ballistic_coefficient(0.0) == 0.0
        assert ballistic_coefficient(0.0, cd=2.0, area_m2=10.0) == 0.0


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
