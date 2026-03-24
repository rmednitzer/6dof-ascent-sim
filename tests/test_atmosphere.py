"""Tests for the US Standard Atmosphere 1976 model (sim.environment.atmosphere).

Validates sea-level reference values, temperature profile across layers,
altitude dependence, and upper-atmosphere / vacuum behaviour.
"""

import numpy.testing as npt

from sim.environment.atmosphere import AtmosphereResult, atmosphere


class TestSeaLevel:
    """Verify US Standard Atmosphere sea-level reference values."""

    def test_temperature_sea_level(self):
        """Sea-level temperature should be 288.15 K."""
        result = atmosphere(0.0)
        npt.assert_allclose(result.temperature_k, 288.15, atol=0.01)

    def test_pressure_sea_level(self):
        """Sea-level pressure should be 101325 Pa."""
        result = atmosphere(0.0)
        npt.assert_allclose(result.pressure_pa, 101325.0, atol=1.0)

    def test_density_sea_level(self):
        """Sea-level density should be 1.225 kg/m^3."""
        result = atmosphere(0.0)
        npt.assert_allclose(result.density_kg_m3, 1.225, atol=0.001)

    def test_speed_of_sound_sea_level(self):
        """Sea-level speed of sound should be ~340.3 m/s."""
        result = atmosphere(0.0)
        # sqrt(1.4 * 287.058 * 288.15) ~ 340.3
        npt.assert_allclose(result.speed_of_sound_ms, 340.3, atol=0.5)

    def test_negative_altitude_returns_sea_level(self):
        """Negative altitudes should be clamped to sea-level values."""
        result = atmosphere(-1000.0)
        npt.assert_allclose(result.temperature_k, 288.15, atol=0.01)
        npt.assert_allclose(result.pressure_pa, 101325.0, atol=1.0)


class TestAltitudeDependence:
    """Verify expected trends with increasing altitude."""

    def test_temperature_decreases_in_troposphere(self):
        """Temperature should decrease from sea level to 11 km (troposphere)."""
        t_sl = atmosphere(0.0).temperature_k
        t_5km = atmosphere(5000.0).temperature_k
        t_10km = atmosphere(10000.0).temperature_k

        assert t_5km < t_sl
        assert t_10km < t_5km

    def test_temperature_at_tropopause(self):
        """At 11 km (tropopause), temperature ~ 216.65 K."""
        result = atmosphere(11000.0)
        npt.assert_allclose(result.temperature_k, 216.65, atol=0.5)

    def test_temperature_constant_in_tropopause(self):
        """Temperature should be roughly constant between 11 and 20 km."""
        t_12km = atmosphere(12000.0).temperature_k
        t_18km = atmosphere(18000.0).temperature_k
        npt.assert_allclose(t_12km, t_18km, atol=0.5)

    def test_pressure_decreases_with_altitude(self):
        """Pressure should monotonically decrease with altitude."""
        altitudes = [0, 5000, 11000, 20000, 32000, 47000, 60000, 80000]
        pressures = [atmosphere(h).pressure_pa for h in altitudes]

        for i in range(len(pressures) - 1):
            assert pressures[i + 1] < pressures[i], (
                f"Pressure at {altitudes[i + 1]}m ({pressures[i + 1]:.2f} Pa) "
                f">= pressure at {altitudes[i]}m ({pressures[i]:.2f} Pa)"
            )

    def test_density_decreases_with_altitude(self):
        """Density should monotonically decrease with altitude."""
        altitudes = [0, 5000, 11000, 20000, 50000, 80000]
        densities = [atmosphere(h).density_kg_m3 for h in altitudes]

        for i in range(len(densities) - 1):
            assert densities[i + 1] < densities[i]

    def test_pressure_at_5500m_roughly_half(self):
        """At ~5500 m the pressure should be roughly half sea-level."""
        result = atmosphere(5500.0)
        ratio = result.pressure_pa / 101325.0
        npt.assert_allclose(ratio, 0.5, atol=0.05)

    def test_speed_of_sound_decreases_in_troposphere(self):
        """Speed of sound decreases as temperature falls in troposphere."""
        a_sl = atmosphere(0.0).speed_of_sound_ms
        a_10km = atmosphere(10000.0).speed_of_sound_ms
        assert a_10km < a_sl


class TestUpperAtmosphere:
    """Behaviour above the tabulated 86 km ceiling."""

    def test_density_very_small_at_100km(self):
        """At 100 km (Karman line) density should be extremely small."""
        result = atmosphere(100_000.0)
        assert result.density_kg_m3 < 1e-5

    def test_density_decreases_above_86km(self):
        """Density should continue to decrease above 86 km."""
        rho_86 = atmosphere(86_000.0).density_kg_m3
        rho_100 = atmosphere(100_000.0).density_kg_m3
        rho_150 = atmosphere(150_000.0).density_kg_m3

        assert rho_100 < rho_86
        assert rho_150 < rho_100

    def test_density_decreases_above_200km(self):
        """Above 200 km density should be extremely small but non-zero
        (multi-layer thermosphere model extends to 1000 km)."""
        result_200 = atmosphere(200_000.0)
        result_500 = atmosphere(500_000.0)
        # Density at 200 km should be ~1e-10 (NRLMSISE-00 moderate solar)
        assert result_200.density_kg_m3 < 1e-8
        assert result_200.density_kg_m3 > 0.0
        # Density continues to decrease with altitude
        assert result_500.density_kg_m3 < result_200.density_kg_m3

    def test_vacuum_above_1000km(self):
        """Above 1000 km the atmosphere should return zero density."""
        result = atmosphere(1_100_000.0)
        assert result.density_kg_m3 == 0.0
        assert result.pressure_pa == 0.0


class TestAtmosphereResult:
    """Verify the named-tuple interface."""

    def test_result_is_named_tuple(self):
        """AtmosphereResult should be a NamedTuple with expected fields."""
        r = atmosphere(0.0)
        assert isinstance(r, AtmosphereResult)
        assert hasattr(r, "density_kg_m3")
        assert hasattr(r, "pressure_pa")
        assert hasattr(r, "temperature_k")
        assert hasattr(r, "speed_of_sound_ms")

    def test_ideal_gas_consistency(self):
        """Check that rho = P / (R_specific * T) holds at a mid-altitude."""
        R_specific = 287.058  # approximate
        for alt in [0, 5000, 11000, 25000, 50000]:
            r = atmosphere(alt)
            if r.temperature_k > 0 and r.pressure_pa > 0:
                rho_expected = r.pressure_pa / (R_specific * r.temperature_k)
                npt.assert_allclose(
                    r.density_kg_m3,
                    rho_expected,
                    rtol=0.01,
                    err_msg=f"Ideal gas law mismatch at {alt} m",
                )
