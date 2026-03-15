"""Tests for the orbit propagator (sim.orbital.propagator).

Validates circular orbit element computation, propagation energy conservation,
and Keplerian element accuracy.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from sim.core.state import VehicleState
from sim.orbital.propagator import OrbitPropagator, OrbitalElements
from sim.config import EARTH_MU, EARTH_RADIUS_M


def _circular_orbit_state(altitude_km: float, inclination_deg: float) -> VehicleState:
    """Create a VehicleState for a circular orbit at the given altitude and inclination.

    The orbit is placed at the ascending node (RAAN=0, arg_periapsis=0,
    true_anomaly=0).
    """
    r = EARTH_RADIUS_M + altitude_km * 1000.0
    v = math.sqrt(EARTH_MU / r)  # circular velocity

    inc = math.radians(inclination_deg)

    # Position at ascending node: x-axis in the equatorial plane
    pos = np.array([r, 0.0, 0.0])

    # Velocity perpendicular to position, tilted by inclination
    vel = np.array([0.0, v * math.cos(inc), v * math.sin(inc)])

    return VehicleState(
        position_eci=pos,
        velocity_eci=vel,
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity_body=np.zeros(3),
        mass_kg=5000.0,
        time_s=0.0,
    )


class TestCircularOrbitElements:
    """Verify orbital element computation for circular orbits."""

    def test_semi_major_axis(self):
        """Semi-major axis should match the orbital radius for circular orbit."""
        alt_km = 400.0
        state = _circular_orbit_state(alt_km, inclination_deg=51.6)
        prop = OrbitPropagator(state)
        elements = prop.state_to_elements()

        expected_a = EARTH_RADIUS_M + alt_km * 1000.0
        npt.assert_allclose(elements.semi_major_axis_m, expected_a, rtol=1e-6)

    def test_eccentricity_near_zero(self):
        """Circular orbit should have eccentricity very close to zero."""
        state = _circular_orbit_state(400.0, inclination_deg=51.6)
        prop = OrbitPropagator(state)
        elements = prop.state_to_elements()

        assert elements.eccentricity < 1e-6

    def test_inclination(self):
        """Inclination should match the input value."""
        inc_deg = 51.6
        state = _circular_orbit_state(400.0, inclination_deg=inc_deg)
        prop = OrbitPropagator(state)
        elements = prop.state_to_elements()

        npt.assert_allclose(elements.inclination_deg, inc_deg, atol=0.01)

    def test_orbital_period(self):
        """Period should match Kepler's third law: T = 2*pi*sqrt(a^3/mu)."""
        alt_km = 400.0
        a = EARTH_RADIUS_M + alt_km * 1000.0
        expected_period = 2.0 * math.pi * math.sqrt(a ** 3 / EARTH_MU)

        state = _circular_orbit_state(alt_km, inclination_deg=51.6)
        prop = OrbitPropagator(state)
        elements = prop.state_to_elements()

        npt.assert_allclose(elements.period_s, expected_period, rtol=1e-6)

    def test_apoapsis_periapsis_altitude(self):
        """For circular orbit, apoapsis and periapsis altitudes should equal the input."""
        alt_km = 400.0
        state = _circular_orbit_state(alt_km, inclination_deg=0.0)
        prop = OrbitPropagator(state)
        elements = prop.state_to_elements()

        npt.assert_allclose(elements.apoapsis_alt_km, alt_km, atol=1.0)
        npt.assert_allclose(elements.periapsis_alt_km, alt_km, atol=1.0)

    def test_different_altitudes(self):
        """Higher altitude orbits should have larger semi-major axes."""
        state_200 = _circular_orbit_state(200.0, inclination_deg=28.5)
        state_800 = _circular_orbit_state(800.0, inclination_deg=28.5)

        el_200 = OrbitPropagator(state_200).state_to_elements()
        el_800 = OrbitPropagator(state_800).state_to_elements()

        assert el_800.semi_major_axis_m > el_200.semi_major_axis_m
        assert el_800.period_s > el_200.period_s

    def test_equatorial_orbit_inclination_zero(self):
        """An equatorial orbit should have inclination near zero."""
        state = _circular_orbit_state(400.0, inclination_deg=0.0)
        elements = OrbitPropagator(state).state_to_elements()
        npt.assert_allclose(elements.inclination_deg, 0.0, atol=0.01)

    def test_polar_orbit_inclination_90(self):
        """A polar orbit should have inclination near 90 deg."""
        state = _circular_orbit_state(400.0, inclination_deg=90.0)
        elements = OrbitPropagator(state).state_to_elements()
        npt.assert_allclose(elements.inclination_deg, 90.0, atol=0.01)


class TestOrbitPropagation:
    """Verify J2-perturbed orbit propagation."""

    def test_propagation_preserves_energy_approximately(self):
        """Specific orbital energy should be approximately conserved.

        J2 is a conservative force, so energy should be well-conserved over
        a short propagation.  We allow small numerical drift.
        """
        state = _circular_orbit_state(400.0, inclination_deg=51.6)
        prop = OrbitPropagator(state)

        # Propagate for one orbit (~5550 s at 400 km)
        duration = 5550.0
        states = prop.propagate(duration_s=duration, dt_s=10.0)

        # Compute energy at start and end
        def specific_energy(s: VehicleState) -> float:
            r = np.linalg.norm(s.position_eci)
            v = np.linalg.norm(s.velocity_eci)
            return 0.5 * v ** 2 - EARTH_MU / r

        e_start = specific_energy(states[0])
        e_end = specific_energy(states[-1])

        # Energy should be conserved to within 0.01%
        npt.assert_allclose(e_end, e_start, rtol=1e-4)

    def test_propagation_returns_multiple_states(self):
        """Propagation should return one state per time step plus endpoints."""
        state = _circular_orbit_state(400.0, inclination_deg=51.6)
        prop = OrbitPropagator(state)

        states = prop.propagate(duration_s=100.0, dt_s=10.0)
        assert len(states) >= 10

    def test_propagation_position_magnitude_stays_bounded(self):
        """For a circular orbit, position magnitude should stay near the orbital radius."""
        alt_km = 400.0
        r_expected = EARTH_RADIUS_M + alt_km * 1000.0

        state = _circular_orbit_state(alt_km, inclination_deg=51.6)
        prop = OrbitPropagator(state)
        states = prop.propagate(duration_s=5550.0, dt_s=10.0)

        for s in states:
            r = np.linalg.norm(s.position_eci)
            # Should stay within ~1% of expected radius for nearly circular orbit
            npt.assert_allclose(r, r_expected, rtol=0.01)

    def test_propagation_time_advances(self):
        """Time in output states should advance correctly."""
        state = _circular_orbit_state(400.0, inclination_deg=51.6)
        prop = OrbitPropagator(state)
        states = prop.propagate(duration_s=100.0, dt_s=10.0)

        times = [s.time_s for s in states]
        # Times should be monotonically increasing
        for i in range(len(times) - 1):
            assert times[i + 1] > times[i]

        # First time should be 0, last should be ~100
        npt.assert_allclose(times[0], 0.0)
        npt.assert_allclose(times[-1], 100.0, atol=1.0)


class TestOrbitSummary:
    """Verify the orbit_summary() convenience method."""

    def test_summary_contains_expected_keys(self):
        """orbit_summary() should return a dict with expected keys."""
        state = _circular_orbit_state(400.0, inclination_deg=51.6)
        prop = OrbitPropagator(state)
        summary = prop.orbit_summary()

        expected_keys = [
            "semi_major_axis_km",
            "eccentricity",
            "inclination_deg",
            "raan_deg",
            "arg_periapsis_deg",
            "true_anomaly_deg",
            "period_min",
            "apoapsis_alt_km",
            "periapsis_alt_km",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_period_in_minutes(self):
        """Period should be reported in minutes."""
        state = _circular_orbit_state(400.0, inclination_deg=51.6)
        prop = OrbitPropagator(state)
        summary = prop.orbit_summary()

        # ISS period is ~92 minutes
        assert 85 < summary["period_min"] < 100
