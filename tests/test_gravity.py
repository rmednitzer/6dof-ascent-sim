"""Tests for the J2 gravity model (sim.environment.gravity).

Verifies surface-level magnitudes, J2 perturbation effects (pole vs equator),
and edge-case behaviour.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from sim.environment.gravity import gravitational_acceleration
from sim import config


class TestGravityMagnitude:
    """Check that gravity magnitudes are physically reasonable."""

    def test_surface_magnitude_equator(self):
        """At the equator on Earth's surface, |g| ~ 9.78 - 9.82 m/s^2."""
        pos = np.array([config.EARTH_RADIUS_M, 0.0, 0.0])
        g = gravitational_acceleration(pos)
        mag = np.linalg.norm(g)
        # Standard gravity is 9.80665; with J2 at equator expect ~9.78-9.82
        assert 9.7 < mag < 9.9, f"Equatorial surface gravity {mag:.4f} out of range"

    def test_surface_magnitude_pole(self):
        """At the pole on Earth's surface, |g| ~ 9.83 m/s^2 (slightly higher)."""
        pos = np.array([0.0, 0.0, config.EARTH_RADIUS_M])
        g = gravitational_acceleration(pos)
        mag = np.linalg.norm(g)
        assert 9.7 < mag < 9.9, f"Polar surface gravity {mag:.4f} out of range"

    def test_gravity_direction_points_toward_center(self):
        """Gravity should point roughly toward Earth's center."""
        pos = np.array([config.EARTH_RADIUS_M, 0.0, 0.0])
        g = gravitational_acceleration(pos)
        # Should point in the -x direction predominantly
        assert g[0] < 0.0
        # Magnitude in x should dominate
        assert abs(g[0]) > abs(g[1])
        assert abs(g[0]) > abs(g[2])

    def test_gravity_decreases_with_altitude(self):
        """Gravity magnitude should decrease with increasing altitude."""
        pos_surface = np.array([config.EARTH_RADIUS_M, 0.0, 0.0])
        pos_400km = np.array([config.EARTH_RADIUS_M + 400_000.0, 0.0, 0.0])

        g_surface = np.linalg.norm(gravitational_acceleration(pos_surface))
        g_400km = np.linalg.norm(gravitational_acceleration(pos_400km))

        assert g_400km < g_surface

    def test_inverse_square_law_approximate(self):
        """Far from Earth (where J2 is negligible), gravity ~ mu/r^2."""
        r1 = 1e8  # 100,000 km - far from Earth, J2 negligible
        r2 = 2e8
        pos1 = np.array([r1, 0.0, 0.0])
        pos2 = np.array([r2, 0.0, 0.0])

        g1 = np.linalg.norm(gravitational_acceleration(pos1))
        g2 = np.linalg.norm(gravitational_acceleration(pos2))

        # g1/g2 should be close to (r2/r1)^2 = 4
        ratio = g1 / g2
        npt.assert_allclose(ratio, 4.0, rtol=1e-4)


class TestJ2Effect:
    """Verify that J2 oblateness perturbation is present and correct."""

    def test_j2_modifies_gravity_differently_at_pole_vs_equator(self):
        """J2 perturbation should cause different gravity at poles vs equator.

        At the same geocentric radius, the J2 correction makes equatorial
        gravity slightly stronger than polar (because the 5*z^2/r^2 - 1
        factor differs).  On the real Earth, the pole has stronger surface
        gravity because the polar radius is smaller, but this model uses
        the same radius for both, showing the J2 correction effect.
        """
        r = config.EARTH_RADIUS_M
        pos_equator = np.array([r, 0.0, 0.0])
        pos_pole = np.array([0.0, 0.0, r])

        g_eq = np.linalg.norm(gravitational_acceleration(pos_equator))
        g_pole = np.linalg.norm(gravitational_acceleration(pos_pole))

        # At the same geocentric radius, J2 gives different magnitudes
        assert g_eq != g_pole, "J2 should break spherical symmetry"

        # At same radius, equatorial gravity is enhanced by J2 (z/r=0)
        # while polar gravity is reduced (z/r=1), so g_eq > g_pole
        assert g_eq > g_pole

    def test_j2_introduces_off_radial_component_at_45deg(self):
        """At 45 deg latitude, J2 should produce a non-zero off-radial component.

        For a purely spherical model, gravity at (x, 0, z) with x=z would
        point exactly radially inward.  J2 breaks this symmetry because the
        z-component acceleration has a different correction factor.
        """
        r = config.EARTH_RADIUS_M
        angle = math.pi / 4.0  # 45 degrees
        pos = np.array([r * math.cos(angle), 0.0, r * math.sin(angle)])

        g = gravitational_acceleration(pos)

        # Radial unit vector
        r_hat = pos / np.linalg.norm(pos)

        # Radial component of g
        g_radial = np.dot(g, r_hat) * r_hat

        # Tangential (off-radial) component
        g_tangential = g - g_radial
        g_tan_mag = np.linalg.norm(g_tangential)

        # Should be non-zero (J2 effect) but small compared to total
        assert g_tan_mag > 1e-4, "Expected non-zero off-radial J2 component"
        assert g_tan_mag < 0.1 * np.linalg.norm(g), "Off-radial too large"

    def test_equatorial_symmetry(self):
        """Gravity at symmetric equatorial points should have equal magnitudes."""
        r = config.EARTH_RADIUS_M
        pos_x = np.array([r, 0.0, 0.0])
        pos_y = np.array([0.0, r, 0.0])

        g_x = np.linalg.norm(gravitational_acceleration(pos_x))
        g_y = np.linalg.norm(gravitational_acceleration(pos_y))

        npt.assert_allclose(g_x, g_y, rtol=1e-12)


class TestGravityEdgeCases:
    """Edge cases and error handling."""

    def test_zero_position_raises(self):
        """Position at origin should raise ValueError."""
        with pytest.raises(ValueError, match="too small"):
            gravitational_acceleration(np.array([0.0, 0.0, 0.0]))

    def test_very_small_position_raises(self):
        """Position with magnitude < 1 m should raise ValueError."""
        with pytest.raises(ValueError):
            gravitational_acceleration(np.array([0.5, 0.0, 0.0]))
