"""Tests for orbital decay estimation (sim.orbital.decay)."""

import math
import sys
from unittest.mock import MagicMock

# Mock numpy before importing simulation modules to allow tests to run in restricted environments.
# This is necessary because the simulation modules depend on numpy which might not be installed.
if 'numpy' not in sys.modules:
    mock_np = MagicMock()
    mock_np.bool_ = bool
    sys.modules['numpy'] = mock_np
    sys.modules['numpy.testing'] = MagicMock()

import pytest  # noqa: E402

from sim.orbital.decay import ballistic_coefficient  # noqa: E402


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
