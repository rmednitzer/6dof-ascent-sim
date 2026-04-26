"""Tests for orbital decay calculations (sim.orbital.decay)."""

import math
import sys
from unittest.mock import MagicMock

# Mock numpy before importing any sim modules that might use it
mock_np = MagicMock()
mock_np.bool_ = bool  # pytest.approx checks for np.bool_
sys.modules["numpy"] = mock_np

import pytest  # noqa: E402

from sim.orbital.decay import ballistic_coefficient  # noqa: E402


class TestBallisticCoefficient:
    """Verify ballistic coefficient calculation and edge cases."""

    def test_standard_values(self):
        """BC = m / (Cd * A) for typical values."""
        mass = 1000.0
        cd = 2.2
        area = 10.0
        expected = mass / (cd * area)
        assert ballistic_coefficient(mass, cd, area) == pytest.approx(expected)

    def test_zero_cd_returns_inf(self):
        """If Cd is zero, BC should be infinite (no drag)."""
        assert ballistic_coefficient(1000.0, 0.0, 10.0) == math.inf

    def test_zero_area_returns_inf(self):
        """If area is zero, BC should be infinite (no drag)."""
        assert ballistic_coefficient(1000.0, 2.2, 0.0) == math.inf

    def test_very_small_drag_area_returns_inf(self):
        """If Cd * A is extremely small, return inf to avoid numerical issues."""
        assert ballistic_coefficient(1000.0, 1e-7, 1e-7) == math.inf

    def test_negative_mass_raises_error(self):
        """Mass must be positive."""
        with pytest.raises(ValueError, match="Mass must be positive"):
            ballistic_coefficient(-100.0, 2.2, 10.0)

    def test_zero_mass_raises_error(self):
        """Mass must be positive."""
        with pytest.raises(ValueError, match="Mass must be positive"):
            ballistic_coefficient(0.0, 2.2, 10.0)

    def test_negative_cd_raises_error(self):
        """Drag coefficient cannot be negative."""
        with pytest.raises(ValueError, match="Drag coefficient and area must be non-negative"):
            ballistic_coefficient(1000.0, -2.2, 10.0)

    def test_negative_area_raises_error(self):
        """Area cannot be negative."""
        with pytest.raises(ValueError, match="Drag coefficient and area must be non-negative"):
            ballistic_coefficient(1000.0, 2.2, -10.0)
