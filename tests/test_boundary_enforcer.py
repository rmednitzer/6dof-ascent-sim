"""Tests for the BoundaryEnforcer (sim.safety.boundary_enforcer).

Covers throttle clamping, TVC deflection and slew-rate limits, and
structural limit checks.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sim.safety.boundary_enforcer import BoundaryEnforcer, BoundaryResult
from sim import config


# ---------------------------------------------------------------------------
# Throttle validation
# ---------------------------------------------------------------------------

class TestThrottleValidation:
    """Throttle command clamping and propellant depletion."""

    def test_nominal_throttle_passes(self):
        """A throttle command of 0.8 with propellant should pass unchanged."""
        be = BoundaryEnforcer()
        result = be.validate_throttle(0.8, propellant_remaining_kg=10000.0)
        assert result.approved is True
        assert result.value == 0.8
        assert result.was_clamped is False

    def test_throttle_clamped_above_one(self):
        """Throttle > 1.0 should be clamped to 1.0."""
        be = BoundaryEnforcer()
        result = be.validate_throttle(1.5, propellant_remaining_kg=10000.0)
        assert result.approved is True
        assert result.value == 1.0
        assert result.was_clamped is True

    def test_throttle_clamped_below_zero(self):
        """Throttle < 0 should be clamped to 0.0."""
        be = BoundaryEnforcer()
        result = be.validate_throttle(-0.5, propellant_remaining_kg=10000.0)
        assert result.approved is True
        assert result.value == 0.0
        assert result.was_clamped is True  # negative -> clamped to 0

    def test_throttle_below_minimum_raised_to_minimum(self):
        """Throttle between 0 and S1_THROTTLE_MIN should be raised."""
        be = BoundaryEnforcer()
        cmd = config.S1_THROTTLE_MIN / 2.0  # e.g., 0.2 if min is 0.4
        result = be.validate_throttle(cmd, propellant_remaining_kg=10000.0)
        assert result.approved is True
        assert result.value == config.S1_THROTTLE_MIN
        assert result.was_clamped is True

    def test_throttle_zero_passes_without_min_enforcement(self):
        """Zero throttle should pass as-is (engine off, no min throttle)."""
        be = BoundaryEnforcer()
        result = be.validate_throttle(0.0, propellant_remaining_kg=10000.0)
        assert result.approved is True
        assert result.value == 0.0
        assert result.was_clamped is False

    def test_throttle_with_no_propellant(self):
        """With zero propellant, throttle should be forced to zero."""
        be = BoundaryEnforcer()
        result = be.validate_throttle(0.8, propellant_remaining_kg=0.0)
        assert result.approved is True
        assert result.value == 0.0
        assert result.was_clamped is True
        assert result.violation_type == "propellant_depleted"

    def test_throttle_with_negative_propellant(self):
        """Negative propellant should also force throttle to zero."""
        be = BoundaryEnforcer()
        result = be.validate_throttle(1.0, propellant_remaining_kg=-10.0)
        assert result.value == 0.0

    def test_violation_count_increments(self):
        """Clamped commands should increment violation_count."""
        be = BoundaryEnforcer()
        assert be.violation_count == 0

        be.validate_throttle(1.5, propellant_remaining_kg=10000.0)
        assert be.violation_count == 1

        be.validate_throttle(0.1, propellant_remaining_kg=10000.0)  # below min
        assert be.violation_count == 2


# ---------------------------------------------------------------------------
# TVC validation
# ---------------------------------------------------------------------------

class TestTVCValidation:
    """TVC gimbal deflection and slew-rate limits."""

    def test_nominal_tvc_passes(self):
        """Small TVC command within limits should pass unchanged with large dt."""
        be = BoundaryEnforcer()
        # Use large dt so slew rate doesn't limit the step from 0 to 1.0
        result = be.validate_tvc(1.0, -1.0, dt=1.0)
        assert result.approved is True
        pitch, yaw = result.value
        npt.assert_allclose(pitch, 1.0)
        npt.assert_allclose(yaw, -1.0)
        assert result.was_clamped is False

    def test_deflection_clamped(self):
        """Commands exceeding TVC_MAX_DEFLECTION_DEG should be clamped."""
        be = BoundaryEnforcer()
        big = config.TVC_MAX_DEFLECTION_DEG + 5.0
        result = be.validate_tvc(big, -big, dt=1.0)

        pitch, yaw = result.value
        assert abs(pitch) <= config.TVC_MAX_DEFLECTION_DEG + 1e-10
        assert abs(yaw) <= config.TVC_MAX_DEFLECTION_DEG + 1e-10
        assert result.was_clamped is True

    def test_slew_rate_limited(self):
        """Large step change should be limited by TVC_MAX_SLEW_RATE_DEG_S."""
        be = BoundaryEnforcer()
        dt = 0.01
        max_delta = config.TVC_MAX_SLEW_RATE_DEG_S * dt  # 0.1 deg at 10 deg/s

        # First call sets the reference to (0, 0)
        be.validate_tvc(0.0, 0.0, dt=dt)

        # Now request a large step
        result = be.validate_tvc(5.0, -5.0, dt=dt)
        pitch, yaw = result.value

        # The change from 0 should be at most max_delta
        assert abs(pitch) <= max_delta + 1e-10
        assert abs(yaw) <= max_delta + 1e-10
        assert result.was_clamped is True

    def test_slew_rate_allows_slow_changes(self):
        """Gradual changes within the slew rate should not be clamped."""
        be = BoundaryEnforcer()
        dt = 0.01
        max_delta = config.TVC_MAX_SLEW_RATE_DEG_S * dt

        # Start at 0
        be.validate_tvc(0.0, 0.0, dt=dt)

        # Request a small change within limits
        small_step = max_delta * 0.5
        result = be.validate_tvc(small_step, small_step, dt=dt)
        pitch, yaw = result.value

        npt.assert_allclose(pitch, small_step, atol=1e-10)
        npt.assert_allclose(yaw, small_step, atol=1e-10)

    def test_tvc_result_is_tuple(self):
        """The TVC result value should be a (pitch, yaw) tuple."""
        be = BoundaryEnforcer()
        result = be.validate_tvc(0.0, 0.0, dt=0.01)
        assert isinstance(result.value, tuple)
        assert len(result.value) == 2


# ---------------------------------------------------------------------------
# Structural limits
# ---------------------------------------------------------------------------

class TestStructuralLimits:
    """Check structural limit evaluation."""

    def test_nominal_conditions_approved(self):
        """Within all limits should be approved."""
        be = BoundaryEnforcer()
        result = be.check_structural_limits(
            axial_g=3.0,
            lateral_g=0.2,
            dynamic_pressure_pa=20000.0,
        )
        assert result.approved is True
        assert result.violation_type is None

    def test_axial_g_exceeded(self):
        """Exceeding MAX_AXIAL_G should not be approved."""
        be = BoundaryEnforcer()
        result = be.check_structural_limits(
            axial_g=config.MAX_AXIAL_G + 1.0,
            lateral_g=0.1,
            dynamic_pressure_pa=10000.0,
        )
        assert result.approved is False
        assert "axial_g_exceeded" in result.violation_type

    def test_lateral_g_exceeded(self):
        """Exceeding MAX_LATERAL_G should not be approved."""
        be = BoundaryEnforcer()
        result = be.check_structural_limits(
            axial_g=1.0,
            lateral_g=config.MAX_LATERAL_G + 0.5,
            dynamic_pressure_pa=10000.0,
        )
        assert result.approved is False
        assert "lateral_g_exceeded" in result.violation_type

    def test_dynamic_pressure_exceeded(self):
        """Exceeding MAX_Q_PA should not be approved."""
        be = BoundaryEnforcer()
        result = be.check_structural_limits(
            axial_g=1.0,
            lateral_g=0.1,
            dynamic_pressure_pa=config.MAX_Q_PA + 5000.0,
        )
        assert result.approved is False
        assert "dynamic_pressure_exceeded" in result.violation_type

    def test_multiple_violations(self):
        """Multiple exceeded limits should all appear in violation_type."""
        be = BoundaryEnforcer()
        result = be.check_structural_limits(
            axial_g=config.MAX_AXIAL_G + 1.0,
            lateral_g=config.MAX_LATERAL_G + 1.0,
            dynamic_pressure_pa=config.MAX_Q_PA + 10000.0,
        )
        assert result.approved is False
        assert "axial_g_exceeded" in result.violation_type
        assert "lateral_g_exceeded" in result.violation_type
        assert "dynamic_pressure_exceeded" in result.violation_type

    def test_evidence_contains_fractions(self):
        """Evidence dict should contain load fraction values."""
        be = BoundaryEnforcer()
        result = be.check_structural_limits(
            axial_g=3.0,
            lateral_g=0.2,
            dynamic_pressure_pa=20000.0,
        )
        assert "axial_g_fraction" in result.evidence
        assert "lateral_g_fraction" in result.evidence
        assert "q_fraction" in result.evidence

        npt.assert_allclose(
            result.evidence["axial_g_fraction"],
            3.0 / config.MAX_AXIAL_G,
        )

    def test_exactly_at_limit_is_approved(self):
        """At exactly the limit boundary, the check should still approve."""
        be = BoundaryEnforcer()
        result = be.check_structural_limits(
            axial_g=config.MAX_AXIAL_G,
            lateral_g=config.MAX_LATERAL_G,
            dynamic_pressure_pa=config.MAX_Q_PA,
        )
        assert result.approved is True
