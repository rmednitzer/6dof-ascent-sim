"""Tests for the Flight Termination System (sim.safety.fts).

Validates that FTS does not trigger under nominal conditions, and correctly
triggers for cross-range, attitude, and structural violations.
"""

import math

import numpy as np

from sim import config
from sim.safety.boundary_enforcer import BoundaryEnforcer
from sim.safety.fts import FlightTerminationSystem


def _nominal_args() -> dict:
    """Return a set of nominal arguments that should NOT trigger FTS."""
    return dict(
        position_ecef=np.array([config.EARTH_RADIUS_M, 0.0, 0.0]),
        nominal_plane_normal=np.array([0.0, 1.0, 0.0]),  # y-axis normal
        nominal_plane_point=np.array([config.EARTH_RADIUS_M, 0.0, 0.0]),
        q_actual=np.array([0.0, 0.0, 0.0, 1.0]),
        q_desired=np.array([0.0, 0.0, 0.0, 1.0]),
        ekf_pos_covariance=np.eye(3) * 100.0,  # 10 m 1-sigma per axis
        axial_g=3.0,
        lateral_g=0.2,
        dynamic_pressure_pa=20000.0,
        sim_time=10.0,
        altitude_m=50000.0,
    )


class TestFTSNominal:
    """FTS should not trigger under nominal flight conditions."""

    def test_nominal_no_trigger(self):
        """Nominal conditions should not trigger FTS."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        triggered = fts.evaluate(**_nominal_args())
        assert triggered is False
        assert fts.fts_triggered is False

    def test_multiple_nominal_calls(self):
        """Repeated nominal evaluations should all return False."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        for i in range(10):
            args = _nominal_args()
            args["sim_time"] = float(i)
            assert fts.evaluate(**args) is False

        assert fts.fts_triggered is False


class TestFTSCrossRange:
    """FTS should trigger when cross-range deviation exceeds the limit."""

    def test_crossrange_violation_triggers(self):
        """Large cross-range deviation should trigger FTS."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        # Move the position far from the nominal plane in the normal direction
        offset = config.FTS_CROSSRANGE_LIMIT_M + 10000.0
        args["position_ecef"] = np.array(
            [
                config.EARTH_RADIUS_M,
                offset,  # cross-range in the y direction (plane normal)
                0.0,
            ]
        )

        triggered = fts.evaluate(**args)
        assert triggered is True
        assert fts.fts_triggered is True
        assert "Cross-range" in fts.state.reason

    def test_crossrange_within_limit_no_trigger(self):
        """Cross-range within limits should not trigger."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        # Small offset in the normal direction
        args["position_ecef"] = np.array(
            [
                config.EARTH_RADIUS_M,
                1000.0,  # well within 200 km limit
                0.0,
            ]
        )

        assert fts.evaluate(**args) is False

    def test_crossrange_not_checked_above_100km(self):
        """Cross-range is only checked below 100 km altitude."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        args["altitude_m"] = 150_000.0  # above 100 km
        args["position_ecef"] = np.array(
            [
                config.EARTH_RADIUS_M,
                config.FTS_CROSSRANGE_LIMIT_M + 50000.0,
                0.0,
            ]
        )

        # Should NOT trigger because altitude > 100 km
        assert fts.evaluate(**args) is False


def _q_thrust_axis_error(deg: float) -> np.ndarray:
    """Quaternion whose body +X (thrust axis) is rotated *deg* about body Z."""
    half = math.radians(deg) / 2.0
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)])


class TestFTSAttitude:
    """FTS attitude criterion is hysteresis-gated (ADR 0014, audit AD-19)."""

    def test_single_frame_violation_does_not_trigger(self):
        """A single marginal-frame thrust-axis excursion is debounced."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)
        assert fts._attitude_hysteresis_s > 0.0  # default is non-zero

        args = _nominal_args()
        args["q_actual"] = _q_thrust_axis_error(90.0)  # 90 deg >> 25 deg limit
        args["q_desired"] = np.array([0.0, 0.0, 0.0, 1.0])

        # One frame over the limit: not yet persisted -> no trigger.
        assert fts.evaluate(**args) is False
        assert fts.fts_triggered is False

    def test_sustained_violation_triggers_after_hysteresis(self):
        """A sustained thrust-axis violation trips after the hysteresis window."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)
        q_bad = _q_thrust_axis_error(90.0)
        dt = config.DT
        n = int(fts._attitude_hysteresis_s / dt) + 2

        triggered = False
        for i in range(n):
            args = _nominal_args()
            args["q_actual"] = q_bad
            args["sim_time"] = i * dt
            triggered = fts.evaluate(**args)
            if triggered:
                break

        assert triggered is True
        assert "Attitude" in fts.state.reason

    def test_intermittent_violation_never_accumulates(self):
        """A violation that clears between frames never reaches the trip time."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)
        q_bad = _q_thrust_axis_error(90.0)
        q_ok = np.array([0.0, 0.0, 0.0, 1.0])
        dt = config.DT

        for i in range(200):  # 2 s, ten hysteresis windows
            args = _nominal_args()
            args["q_actual"] = q_bad if (i % 2 == 0) else q_ok
            args["sim_time"] = i * dt
            assert fts.evaluate(**args) is False

        assert fts.fts_triggered is False

    def test_hysteresis_zero_is_instantaneous(self):
        """FTS_ATTITUDE_HYSTERESIS_S = 0 recovers the original instant trip."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)
        fts._attitude_hysteresis_s = 0.0  # back-compatible setting

        args = _nominal_args()
        args["q_actual"] = _q_thrust_axis_error(90.0)
        assert fts.evaluate(**args) is True
        assert "Attitude" in fts.state.reason

    def test_small_attitude_error_no_trigger(self):
        """Small thrust-axis error should not trigger FTS."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        # 5 degree rotation about z-axis (well under the limit)
        half_angle = math.radians(2.5)
        args["q_actual"] = np.array([0.0, 0.0, math.sin(half_angle), math.cos(half_angle)])

        assert fts.evaluate(**args) is False

    def test_pure_roll_about_thrust_axis_does_not_trigger(self):
        """A large roll about the thrust axis is NOT a loss of control.

        The vehicle is axisymmetric with no roll authority, so roll
        drifts freely. The FTS attitude criterion must measure
        thrust-axis divergence only — a 180 deg roll about +X leaves the
        thrust axis exactly on the commanded direction and must not abort.
        """
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        # q = [1,0,0,0] is a 180 deg rotation about body +X (pure roll).
        args["q_actual"] = np.array([1.0, 0.0, 0.0, 0.0])
        args["q_desired"] = np.array([0.0, 0.0, 0.0, 1.0])

        assert fts.evaluate(**args) is False


class TestFTSStructural:
    """FTS should trigger when structural limits are exceeded."""

    def test_structural_violation_triggers(self):
        """Exceeding structural limits should trigger FTS."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        args["axial_g"] = config.MAX_AXIAL_G + 2.0
        args["lateral_g"] = 0.1
        args["dynamic_pressure_pa"] = 10000.0

        triggered = fts.evaluate(**args)
        assert triggered is True
        assert "Structural" in fts.state.reason

    def test_max_q_exceeded_triggers(self):
        """Dynamic pressure exceeding MAX_Q_PA should trigger FTS."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        args["dynamic_pressure_pa"] = config.MAX_Q_PA + 10000.0

        triggered = fts.evaluate(**args)
        assert triggered is True


class TestFTSLatch:
    """FTS should latch irrevocably once triggered."""

    def test_latch_stays_triggered(self):
        """Once triggered, FTS stays triggered even with nominal args."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        # Trigger with structural violation
        args = _nominal_args()
        args["axial_g"] = config.MAX_AXIAL_G + 5.0
        fts.evaluate(**args)
        assert fts.fts_triggered is True

        # Now pass nominal conditions
        nominal = _nominal_args()
        result = fts.evaluate(**nominal)
        assert result is True  # still triggered
        assert fts.fts_triggered is True

    def test_state_snapshot_recorded(self):
        """Trigger event should record reason and snapshot."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        args["axial_g"] = config.MAX_AXIAL_G + 5.0
        fts.evaluate(**args)

        assert fts.state.reason is not None
        assert fts.state.trigger_time is not None
        assert "sim_time" in fts.state.snapshot


class TestFTSCovariance:
    """FTS should trigger when EKF position uncertainty is too large."""

    def test_large_covariance_triggers(self):
        """EKF covariance exceeding the limit should trigger."""
        be = BoundaryEnforcer()
        fts = FlightTerminationSystem(be)

        args = _nominal_args()
        # Make eigenvalues exceed FTS_COVARIANCE_LIMIT_M^2
        big_var = (config.FTS_COVARIANCE_LIMIT_M + 1000.0) ** 2
        args["ekf_pos_covariance"] = np.eye(3) * big_var

        triggered = fts.evaluate(**args)
        assert triggered is True
        assert "uncertainty" in fts.state.reason.lower()
