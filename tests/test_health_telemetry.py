"""Regression tests: health status must reach telemetry (finding Q-01).

The recorder previously read a non-existent ``.status`` attribute on
``HealthMonitor`` via ``getattr(..., "status", "NOMINAL")``, so the telemetry
``health_status`` field was permanently ``"NOMINAL"`` regardless of actual
vehicle health (confirmed across 48,816 frames of a nominal run whose peak
dynamic pressure reached 92.6% of the structural limit). These tests pin the
fix so the channel cannot silently regress.
"""

from __future__ import annotations

import numpy as np

from sim import config
from sim.core.state import VehicleState
from sim.safety.boundary_enforcer import BoundaryEnforcer
from sim.safety.health_monitor import HealthMonitor
from sim.telemetry.recorder import TelemetryRecorder
from sim.telemetry.schemas import HEALTH_CRITICAL, HEALTH_NOMINAL, HEALTH_WARNING


def _nominal_cov() -> np.ndarray:
    """1-sigma = 1 m position covariance, far below the FTS limit (EKF NOMINAL)."""
    return np.eye(3)


def _state() -> VehicleState:
    return VehicleState(
        position_eci=np.array([config.EARTH_RADIUS_M + 50_000.0, 0.0, 0.0]),
        velocity_eci=np.array([0.0, 500.0, 0.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity_body=np.zeros(3),
        mass_kg=300_000.0,
        time_s=12.34,
    )


class TestHealthStatusString:
    """HealthMonitor.status must map the worst channel to the schema string."""

    def test_nominal(self):
        hm = HealthMonitor()
        hm.update(_nominal_cov(), dynamic_pressure_pa=0.0, propellant_remaining_kg=100.0, propellant_initial_kg=100.0)
        assert hm.status == HEALTH_NOMINAL

    def test_warning_on_high_q(self):
        hm = HealthMonitor()
        # 85% of the q limit is between the 80% WARNING and 95% ALERT thresholds.
        hm.update(
            _nominal_cov(),
            dynamic_pressure_pa=0.85 * config.MAX_Q_PA,
            propellant_remaining_kg=100.0,
            propellant_initial_kg=100.0,
        )
        assert hm.status == HEALTH_WARNING

    def test_critical_on_exceeded_q(self):
        hm = HealthMonitor()
        hm.update(
            _nominal_cov(),
            dynamic_pressure_pa=1.5 * config.MAX_Q_PA,
            propellant_remaining_kg=100.0,
            propellant_initial_kg=100.0,
        )
        assert hm.status == HEALTH_CRITICAL


class TestRecorderSurfacesHealth:
    """The recorder must record the real health status, not a hardcoded default."""

    def test_frame_reflects_warning(self):
        hm = HealthMonitor()
        hm.update(
            _nominal_cov(),
            dynamic_pressure_pa=0.85 * config.MAX_Q_PA,
            propellant_remaining_kg=100.0,
            propellant_initial_kg=100.0,
        )
        assert hm.status == HEALTH_WARNING  # precondition

        rec = TelemetryRecorder()
        state = _state()
        rec.record(
            true_state=state,
            estimated_state=state,
            health_monitor=hm,
            boundary_enforcer=BoundaryEnforcer(),
            time_s=state.time_s,
            sim_context={},
        )
        assert rec.internal_frames[-1].health_status == HEALTH_WARNING

    def test_frame_nominal_when_healthy(self):
        hm = HealthMonitor()
        hm.update(_nominal_cov(), dynamic_pressure_pa=0.0, propellant_remaining_kg=100.0, propellant_initial_kg=100.0)

        rec = TelemetryRecorder()
        state = _state()
        rec.record(
            true_state=state,
            estimated_state=state,
            health_monitor=hm,
            boundary_enforcer=BoundaryEnforcer(),
            time_s=state.time_s,
            sim_context={},
        )
        assert rec.internal_frames[-1].health_status == HEALTH_NOMINAL
