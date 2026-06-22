"""Regression tests for the 2026-06-14 fidelity / model-quality fixes.

Each test pins the corrected behaviour described in
``audit/04-adversarial-findings.md`` so the defect cannot silently return.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from sim import config


# --------------------------------------------------------------------------- #
# AD-03 / AD-11 — J3/J5 zonal gravity verified against the geopotential gradient
# --------------------------------------------------------------------------- #
class TestGravityZonalHarmonics:
    JS = {
        2: config.EARTH_J2,
        3: config.EARTH_J3,
        4: config.EARTH_J4,
        5: config.EARTH_J5,
        6: config.EARTH_J6,
    }

    @staticmethod
    def _legendre(n: int, s: float) -> float:
        return {
            2: (3 * s**2 - 1) / 2,
            3: (5 * s**3 - 3 * s) / 2,
            4: (35 * s**4 - 30 * s**2 + 3) / 8,
            5: (63 * s**5 - 70 * s**3 + 15 * s) / 8,
            6: (231 * s**6 - 315 * s**4 + 105 * s**2 - 5) / 16,
        }[n]

    def _potential(self, rv: np.ndarray) -> float:
        mu, re = config.EARTH_MU, config.EARTH_RADIUS_M
        r = np.linalg.norm(rv)
        s = rv[2] / r
        return (mu / r) * (1.0 - sum(jn * (re / r) ** n * self._legendre(n, s) for n, jn in self.JS.items()))

    def _num_grad(self, rv: np.ndarray, h: float = 2.0) -> np.ndarray:
        g = np.zeros(3)
        for i in range(3):
            a, b = rv.copy(), rv.copy()
            a[i] += h
            b[i] -= h
            g[i] = (self._potential(a) - self._potential(b)) / (2 * h)
        return g

    @pytest.mark.parametrize("lat_deg", [0.0, 28.5, 45.0, 60.0, 89.0])
    def test_matches_numerical_gradient(self, lat_deg):
        from sim.environment.gravity import gravitational_acceleration

        lat = math.radians(lat_deg)
        r = config.EARTH_RADIUS_M + 400_000.0
        rv = np.array([r * math.cos(lat), 0.0, r * math.sin(lat)])
        code = gravitational_acceleration(rv)
        ref = self._num_grad(rv)
        relerr = np.linalg.norm(code - ref) / np.linalg.norm(ref)
        assert relerr < 1e-6, f"gravity off by {relerr:.2e} at {lat_deg} deg"

    def test_odd_zonals_are_live(self):
        """J3/J5 must actually contribute (were ~1/r too small, ~zero before)."""
        from sim.environment import gravity

        lat = math.radians(45.0)
        r = config.EARTH_RADIUS_M + 400_000.0
        rv = np.array([r * math.cos(lat), 0.0, r * math.sin(lat)])
        full = gravity.gravitational_acceleration(rv)
        saved = (config.EARTH_J3, config.EARTH_J5)
        config.EARTH_J3, config.EARTH_J5 = 0.0, 0.0
        try:
            no_odd = gravity.gravitational_acceleration(rv)
        finally:
            config.EARTH_J3, config.EARTH_J5 = saved
        # The J3+J5 contribution should be ~1e-5 m/s^2, not ~1e-12.
        assert np.linalg.norm(full - no_odd) > 1e-6


# --------------------------------------------------------------------------- #
# AD-02 — TVC actuator settles (no rate-limited limit cycle)
# --------------------------------------------------------------------------- #
class TestActuatorStability:
    def test_step_response_settles(self):
        from sim.vehicle.actuator import TVCActuator

        act = TVCActuator()
        positions = [act.update(2.0, 0.01) for _ in range(400)]
        steady = positions[-40:]
        peak_to_peak = max(steady) - min(steady)
        assert peak_to_peak < 0.02, f"actuator limit-cycles (p2p={peak_to_peak:.3f} deg)"
        assert abs(positions[-1] - 2.0) < 0.02, "actuator did not converge to command"

    def test_overshoot_is_bounded(self):
        """zeta=0.7 second-order response: modest overshoot, not divergence."""
        from sim.vehicle.actuator import TVCActuator

        act = TVCActuator()
        positions = [act.update(1.0, 0.01) for _ in range(400)]
        assert max(positions) < 1.3  # ~5% overshoot expected, certainly < 30%


# --------------------------------------------------------------------------- #
# AD-05 — gain schedule continuous across q = 100 Pa
# --------------------------------------------------------------------------- #
class TestGainScheduleContinuity:
    def test_no_step_at_100pa(self):
        from sim.gnc.control import AttitudeController

        c = AttitudeController()
        c._schedule_gains(99.999, config.CONTROL_MASS_REF_KG)
        kp_below = c._kp
        c._schedule_gains(100.001, config.CONTROL_MASS_REF_KG)
        kp_above = c._kp
        assert abs(kp_above - kp_below) / kp_below < 0.005, "gain step at 100 Pa"

    def test_monotonic_through_crossing(self):
        from sim.gnc.control import AttitudeController

        c = AttitudeController()
        kps = []
        for q in [50, 90, 100, 110, 200, 500]:
            c._schedule_gains(float(q), config.CONTROL_MASS_REF_KG)
            kps.append(c._kp)
        # Boosted/flat at low q then decreasing — no jump up crossing 100 Pa.
        assert kps[3] <= kps[2] + 1e-9


# --------------------------------------------------------------------------- #
# AD-12 — propellant mass flow conserved across ambient pressure
# --------------------------------------------------------------------------- #
class TestPropulsionMassFlow:
    def test_mdot_independent_of_pressure(self):
        from sim.vehicle.propulsion import EngineModel
        from sim.vehicle.vehicle import Vehicle

        eng = EngineModel(Vehicle().current_stage)
        eng.ignite()
        for _ in range(120):  # clear the ignition ramp
            eng.update(0.01, 101325.0)
        _, mdot_sl = eng.update(0.01, 101325.0)
        _, mdot_vac = eng.update(0.01, 0.0)
        assert abs(mdot_sl - mdot_vac) / mdot_vac < 1e-9, "mdot drifts with ambient pressure"


# --------------------------------------------------------------------------- #
# AD-07 — correction budget counts the altitude raise for elliptical orbits
# --------------------------------------------------------------------------- #
class TestCorrectionBudget:
    def test_elliptical_includes_altitude_raise(self):
        from sim.config import EARTH_MU, EARTH_RADIUS_M
        from sim.orbital.maneuvers import total_correction_budget
        from sim.orbital.propagator import OrbitalElements

        peri = EARTH_RADIUS_M + 142_000
        apo = EARTH_RADIUS_M + 658_000
        a = (peri + apo) / 2
        e = (apo - peri) / (apo + peri)
        el = OrbitalElements(
            semi_major_axis_m=a,
            eccentricity=e,
            inclination_deg=51.6,
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            true_anomaly_deg=0.0,
            period_s=2 * math.pi * math.sqrt(a**3 / EARTH_MU),
            apoapsis_alt_km=658.0,
            periapsis_alt_km=142.0,
        )
        budget = total_correction_budget(el, 400_000, 51.6)  # inclination matched
        # Must reflect circularise-at-periapsis + Hohmann 142->400 km (~298 m/s),
        # not the ~147 m/s the SMA-reference bug produced.
        assert budget > 250.0


# --------------------------------------------------------------------------- #
# AD-08 — coast (zero throttle, no propellant) is not a boundary violation
# --------------------------------------------------------------------------- #
class TestBoundaryCoast:
    def test_zero_throttle_coast_not_counted(self):
        from sim.safety.boundary_enforcer import BoundaryEnforcer

        enf = BoundaryEnforcer()
        for _ in range(500):
            enf.validate_throttle(0.0, 0.0)
        assert enf.violation_count == 0

    def test_positive_throttle_no_propellant_is_violation(self):
        from sim.safety.boundary_enforcer import BoundaryEnforcer

        enf = BoundaryEnforcer()
        res = enf.validate_throttle(0.6, 0.0)
        assert enf.violation_count == 1
        assert res.value == 0.0

    def test_out_of_range_throttle_when_depleted_is_flagged(self):
        # A malformed command (negative or >1) while depleted is still forced
        # to 0.0 and flagged as out-of-range, so it cannot mask an upstream
        # fault during coast (Copilot review on PR #54).
        from sim.safety.boundary_enforcer import BoundaryEnforcer

        enf = BoundaryEnforcer()

        res = enf.validate_throttle(-0.5, 0.0)
        assert res.value == 0.0
        assert res.was_clamped is True
        assert res.violation_type == "throttle_out_of_range"
        assert enf.violation_count == 1

        res = enf.validate_throttle(1.5, 0.0)
        assert res.value == 0.0
        assert res.violation_type == "throttle_out_of_range"
        assert enf.violation_count == 2


# --------------------------------------------------------------------------- #
# AD-09 — staging SEPARATION abort recovers instead of looping forever
# --------------------------------------------------------------------------- #
class TestStagingAbortRecovery:
    def test_abort_does_not_latch_in_separation(self):
        from sim.vehicle.propulsion import EngineModel
        from sim.vehicle.staging import StagingPhase, StagingSequencer
        from sim.vehicle.vehicle import Vehicle

        veh = Vehicle()
        s1 = EngineModel(veh.current_stage)
        s2 = EngineModel(veh.stages[1])
        seq = StagingSequencer(veh, s1, s2)
        # Force SEPARATION with the engine still at full thrust (interlock fails).
        s1.ignite()
        for _ in range(120):
            s1.update(0.01, 0.0)  # ramp to full
        seq._phase = StagingPhase.SEPARATION
        event = seq.update(0.01)
        assert "ABORT" in event
        # Must have left SEPARATION (re-entered TAIL_OFF) and commanded shutdown,
        # rather than latching and emitting ABORT forever.
        assert seq.phase is StagingPhase.TAIL_OFF


# --------------------------------------------------------------------------- #
# ADR 0025 — post-separation ullage-settling coast (cold staging)
# --------------------------------------------------------------------------- #
class TestPostSeparationCoast:
    """S2 is not lit at the instant of separation: the vehicle coasts
    ~POST_SEP_COAST_DURATION (ullage settling / stage clearance) first."""

    @staticmethod
    def _seq_at_separation():
        from sim.vehicle.propulsion import EngineModel
        from sim.vehicle.staging import StagingPhase, StagingSequencer
        from sim.vehicle.vehicle import Vehicle

        veh = Vehicle()
        s1 = EngineModel(veh.current_stage)  # never ignited → thrust below interlock
        s2 = EngineModel(veh.stages[1])
        seq = StagingSequencer(veh, s1, s2)
        seq._phase = StagingPhase.SEPARATION
        return seq, s2

    def test_s2_not_ignited_at_separation(self):
        from sim.vehicle.staging import StagingPhase

        seq, s2 = self._seq_at_separation()
        seq.update(0.01)  # process SEPARATION
        assert seq.phase is StagingPhase.SETTLING
        assert not s2.is_ignited  # cold staging — ignition deferred

    def test_s2_ignites_only_after_settling_coast(self):
        import sim.vehicle.staging as st
        from sim.vehicle.staging import StagingPhase

        seq, s2 = self._seq_at_separation()
        seq.update(0.01)  # -> SETTLING
        for _ in range(int(0.5 * st.POST_SEP_COAST_DURATION / 0.01)):
            seq.update(0.01)
        assert not s2.is_ignited  # still settling, halfway through the coast
        for _ in range(int(st.POST_SEP_COAST_DURATION / 0.01) + 2):
            seq.update(0.01)
        assert s2.is_ignited
        assert seq.phase in (StagingPhase.S2_IGNITION, StagingPhase.COMPLETE)


# --------------------------------------------------------------------------- #
# AD-13 — cop_com_margin sign convention
# --------------------------------------------------------------------------- #
class TestCopComMargin:
    def test_positive_when_cop_aft_of_com(self):
        from sim.vehicle.aerodynamics import AerodynamicsModel

        aero = AerodynamicsModel()
        cop = aero.cop_offset_from_nose
        # CoM forward of CoP (smaller from-nose) => statically stable => positive.
        assert aero.cop_com_margin(cop - 2.0) > 0
        # CoM aft of CoP => unstable => negative.
        assert aero.cop_com_margin(cop + 2.0) < 0


# --------------------------------------------------------------------------- #
# AD-14 — compute_statistics tolerates an empty result list
# --------------------------------------------------------------------------- #
class TestStatisticsEmpty:
    def test_empty_results_no_crash(self):
        from sim.montecarlo.statistics import compute_statistics

        stats = compute_statistics([])
        assert stats["total_runs"] == 0
        assert stats["limit_proximity"]["peak_q_pct"]["max"] == 0.0
        assert stats["boundary_clamps"]["max"] == 0


# --------------------------------------------------------------------------- #
# AD-16 — eci_to_ned accounts for Earth rotation (and takes time_s)
# --------------------------------------------------------------------------- #
class TestEciToNed:
    def test_zero_inertial_velocity_sees_earth_rotation(self):
        from sim.core.reference_frames import eci_to_ned

        # A point on the equator with zero ECI velocity is moving west in NED at
        # the surface speed (it is the rotating frame catching up). With the old
        # transport-term-free code the result was exactly zero.
        r = config.EARTH_RADIUS_M
        pos = np.array([r, 0.0, 0.0])
        ned = eci_to_ned(pos, np.zeros(3), 0.0, 0.0, 0.0)
        v_surface = config.EARTH_OMEGA * r
        assert abs(abs(ned[1]) - v_surface) < 1e-6  # East component magnitude
        assert np.linalg.norm(ned) > 1.0


# --------------------------------------------------------------------------- #
# AD-06 / AD-15 — telemetry: FTS source and t=0 downlink frame
# --------------------------------------------------------------------------- #
def _state(t=0.0):
    from sim.core.state import VehicleState

    return VehicleState(
        position_eci=np.array([config.EARTH_RADIUS_M, 0.0, 0.0]),
        velocity_eci=np.array([0.0, 465.0, 0.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity_body=np.zeros(3),
        mass_kg=500_000.0,
        time_s=t,
    )


class TestTelemetryFixes:
    def _record_run(self, fts):
        from sim.safety.boundary_enforcer import BoundaryEnforcer
        from sim.safety.health_monitor import HealthMonitor
        from sim.telemetry.recorder import TelemetryRecorder

        rec = TelemetryRecorder()
        hm = HealthMonitor()
        be = BoundaryEnforcer()
        ctx = {"throttle": 1.0, "thrust_n": 1e6, "dynamic_pressure_pa": 0.0, "stage": 1}
        for i in range(25):
            st = _state(i * 0.01)
            rec.record(
                true_state=st,
                estimated_state=st,
                health_monitor=hm,
                boundary_enforcer=be,
                time_s=i * 0.01,
                sim_context=ctx,
                fts=fts,
            )
        return rec

    def test_downlink_includes_t0(self):
        rec = self._record_run(fts=None)
        assert rec.downlink_frames, "no downlink frames recorded"
        assert rec.downlink_frames[0].time_s == 0.0

    def test_fts_triggered_reads_fts_object(self):
        rec = self._record_run(fts=SimpleNamespace(fts_triggered=True))
        assert all(f.fts_triggered for f in rec.internal_frames)

    def test_fts_false_without_trigger(self):
        rec = self._record_run(fts=SimpleNamespace(fts_triggered=False))
        assert not any(f.fts_triggered for f in rec.internal_frames)
