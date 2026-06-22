"""End-to-end simulation regression tests.

Closes the long-standing coverage gap (Q-04): the full
ignition -> ascent -> staging -> LEO-insertion pipeline had no automated test,
so a regression spanning physics/GNC/safety/telemetry could pass CI. These run
the real ``run_simulation`` and assert outcome and physical invariants.

``config_override={}`` runs the in-process (Monte-Carlo-style) path, which skips
file and plot I/O but exercises the identical physics loop and is deterministic
for a fixed seed.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from sim import config
from sim.main import run_simulation


class TestEndToEndAscent:
    def test_nominal_reaches_orbit(self):
        """A full nominal ascent inserts into a sustainable LEO within limits."""
        result = run_simulation(config_override={}, quiet=True)

        assert result.outcome == "SUCCESS", f"expected orbit, got {result.outcome}"
        # Insertion near the 400 km / 7.67 km/s target.
        assert (
            config.INSERTION_MIN_ALTITUDE_FRAC * config.TARGET_ALTITUDE_M
            <= result.insertion_altitude_m
            <= 1.10 * config.TARGET_ALTITUDE_M
        )
        assert result.insertion_velocity_ms >= config.INSERTION_MIN_VELOCITY_FRAC * config.TARGET_VELOCITY_MS
        # Structural limits respected (the boundary enforcer / throttle managers
        # keep peak loads under the limits).
        assert result.peak_q_pa < config.MAX_Q_PA
        assert result.peak_axial_g <= config.MAX_AXIAL_G * 1.001
        # No abort on the nominal path.
        assert result.fts_trigger_time_s is None
        assert result.boundary_clamp_count >= 0

    def test_run_is_deterministic_for_fixed_seed(self, monkeypatch):
        """Identical seed -> identical result (guards the seeding/RNG plumbing)."""
        monkeypatch.setattr(config, "T_MAX", 30.0)
        a = run_simulation(config_override={}, quiet=True)
        b = run_simulation(config_override={}, quiet=True)
        assert asdict(a) == asdict(b)

    def test_pipeline_runs_without_flex_or_slosh(self, monkeypatch):
        """The --no-flex/--no-slosh path runs end to end and ascends."""
        monkeypatch.setattr(config, "T_MAX", 60.0)
        result = run_simulation(
            config_override={"FLEX_ENABLED": False, "SLOSH_ENABLED": False},
            quiet=True,
        )
        assert result.total_time_s > 0.0
        assert result.peak_q_pa > 0.0  # accelerated through the atmosphere
        assert result.peak_axial_g <= config.MAX_AXIAL_G * 1.001

    def test_dispersed_run_completes(self, monkeypatch):
        """A dispersed run (the Monte Carlo path) returns a valid outcome."""
        monkeypatch.setattr(config, "T_MAX", 60.0)
        import numpy as np

        from sim.montecarlo.dispersions import DEFAULT_DISPERSIONS, generate_dispersed_config

        override = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(7))
        override["_seed"] = 7
        override["_run_index"] = 0
        result = run_simulation(config_override=override, quiet=True)
        assert result.outcome in {"SUCCESS", "TIMEOUT", "FTS_ABORT"}
        assert not result.outcome.startswith("ERROR")  # AD-18: no scale<0 crashes


class TestEstimatedAttitudeClosedLoop:
    """ADR 0020 Stage 2: ``USE_ESTIMATED_ATTITUDE`` defaults to True, so guidance,
    the attitude controller, and the FTS fly on the EKF's *estimated* attitude.

    This is a fast, deterministic regression guard for the Stage-2 property that
    closing the loop on the estimate does not regress the mission: on a dispersed
    seed that flies cleanly on the true attitude, the estimated-attitude run must
    still insert to orbit without an FTS trip. The statistical abort-rate /
    false-trip analysis over the full paired (true- vs estimated-attitude)
    campaign is recorded in ADR 0020 — over 24 paired seeds the two authorities
    matched within Monte-Carlo noise (equal success rate; a few marginal cases
    near the FTS covariance limit flip in both directions). The seeds below are
    pinned true-attitude successes, so a regression in estimated-attitude control
    surfaces here as a failure.
    """

    # Dispersed seeds that fly cleanly on both authorities (campaign-verified).
    CLEAN_SEEDS = (47, 51)

    @staticmethod
    def _dispersed_override(seed: int) -> dict:
        import numpy as np

        from sim.montecarlo.dispersions import DEFAULT_DISPERSIONS, generate_dispersed_config

        override = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(seed))
        override["_seed"] = seed
        override["_run_index"] = 0
        override["USE_ESTIMATED_ATTITUDE"] = True  # explicit; this is the Stage-2 default
        return override

    @pytest.mark.parametrize("seed", CLEAN_SEEDS)
    def test_estimated_attitude_inserts_to_orbit_under_dispersion(self, seed):
        """Flying the closed loop on the estimated attitude still reaches orbit
        on a dispersed seed (the core Stage-2 capability), with no FTS trip."""
        result = run_simulation(config_override=self._dispersed_override(seed), quiet=True)
        assert result.outcome == "SUCCESS", f"seed {seed}: estimated-attitude run got {result.outcome}"
        assert result.insertion_altitude_m >= config.INSERTION_MIN_ALTITUDE_FRAC * config.TARGET_ALTITUDE_M
        assert result.insertion_velocity_ms >= config.INSERTION_MIN_VELOCITY_FRAC * config.TARGET_VELOCITY_MS
        assert result.fts_trigger_time_s is None  # no false trip


class TestPerformanceMargin:
    """ADR 0021 (BACKLOG N-01): the Stage-2 dispersed abort rate was dominated by
    the upper stage reaching insertion just short on performance under adverse
    propulsion/drag dispersion — the tank emptied a few tens of m/s shy of orbit
    and the unpowered vehicle tumbled into an FTS attitude trip. The margin was
    added via S2 Isp (348 → 356 s), not propellant: a propellant increase changes
    liftoff mass and the nominal trajectory is chaotically sensitive to it
    (flips SUCCESS/abort across small mass steps), whereas the Isp lever leaves
    liftoff mass — and the nominal — intact.

    This guards that margin: a dispersed seed that previously fell short of orbit
    now reaches it. If the S2 Isp default is cut back, this fails.
    """

    def test_former_performance_shortfall_seed_now_reaches_orbit(self):
        import numpy as np

        from sim.montecarlo.dispersions import DEFAULT_DISPERSIONS, generate_dispersed_config

        seed = 49  # pre-margin this reached insertion just short of orbit and tumbled
        override = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(seed))
        override["_seed"] = seed
        override["_run_index"] = 0
        result = run_simulation(config_override=override, quiet=True)
        assert result.outcome == "SUCCESS", f"seed {seed}: expected orbit with margin, got {result.outcome}"
        assert result.fts_trigger_time_s is None
        assert result.insertion_velocity_ms >= config.INSERTION_MIN_VELOCITY_FRAC * config.TARGET_VELOCITY_MS


class TestFlexControlStructureInteraction:
    """AD-04: the flex model is now live in the control loop, stabilised by a
    frequency-scheduled structural notch filter.

    Previously the bending rate was added to the gyro *after* the EKF/controller
    had already read it, so the whole flex subsystem was a no-op (flex-on
    produced byte-identical telemetry to flex-off). It now couples through the
    controller's measured rate feedback; the notch keeps that coupling from
    fluttering the vehicle.
    """

    def test_flex_is_live_and_notch_stabilizes_it(self, monkeypatch):
        # A truncated ascent through max-q is enough: the lightly-damped first
        # bending mode flutters during atmospheric flight when unfiltered.
        monkeypatch.setattr(config, "T_MAX", 100.0)

        flex_off = run_simulation(config_override={"FLEX_ENABLED": False}, quiet=True)
        flex_on_notch = run_simulation(
            config_override={"FLEX_ENABLED": True, "FLEX_NOTCH_ENABLED": True},
            quiet=True,
        )
        flex_on_no_notch = run_simulation(
            config_override={"FLEX_ENABLED": True, "FLEX_NOTCH_ENABLED": False},
            quiet=True,
        )

        base = flex_off.boundary_clamp_count
        # Coupling is live: without the notch, the bending mode floods the TVC
        # with flutter-driven clamp events — impossible on the old inert wiring,
        # where flex-on was identical to flex-off.
        assert flex_on_no_notch.boundary_clamp_count > 10 * (base + 50)
        # The notch suppresses that flutter back toward the flex-off baseline.
        assert flex_on_notch.boundary_clamp_count < flex_on_no_notch.boundary_clamp_count / 10

    def test_live_flex_does_not_regress_nominal(self, monkeypatch):
        """With the notch, the live flex coupling still inserts into orbit."""
        monkeypatch.setattr(config, "T_MAX", 600.0)
        result = run_simulation(
            config_override={"FLEX_ENABLED": True, "FLEX_NOTCH_ENABLED": True},
            quiet=True,
        )
        assert result.outcome == "SUCCESS"
        assert result.fts_trigger_time_s is None
