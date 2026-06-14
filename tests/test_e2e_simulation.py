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
