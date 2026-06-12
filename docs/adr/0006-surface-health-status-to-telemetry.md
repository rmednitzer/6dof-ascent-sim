# 0006. Surface health status to telemetry

- Status: accepted
- Date: 2026-06-12
- Deciders: audit (2026-06-12)

## Context and Problem Statement

`HealthMonitor` computes a multi-channel health assessment every timestep, but
the telemetry recorder read `getattr(health_monitor, "status", "NOMINAL")` — and
`HealthMonitor` exposed no `status` attribute. Result: telemetry `health_status`
was permanently `"NOMINAL"`, confirmed across all 48,816 frames of a nominal run
whose peak dynamic pressure reached 92.6% of the structural limit (the WARNING
threshold is 80%). The health assessment was silently discarded (finding Q-01).

## Considered Options

1. Add a `status` property to `HealthMonitor` returning `overall_status().name`.
2. Change the recorder to call `health_monitor.overall_status().name` directly.
3. Pass a precomputed status string into `record()`.

## Decision Outcome

Option 1. It satisfies the recorder's already-documented interface
("Object exposing `status` (str)") with zero changes to any call site, the
smallest blast radius, and keeps the recorder decoupled from the health enum.

## Consequences

- Positive: telemetry now reflects WARNING/ALERT/CRITICAL; the existing
  `getattr` default still protects against a non-health object being passed.
- Behavioral change: internal telemetry frames during high-q now carry WARNING,
  so the per-run telemetry SHA-256 hash changes. `health_status_final` is
  unchanged for a nominal run (q -> 0 at insertion). No test pinned the hash, so
  nothing regresses; a new test (`tests/test_health_telemetry.py`) pins the
  behavior.
- Out of scope: the `engine_health` and `sensor_status` channels are still not
  fed real data by the main loop (finding Q-03); they remain NOMINAL until the
  loop passes thrust/sensor data. Tracked separately.

## Notes / Evidence

Implemented in commit "fix(telemetry): surface real health status...". Suite went
202 -> 207 passed, 0 failed.
