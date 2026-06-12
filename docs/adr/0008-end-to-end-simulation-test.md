# 0008. Add an end-to-end simulation regression test

- Status: proposed
- Date: 2026-06-12
- Deciders: maintainers (pending)

## Context and Problem Statement

`run_simulation` (the orchestration in `sim/main.py`, 379 statements) and
`sim/telemetry/recorder.py` are barely covered (15% and 28% at baseline; finding
Q-04). No test runs the full loop and asserts on the outcome, peak loads, or
telemetry artifacts. Regressions in the integration of physics, GNC, safety, and
telemetry could pass CI undetected. The existing suite tests components in
isolation and pins specific prior bugs, but never the whole pipeline.

## Considered Options

1. A fast end-to-end test that runs a shortened or `--no-flex --no-slosh`
   ascent and asserts coarse invariants (outcome `SUCCESS`, periapsis above the
   atmosphere, peak q/G within limits, telemetry files written, hash stable for
   a fixed seed).
2. A characterization test pinning the full telemetry hash for a fixed seed.
3. Leave integration uncovered.

## Decision Outcome (proposed)

Option 1, possibly with a reduced `T_MAX` or a deterministic seed to keep
runtime within a few seconds. Avoid pinning the exact hash (Option 2) initially
because it is brittle to legitimate model changes; assert structural invariants
instead.

## Consequences

- Positive: catches integration regressions; raises `main.py`/`recorder.py`
  coverage; documents the expected nominal outcome.
- Negative: slower than unit tests (a nominal run is ~45 s; a reduced run must be
  tuned). Must inject a fixed RNG seed so the default `secrets`-seeded sensors do
  not make it flaky.

## Notes / Evidence

Baseline coverage table in `audit/01-baseline.md`. The Q-01 fix added the first
recorder test (`tests/test_health_telemetry.py`) but not a full-loop test.
