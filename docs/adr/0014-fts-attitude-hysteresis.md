# 0014. Hysteresis on the FTS attitude criterion

- Status: accepted
- Date: 2026-06-21
- Deciders: GNC/safety eng (cross-repo lessons pass, `audit/05`)

## Context and Problem Statement

The Flight Termination System (`sim/safety/fts.py`) triggers if the thrust-axis
pointing error exceeds `FTS_ATTITUDE_LIMIT_DEG = 25°` on **any single
timestep**. The adversarial audit (`audit/04`, finding **AD-19**, tracked in
`BACKLOG.md`) found this is the dominant Monte-Carlo failure mode: ~19% of
dispersed runs FTS-abort, and *every observed dispersed abort is a marginal
~25.0x° thrust-axis hit* right at the limit during the PEG terminal phase near
insertion. The nominal run tracks to <5°, so the 25° limit itself is correct; the
problem is that a single marginal-frame excursion latches an irreversible abort.

PX4's failure detector (`audit/05`, finding A3.1/A3.3) gates every failure
condition through a `Hysteresis` (persistence) timer for exactly this reason.

## Decision

Apply a persistence (debounce) timer to the **attitude criterion only**: the
thrust-axis error must exceed `FTS_ATTITUDE_LIMIT_DEG` *continuously* for
`FTS_ATTITUDE_HYSTERESIS_S` (default `0.2 s`, ~20 frames at 100 Hz) before it
contributes an abort. The persistence clock resets whenever the error returns
within the limit. The cross-range, EKF-covariance, and structural criteria remain
instantaneous. Setting `FTS_ATTITUDE_HYSTERESIS_S = 0.0` recovers the original
instantaneous behaviour exactly (on the first violating frame, elapsed
`0.0 ≥ 0.0`).

This is an **FTS-side mitigation**, not the AD-19 fix. AD-19's root cause is a
PEG terminal-guidance attitude-margin fragility; hardening that (smoother
GT→PEG handover, command-rate vs. low-propellant control authority) remains a
separate control-design task. Hysteresis reduces single-sample false trips so the
FTS attitude check reflects genuine sustained loss of control.

## Consequences

- Positive: filters the marginal single-frame trips that dominate dispersed
  aborts without weakening detection of a real divergence (which persists for
  many frames and still trips after a fixed ~0.2 s delay).
- Positive: nominal flight is unaffected (it never approaches the limit), so
  nominal telemetry and committed artifacts are unchanged.
- Negative / to validate: the exact `FTS_ATTITUDE_HYSTERESIS_S` value should be
  confirmed against a dispersed Monte-Carlo sweep; too long a window could delay
  a genuine abort. The default is deliberately short. The dispersed abort-rate
  change is a robustness *metric* shift, not a nominal-physics change.
- The FTS now holds a small amount of per-run state (`_attitude_violation_start_s`);
  it is reset implicitly by constructing a fresh FTS per run (as the main loop and
  Monte-Carlo dispatcher already do).

## Notes / Evidence

`sim/safety/fts.py:evaluate` (attitude block) and `__init__`; config
`FTS_ATTITUDE_HYSTERESIS_S`. Regression tests in `tests/test_fts.py`
(`TestFTSAttitude`): a single marginal frame does not trip, a sustained
violation trips after the hysteresis window, an intermittent (resetting)
violation never accumulates a trip, a gross sustained excursion still trips, and
`FTS_ATTITUDE_HYSTERESIS_S = 0.0` reproduces instantaneous triggering. Cross-repo
rationale: `audit/05-cross-repo-lessons.md` §A3.1.
