# Backlog

Deferred items from the 2026-06-12 audit (`audit/02-security-findings.md`) and
the 2026-06-14 adversarial physics/numerics/GNC re-audit
(`audit/04-adversarial-findings.md`). Each links to its findings register and,
where relevant, an ADR (`docs/adr/`). Ordered by severity then effort within
each section.

Effort: S (hours) / M (about a day) / L (multi-day or structural).

## Physics / Numerics / GNC (adversarial audit, 2026-06-14)

All verified with repros in `audit/04-adversarial-findings.md`. AD-01 was fixed
on this branch; the rest change nominal physics/telemetry (so they require
regenerating committed example outputs) or need a design decision, and are
deferred to focused PRs.

| ID | Title | Sev | Effort | Approach | Owner role |
|----|-------|-----|--------|----------|------------|
| [AD-02](audit/04-adversarial-findings.md) | TVC actuator unstable at 100 Hz (limit cycle) | high | S | Sub-step the actuator ODE (`ωₙ·dt_sub ≲ 0.3`) or use exact ZOH `expm` discretization; regenerate examples. | GNC/sim eng |
| [AD-03](audit/04-adversarial-findings.md) | J3 gravity term suppressed ~1/r (effectively zero) | medium | S | Delete the extra `* r_inv` on `gravity.py:75-77`; add a term-isolated test vs numerical gradient. Overturns the prior "dismissed" J3 note. | Astrodynamics eng |
| [AD-04](audit/04-adversarial-findings.md) | Flex-body model is inert (gyro coupling dead) | medium | M | Feed flex gyro contamination before EKF/controller read it **and** pass a realistic `modal_mass_kg`; add a flex-on≠flex-off test. | GNC/sim eng |
| [AD-05](audit/04-adversarial-findings.md) | Gain-schedule 2× discontinuity at q=100 Pa | medium | S | Make `q_factor` continuous across 100 Pa; add a continuity test. | GNC eng |
| [AD-06](audit/04-adversarial-findings.md) | `fts_triggered` always False in telemetry | medium | S | Thread the FTS instance into the recorder; read `fts.fts_triggered`; regression test (mirror of Q-01). | Telemetry eng |
| [AD-07](audit/04-adversarial-findings.md) | Correction budget underestimates elliptical orbits ~50% | medium | S | Use periapsis radius as the Hohmann reference; add an elliptical-orbit test. | Astrodynamics eng |
| [AD-08](audit/04-adversarial-findings.md) | `validate_throttle` counts coast ticks as violations | medium | S | Only count a violation when `throttle_cmd > 0` with no propellant. | Safety eng |
| [AD-09](audit/04-adversarial-findings.md) | Staging SEPARATION abort = infinite loop | medium | M | Decide recovery semantics (re-enter TAIL_OFF + force shutdown, or latch fault) then implement + test. | Sim eng |
| [AD-10](audit/04-adversarial-findings.md) | PEG uses unclamped `T` after clamping `ratio` | medium | S | Use `T_eff = ratio·tau` consistently, or hold coefficients when `T ≥ 0.95·tau`. | GNC eng |
| [AD-11](audit/04-adversarial-findings.md) | J5 gravity term suppressed ~1/r | low | S | Delete the extra `* r_inv` on `gravity.py:88-89` (with AD-03). | Astrodynamics eng |
| [AD-12](audit/04-adversarial-findings.md) | Propulsion mdot not conserved across pressure (1.35%) | low | S | Derive Isp/thrust interpolation so `F/(Isp·g0)` is constant at fixed throttle. | Propulsion eng |
| [AD-13](audit/04-adversarial-findings.md) | `cop_com_margin` inverted polarity (unused) | low | S | Flip the subtraction or fix the docstring. | Aero eng |
| [AD-14](audit/04-adversarial-findings.md) | `compute_statistics([])` crashes | low | S | Extend the `n > 0` guard to all reductions. | Sim eng |
| [AD-15](audit/04-adversarial-findings.md) | Downlink telemetry omits t=0 frame | low | S | Test the decimation modulo before incrementing the counter. | Telemetry eng |
| [AD-16](audit/04-adversarial-findings.md) | `eci_to_ned` ignores ECI→ECEF rotation (unused) | low | S | Add `time_s`, rotate velocity to ECEF (minus transport term) before NED. | Sim eng |
| [AD-17](audit/04-adversarial-findings.md) | Launch azimuth ignores Earth-rotation inclination term | info | M | Correct targeting for launch-site eastward velocity, or document as an accepted simplification. | GNC eng |

## Security

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| [S-04](audit/02-security-findings.md) | Adopt a dependency lockfile | low | M | Floors-only deps + no lock means non-reproducible installs and a no-op Renovate `lockFileMaintenance`. Generate a hashed `requirements.lock` (pip-compile), install it in CI, let Renovate bump it. See [ADR 0007](docs/adr/0007-adopt-dependency-lockfile.md). | — | Build/release eng |
| [S-06](audit/02-security-findings.md) | Make `lockFileMaintenance` effective | info | S | Renovate is configured to maintain a lock that does not exist. Resolved automatically once S-04 lands. | S-04 | Build/release eng |
| [S-07](audit/02-security-findings.md) | Confirm or add CodeQL scanning | info | S | Commit `9a2e67e` mentions CodeQL but there is no workflow file; default-setup status is not visible in-repo. Either commit a `codeql.yml` (SHA-pinned) or document that default setup is enabled in repo settings. | — | Security eng |

## Reliability

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| [Q-03](audit/02-security-findings.md) | Feed engine/sensor health channels in the main loop | low | M | `HealthMonitor.update` is called without `commanded_thrust_n`/`actual_thrust_n` or sensor flags (`main.py:438`), so `engine_health` and `sensor_status` are always NOMINAL. Thread thrust and sensor-degradation data into the call; extend the Q-01 test to cover them. | — | GNC/sim eng |
| [Q-05](audit/02-security-findings.md) | Replace PEG `except UnboundLocalError` control flow | low | S | `guidance.py:463-467` retains stale PEG coefficients via exception control flow with no signal when the iteration degenerates. Initialize `A`/`B` from the stored values before the loop and drop the try/except; optionally count non-convergence for telemetry. Behavior-preserving; add a unit test. | Q-04 (test harness) | GNC eng |

## Quality

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| [Q-04](audit/02-security-findings.md) | Add an end-to-end simulation regression test | medium | M | `run_simulation` (15%) and `recorder` (28%) lack any full-pipeline test. Add a fast, seeded nominal run asserting outcome/peak-load/telemetry invariants. See [ADR 0008](docs/adr/0008-end-to-end-simulation-test.md). | — | Sim eng / QA |
| [Q-02](audit/02-security-findings.md) | Remove `_save_config` key-list drift | low | S | The hardcoded save/restore key list (`main.py:86-103`) must track `DEFAULT_DISPERSIONS` by hand. Interim: snapshot exactly the override keys dynamically. Long-term: [ADR 0009](docs/adr/0009-explicit-dispersion-parameters.md). | — | Sim eng |
| [Q-06](audit/02-security-findings.md) | Remove dead `_v_p` in decay.py | info | S | `decay.py:231` computes vis-viva periapsis velocity, suppresses F841, never uses it. The King-Hele formula was verified correct without it; delete the line (or use it intentionally). | — | Orbital eng |
| [Q-07](audit/02-security-findings.md) | Document the yaw sign convention | info | S | `control.py:177` negates the yaw PID output (`cmd_yaw_rad = -(...)`) with no comment. Add a one-line note on the body/TVC sign convention to prevent an accidental flip in future refactors. | — | GNC eng |

## Documentation

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| D-06 | Regenerate committed example outputs after Q-01 | low | S | The Q-01 fix changes internal telemetry during high-q, so `examples/output/mission_summary.txt`'s telemetry hash is now stale (its `health_status_final` line stays NOMINAL and remains correct). Re-run `examples/run_and_visualize.py` and commit refreshed artifacts, as done previously in commit `fea2235`. | Q-01 (done) | Sim eng |

## Tooling

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| [S-05](audit/02-security-findings.md)b | Single-source the ruff version | low | S | CI and pre-commit pinned different ruff versions (now realigned to 0.15.16). Prevent recurrence by deriving CI's ruff from `.pre-commit-config.yaml` (e.g. run `pre-commit run` in CI, or read the rev), so they cannot drift again. See [ADR 0005](docs/adr/0005-quality-tooling-strategy.md). | — | DevEx |
| T-02 | Make lint job match the support matrix | low | S | CI lints on Python 3.14 while the project supports 3.11-3.13. Pin the lint job to a supported version for consistency (ruff is a standalone binary, so impact is low). | — | DevEx |
| T-03 | Decide on mypy adoption | info | M | mypy is installed in the environment but not part of the project toolchain; it surfaces only false positives today (`propagator.py` union-attr). Either adopt it with config + stubs (`numpy` ships types) and wire into CI/pre-commit, or document that it is intentionally out of scope. | — | DevEx |
