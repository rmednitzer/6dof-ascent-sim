# Backlog

Deferred items from the 2026-06-12 audit (`audit/02-security-findings.md`) and
the 2026-06-14 adversarial physics/numerics/GNC re-audit
(`audit/04-adversarial-findings.md`). Each links to its findings register and,
where relevant, an ADR (`docs/adr/`). Ordered by severity then effort within
each section.

Effort: S (hours) / M (about a day) / L (multi-day or structural).

## Physics / Numerics / GNC (adversarial audit, 2026-06-14)

All verified with repros in `audit/04-adversarial-findings.md`. **AD-01–AD-16
and AD-18 are fixed** — AD-01/AD-18 and AD-02/03/05–16 in earlier passes (each
with a regression test in `tests/test_fidelity_fixes.py`), and **AD-04 in this
pass**: the flex model is now live in the control loop, stabilised by a
frequency-scheduled structural notch filter (`sim/gnc/notch_filter.py`; see
[ADR 0012](docs/adr/0012-live-flex-structural-notch.md), tests in
`tests/test_notch_filter.py` and `tests/test_e2e_simulation.py`). **AD-17 is
documented as an accepted simplification**: the corrections are physically
correct and reach the target inclination, but they regress Monte-Carlo
robustness via a pre-existing PEG attitude-margin fragility, now tracked as
AD-19.

| ID | Title | Sev | Effort | Approach | Owner role |
|----|-------|-----|--------|----------|------------|
| [AD-17](audit/04-adversarial-findings.md) | Launch azimuth ignores Earth-rotation inclination term | info | M | **Accepted simplification.** The rotating-frame azimuth correction *and* active out-of-plane yaw nulling were both prototyped: they reach the 51.6° target (achieved 45°→51.4°, insertion correction-dv ~995→~266 m/s) but every variant *systematically* increases dispersed FTS aborts (48-seed: 39→35 azimuth-only, worse with nulling) because the PEG terminal phase already rides the 25° attitude limit (AD-19). Re-attempt once AD-19 lands. | GNC eng |
| [AD-19](audit/04-adversarial-findings.md) | PEG terminal phase rides the 25° FTS attitude limit under dispersions | medium | L | Discovered during the AD-17 investigation: ~19% of dispersed runs FTS-abort on a marginal thrust-axis error near insertion (every observed dispersed abort is 25.0x° — right at `FTS_ATTITUDE_LIMIT_DEG`). Harden the PEG terminal transient (smoother GT→PEG handover, command-rate vs low-propellant control authority) so the margin is real; this also unblocks AD-17. | GNC/controls eng |

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
| [Q-05](audit/02-security-findings.md) | Replace PEG `except UnboundLocalError` control flow | low | S | **✅ Resolved (ADR 0015):** `A`/`B` are now seeded from the stored coefficients before the iteration and the `try/except UnboundLocalError` is gone (behaviour-preserving — the nominal golden trajectory is unchanged); surfaced and guarded by the pyright `reportPossiblyUnbound` check. _Original:_ `guidance.py` retained stale PEG coefficients via exception control flow with no signal when the iteration degenerated. | Q-04 (test harness) | GNC eng |
| N-01 | Dispersed FTS aborts near insertion | medium | M | **✅ Resolved (ADR 0021 + 0022): abort rate 33 % → 4.2 % (23/24).** _Corrected diagnosis:_ a per-seed look showed the ~33 % Stage-2 abort rate was **not** nav-covariance driven (only 1 of 8 was a covariance trip). 7 of 8 were thrust-axis attitude trips near insertion: the upper stage under-performed under adverse propulsion/drag dispersion, the S2 tank emptied a few tens of m/s short of orbit, and the **unpowered** vehicle tumbled at its residual ~2.4°/s rate (no thrust ⇒ no TVC) past the 25° FTS limit — a performance-margin problem, consistent with AD-17. _Fix 1 (ADR 0021):_ raised the S2 Isp default 348 → 356 s (`config_schema.py`); the nominal is preserved (Isp, unlike propellant, leaves liftoff mass unchanged — the nominal is chaotically mass-sensitive) and the dispersed abort rate dropped to ~12.5 %, leaving 2 marginal covariance trips + 1 staging transient. _Fix 2 (ADR 0022):_ tightened the implausibly-wide IMU bias dispersion (σ 2× nominal, ~7× tail → ~30 % of nominal, ~2× tail); recovered the IMU-driven covariance trip (seed 42) and the staging transient (seed 54), → 23/24. _Residual (1 seed):_ seed 56 — the EKF position covariance grows unbounded through the ~300 s GPS-denied coast and the longest/most-lofted dispersions still reach the 10 km limit; bounding it needs position aiding above the COCOM ceiling (e.g. ground-station range tracking), a larger effort deferred here. | ADR 0020, 0021, 0022 | GNC eng |

## Quality

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| [Q-04](audit/02-security-findings.md) | Add an end-to-end simulation regression test | medium | M | **✅ Resolved:** `tests/test_e2e_simulation.py` covers the full pipeline with range invariants; `tests/test_golden_trajectory.py` (this PR) adds a pinned golden-summary regression plus an independent `solve_ivp` integrator cross-oracle. See [ADR 0008](docs/adr/0008-end-to-end-simulation-test.md). | — | Sim eng / QA |
| [Q-02](audit/02-security-findings.md) | Remove `_save_config` key-list drift | low | S | **✅ Resolved ([ADR 0016](docs/adr/0016-validated-config-override-schema.md)):** `_save_config` now derives its key list from the single `OVERRIDABLE_PARAM_NAMES` declaration in `sim/config_schema.py`, so it cannot drift from the dispersion set. A test (`test_config_schema.py`) asserts the overridable set covers every `DEFAULT_DISPERSIONS` parameter. Step 1 of [ADR 0009](docs/adr/0009-explicit-dispersion-parameters.md). | — | Sim eng |
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
| T-03 | Decide on mypy adoption | info | M | **✅ Resolved (ADR 0015):** adopted **pyright** (handles NumPy typing better than mypy) — `pyrightconfig.json` + a pinned CI `typecheck` job, gating `sim/` at zero findings. The adoption pass fixed two latent issues (Q-05; an unnarrowed `Optional` in `propagator.py`). `tests/` are scoped out for now (Optional-narrowing noise) — a documented follow-up. | — | DevEx |
