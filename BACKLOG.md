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
`tests/test_notch_filter.py` and `tests/test_e2e_simulation.py`). **AD-17 and
AD-19 are now resolved**: the N-01 performance-margin work
([ADR 0021](docs/adr/0021-stage2-performance-margin.md)/[0023](docs/adr/0023-ground-station-range-tracking.md))
removed the PEG terminal attitude-margin fragility (AD-19; dispersed abort rate
19%→0%), which unblocked inclination targeting (AD-17 →
[ADR 0024](docs/adr/0024-inclination-targeting.md): azimuth correction + yaw
out-of-plane steering, 45°→51°, correction-dv ~995→74 m/s, with no robustness
regression).

| ID | Title | Sev | Effort | Approach | Owner role |
|----|-------|-----|--------|----------|------------|
| [AD-17](audit/04-adversarial-findings.md) | Launch azimuth ignores Earth-rotation inclination term | info | M | **✅ Resolved ([ADR 0024](docs/adr/0024-inclination-targeting.md)).** Earth-rotation azimuth correction + terminal yaw out-of-plane steering reach the target inclination (45°→51.04°, correction-dv ~995→74 m/s) with peak loads unchanged. Originally deferred because every variant regressed dispersed FTS aborts via the PEG terminal margin (AD-19); the N-01 work removed that fragility, so this landed with **no robustness regression (24/24)**. | GNC eng |
| [AD-19](audit/04-adversarial-findings.md) | PEG terminal phase rides the 25° FTS attitude limit under dispersions | medium | L | **✅ Resolved (N-01: [ADR 0021](docs/adr/0021-stage2-performance-margin.md) + [0023](docs/adr/0023-ground-station-range-tracking.md)).** Discovered during the AD-17 investigation: ~19% of dispersed runs FTS-aborted on a marginal thrust-axis error near insertion (every observed abort 25.0x° — right at `FTS_ATTITUDE_LIMIT_DEG`). Root cause was S2 propellant depleting just short of orbit (post-depletion tumble); the S2 Isp margin + ground-station tracking took the dispersed abort rate **19%→0%** and moved the terminal phase off the limit, unblocking AD-17 (ADR 0024). | GNC/controls eng |

## Security

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| [S-04](audit/02-security-findings.md) | Adopt a dependency lockfile | low | M | Floors-only deps + no lock means non-reproducible installs and a no-op Renovate `lockFileMaintenance`. Generate a hashed `requirements.lock` (pip-compile), install it in CI, let Renovate bump it. See [ADR 0007](docs/adr/0007-adopt-dependency-lockfile.md). | — | Build/release eng |
| [S-06](audit/02-security-findings.md) | Make `lockFileMaintenance` effective | info | S | Renovate is configured to maintain a lock that does not exist. Resolved automatically once S-04 lands. | S-04 | Build/release eng |
| [S-07](audit/02-security-findings.md) | Confirm or add CodeQL scanning | info | S | Commit `9a2e67e` mentions CodeQL but there is no workflow file; default-setup status is not visible in-repo. Either commit a `codeql.yml` (SHA-pinned) or document that default setup is enabled in repo settings. | — | Security eng |

## Reliability

| ID | Title | Sev | Effort | Rationale / approach | Depends on | Owner role |
|----|-------|-----|--------|----------------------|------------|------------|
| [Q-03](audit/02-security-findings.md) | Feed engine/sensor health channels in the main loop | low | M | **✅ Resolved.** `main.py` now feeds the health monitor commanded-vs-actual thrust (gated to steady firing, so ignition/tail-off ramps and the inter-stage coast don't false-positive) and per-sensor degradation flags. `engine_health` and `sensor_status` are now live — NOMINAL in fault-free flight but responsive: a steady-state thrust deviation and genuine *in-envelope* GPS/baro dropouts are detected (expected loss of fix above the COCOM ceiling / baro altitude is not flagged). Tests in `test_health_telemetry.py`. Star-tracker / ground-network health (complex envelopes) is a documented follow-up. | — | GNC/sim eng |
| [Q-05](audit/02-security-findings.md) | Replace PEG `except UnboundLocalError` control flow | low | S | **✅ Resolved (ADR 0015):** `A`/`B` are now seeded from the stored coefficients before the iteration and the `try/except UnboundLocalError` is gone (behaviour-preserving — the nominal golden trajectory is unchanged); surfaced and guarded by the pyright `reportPossiblyUnbound` check. _Original:_ `guidance.py` retained stale PEG coefficients via exception control flow with no signal when the iteration degenerated. | Q-04 (test harness) | GNC eng |
| N-01 | Dispersed FTS aborts near insertion | medium | M | **✅ Resolved (ADR 0021 + 0022 + 0023): abort rate 33 % → 0 % (24/24).** _Corrected diagnosis:_ a per-seed look showed the ~33 % Stage-2 abort rate was **not** nav-covariance driven (only 1 of 8 was a covariance trip). 7 of 8 were thrust-axis attitude trips near insertion: the upper stage under-performed under adverse propulsion/drag dispersion, the S2 tank emptied a few tens of m/s short of orbit, and the **unpowered** vehicle tumbled at its residual ~2.4°/s rate (no thrust ⇒ no TVC) past the 25° FTS limit — a performance-margin problem, consistent with AD-17. _Fix 1 (ADR 0021):_ raised the S2 Isp default 348 → 356 s (`config_schema.py`); the nominal is preserved (Isp, unlike propellant, leaves liftoff mass unchanged — the nominal is chaotically mass-sensitive) and the dispersed abort rate dropped to ~12.5 %, leaving 2 marginal covariance trips + 1 staging transient. _Fix 2 (ADR 0022):_ tightened the implausibly-wide IMU bias dispersion (σ 2× nominal, ~7× tail → ~30 % of nominal, ~2× tail); recovered the IMU-driven covariance trip (seed 42) and the staging transient (seed 54), → 23/24. _Fix 3 (ADR 0023):_ the last residual (seed 56) had its covariance reach 10 km **even with a nominal IMU** — its high-drag trajectory lengthens the GPS-denied coast, where there is no position aiding at all (GPS COCOM-denied, baro gone, star tracker attitude-only). Added a **ground-station range-tracking network** (KSC + Bermuda) — independent of GPS (the vehicle is *tracked*, not self-locating), so it is not COCOM-bound and bounds EKF position covariance through the coast (nominal peak 1.7 km → 0.52 km; seed 56 14.1 km → 0.57 km) → **24/24, 0 % aborts**. N-01 fully resolved. | ADR 0020, 0021, 0022, 0023 | GNC eng |

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
