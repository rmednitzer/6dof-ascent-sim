# Audit 03 — Final Report

Date: 2026-06-12
Repository: rmednitzer/6dof-ascent-sim
Branch: `claude/blissful-ritchie-1575b6`

## Branch-name note

The task prompt requested an `audit/YYYY-MM-DD-full-pass` branch. The hard
environment constraint for this session is to develop on
`claude/blissful-ritchie-1575b6` and never push elsewhere without explicit
permission, so all work landed there. The intent of both (work on a branch, open
a PR, never touch `main`) is satisfied.

## Executive summary

The repository is in good shape. It is a self-contained numerical simulation with
an intrinsically small attack surface: no network, no authentication, no
persistence of untrusted input, and no deserialization. Independent tooling found
**no dependency vulnerabilities** (`pip-audit`), **no secrets** in the working
tree or 57-commit history (`gitleaks`), and **no dangerous execution sinks**
(no `eval`/`exec`/`pickle`/`subprocess`/`yaml.load`). GitHub Actions are
SHA-pinned with least-privilege permissions, and there is no container/IaC to
harden.

The most material defect found was a **correctness bug in a safety-adjacent
signal**: telemetry `health_status` was permanently `"NOMINAL"` because the
recorder read a non-existent attribute, silently discarding the health monitor's
per-step assessment (finding Q-01). This was fixed with a regression test. Five
documentation/comment drifts were corrected (including a quaternion-convention
error in the FTS docstring), and a CI/pre-commit ruff-version inconsistency was
realigned.

Two candidate "physics correctness" findings raised by an automated sweep — a J3
gravity coefficient and the King-Hele decay formula — were **investigated and
disproved** (the J3 model matches a numerical geopotential gradient to 1e-6
relative error; the King-Hele per-revolution form is correct by derivation).
They are documented as dismissed so they are not re-flagged. No correct code was
changed on the strength of an unverified claim.

The test suite runs and stayed green throughout; no stop conditions were hit.

## Baseline vs post-fix metrics

| Metric | Baseline | Post-fix | Delta |
|--------|----------|----------|-------|
| Tests passing | 202 | 207 | +5 |
| Tests failing | 0 | 0 | 0 |
| Coverage (line) | 53% (1411 missed / 3026) | 55% (1364 missed / 3029) | +2 pts |
| ruff lint violations | 0 | 0 | 0 |
| ruff format issues | 0 | 0 | 0 |
| Dependency vulns (pip-audit) | 0 | 0 | 0 |
| Secrets (gitleaks) | 0 | 0 | 0 |
| End-to-end run | SUCCESS | SUCCESS | unchanged |

Vulnerability counts by severity (pip-audit, both runs): critical 0, high 0,
medium 0, low 0. No dependency versions were changed by this audit.

## Findings summary

| Severity | Count | Fixed | Deferred |
|----------|-------|-------|----------|
| Critical | 0 | 0 | 0 |
| High | 0 | 0 | 0 |
| Medium | 2 (Q-01, Q-04) | 1 (Q-01) | 1 (Q-04) |
| Low | 8 (S-04, S-05, S-08, Q-02, Q-03, Q-05, D-01..D-05*) | 5 (S-05, S-08/D-05, D-01..D-04) | rest |
| Info | 5 (S-06, S-07, Q-06, Q-07, plus dismissed) | — | tracked |

\* D-01..D-05 are documentation drifts; all five were fixed. See
`audit/02-security-findings.md` for the full register and the
"investigated and dismissed" section.

## Commits (this branch, in order) with rationale

| Commit | Rationale |
|--------|-----------|
| `0c136b0` docs(audit): Phase 0-3 evidence | Inventory, baseline, and the security/quality findings register (read-only output, no source changes). |
| `fac31c6` fix(telemetry): surface real health status | Q-01: recorder read a non-existent `.status`; add a `status` property + regression test so telemetry reflects WARNING/ALERT/CRITICAL. |
| `5b76df5` docs: correct drifted source comments | D-01 quaternion order in FTS, D-02 atmosphere 200->1000 km, D-03 gain-schedule `sqrt` formula; comment-only. |
| `7a4c844` docs: fix command/parameter drift | D-04 runbook TVC slew 10->20, D-05 SECURITY.md `pip audit`->`pip-audit`. |
| `79b209c` chore(ci): align ruff pin | S-05: CI `0.9.7` vs pre-commit `0.15.16`; realign CI to 0.15.16 (verified passing). |
| `2954986` docs(adr): ADR set | Five backfilled accepted ADRs, one audit-fix ADR, three proposed ADRs + index. |
| `8585302` docs: add BACKLOG.md | Deferred findings/proposals by section, ordered by severity then effort. |
| (this report) | Phase 8 final report. |

## Residual risk statement

- **Reproducibility / supply chain (low):** dependencies are unpinned floors with
  no lockfile, so installs drift to latest. Mitigated by `pip-audit` and Renovate
  OSV alerts; a lockfile is proposed (ADR 0007, S-04). No known-vulnerable
  versions today.
- **Integration test gap (medium-low):** `run_simulation` and the recorder are
  thinly covered; a full-pipeline regression could pass CI. The Q-01 fix added
  the first recorder test; a seeded end-to-end test is proposed (ADR 0008, Q-04).
- **Partially-wired health channels (low):** engine and sensor health are never
  fed real data by the main loop, so they read NOMINAL regardless (Q-03). The
  fixed channels (EKF, dynamic pressure, propellant) now reach telemetry.
- **Model-fidelity items unchanged:** the audit deliberately changed no physics.
  Modeling simplifications (zero-order-hold forces across RK4 sub-steps, fitted
  thermosphere, pendulum slosh) are documented in `docs/assumptions.md` and ADR
  0003 and are appropriate for this fidelity tier; they are not defects.

No critical or high-severity security risk remains. No data migration, history
rewrite, or major dependency bump was performed; such items are recorded as
proposals only.

## Top 5 backlog items

1. **Q-04 — End-to-end simulation regression test** (medium, M). The largest
   coverage gap; guards the whole physics/GNC/safety/telemetry pipeline. ADR 0008.
2. **S-04 — Adopt a dependency lockfile** (low, M). Reproducible builds and a
   real artifact for Renovate/pip-audit to maintain. ADR 0007.
3. **Q-03 — Feed engine/sensor health channels in the main loop** (low, M).
   Makes two health channels meaningful instead of constant NOMINAL.
4. **Q-02 — Remove `_save_config` key-list drift** (low, S). Closes a latent
   Monte Carlo restore bug; interim step toward ADR 0009.
5. **D-06 — Regenerate committed example outputs** (low, S). The Q-01 fix makes
   the committed example telemetry hash stale; refresh artifacts.

## Method note

Every quantitative claim here is backed by a command run this session (recorded
in `00-inventory.md`, `01-baseline.md`, and `02-security-findings.md`). Where a
fact could not be verified from repository contents (e.g. CodeQL default-setup
status) it is marked `[UNVERIFIED]` in the inventory rather than asserted.
