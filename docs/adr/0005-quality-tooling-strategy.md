# 0005. Quality tooling: ruff + pytest matrix + pre-commit + Renovate

- Status: accepted
- Date: 2026-06-12 (backfilled)
- Deciders: original authors

## Context and Problem Statement

A scientific Python project needs consistent style, fast linting, regression
testing across supported interpreters, and controlled dependency updates,
without a heavyweight toolchain.

## Decision

- Lint and format with **ruff** (single tool for both), rule set
  `E,W,F,I,UP,B,SIM`, line length 120, with deliberate ignores documented in
  `pyproject.toml` (e.g. `SIM102` for safety-critical nested ifs).
- Test with **pytest** + **pytest-cov** across a Python **3.11/3.12/3.13**
  matrix in GitHub Actions.
- Enforce hygiene locally with **pre-commit** (ruff, ruff-format, plus
  whitespace/EOF/YAML/TOML/large-file/merge-conflict hooks).
- Manage dependency and Action updates with **Renovate** (`config:best-practices`,
  weekly schedule, OSV alerts, grouped PRs).
- Pin all GitHub Actions to commit SHAs; grant CI least privilege
  (`permissions: contents: read`).

## Consequences

- Positive: one fast linter/formatter; reproducible CI; supply-chain hygiene
  (pinned actions, automated security alerts).
- Negative: the ruff version must be kept identical between CI and pre-commit or
  they disagree. The audit found exactly this drift (CI `0.9.7` vs pre-commit
  `0.15.16`, finding S-05) and realigned CI. A single source for the ruff
  version (or `pre-commit` running in CI) would prevent recurrence.
- Note: the lint job runs on Python 3.14, outside the 3.11-3.13 support matrix;
  low impact since ruff is a standalone binary, but inconsistent.

## Notes / Evidence

`.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `renovate.json5`,
`pyproject.toml` `[tool.ruff]`. CI actions are SHA-pinned (verified by reading
the workflow).
