# 0007. Adopt a dependency lockfile

- Status: proposed
- Date: 2026-06-12
- Deciders: maintainers (pending)

## Context and Problem Statement

`pyproject.toml` declares only lower-bound floors (`numpy>=1.24`, `scipy>=1.10`,
`matplotlib>=3.7`, plus dev tools) and there is no lockfile. Installs resolve to
whatever is latest at install time — this audit pulled numpy 2.4.6, scipy 1.17.1,
matplotlib 3.11.0. A future transitive release could change numerical results or
introduce a regression with no pinned baseline to diff against. Renovate's
`lockFileMaintenance` is currently a no-op because there is no lockfile (findings
S-04, S-06).

## Considered Options

1. `pip-compile` (pip-tools) producing `requirements.lock`, installed in CI.
2. Commit a `constraints.txt` from `pip freeze` of the CI matrix.
3. Status quo (floors only), relying on `pip-audit` + Renovate alerts.

## Decision Outcome (proposed)

Option 1. Generate and commit a hashed lock, have CI install from it, and let
Renovate bump it on the weekly schedule. This makes builds reproducible and
gives `pip-audit`/Renovate a concrete artifact to scan and maintain.

## Consequences

- Positive: reproducible numerics; meaningful `lockFileMaintenance`; clearer
  diffs when a dependency bump changes behavior.
- Negative: a lock to maintain; contributors must regenerate it when changing
  dependencies. Application-vs-library tension (this is effectively an
  application, so pinning is appropriate).

## Notes / Evidence

`pyproject.toml:22-26`; `ls` confirms no lockfile this session; `renovate.json5`
enables `lockFileMaintenance`.
