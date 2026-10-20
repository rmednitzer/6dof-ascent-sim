# Contributing

Thank you for your interest in contributing to the 6-DOF Launch Vehicle Ascent Simulation!

## Getting Started

1. Fork the repository and create a branch from `main`.
2. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

- Keep changes focused — one logical change per pull request.
- Follow the existing code style. The pre-commit hooks will run automatically, or run manually:
  ```bash
  ruff check .
  ruff format .
  ```
- Add or update tests in `tests/` for any new functionality.
- Ensure all tests pass:
  ```bash
  pytest tests/ -v
  ```

## Pull Request Process

1. Fill in the pull request template.
2. Ensure CI passes (lint, type checking, and tests on Python 3.11, 3.12, 3.13, and 3.14).
3. Describe what you changed and why in the PR description.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## License

By contributing you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).


## Dependency automation

Renovate policy is owned by this repository in `renovate.json5`. It does not
inherit another project’s policy. Keep update grouping, release-age delays,
major-version review and lock-maintenance behavior explicit here. Built-in
Renovate presets supply standard defaults; repository-specific decisions remain
local. Review policy changes and actual dependency extraction before merging.


## CI run ownership

The PR-validation workflows updated here coalesce first attempts for the same PR.
Manual reruns use separate run-ID/attempt groups, so rerunning an old revision
cannot cancel or replace current-head validation. Non-PR invocations also use
unique run-ID/attempt groups to preserve each revision's evidence.
GitHub does not guarantee chronological ordering of initial attempts: a late event
can still interrupt another initial attempt. Required checks bind to the current
head; a cancelled current-head check needs a successful rerun before merge.
Explicit diagnostic/publication serialization in other workflows remains separate. Reusable
workflows have distinct file prefixes to avoid cancelling a caller or sibling.
Cancelling an obsolete PR run is not a successful check; the current head still
needs all required results. Revision identity and signing gates remain separate.
Publication workflows retain their existing serialization and cache isolation.
Revert this repository's workflow change to restore the earlier scheduling;
no service deployment or application-data migration is involved.
