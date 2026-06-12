# Audit 00 — Recon and Inventory

Date: 2026-06-12
Scope: full repository, read-only recon phase.
Branch: `claude/blissful-ritchie-1575b6` (see note in `03-final-report.md` on the
branch-name deviation from the `audit/YYYY-MM-DD-full-pass` convention).

All facts below are backed by commands run in this session. Where a fact could
not be verified from repository contents it is marked `[UNVERIFIED]`.

## Toolchain actually available in this environment

| Tool | Version | Source command |
|------|---------|----------------|
| Python | 3.11.15 | `python --version` |
| pip | 24.0 (system), 26.1.2 (venv) | `pip --version` |
| git | 2.43.0 | `git --version` |
| pytest | 9.0.3 (venv) | `pytest --version` |
| ruff | 0.15.17 (venv) | `ruff --version` |
| mypy | 1.19.1 | `mypy --version` (not part of project config) |
| gitleaks | present (`/usr/bin/gitleaks`) | `which gitleaks` |
| pip-audit | 2.10.1 (installed into venv this session) | `pip-audit --version` |
| semgrep | not available | `which semgrep` (no output) |
| trufflehog | not available | `which trufflehog` (no output) |

Notes:
- The host Python is Debian-managed; `pip install -e ".[dev]"` against it fails at
  the `platformdirs` uninstall step ("Cannot uninstall ... RECORD file not found").
  A virtualenv (`.venv/`, already git-ignored) was used for all build/test/audit
  steps. See `01-baseline.md`.
- Network egress via `curl`/`wget` is denied by `.claude/settings.json`; PyPI
  access via `pip` works.

## Languages and frameworks

- Single language: Python (>= 3.11, per `pyproject.toml` `requires-python`).
- No compiled extensions, no other languages.
- Scientific stack: NumPy, SciPy, Matplotlib. No web framework, no database, no
  network server, no ORM.

## Build system

- PEP 621 `pyproject.toml`, build backend `setuptools.build_meta`
  (`setuptools>=68.0`, `wheel`).
- Installable as editable; console script `6dof-sim = sim.main:main`.
- No `setup.py`, `setup.cfg`, `requirements*.txt`, `poetry.lock`, `Pipfile`, or
  `uv.lock` (verified by `ls`). **There is no dependency lockfile of any kind.**

## Entry points

| Entry point | Invocation |
|-------------|------------|
| Main simulation | `python -m sim.main` (`--no-flex`, `--no-slosh`) |
| Console script | `6dof-sim` |
| Monte Carlo campaign | `python -m sim.montecarlo.dispatcher --runs N --seed S --workers W` |
| Example + visualization | `python examples/run_and_visualize.py` |
| Microbenchmark | `python benchmark.py` |

## Component map (`sim/` package)

43 Python files under `sim/` + `tests/`; 10,317 LOC total
(`find sim tests -name '*.py' -exec wc -l`). ~3,026 executable statements in
`sim/` (from coverage run, `01-baseline.md`).

| Subpackage | Files | Responsibility |
|------------|-------|----------------|
| `sim/` (root) | `config.py`, `main.py`, `__init__.py` | Parameter source-of-truth; 100 Hz RK4 loop; CLI |
| `sim/core/` | `state`, `integrator`, `reference_frames`, `fast_math` | State vector, RK4, frame/quaternion math, hot-path vector helpers |
| `sim/environment/` | `atmosphere`, `gravity`, `wind` | US Std Atmosphere 1976, WGS84+J2..J6 gravity, wind/gusts |
| `sim/vehicle/` | `vehicle`, `propulsion`, `aerodynamics`, `staging`, `actuator` | Mass ledger, engine model, aero, staging FSM, TVC actuator |
| `sim/dynamics/` | `flex_body`, `slosh` | Bending modes, pendulum-analogy slosh |
| `sim/gnc/` | `guidance`, `control`, `navigation`, `sensors` | 3-phase guidance, PID+TVC, 12-state EKF, IMU/GPS/baro |
| `sim/safety/` | `boundary_enforcer`, `fts`, `health_monitor` | Command clamping, Flight Termination System, health channels |
| `sim/telemetry/` | `recorder`, `schemas` | Dual-rate recorder, SHA-256 integrity hash, frame/summary dataclasses |
| `sim/orbital/` | `propagator`, `maneuvers`, `decay` | Cartesian->Keplerian, J2 propagation, dv budget, decay |
| `sim/montecarlo/` | `dispatcher`, `dispersions`, `statistics` | Multiprocessing campaign, dispersions, stats/plots |
| `sim/analysis/` | `postflight` | Trajectory plots |

## Dependency graph summary

Direct runtime dependencies (3), from `pyproject.toml`:

| Package | Declared constraint | Resolved in clean venv |
|---------|--------------------|------------------------|
| numpy | `>=1.24` | 2.4.6 |
| scipy | `>=1.10` | 1.17.1 |
| matplotlib | `>=3.7` | 3.11.0 |

Direct dev dependencies (4): `pytest>=7.0`, `pytest-cov>=4.0`, `ruff>=0.4`,
`pre-commit>=3.0`.

Transitive footprint (clean venv install of `.[dev]`): ~30 packages
(contourpy, cycler, fonttools, kiwisolver, pillow, pyparsing, python-dateutil,
six for matplotlib; pluggy/iniconfig for pytest; coverage for pytest-cov;
cfgv/identify/nodeenv/virtualenv/distlib/filelock/platformdirs for pre-commit).

Lockfile state: **none**. All constraints are lower-bound floors, so installs
resolve to the latest compatible release at install time. Renovate is configured
with `lockFileMaintenance` enabled, but with no lockfile present that rule is a
no-op. See finding S-06 in `02-security-findings.md`.

## CI / automation / governance

- GitHub Actions: single workflow `.github/workflows/ci.yml`.
  - `lint` job: Python 3.14, `pip install "ruff==0.9.7"`, `ruff check` + `ruff format --check`.
  - `test` job: matrix Python 3.11/3.12/3.13, `pip install -e ".[dev]"`,
    `pytest --cov`, uploads `coverage.xml` artifact on 3.12.
  - Top-level `permissions: contents: read` (least privilege).
  - All `uses:` actions are pinned to 40-hex commit SHAs with version comments.
  - Triggers: push and pull_request to `main`; concurrency group cancels in-progress.
- Renovate (`renovate.json5`): `config:best-practices`, semantic commits, weekly
  schedule, OSV vulnerability alerts, grouped updates, `pre-commit` manager enabled.
- pre-commit (`.pre-commit-config.yaml`): standard hygiene hooks + ruff `v0.15.16`
  (lint with `--fix`) + ruff-format.
- Governance files present: `CODEOWNERS`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `LICENSE` (Apache-2.0), `.github/ISSUE_TEMPLATE/*`,
  `pull_request_template.md`, `.github/copilot-instructions.md`, `CLAUDE.md`.
- CodeQL: **no workflow file** under `.github/workflows/` (`find .github -iname '*codeql*'`
  returns nothing) although commit `9a2e67e` mentions adding CodeQL. If active it is
  via GitHub "default setup" (UI-configured), which cannot be confirmed from repo
  contents. `[UNVERIFIED]`

## Container / IaC files

None. No `Dockerfile`, no Kubernetes manifests, no Terraform, no systemd units
(verified by `find`). The only YAML in the tree is GitHub issue templates, the CI
workflow, and the pre-commit config. This materially reduces the deployment attack
surface.

## Test layout

- `tests/` with 15 modules (`test_*.py`) + `__init__.py`, 2,724 LOC.
- pytest configured in `pyproject.toml`: `testpaths=["tests"]`, `addopts="--strict-markers"`.
- Two regression-oriented suites: `test_audit_regressions.py` (pins prior audit
  fixes) and `test_improvements.py` (pins optimization-equivalence).
- Coverage tooling present via `pytest-cov`.
