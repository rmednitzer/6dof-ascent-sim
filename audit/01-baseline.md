# Audit 01 — Validation Baseline

Date: 2026-06-12
Phase: read-only. This baseline is the regression reference for every later change.
All numbers below come from commands run in this session in a clean virtualenv.

## Environment setup

The Debian-managed host Python cannot complete `pip install -e ".[dev]"` (it
fails uninstalling the OS-managed `platformdirs`: "Cannot uninstall ... RECORD
file not found"). A clean virtualenv was therefore used:

```
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Result: success. Resolved versions: numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.0,
pytest 9.0.3, pytest-cov 7.1.0, ruff 0.15.17, coverage 7.14.1.

`.venv/` is already in `.gitignore`, so no artifacts are committed.

## Build

`pip install -e ".[dev]"` builds an editable wheel cleanly
(`6dof_ascent_sim-0.1.0-0.editable`). No build warnings or errors.

## Test suite

Command:

```
.venv/bin/python -m pytest tests/ -q --cov=sim --cov-report=term-missing
```

| Metric | Value |
|--------|-------|
| Result | **202 passed, 0 failed, 0 skipped** |
| Wall time | ~2.5 s (test session); ~3.0 s including process start |
| Flaky candidates | none observed (sensor RNG default uses `secrets`, but every test that needs determinism injects a seeded `np.random.default_rng`) |
| Coverage | **53%** overall (3,026 statements, 1,411 missed) |

### Coverage hot spots (lowest coverage of executable modules)

| Module | Coverage | Note |
|--------|----------|------|
| `sim/montecarlo/dispatcher.py` | 0% | exercised only via multiprocessing entry point |
| `sim/montecarlo/dispersions.py` | 0% | |
| `sim/montecarlo/statistics.py` | 0% | plotting/stats |
| `sim/analysis/postflight.py` | 0% | plotting |
| `sim/main.py` | 15% | **no end-to-end test of `run_simulation`** (see Q-04) |
| `sim/telemetry/recorder.py` | 28% | **recorder untested** (see Q-01, Q-04) |
| `sim/gnc/control.py` | 18% | controller exercised mostly via integration paths |
| `sim/gnc/guidance.py` | 34% | |

High-coverage core (gravity 100%, config 100%, integrator 98%, navigation 98%,
atmosphere 99%, propagator 93%, reference_frames 93%, fts 96%, decay 90%) shows
the numerically critical paths are well tested; the gaps are concentrated in
orchestration (main loop), telemetry I/O, Monte Carlo, and plotting.

## Lint / format / type-check (check-only)

| Tool | Command | Result |
|------|---------|--------|
| ruff lint | `ruff check .` | **All checks passed** (ruff 0.15.17) |
| ruff format | `ruff format --check .` | **60 files already formatted** |
| mypy | `mypy sim/` | 39 errors, but **mypy is not part of the project toolchain** (absent from `pyproject.toml`, `.pre-commit-config.yaml`, and CI). 37 of 39 are "missing stub for numpy/matplotlib" from running the host mypy outside the venv; the 2 substantive ones are `union-attr` false positives in `orbital/propagator.py:303-311` where `self._elements` is narrowed via a method call (`orbit_summary` calls `state_to_elements()` first). Informational only. |

## Smoke run of entry points

`python -m sim.main` runs end to end:

```
Outcome: SUCCESS
ORBITAL INSERTION at t=488.2s
Final altitude 407.0 km, final velocity 7603.4 m/s
Peak q 32425 Pa (92.6% of limit); peak axial 5.40 g (90.0% of limit)
Boundary violations: 319; FTS triggered: False
Telemetry hash (SHA-256): 38522a9f...3243bb02
```

Wall time ~45 s. Writes `output/telemetry_internal.json`,
`output/telemetry_downlink.json`, `output/mission_summary.json`, and
`output/plots/*.png`. A benign matplotlib `INFO` about a non-scalable emoji font
is printed; it does not affect output.

Observation (not a defect): 319 boundary clamp events occur on the nominal
trajectory. This is by design (the enforcer counts every clamp, including
routine max-q / G-limit throttle management), but the magnitude is worth a note
for anyone interpreting `total_boundary_violations` as an anomaly count.

## CI drift vs. what actually runs

Reproducing CI locally surfaced two config drifts (carried into
`02-security-findings.md` as S-05 and noted for `04`/tooling):

1. **Ruff version mismatch.** `.github/workflows/ci.yml` installs `ruff==0.9.7`,
   but `.pre-commit-config.yaml` pins `ruff` at `v0.15.16` — despite a CI comment
   stating the pin exists "to match `.pre-commit-config.yaml` so local pre-commit
   and CI never disagree." 0.9.7 and 0.15.16 are many releases apart and can
   diverge on lint/format. This is a real local-vs-CI inconsistency.
2. **Lint runs on Python 3.14**, which is outside the declared support matrix
   (`pyproject.toml` classifiers and the CI test matrix are 3.11/3.12/3.13).
   Low impact (ruff is a standalone binary), but inconsistent.

Baseline summary: green build, green tests (202/0), clean lint/format, 53%
coverage, one working end-to-end run. No blocking issues; remediation in Phase 4
must hold all of the above.
