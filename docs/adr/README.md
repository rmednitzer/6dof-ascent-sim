# Architecture Decision Records

This directory records the significant architecture decisions for the 6-DOF
ascent simulation, in [MADR](https://adr.github.io/madr/)-style format.

Records numbered and immutable once accepted; supersede rather than rewrite.
ADRs `0001`-`0005` are **backfilled** from decisions already embodied in the
code (status `accepted`), written during the 2026-06-12 audit. Where the
original rationale is not recoverable from the repository it is marked as such
rather than invented. `0006` documents an audit fix. `0007`-`0009` are
`proposed` and need a maintainer decision (see `BACKLOG.md`). `0010` documents
the experimental SLURM HPC Monte Carlo backend; `0011` documents the verified
zonal-harmonic gravity rewrite and `0012` the live flex/control coupling with a
structural notch filter, both from the 2026-06-14 fidelity pass. `0013` (NIS
innovation gating) and `0014` (FTS attitude hysteresis) are GNC-robustness
changes from the 2026-06-21 cross-repo lessons pass (`audit/05`).

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-single-config-source-of-truth.md) | Single `config.py` source of truth for all parameters | accepted |
| [0002](0002-scalar-last-quaternion-convention.md) | Scalar-last `[x, y, z, w]` quaternion convention | accepted |
| [0003](0003-fixed-step-rk4-100hz.md) | Fixed-step 100 Hz RK4 with a physics-agnostic integrator | accepted |
| [0004](0004-monte-carlo-process-isolation.md) | Monte Carlo via process isolation + transient global-config override | accepted |
| [0005](0005-quality-tooling-strategy.md) | Quality tooling: ruff + pytest matrix + pre-commit + Renovate | accepted |
| [0006](0006-surface-health-status-to-telemetry.md) | Surface health status to telemetry | accepted |
| [0007](0007-adopt-dependency-lockfile.md) | Adopt a dependency lockfile | proposed |
| [0008](0008-end-to-end-simulation-test.md) | Add an end-to-end simulation regression test | proposed |
| [0009](0009-explicit-dispersion-parameters.md) | Replace mutable-global config override with explicit parameter passing | proposed |
| [0010](0010-slurm-hpc-monte-carlo.md) | SLURM HPC Monte Carlo via job array + shard/collect | accepted (experimental) |
| [0011](0011-verified-zonal-gravity.md) | Zonal-harmonic gravity via the verified analytic geopotential gradient | accepted |
| [0012](0012-live-flex-structural-notch.md) | Live flex/control coupling stabilised by a frequency-scheduled structural notch | accepted |
| [0013](0013-chi-square-innovation-gating.md) | Chi-square (NIS) innovation gating in the navigation EKF | accepted |
| [0014](0014-fts-attitude-hysteresis.md) | Hysteresis on the FTS attitude criterion | accepted |
| [0015](0015-adopt-pyright-type-checking.md) | Adopt pyright for static type checking (production code) | accepted |
| [0016](0016-validated-config-override-schema.md) | Validated Monte-Carlo override schema (ADR-0009 step 1) | accepted |
| [0017](0017-defer-numba-jit-profile-driven.md) | Evaluate numba-JIT for the hot loop; defer (profile-driven) | accepted |
