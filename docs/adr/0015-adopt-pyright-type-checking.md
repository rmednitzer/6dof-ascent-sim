# 0015. Adopt pyright for static type checking (production code)

- Status: accepted
- Date: 2026-06-21
- Deciders: DevEx (cross-repo lessons pass, `audit/05`)

## Context and Problem Statement

The project is fully type-annotated but had no static type checker in CI.
`BACKLOG` item **T-03** left the decision open: "mypy is installed ... surfaces
only false positives today; either adopt it with config + stubs and wire into
CI, or document that it is intentionally out of scope." The cross-repo audit
(`audit/05`, §C5) recommended **pyright**, which handles NumPy's typing far
better than mypy in practice and reads inline annotations without extra stubs.

Running pyright (basic mode) surfaced 46 findings. Triaging them was itself
valuable: `sim/gnc/guidance.py` flagged `A`/`B` as *possibly unbound* — exactly
the latent **Q-05** PEG `except UnboundLocalError` control-flow issue — and
`sim/orbital/propagator.py` flagged an unnarrowed `Optional` element access.

## Decision

- Add `pyrightconfig.json`: `typeCheckingMode: "basic"`, `pythonVersion: "3.11"`,
  scoped to **`sim/`** (production code).
- Reach **zero** pyright findings in `sim/` by fixing the genuine issues:
  - `guidance.py`: seed PEG `A`/`B` from the stored coefficients before the
    iteration and drop the `try/except UnboundLocalError` (closes **Q-05**).
  - `propagator.py`: narrow the `Optional[OrbitalElements]` before access and
    cast NumPy scalars to `float` at the dataclass boundary.
  - `dispersions.py`: assert the per-distribution `sigma`/`bounds` invariants.
  - `maneuvers.py`: cast a NumPy scalar return to `float`.
- Add a pinned `typecheck` CI job (`pyright==1.1.410`) so a future pyright
  release cannot silently introduce new findings.

`tests/` are **not** yet gated. Their remaining findings are almost entirely
`Optional`-narrowing noise around the deliberately-`Optional` `MonteCarloResult`
and `BoundaryResult` fields (e.g. asserting `"x" in result.violation_type`),
plus a duplicate `OrbitalElements` mock in `tests/test_orbital_decay.py`. Gating
those is low value relative to the churn and is left as a follow-up.

## Consequences

- Positive: production code is statically type-checked in CI; the adoption pass
  already surfaced and fixed two latent issues (Q-05 and the propagator Optional
  access). Resolves the long-open T-03.
- Positive: pinning the checker version makes the gate reproducible.
- Negative: `tests/` type errors remain unaddressed (documented, scoped out);
  the CI gate will not catch type regressions in test code yet.
- Note: this extends the quality-tooling decision in ADR 0005 (ruff + pytest +
  pre-commit + Renovate) with a type checker; it does not supersede it.

## Notes / Evidence

`pyrightconfig.json`, `.github/workflows/ci.yml` (`typecheck` job). Source fixes
in `sim/gnc/guidance.py`, `sim/orbital/propagator.py`,
`sim/montecarlo/dispersions.py`, `sim/orbital/maneuvers.py`. Cross-repo
rationale: `audit/05-cross-repo-lessons.md` §C5. Verified locally: `pyright`
reports 0 errors; full pytest suite (316 tests) green; ruff clean.
