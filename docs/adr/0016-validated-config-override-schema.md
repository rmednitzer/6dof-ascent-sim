# 0016. Validated Monte-Carlo override schema (ADR-0009 step 1)

- Status: accepted
- Date: 2026-06-21
- Deciders: Sim eng (cross-repo lessons pass, `audit/05` X-10)

## Context and Problem Statement

Monte-Carlo runs perturb parameters by overriding `sim.config` attributes
(ADR-0004). Two long-standing weaknesses:

- **Silent mis-configuration.** `generate_dispersed_config` resolves each
  dispersion with `getattr(config, name, None)` and skips it if `None`, so a
  dispersion whose `parameter` is mis-spelled is *silently disabled* — the
  campaign runs with one fewer dispersion and no warning. Override values are
  likewise unchecked (a non-positive sensor-noise scale crashed ~half of all
  runs — AD-18).
- **Key-list drift (Q-02).** `main._save_config` hand-maintained a copy of the
  overridable-parameter names that had to track `DEFAULT_DISPERSIONS` manually.

ADR-0009 (`proposed`) calls for replacing the mutable-global override with
explicit parameter passing. That is a large, structural change (every module
reads `from sim import config`). The cross-repo audit (`audit/05` §B3, validated
against genesis-world's pydantic `Options`) recommends a typed/validated config;
this ADR lands the **low-risk first step**.

## Decision

Add `sim/config_schema.py` — a **validation layer** that does **not** change how
the simulation reads configuration:

- `OverridableParams` (pydantic `BaseModel`, `extra="forbid"`): a typed, bounded
  declaration of every per-run overridable parameter. Field *values* default
  from `sim.config` (still the single source of truth, ADR-0001); the schema
  adds only types, bounds, and units. Strictly-positive bounds on the
  sensor-noise scales encode the AD-18 invariant.
- `validate_dispersions(...)`: rejects a dispersion targeting an unknown /
  non-overridable parameter (the silent-skip bug) — wired into
  `MonteCarloDispatcher.__init__`.
- `validate_overrides(...)`: range/type-checks a drawn override dict (ignoring
  `_`-prefixed bookkeeping keys) — wired into
  `MonteCarloDispatcher.generate_run_configs`.
- `OVERRIDABLE_PARAM_NAMES`: the single declaration of the overridable set;
  `main._save_config` derives its key list from it, closing **Q-02**.

Adds a runtime dependency on **pydantic >= 2** (already the configuration
mechanism in the sibling genesis-world and quadrants repos).

## Consequences

- Positive: a mis-spelled or out-of-range dispersion now fails fast *before* a
  campaign launches, instead of silently degrading it or crashing workers
  mid-run. The Q-02 drift is structurally impossible (one declaration).
- Positive: leaves the hot read path (`config.X`) untouched, so this is a
  behaviour-preserving, low-risk change — the nominal golden trajectory is
  unchanged and existing direct `run_simulation(config_override=...)` calls are
  not newly constrained (validation is at the Monte-Carlo campaign boundary).
- Negative: adds a runtime dependency (pydantic). Acceptable — it is small,
  ubiquitous, and the cited exemplars use it; note the repo still has no
  lockfile (S-04).
- This is **step 1**. ADR-0009's core goal — removing the global-mutation
  override by threading a config object explicitly (which also unblocks
  threaded / vectorised Monte Carlo, `audit/05` X-14) — remains open.

## Notes / Evidence

`sim/config_schema.py`; wiring in `sim/montecarlo/dispatcher.py` and
`sim/main.py:_save_config`; dependency in `pyproject.toml`. Tests:
`tests/test_config_schema.py` (typo/bounds/type rejection, unknown-dispersion
rejection, all 200-seed default overrides validate, overridable-set covers the
dispersions, dispatcher rejects a bad campaign). Cross-repo rationale:
`audit/05-cross-repo-lessons.md` §B3 / §E. Verified: pyright 0 errors on `sim/`;
full suite 329 passed; ruff clean.
