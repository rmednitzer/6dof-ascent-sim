# 0018. Context-local overridable config (ADR-0009 step 2)

- Status: accepted
- Date: 2026-06-21
- Deciders: Sim eng (cross-repo lessons pass, `audit/05` X-10)
- Supersedes: the global-override mechanism of ADR-0004; completes ADR-0009

## Context and Problem Statement

ADR-0004 implemented Monte-Carlo dispersion by **mutating the global
`sim.config` module** per run (`_apply_overrides` / `_save_config` /
`_restore_config`), correct only because each run executes in its own OS process
(`multiprocessing`). ADR-0009 (`proposed`) called for replacing this with explicit
parameter passing; X-10 step 1 (ADR 0016) added a validated schema for the
overridable parameters. This is **step 2**: remove the global mutation.

A full thread-through of an explicit config object would touch every module's
signature and the hot force functions (large, risky). The audit (`audit/05` X-10)
endorsed a staged path.

## Decision

Make the **overridable** parameters context-local, scoped to exactly the ~18
parameters a run may disperse (the rest stay as plain, typed module globals):

- The overridable params are no longer defined in `sim.config`. Their defaults,
  types, and bounds live in `OverridableParams` (`sim/config_schema.py`).
- `sim.config` holds a `contextvars.ContextVar[OverridableParams | None]` and a
  PEP-562 module `__getattr__` that resolves `config.<NAME>` for those names from
  the active context-local config (defaulting to the nominal frozen singleton).
  **Call sites are unchanged** — code still reads `config.<NAME>`.
- `run_simulation` applies a run's overrides with
  `with config.override_context(overrides): …` — which builds (and validates) an
  `OverridableParams` and installs it for the duration, restoring on exit. No
  global is mutated. `_save_config` / `_restore_config` / `_apply_overrides` are
  removed; `run_simulation` gains an explicit `write_output` flag (previously
  inferred from `config_override is None`).

## Consequences

- Positive: **concurrent in-process runs are isolated** (regression test:
  two threads with different overrides do not interfere) — impossible under the
  old global mutation. Removes the save/restore dance and the Q-02 drift surface
  entirely. Monte Carlo still uses processes for CPU parallelism, but no longer
  *needs* isolation for correctness. Unblocks X-14 (vectorised MC), itself gated
  per ADR 0017.
- Behaviour-preserving: the nominal run applies no overrides → the frozen default
  config → **bit-identical** telemetry (SHA-256 unchanged; committed example
  artifacts unchanged; golden-trajectory test passes).
- Trade-offs (deliberately scoped to the 18 overridable params to bound both):
  - **Type info:** `config.<overridable>` resolves through `__getattr__` → `Any`,
    so pyright no longer type-checks those 18 reads (the fixed physical constants
    keep their types). A `.pyi` stub could restore them; deferred.
  - **Overhead:** each overridable read is a `__getattr__` + `ContextVar.get()`.
    Measured at ~3 % on a full ascent (`benchmark.py --full`) — acceptable.
- `sim.config` now imports `sim.config_schema` (one-directional; the schema no
  longer imports config). Source of truth is split by role: fixed constants in
  `config.py`, overridable/dispersible params in `config_schema.py`.

## Notes / Evidence

`sim/config.py` (`__getattr__`, `override_context`, `active_overrides`),
`sim/config_schema.py` (frozen `OverridableParams` with literal defaults),
`sim/main.py` (`run_simulation` / `_run_inner` / CLI). Tests:
`tests/test_config_schema.py::TestContextLocalConfig` (apply/revert, internal-key
handling, validation, concurrent isolation). Verified: full suite 335 passed;
pyright 0 errors on `sim/`; ruff clean; nominal telemetry hash unchanged.
Cross-repo rationale: `audit/05-cross-repo-lessons.md` §B3 / §E (X-10).
