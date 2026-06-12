# 0009. Replace mutable-global config override with explicit parameter passing

- Status: proposed
- Date: 2026-06-12
- Deciders: maintainers (pending)

## Context and Problem Statement

Monte Carlo dispersion currently mutates the global `sim.config` module and
restores it in a `finally` block (ADR 0004). This works under process isolation
but has sharp edges:

- The save/restore key list in `sim/main.py:_save_config` is hardcoded and must
  be kept in sync with `DEFAULT_DISPERSIONS`; a custom dispersion on an unlisted
  key would leak across in-process calls (finding Q-02).
- In-process callers (tests, notebooks, any future single-process driver) can
  observe mutated globals if a run raises before restore.
- It blocks in-process parallelism (threads) entirely.

## Considered Options

1. Keep globals; make `_save_config` snapshot exactly the override keys
   dynamically (small, removes Q-02 but keeps global mutation).
2. Thread an immutable per-run parameter object (e.g. a frozen dataclass) from
   `run_simulation` down through the models, removing reliance on module globals
   for dispersed values.
3. Status quo.

## Decision Outcome (proposed)

Prefer Option 2 long-term (clean, enables in-process concurrency, no
save/restore), with Option 1 as a low-risk interim that immediately closes the
Q-02 drift. Option 2 is a structural refactor touching many modules and should
be sequenced behind ADR 0008 (an end-to-end test) so the refactor is guarded.

## Consequences

- Positive (Option 2): no global mutation; deterministic in-process runs;
  removes a class of latent bugs.
- Negative (Option 2): broad change surface (every model reads some `config`
  values); needs the integration test from ADR 0008 first.

## Notes / Evidence

`sim/main.py:84-122` (save/restore/apply); `sim/montecarlo/dispersions.py`
(`DEFAULT_DISPERSIONS`); findings Q-02, Q-03, Q-05.
