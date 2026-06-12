# 0004. Monte Carlo via process isolation + transient global-config override

- Status: accepted
- Date: 2026-06-12 (backfilled)
- Deciders: original authors

## Context and Problem Statement

Monte Carlo dispersion analysis runs the full simulation many times (default
`MC_NUM_RUNS = 1000`) with per-run perturbed parameters. Parameters live in the
global `sim.config` module (ADR 0001), so each run needs its own parameter set
without corrupting others, and runs should use multiple CPU cores.

## Decision

Use `multiprocessing.Pool` (`sim/montecarlo/dispatcher.py`) so each run executes
in a separate OS process with its own copy of the `sim.config` module. Within a
process, `run_simulation` applies a dispersion dict by setting `config`
attributes and restores them in a `finally` block (`sim/main.py:_save_config /
_restore_config / _apply_overrides`).

## Consequences

- Positive: near-linear speedup across cores; per-process config isolation makes
  the global-state override safe in the parallel path.
- Negative: the in-process save/restore relies on a hardcoded key list that must
  stay in sync with the dispersion set (finding Q-02). It is correct today but
  fragile; a custom dispersion targeting an unlisted key would not be restored
  after an in-process call. ADR 0009 proposes removing the global-override
  pattern.
- Negative: results cross the process boundary as dataclass dicts, requiring
  `MonteCarloResult` to be picklable (it is — plain fields).

## Notes / Evidence

`dispatcher.execute` uses `pool.imap_unordered(_run_single, configs)`;
`generate_run_configs` seeds each run as `MC_SEED + i` for reproducibility. The
save/restore key list is `sim/main.py:86-103`; all 13 `DEFAULT_DISPERSIONS` keys
are covered by it today (verified this session).
