# 0001. Single `config.py` source of truth for all parameters

- Status: accepted
- Date: 2026-06-12 (backfilled)
- Deciders: original authors (rationale inferred from code; see Notes)

## Context and Problem Statement

A 6-DOF ascent simulation has ~100 physical, vehicle, GNC, safety, and Monte
Carlo constants. These must be consistent across many modules and overridable
for dispersion analysis.

## Decision

All tunable constants live in a single flat module, `sim/config.py`, imported by
every other module. There is no layered config system, no YAML/TOML runtime
config, and no per-module defaults that could drift.

## Consequences

- Positive: one place to read/change any parameter; trivially greppable;
  `config.py` reaches 100% test coverage because everything imports it.
- Positive: Monte Carlo can override parameters by setting module attributes
  (see ADR 0004).
- Negative: the module is global mutable state. In-process overrides must be
  saved/restored carefully (see finding Q-02), and concurrent in-process runs
  would interfere — mitigated today by process isolation (ADR 0004).

## Notes / Evidence

`sim/config.py` is imported as `from sim import config` across the package; the
README and `docs/architecture.md` both describe it as the "single source of
truth." The specific tradeoff against a structured config object is not
documented in history and is inferred from the code.
