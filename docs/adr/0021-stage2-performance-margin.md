# 0021. Stage-2 Isp performance margin (N-01)

- Status: accepted
- Date: 2026-06-22
- Deciders: GNC / vehicle eng (BACKLOG N-01)
- Related: ADR 0020 (estimated-attitude closed loop, surfaced the abort rate),
  ADR 0014 (FTS attitude limit), AD-17 (PEG terminal margin, assumptions.md)

## Context and Problem Statement

The Stage-2 paired Monte-Carlo campaign (ADR 0020) recorded a ~33% FTS-abort rate
over the dispersed seeds. BACKLOG N-01 attributed this to the `IMU_*_BIAS`
dispersions inflating the EKF position covariance past the 10 km FTS limit. A
per-seed diagnosis **overturned that hypothesis**: of the 8 aborts, 7 were
thrust-axis attitude trips (25–27°) near insertion and only 1 was a covariance
trip. Tracing a representative seed showed the mechanism — the upper stage
reaches the terminal phase slightly under-performing, the S2 tank empties a few
tens of m/s short of orbit, and the **unpowered** vehicle then drifts at its
residual ~2.4°/s body rate (no thrust ⇒ no TVC authority) until the thrust-axis
error crosses the 25° FTS limit. The dominant cause is therefore a **performance
shortfall**, not navigation, consistent with the AD-17 note that "every dispersed
abort is a marginal ~25° thrust-axis hit near insertion." The nominal had almost
no margin: it inserted into a 141 × 434 km orbit, periapsis just 0.7 km above the
140 km survivable floor.

## Decision

**Raise the Stage-2 vacuum Isp default from 348 to 356 s (+2.3%)**
(`OverridableParams.S2_ISP_VAC_S` in `sim/config_schema.py`) — a higher-Isp upper
stage that reaches orbit with margin, so adverse propulsion/drag dispersions no
longer fall short.

The N-01 options were "tighten IMU dispersion", "revisit the FTS limit", or
"improve performance margin"; the chosen direction was performance margin, and
between its two natural levers — propellant vs engine performance — **Isp is the
correct one and propellant is not**:

- **Propellant (rejected).** Adding S2 propellant raises liftoff mass, and the
  nominal trajectory is *chaotically sensitive* to it: a clean fresh-process
  sweep flips SUCCESS↔abort across small steps (92.7 t→ok, 94 t→abort, 96 t→ok,
  98–100 t→abort, 102 t→ok, 104 t→abort), and the one "success island" (102 t)
  flew a 30 s-longer, more-lofted trajectory with 3.4× the nominal EKF
  uncertainty (1.66 km → 5.6 km). That is a fragile knife-edge, not a margin.
- **Isp (chosen).** Raising Isp adds upper-stage impulse *without* changing
  liftoff mass, so the nominal trajectory is preserved (across 350–360 s it stays
  SUCCESS with EKF uncertainty ~1.7 km and burn time ~490 s). Changing the schema
  default updates both the real engine (`STAGE_2.isp_vac`, set from the default at
  import — lower `mdot`, more total impulse) and the guidance Isp model
  (`config.S2_ISP_VAC_S`), and keeps the Monte-Carlo dispersion centred on the new
  value.

356 s is the knee: it recovers the propellant-shortfall aborts while leaving the
nominal at baseline; the residual aborts are not Isp-limited.

## Validation

- **Nominal preserved** — SUCCESS, insertion 407.3 km / 7602 m/s, peak-q and
  axial-g unchanged, EKF uncertainty ~1.7 km (vs 1.66 km baseline), burn time
  ~492 s (vs 488 s). Golden baseline and `examples/output/` artifacts regenerated.
- **Dispersed abort rate** — fresh 24-seed estimated-attitude campaign:
  **33% → 12.5%** (success 16/24 → 21/24); no previously-successful seed regressed.
- **Regression guard** —
  `tests/test_e2e_simulation.py::TestPerformanceMargin` pins a former
  performance-shortfall seed (49) to SUCCESS.
- Full `pytest`, `ruff`, `pyright` pass.

## Consequences

- **Positive:** the dominant dispersed failure mode is cleared with a single,
  stable, low-blast-radius parameter change; the nominal is untouched.
- **Cost:** the modelled S2 engine is 2.3% more efficient — a design assumption,
  recorded here, not a free lunch.
- **Residual (out of scope):** a few minority abort modes remain — the terminal
  control sensitivity (seed 42, aborts even with favourable propulsion) and, run
  to run, an occasional IMU-covariance or staging-region trip. These are distinct
  from performance margin and, with the nominal's chaotic mass-sensitivity noted
  above, point at the deeper AD-17 PEG terminal-margin hardening as the next
  follow-up. Tracked under BACKLOG N-01.

## Notes / Evidence

Diagnosis: per-seed FTS-reason categorisation; a terminal-phase trace showing the
post-cutoff tumble (TVC saturated at ±5° producing zero torque, body rate frozen
at the residual ~2.4°/s); and clean fresh-process (`max_tasks_per_child=1`)
parameter sweeps — earlier worker-reuse / config-vs-stage-inconsistent sweeps gave
misleading results and were discarded.
