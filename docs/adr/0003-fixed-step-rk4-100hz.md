# 0003. Fixed-step 100 Hz RK4 with a physics-agnostic integrator

- Status: accepted
- Date: 2026-06-12 (backfilled)
- Deciders: original authors

## Context and Problem Statement

The vehicle state (position, velocity, attitude quaternion, body rate, mass)
must be integrated through powered ascent. Options include fixed-step explicit
schemes (Euler, RK4) and adaptive integrators (`scipy.integrate.solve_ivp`).

## Decision

Integrate with a fixed-step 4th-order Runge-Kutta scheme at `DT = 0.01 s`
(100 Hz), implemented in `sim/core/integrator.py`. The integrator is
physics-agnostic: the main loop builds a `derivatives_fn` closure capturing all
forces/torques for the step and passes it to `rk4_step`. Forces are held
constant across the four RK4 sub-stages (zero-order hold). After each step the
quaternion is re-normalized and a NaN/Inf guard runs (raising `RuntimeError`).

## Consequences

- Positive: deterministic, real-time-like cadence matching the GNC/telemetry
  rates (100 Hz internal, 10 Hz downlink); simple and fast; the NaN/Inf guard
  turns divergence into an explicit error instead of silent garbage.
- Positive: separating integration from physics keeps `integrator.py` at 98%
  coverage and independently testable.
- Negative: zero-order-hold on forces introduces O(dt) error in the force model
  even though the state integration is O(dt^4); acceptable at 100 Hz for this
  fidelity. Fixed step cannot adapt to stiff transients (staging, max-q).

## Notes / Evidence

`sim/core/integrator.py:136-147` performs the re-normalization and finite
checks. `docs/architecture.md` documents the closure pattern. The 100 Hz / 10 Hz
choice ties to `INTERNAL_HZ`/`TELEMETRY_HZ` in `config.py`.
