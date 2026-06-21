"""X-14 feasibility spike: is batched / vectorized Monte Carlo worth it?

Standalone micro-experiment backing ADR 0019. It measures the *ceiling* speedup
of vectorizing N trajectories over a leading batch dimension versus running them
one-at-a-time in a Python loop, on a representative translational-dynamics step
(J2 gravity + exponential-atmosphere drag, RK4 — the same integration structure
sim.core.integrator uses, with pure-array force math like the vectorizable part
of the real force model).

It deliberately models only the *vectorizable* core. The decision (ADR 0019)
weighs this ceiling against:
  (a) the existing multiprocessing Monte Carlo, which already scales
      near-linearly across CPU cores (so the bar is ~P x, not 1 x); and
  (b) the cost of vectorizing the *entire* force model + 12-state EKF and
      masking the per-trajectory branchy control flow (staging FSM, FTS abort,
      insertion detection, max-q / G throttle management), which run to
      different step counts per trajectory.

Run: ``python spike_vectorized_mc.py``
"""

from __future__ import annotations

import time

import numpy as np

MU = 3.986004418e14
J2 = 1.08262668e-3
RE = 6_378_137.0
RHO0 = 1.225
SCALE_H = 8500.0
BALLISTIC_COEFF = 2000.0  # m / (Cd*A); drag accel = -0.5 rho |v| v / BC


def accel(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Specific force (...,3): J2 gravity + exponential-atmosphere drag.

    Works for a single trajectory (shape (3,)) or a batch (shape (N,3)) via
    last-axis reductions — identical math either way.
    """
    r = np.sqrt((pos * pos).sum(-1, keepdims=True))
    rhat = pos / r
    g = -MU / r**2 * rhat
    zr2 = (pos[..., 2:3] / r) ** 2
    fac = 1.5 * J2 * (RE / r) ** 2
    # x,y components carry the (5 z^2/r^2 - 1) factor; z carries (5 z^2/r^2 - 3).
    gj2 = -MU / r**2 * fac * rhat * (5.0 * zr2 - 1.0)
    gj2[..., 2:3] = -MU / r**2 * fac * rhat[..., 2:3] * (5.0 * zr2 - 3.0)
    alt = np.clip(r - RE, 0.0, None)
    rho = RHO0 * np.exp(-alt / SCALE_H)
    vmag = np.sqrt((vel * vel).sum(-1, keepdims=True))
    drag = -0.5 * rho * vmag * vel / BALLISTIC_COEFF
    return g + gj2 + drag


def rk4_step(pos: np.ndarray, vel: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    a1 = accel(pos, vel)
    p2, v2 = pos + vel * (dt / 2), vel + a1 * (dt / 2)
    a2 = accel(p2, v2)
    p3, v3 = pos + v2 * (dt / 2), vel + a2 * (dt / 2)
    a3 = accel(p3, v3)
    p4, v4 = pos + v3 * dt, vel + a3 * dt
    a4 = accel(p4, v4)
    pos_n = pos + (dt / 6) * (vel + 2 * v2 + 2 * v3 + v4)
    vel_n = vel + (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
    return pos_n, vel_n


def _initial(n: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.tile([RE + 200_000.0, 0.0, 0.0], (n, 1))
    vel = np.tile([0.0, 7000.0, 1000.0], (n, 1))
    return pos, vel


def run_vectorized(n: int, steps: int, dt: float) -> float:
    pos, vel = _initial(n)
    t0 = time.perf_counter()
    for _ in range(steps):
        pos, vel = rk4_step(pos, vel, dt)
    return time.perf_counter() - t0


def run_sequential(n: int, steps: int, dt: float) -> float:
    pos_all, vel_all = _initial(n)
    t0 = time.perf_counter()
    for i in range(n):
        pos, vel = pos_all[i], vel_all[i]
        for _ in range(steps):
            pos, vel = rk4_step(pos, vel, dt)
    return time.perf_counter() - t0


def main() -> None:
    n, steps, dt = 2000, 2000, 0.01
    print(f"X-14 spike: {n} trajectories x {steps} steps (representative force-math core)\n")

    # warm-up
    run_vectorized(64, 10, dt)
    run_sequential(8, 10, dt)

    t_vec = run_vectorized(n, steps, dt)
    t_seq = run_sequential(n, steps, dt)

    traj_vec = n / t_vec
    traj_seq = n / t_seq
    speedup = t_seq / t_vec

    print(f"  vectorized batch : {t_vec:7.3f} s  ({traj_vec:8.1f} traj/s)")
    print(f"  sequential loop  : {t_seq:7.3f} s  ({traj_seq:8.1f} traj/s)")
    print(f"  vectorization speedup (seq/vec): {speedup:6.1f} x\n")

    import os

    cores = os.cpu_count() or 1
    print(f"  CPU cores available: {cores}")
    print(f"  multiprocessing MC scales sequential by ~cores => bar to beat ~ {cores} x")
    print(
        f"  => vectorization is ~{speedup:.0f}x vs 1 core, ~{speedup / cores:.1f}x vs the "
        f"{cores}-core multiprocessing pool on THIS (fully vectorizable) core."
    )
    print("\n  Caveat: only the translational force math is modelled here. The full sim also")
    print("  needs the 12-state EKF, the Mach-spline aero, sensors, and the branchy per-")
    print("  trajectory control flow (staging / FTS abort / insertion / throttle limits),")
    print("  which do not vectorize without a large rewrite + masking. See ADR 0019.")


if __name__ == "__main__":
    main()
