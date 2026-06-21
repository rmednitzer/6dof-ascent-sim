"""Performance benchmarks for the ascent simulation.

Run all: ``python benchmark.py``
End-to-end only: ``python benchmark.py --full [--t-max 120] [--repeats 3]``
Profile the hot loop: ``python benchmark.py --profile [--t-max 120]``

The end-to-end benchmark exercises the real per-step hot loop (environment,
propulsion, aero, GNC, EKF, structural dynamics, RK4) — i.e. where time is
actually spent. ``--profile`` prints a ``cProfile`` breakdown so optimisation
targets are chosen from evidence (the "profile before optimizing" discipline,
audit/05 §C4). It was this harness that showed numba-JIT of the integrator
alone is not worth a heavy LLVM dependency (the RK4 path is ~8% of runtime; the
cost is spread across the force model, the EKF, and small-array allocations) —
see ADR 0017.
"""

from __future__ import annotations

import argparse
import timeit


def benchmark_full_run(t_max: float = 120.0, repeats: int = 3) -> None:
    """End-to-end wall-clock benchmark of the nominal ascent loop."""
    import time

    from sim import config
    from sim.main import run_simulation

    steps = int(t_max / config.DT)
    original_t_max = config.T_MAX
    config.T_MAX = t_max
    try:
        run_simulation(config_override={}, quiet=True)  # warm-up (imports, caches)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_simulation(config_override={}, quiet=True)
            best = min(best, time.perf_counter() - t0)
    finally:
        config.T_MAX = original_t_max

    print(f"Full ascent run (T_MAX={t_max:.0f}s, {steps} steps, best of {repeats}):")
    print(f"  wall:       {best:.3f} s")
    print(f"  per step:   {best / steps * 1e6:.1f} µs")
    print(f"  throughput: {steps / best:,.0f} steps/s")


def profile_full_run(t_max: float = 120.0, top: int = 25) -> None:
    """Print a cProfile breakdown of a nominal run (by internal time)."""
    import cProfile
    import io
    import pstats

    from sim import config
    from sim.main import run_simulation

    original_t_max = config.T_MAX
    config.T_MAX = t_max
    try:
        pr = cProfile.Profile()
        pr.enable()
        run_simulation(config_override={}, quiet=True)
        pr.disable()
    finally:
        config.T_MAX = original_t_max

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
    print(s.getvalue())


def benchmark_flex_body() -> None:
    from sim.dynamics.flex_body import FlexBody

    fb = FlexBody()

    def test_velocities():
        return fb.modal_velocities()

    def test_displacements():
        return fb.modal_displacements()

    def test_update():
        fb.update(dt=0.01, tvc_force_n=1000.0, propellant_fraction=0.8)

    n = 100000
    t_vel = timeit.timeit(test_velocities, number=n)
    t_disp = timeit.timeit(test_displacements, number=n)
    t_upd = timeit.timeit(test_update, number=n)

    print(f"FlexBody (n={n}):")
    print(f"  modal_velocities:    {t_vel:.4f}s ({t_vel / n * 1e6:.2f} µs/call)")
    print(f"  modal_displacements: {t_disp:.4f}s ({t_disp / n * 1e6:.2f} µs/call)")
    print(f"  update:              {t_upd:.4f}s ({t_upd / n * 1e6:.2f} µs/call)")


def benchmark_slosh() -> None:
    from sim.dynamics.slosh import SloshModel

    sm = SloshModel(n_tanks=2)

    def test_angles():
        return sm.pendulum_angles()

    def test_rates():
        return sm.pendulum_rates()

    def test_update():
        sm.update(dt=0.01, lateral_accel_mps2=5.0, propellant_mass_kg=100000.0, propellant_fraction=0.8)

    n = 100000
    t_ang = timeit.timeit(test_angles, number=n)
    t_rate = timeit.timeit(test_rates, number=n)
    t_upd = timeit.timeit(test_update, number=n)

    print(f"SloshModel (n={n}):")
    print(f"  pendulum_angles: {t_ang:.4f}s ({t_ang / n * 1e6:.2f} µs/call)")
    print(f"  pendulum_rates:  {t_rate:.4f}s ({t_rate / n * 1e6:.2f} µs/call)")
    print(f"  update:          {t_upd:.4f}s ({t_upd / n * 1e6:.2f} µs/call)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ascent simulation benchmarks")
    parser.add_argument("--full", action="store_true", help="End-to-end run benchmark only")
    parser.add_argument("--profile", action="store_true", help="cProfile breakdown of a nominal run")
    parser.add_argument("--t-max", type=float, default=120.0, help="Sim duration for the full/profile run (s)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats for the full-run benchmark")
    args = parser.parse_args()

    if args.profile:
        profile_full_run(t_max=args.t_max)
        return
    if args.full:
        benchmark_full_run(t_max=args.t_max, repeats=args.repeats)
        return

    benchmark_full_run(t_max=args.t_max, repeats=args.repeats)
    print()
    benchmark_flex_body()
    print()
    benchmark_slosh()


if __name__ == "__main__":
    main()
