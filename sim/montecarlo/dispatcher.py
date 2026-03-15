"""Monte Carlo run dispatcher with multiprocessing."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sim import config
from sim.montecarlo.dispersions import (
    DEFAULT_DISPERSIONS,
    Dispersion,
    generate_dispersed_config,
)


@dataclass
class MonteCarloResult:
    """Result from a single Monte Carlo simulation run."""

    run_index: int
    seed: int
    outcome: str
    dispersed_params: dict
    insertion_altitude_m: float | None
    insertion_velocity_ms: float | None
    insertion_fpa_deg: float | None
    peak_q_pa: float
    peak_axial_g: float
    peak_ekf_uncertainty_m: float
    boundary_clamp_count: int
    fts_trigger_time_s: float | None
    total_time_s: float


def _run_single(args: tuple[int, int, dict]) -> dict:
    """Worker function for a single Monte Carlo run.

    Args:
        args: (run_index, seed, config_override).

    Returns:
        MonteCarloResult as dict.
    """
    run_index, seed, config_override = args
    try:
        from sim.main import run_simulation

        result = run_simulation(config_override=config_override, quiet=True)
        return asdict(result)
    except Exception as e:
        return asdict(
            MonteCarloResult(
                run_index=run_index,
                seed=seed,
                outcome=f"ERROR: {e!s}",
                dispersed_params=config_override,
                insertion_altitude_m=None,
                insertion_velocity_ms=None,
                insertion_fpa_deg=None,
                peak_q_pa=0.0,
                peak_axial_g=0.0,
                peak_ekf_uncertainty_m=0.0,
                boundary_clamp_count=0,
                fts_trigger_time_s=None,
                total_time_s=0.0,
            )
        )


class MonteCarloDispatcher:
    """Dispatch and manage Monte Carlo simulation campaigns."""

    def __init__(
        self,
        num_runs: int = config.MC_NUM_RUNS,
        dispersions: list[Dispersion] | None = None,
        seed: int = config.MC_SEED,
    ):
        self.num_runs = num_runs
        self.dispersions = dispersions or DEFAULT_DISPERSIONS
        self.seed = seed

    def generate_run_configs(self) -> list[tuple[int, int, dict]]:
        """Generate dispersed config for each run.

        Returns:
            List of (run_index, seed, config_override) tuples.
        """
        configs = []
        for i in range(self.num_runs):
            run_seed = self.seed + i
            rng = np.random.default_rng(run_seed)
            override = generate_dispersed_config(self.dispersions, rng)
            override["_run_index"] = i
            override["_seed"] = run_seed
            configs.append((i, run_seed, override))
        return configs

    def execute(self, workers: int | None = None) -> list[MonteCarloResult]:
        """Run all Monte Carlo simulations.

        Args:
            workers: Number of parallel workers. None = cpu_count.

        Returns:
            List of MonteCarloResult.
        """
        if workers is None:
            workers = config.MC_WORKERS or multiprocessing.cpu_count()

        configs = self.generate_run_configs()
        results: list[MonteCarloResult] = []
        completed = 0
        start_time = time.time()
        last_report = start_time

        print(f"Monte Carlo: launching {self.num_runs} runs on {workers} workers...")

        with multiprocessing.Pool(processes=workers) as pool:
            for result_dict in pool.imap_unordered(_run_single, configs):
                completed += 1
                now = time.time()
                pct = completed / self.num_runs
                if now - last_report > 60 or pct % 0.1 < 1.0 / self.num_runs:
                    elapsed = now - start_time
                    eta = elapsed / pct - elapsed if pct > 0 else 0
                    print(f"  [{completed}/{self.num_runs}] {pct:.0%} complete, elapsed {elapsed:.0f}s, ETA {eta:.0f}s")
                    last_report = now
                results.append(MonteCarloResult(**result_dict))

        elapsed = time.time() - start_time
        print(f"Monte Carlo complete: {self.num_runs} runs in {elapsed:.1f}s")
        return results


def main() -> None:
    """Entry point for Monte Carlo campaign."""
    parser = argparse.ArgumentParser(description="Monte Carlo ascent simulation")
    parser.add_argument("--runs", type=int, default=config.MC_NUM_RUNS, help="Number of runs")
    parser.add_argument("--seed", type=int, default=config.MC_SEED, help="Base random seed")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    args = parser.parse_args()

    dispatcher = MonteCarloDispatcher(num_runs=args.runs, seed=args.seed)
    results = dispatcher.execute(workers=args.workers)

    # Write results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "montecarlo_results.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"Results written to {results_path}")

    # Print summary
    from sim.montecarlo.statistics import print_summary

    print_summary(results)


if __name__ == "__main__":
    main()
