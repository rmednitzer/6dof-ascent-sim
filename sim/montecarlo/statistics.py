"""Monte Carlo statistics and dispersion analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.montecarlo.dispatcher import MonteCarloResult


def compute_statistics(results: list[MonteCarloResult]) -> dict:
    """Compute summary statistics from Monte Carlo results.

    Args:
        results: List of MonteCarloResult from campaign.

    Returns:
        Dictionary of computed statistics.
    """
    n = len(results)
    outcomes = [r.outcome for r in results]
    n_success = sum(1 for o in outcomes if o == "SUCCESS")
    n_fts = sum(1 for o in outcomes if o == "FTS_ABORT")
    n_timeout = sum(1 for o in outcomes if o == "TIMEOUT")
    n_error = n - n_success - n_fts - n_timeout

    stats: dict = {
        "total_runs": n,
        "success_count": n_success,
        "success_rate": n_success / n if n > 0 else 0,
        "fts_abort_count": n_fts,
        "fts_abort_rate": n_fts / n if n > 0 else 0,
        "timeout_count": n_timeout,
        "timeout_rate": n_timeout / n if n > 0 else 0,
        "error_count": n_error,
    }

    # Insertion accuracy for successful runs
    successful = [r for r in results if r.outcome == "SUCCESS"]
    if successful:
        alts = np.array([r.insertion_altitude_m for r in successful if r.insertion_altitude_m is not None])
        vels = np.array([r.insertion_velocity_ms for r in successful if r.insertion_velocity_ms is not None])
        fpas = np.array([r.insertion_fpa_deg for r in successful if r.insertion_fpa_deg is not None])

        if len(alts) > 0:
            stats["insertion_altitude"] = {
                "mean_km": float(np.mean(alts) / 1000),
                "std_km": float(np.std(alts) / 1000),
                "min_km": float(np.min(alts) / 1000),
                "max_km": float(np.max(alts) / 1000),
                "three_sigma_low_km": float((np.mean(alts) - 3 * np.std(alts)) / 1000),
                "three_sigma_high_km": float((np.mean(alts) + 3 * np.std(alts)) / 1000),
            }
        if len(vels) > 0:
            stats["insertion_velocity"] = {
                "mean_ms": float(np.mean(vels)),
                "std_ms": float(np.std(vels)),
                "three_sigma_low_ms": float(np.mean(vels) - 3 * np.std(vels)),
                "three_sigma_high_ms": float(np.mean(vels) + 3 * np.std(vels)),
            }
        if len(fpas) > 0:
            stats["insertion_fpa"] = {
                "mean_deg": float(np.mean(fpas)),
                "std_deg": float(np.std(fpas)),
            }

    # Limit proximity (all runs)
    from sim import config

    peak_q = np.array([r.peak_q_pa for r in results])
    peak_g = np.array([r.peak_axial_g for r in results])
    peak_ekf = np.array([r.peak_ekf_uncertainty_m for r in results])
    clamps = np.array([r.boundary_clamp_count for r in results])

    stats["limit_proximity"] = {
        "peak_q_pct": {
            "mean": float(np.mean(peak_q) / config.MAX_Q_PA * 100),
            "max": float(np.max(peak_q) / config.MAX_Q_PA * 100),
            "p99": float(np.percentile(peak_q, 99) / config.MAX_Q_PA * 100) if n > 0 else 0,
        },
        "peak_g_pct": {
            "mean": float(np.mean(peak_g) / config.MAX_AXIAL_G * 100),
            "max": float(np.max(peak_g) / config.MAX_AXIAL_G * 100),
            "p99": float(np.percentile(peak_g, 99) / config.MAX_AXIAL_G * 100) if n > 0 else 0,
        },
    }
    stats["boundary_clamps"] = {
        "mean": float(np.mean(clamps)),
        "max": int(np.max(clamps)),
    }

    return stats


def print_summary(results: list[MonteCarloResult]) -> None:
    """Print Monte Carlo summary to stdout.

    Args:
        results: List of MonteCarloResult.
    """
    stats = compute_statistics(results)
    n = stats["total_runs"]

    print(f"\nMonte Carlo Summary (N={n})")
    print("=" * 40)
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"FTS abort rate: {stats['fts_abort_rate']:.1%}")
    print(f"Timeout rate: {stats['timeout_rate']:.1%}")
    if stats["error_count"] > 0:
        print(f"Error count: {stats['error_count']}")

    if "insertion_altitude" in stats:
        ia = stats["insertion_altitude"]
        print(f"\nInsertion accuracy (successful runs):")
        print(f"  Altitude:  mean {ia['mean_km']:.1f} km, std {ia['std_km']:.1f} km, "
              f"3σ range [{ia['three_sigma_low_km']:.1f}, {ia['three_sigma_high_km']:.1f}] km")
    if "insertion_velocity" in stats:
        iv = stats["insertion_velocity"]
        print(f"  Velocity:  mean {iv['mean_ms']:.1f} m/s, std {iv['std_ms']:.1f} m/s, "
              f"3σ range [{iv['three_sigma_low_ms']:.1f}, {iv['three_sigma_high_ms']:.1f}] m/s")
    if "insertion_fpa" in stats:
        fp = stats["insertion_fpa"]
        print(f"  FPA:       mean {fp['mean_deg']:.2f}°, std {fp['std_deg']:.2f}°")

    lp = stats["limit_proximity"]
    print(f"\nLimit proximity (all runs):")
    print(f"  Peak Q:    mean {lp['peak_q_pct']['mean']:.1f}%, "
          f"max {lp['peak_q_pct']['max']:.1f}%, P99 {lp['peak_q_pct']['p99']:.1f}%")
    print(f"  Peak G:    mean {lp['peak_g_pct']['mean']:.1f}%, "
          f"max {lp['peak_g_pct']['max']:.1f}%, P99 {lp['peak_g_pct']['p99']:.1f}%")

    bc = stats["boundary_clamps"]
    print(f"\nBoundary clamps: mean {bc['mean']:.1f}, max {bc['max']}")


def generate_plots(results: list[MonteCarloResult], output_dir: str = "output/montecarlo_plots") -> None:
    """Generate Monte Carlo dispersion plots.

    Args:
        results: List of MonteCarloResult.
        output_dir: Directory for plot output.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    successful = [r for r in results if r.outcome == "SUCCESS"]
    if not successful:
        print("No successful runs — skipping Monte Carlo plots.")
        return

    # Dispersion ellipse: altitude vs velocity
    alts = np.array([r.insertion_altitude_m / 1000 for r in successful
                     if r.insertion_altitude_m is not None])
    vels = np.array([r.insertion_velocity_ms for r in successful
                     if r.insertion_velocity_ms is not None])

    if len(alts) > 1 and len(vels) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(vels, alts, alpha=0.3, s=10, label="Runs")
        from sim import config
        ax.axhline(config.TARGET_ALTITUDE_M / 1000, color="r", linestyle="--", label="Target altitude")
        ax.axvline(config.TARGET_VELOCITY_MS, color="g", linestyle="--", label="Target velocity")
        ax.set_xlabel("Insertion Velocity (m/s)")
        ax.set_ylabel("Insertion Altitude (km)")
        ax.set_title("Monte Carlo Dispersion: Insertion State")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out / "dispersion_ellipse.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Limit proximity CDF
    from sim import config
    peak_q_frac = np.array([r.peak_q_pa / config.MAX_Q_PA for r in results])
    peak_g_frac = np.array([r.peak_axial_g / config.MAX_AXIAL_G for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(peak_q_frac * 100, bins=50, cumulative=True, density=True, alpha=0.7)
    ax1.set_xlabel("Peak Q / Structural Limit (%)")
    ax1.set_ylabel("CDF")
    ax1.set_title("Dynamic Pressure Limit Proximity")
    ax1.axvline(100, color="r", linestyle="--", label="Limit")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(peak_g_frac * 100, bins=50, cumulative=True, density=True, alpha=0.7)
    ax2.set_xlabel("Peak G / G Limit (%)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Axial G Limit Proximity")
    ax2.axvline(100, color="r", linestyle="--", label="Limit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(out / "limit_proximity_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Boundary clamp histogram
    clamps = [r.boundary_clamp_count for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clamps, bins=max(1, max(clamps) - min(clamps) + 1) if clamps else 10, alpha=0.7)
    ax.set_xlabel("Boundary Clamp Count")
    ax.set_ylabel("Number of Runs")
    ax.set_title("Boundary Enforcer Activity Distribution")
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "boundary_clamp_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Success/fail scatter by outcome
    outcomes_color = {"SUCCESS": "green", "FTS_ABORT": "red", "TIMEOUT": "orange"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for outcome, color in outcomes_color.items():
        subset = [r for r in results if r.outcome == outcome]
        if subset:
            ax.scatter(
                [r.total_time_s for r in subset],
                [r.peak_q_pa / 1000 for r in subset],
                c=color, alpha=0.4, s=10, label=outcome,
            )
    ax.set_xlabel("Total Simulation Time (s)")
    ax.set_ylabel("Peak Dynamic Pressure (kPa)")
    ax.set_title("Outcomes by Time and Peak Q")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "outcome_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Monte Carlo plots saved to {out}/")
