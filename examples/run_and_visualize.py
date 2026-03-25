#!/usr/bin/env python3
"""Example: run a nominal ascent simulation and generate a visualization dashboard.

Usage:
    python examples/run_and_visualize.py
    python examples/run_and_visualize.py --no-flex --no-slosh

Produces:
    examples/output/dashboard.png   — 8-panel mission dashboard
    examples/output/ground_track.png — ground track plot
    examples/output/mission_summary.txt — text summary
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Ensure the repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import config
from sim.core.reference_frames import ecef_to_lla, eci_to_ecef
from sim.main import run_simulation


def _extract_ground_track(frames):
    """Compute lat/lon ground track from ECI positions."""
    lats, lons = [], []
    for f in frames:
        pos_eci = np.array(f.position_eci_m)
        pos_ecef = eci_to_ecef(pos_eci, f.time_s)
        lat, lon, _ = ecef_to_lla(pos_ecef)
        lats.append(math.degrees(lat))
        lons.append(math.degrees(lon))
    return lats, lons


def _find_staging_time(frames):
    """Return time of stage separation (stage 1 -> 2 transition)."""
    for i in range(1, len(frames)):
        if frames[i].stage != frames[i - 1].stage:
            return frames[i].time_s
    return None


def generate_dashboard(frames, summary, output_dir: Path) -> None:
    """Create an 8-panel mission dashboard figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    output_dir.mkdir(parents=True, exist_ok=True)

    times = [f.time_s for f in frames]
    if not times:
        print("No telemetry frames to plot.")
        return

    staging_t = _find_staging_time(frames)

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle("6-DOF Ascent Simulation — Mission Dashboard", fontsize=16, fontweight="bold", y=0.98)

    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.30)

    # Color scheme
    c_primary = "#2563eb"
    c_limit = "#dc2626"
    c_staging = "#f59e0b"
    c_maxq = "#8b5cf6"

    def _add_staging_line(ax):
        if staging_t is not None:
            ax.axvline(staging_t, color=c_staging, linestyle="--", alpha=0.6, label=f"Staging @ {staging_t:.0f}s")

    # --- 1. Altitude vs Time ---
    ax = fig.add_subplot(gs[0, 0])
    alts = [f.altitude_m / 1000 for f in frames]
    ax.plot(times, alts, color=c_primary, linewidth=1.2)
    ax.axhline(config.TARGET_ALTITUDE_M / 1000, color=c_limit, linestyle="--", alpha=0.7, label="Target")
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude Profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. Velocity vs Time ---
    ax = fig.add_subplot(gs[0, 1])
    vels = [f.velocity_mag_ms for f in frames]
    ax.plot(times, vels, color=c_primary, linewidth=1.2)
    ax.axhline(config.TARGET_VELOCITY_MS, color=c_limit, linestyle="--", alpha=0.7, label="Target")
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Inertial Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 3. Dynamic Pressure ---
    ax = fig.add_subplot(gs[1, 0])
    q_vals = [f.dynamic_pressure_pa / 1000 for f in frames]
    ax.plot(times, q_vals, color=c_primary, linewidth=1.2)
    ax.axhline(config.MAX_Q_PA / 1000, color=c_limit, linestyle="--", alpha=0.7, label="Structural Limit")
    max_q_idx = int(np.argmax(q_vals))
    ax.axvline(times[max_q_idx], color=c_maxq, linestyle=":", alpha=0.7, label=f"Max-Q @ {times[max_q_idx]:.0f}s")
    ax.fill_between(times, q_vals, alpha=0.15, color=c_primary)
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dynamic Pressure (kPa)")
    ax.set_title("Dynamic Pressure")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 4. Axial G-Load ---
    ax = fig.add_subplot(gs[1, 1])
    g_vals = [f.axial_g for f in frames]
    ax.plot(times, g_vals, color=c_primary, linewidth=1.2)
    ax.axhline(config.MAX_AXIAL_G, color=c_limit, linestyle="--", alpha=0.7, label="Structural Limit")
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Axial Acceleration (g)")
    ax.set_title("Axial G-Load")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 5. Mass vs Time ---
    ax = fig.add_subplot(gs[2, 0])
    masses = [f.mass_kg / 1000 for f in frames]
    ax.plot(times, masses, color=c_primary, linewidth=1.2)
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (tonnes)")
    ax.set_title("Vehicle Mass")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 6. Throttle & Thrust ---
    ax = fig.add_subplot(gs[2, 1])
    throttles = [f.throttle * 100 for f in frames]
    ax.plot(times, throttles, color=c_primary, linewidth=1.2, label="Throttle")
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throttle (%)")
    ax.set_title("Throttle Command")
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 7. EKF Position Uncertainty ---
    ax = fig.add_subplot(gs[3, 0])
    ekf_vals = [f.ekf_position_uncertainty_m for f in frames]
    ax.plot(times, ekf_vals, color=c_primary, linewidth=1.2)
    ax.axhline(config.FTS_COVARIANCE_LIMIT_M, color=c_limit, linestyle="--", alpha=0.7, label="FTS Limit")
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Uncertainty (m)")
    ax.set_title("EKF Navigation Uncertainty")
    if max(ekf_vals) > 100:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 8. Mach Number ---
    ax = fig.add_subplot(gs[3, 1])
    machs = [f.mach_number for f in frames]
    ax.plot(times, machs, color=c_primary, linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Mach 1")
    _add_staging_line(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mach Number")
    ax.set_title("Mach Number")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Summary text box ---
    summary_text = (
        f"Outcome: {summary.outcome}\n"
        f"Flight time: {summary.final_time_s:.1f} s\n"
        f"Final alt: {summary.final_altitude_m / 1000:.1f} km\n"
        f"Final vel: {summary.final_velocity_ms:.0f} m/s\n"
        f"Peak Q: {summary.peak_dynamic_pressure_pa / 1000:.1f} kPa\n"
        f"Peak G: {summary.peak_axial_g:.2f} g\n"
        f"Boundary violations: {summary.total_boundary_violations}"
    )
    fig.text(
        0.5,
        0.005,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f9ff", "edgecolor": "#93c5fd", "alpha": 0.9},
    )

    fig.savefig(output_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved to {output_dir / 'dashboard.png'}")


def generate_ground_track(frames, output_dir: Path) -> None:
    """Create a ground track plot showing the trajectory over Earth's surface."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats, lons = _extract_ground_track(frames)
    if not lats:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot trajectory colored by altitude
    alts = [f.altitude_m / 1000 for f in frames]
    scatter = ax.scatter(lons, lats, c=alts, cmap="plasma", s=1, zorder=2)
    cbar = fig.colorbar(scatter, ax=ax, label="Altitude (km)", shrink=0.8)  # noqa: F841

    # Mark launch site and insertion
    ax.plot(lons[0], lats[0], "g^", markersize=12, label="Launch", zorder=3)
    ax.plot(lons[-1], lats[-1], "r*", markersize=14, label="Insertion", zorder=3)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Ground Track (colored by altitude)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.savefig(output_dir / "ground_track.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ground track saved to {output_dir / 'ground_track.png'}")


def write_summary_text(summary, output_dir: Path) -> None:
    """Write a plain-text mission summary."""
    lines = [
        "6-DOF Ascent Simulation — Mission Summary",
        "=" * 50,
        f"Outcome:               {summary.outcome}",
        f"Flight time:           {summary.final_time_s:.1f} s",
        f"Final altitude:        {summary.final_altitude_m / 1000:.1f} km",
        f"Final velocity:        {summary.final_velocity_ms:.1f} m/s",
        f"Final mass:            {summary.final_mass_kg:.0f} kg",
        f"Final stage:           {summary.final_stage}",
        "",
        "Peak Values",
        "-" * 50,
        f"Peak altitude:         {summary.peak_altitude_m / 1000:.1f} km",
        f"Peak velocity:         {summary.peak_velocity_ms:.1f} m/s",
        f"Peak dynamic pressure: {summary.peak_dynamic_pressure_pa:.0f} Pa ({summary.peak_dynamic_pressure_pa / config.MAX_Q_PA * 100:.1f}% of limit)",
        f"Peak axial G:          {summary.peak_axial_g:.2f} g ({summary.peak_axial_g / config.MAX_AXIAL_G * 100:.1f}% of limit)",
        f"Peak lateral G:        {summary.peak_lateral_g:.3f} g",
        f"Peak Mach:             {summary.peak_mach_number:.1f}",
        "",
        "Safety",
        "-" * 50,
        f"Boundary violations:   {summary.total_boundary_violations}",
        f"FTS triggered:         {summary.fts_triggered}",
        f"Health status:         {summary.health_status_final}",
        "",
        f"Telemetry frames:      {summary.total_frames_internal} (internal), {summary.total_frames_downlink} (downlink)",
        f"Telemetry hash:        {summary.telemetry_hash_sha256}",
    ]
    text = "\n".join(lines) + "\n"

    path = output_dir / "mission_summary.txt"
    path.write_text(text)
    print(f"  Summary saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 6-DOF ascent simulation with visualization")
    parser.add_argument("--no-flex", action="store_true", help="Disable flex body model")
    parser.add_argument("--no-slosh", action="store_true", help="Disable propellant slosh model")
    args = parser.parse_args()

    if args.no_flex:
        config.FLEX_ENABLED = False
    if args.no_slosh:
        config.SLOSH_ENABLED = False

    output_dir = Path(__file__).resolve().parent / "output"

    print("6-DOF Ascent Simulation — Example Run")
    print("=" * 50)
    print(f"Target orbit: {config.TARGET_ALTITUDE_M / 1000:.0f} km, {config.TARGET_INCLINATION_DEG} deg inclination")
    print(f"Flex body: {'ON' if config.FLEX_ENABLED else 'OFF'}")
    print(f"Slosh model: {'ON' if config.SLOSH_ENABLED else 'OFF'}")
    print()

    # Run simulation
    result = run_simulation()

    print(f"\nOutcome: {result.outcome}")
    print(f"Flight time: {result.total_time_s:.1f} s")
    if result.insertion_altitude_m is not None:
        print(f"Insertion altitude: {result.insertion_altitude_m / 1000:.1f} km")
        print(f"Insertion velocity: {result.insertion_velocity_ms:.1f} m/s")
    print()

    # Load the telemetry frames written by run_simulation
    import json

    telemetry_path = Path("output/telemetry_internal.json")
    summary_path = Path("output/mission_summary.json")

    if not telemetry_path.exists():
        print("Error: telemetry output not found. Simulation may have failed.")
        sys.exit(1)

    from sim.telemetry.schemas import MissionSummary, TelemetryFrame

    with open(telemetry_path) as f:
        raw_frames = json.load(f)
    frames = [TelemetryFrame(**frame) for frame in raw_frames]

    with open(summary_path) as f:
        raw_summary = json.load(f)
    summary = MissionSummary(**raw_summary)

    # Downsample to every 10th frame for faster plotting (still 10 Hz)
    plot_frames = frames[::10]

    print("Generating visualizations...")
    generate_dashboard(plot_frames, summary, output_dir)
    generate_ground_track(plot_frames, output_dir)
    write_summary_text(summary, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
