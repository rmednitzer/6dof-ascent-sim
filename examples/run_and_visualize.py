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

# --- Palette (a calm, modern light theme) ------------------------------------
INK = "#111827"  # near-black for titles
SUBINK = "#6b7280"  # muted grey for secondary text
PRIMARY = "#2563eb"  # blue — primary trace
LIMIT = "#ef4444"  # red — structural / safety limits
STAGING = "#f59e0b"  # amber — stage separation
MAXQ = "#8b5cf6"  # violet — max dynamic pressure
OK = "#10b981"  # green — success / launch
DANGER = "#dc2626"  # deep red — abort / insertion marker


def _apply_style(plt) -> None:
    """Apply a consistent, polished Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.facecolor": "#f7f8fa",
            "axes.edgecolor": "#d1d5db",
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 9,
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#374151",
            "xtick.color": SUBINK,
            "ytick.color": SUBINK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


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


def _panel(
    ax,
    times,
    ys,
    *,
    title,
    ylabel,
    color=PRIMARY,
    fill=True,
    limit=None,
    limit_label=None,
    staging_t=None,
    maxq_t=None,
    annotate_peak=None,
):
    """Draw one styled time-series panel and return the axis."""
    ax.plot(times, ys, color=color, linewidth=1.8, zorder=4, solid_capstyle="round")
    if fill:
        ax.fill_between(times, ys, np.min(ys), color=color, alpha=0.10, zorder=2)
    if limit is not None:
        ax.axhline(limit, color=LIMIT, linestyle=(0, (6, 4)), linewidth=1.3, alpha=0.9, label=limit_label or "Limit")
    if staging_t is not None:
        ax.axvline(
            staging_t, color=STAGING, linestyle=(0, (5, 4)), linewidth=1.2, alpha=0.8, label=f"Staging {staging_t:.0f}s"
        )
    if maxq_t is not None:
        ax.axvline(maxq_t, color=MAXQ, linestyle=(0, (1, 2)), linewidth=1.6, alpha=0.9, label=f"Max-Q {maxq_t:.0f}s")
    if annotate_peak is not None:
        i = int(np.argmax(ys))
        ax.scatter([times[i]], [ys[i]], color=color, s=22, zorder=5, edgecolor="white", linewidth=0.8)
        ax.annotate(
            annotate_peak.format(ys[i]),
            (times[i], ys[i]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.01)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    return ax


def generate_dashboard(frames, summary, output_dir: Path) -> None:
    """Create an 8-panel mission dashboard figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _apply_style(plt)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = [f.time_s for f in frames]
    if not times:
        print("No telemetry frames to plot.")
        return

    staging_t = _find_staging_time(frames)
    q_vals = [f.dynamic_pressure_pa / 1000 for f in frames]
    maxq_t = times[int(np.argmax(q_vals))]

    fig = plt.figure(figsize=(16, 19))
    success = summary.outcome == "SUCCESS"
    badge = OK if success else DANGER

    # --- Header -------------------------------------------------------------
    fig.text(
        0.5,
        0.975,
        "6-DOF Launch Vehicle — Ascent Mission Dashboard",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.957,
        f"Target {config.TARGET_ALTITUDE_M / 1000:.0f} km × {config.TARGET_INCLINATION_DEG:.1f}°   ·   "
        f"two-stage to LEO   ·   100 Hz RK4",
        ha="center",
        fontsize=11.5,
        color=SUBINK,
    )
    fig.text(
        0.5,
        0.935,
        f"  {summary.outcome}  ",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": badge, "edgecolor": "none"},
    )

    gs = GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.22, top=0.915, bottom=0.085, left=0.07, right=0.965)

    alts = [f.altitude_m / 1000 for f in frames]
    _panel(
        fig.add_subplot(gs[0, 0]),
        times,
        alts,
        title="Altitude Profile",
        ylabel="Altitude (km)",
        limit=config.TARGET_ALTITUDE_M / 1000,
        limit_label="Target",
        staging_t=staging_t,
        annotate_peak="{:.0f} km",
    )

    vels = [f.velocity_mag_ms for f in frames]
    _panel(
        fig.add_subplot(gs[0, 1]),
        times,
        vels,
        title="Inertial Velocity",
        ylabel="Velocity (m/s)",
        limit=config.TARGET_VELOCITY_MS,
        limit_label="Target",
        staging_t=staging_t,
    )

    _panel(
        fig.add_subplot(gs[1, 0]),
        times,
        q_vals,
        title="Dynamic Pressure",
        ylabel="Dynamic Pressure (kPa)",
        limit=config.MAX_Q_PA / 1000,
        limit_label="Structural limit",
        staging_t=staging_t,
        maxq_t=maxq_t,
        annotate_peak="{:.1f} kPa",
    )

    g_vals = [f.axial_g for f in frames]
    _panel(
        fig.add_subplot(gs[1, 1]),
        times,
        g_vals,
        title="Axial G-Load",
        ylabel="Axial Acceleration (g)",
        limit=config.MAX_AXIAL_G,
        limit_label="Structural limit",
        staging_t=staging_t,
        annotate_peak="{:.2f} g",
    )

    masses = [f.mass_kg / 1000 for f in frames]
    _panel(
        fig.add_subplot(gs[2, 0]),
        times,
        masses,
        title="Vehicle Mass",
        ylabel="Mass (tonnes)",
        staging_t=staging_t,
    )

    throttles = [f.throttle * 100 for f in frames]
    ax = _panel(
        fig.add_subplot(gs[2, 1]),
        times,
        throttles,
        title="Throttle Command",
        ylabel="Throttle (%)",
        staging_t=staging_t,
    )
    ax.set_ylim(-5, 110)

    ekf_vals = [f.ekf_position_uncertainty_m for f in frames]
    ax = _panel(
        fig.add_subplot(gs[3, 0]),
        times,
        ekf_vals,
        title="EKF Navigation Uncertainty",
        ylabel="Position Uncertainty (m)",
        fill=False,
        limit=config.FTS_COVARIANCE_LIMIT_M,
        limit_label="FTS limit",
        staging_t=staging_t,
    )
    if max(ekf_vals) > 100:
        ax.set_yscale("log")

    machs = [f.mach_number for f in frames]
    ax = _panel(
        fig.add_subplot(gs[3, 1]),
        times,
        machs,
        title="Mach Number",
        ylabel="Mach",
        staging_t=staging_t,
        maxq_t=maxq_t,
    )
    ax.axhline(1.0, color=SUBINK, linestyle=":", alpha=0.6, linewidth=1.1)

    # --- Metrics strip ------------------------------------------------------
    metrics = [
        ("FLIGHT TIME", f"{summary.final_time_s:.0f} s"),
        ("FINAL ALT", f"{summary.final_altitude_m / 1000:.0f} km"),
        ("FINAL VEL", f"{summary.final_velocity_ms:.0f} m/s"),
        ("PEAK Q", f"{summary.peak_dynamic_pressure_pa / 1000:.1f} kPa"),
        ("PEAK G", f"{summary.peak_axial_g:.2f} g"),
        ("PEAK MACH", f"{summary.peak_mach_number:.1f}"),
        ("FTS", "ABORT" if summary.fts_triggered else "NOMINAL"),
    ]
    n = len(metrics)
    for i, (label, value) in enumerate(metrics):
        x = 0.07 + (0.965 - 0.07) * (i + 0.5) / n
        fig.text(x, 0.045, value, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
        fig.text(x, 0.022, label, ha="center", va="center", fontsize=8.5, color=SUBINK)
    fig.patches.append(
        plt.Rectangle(
            (0.05, 0.008),
            0.935,
            0.058,
            transform=fig.transFigure,
            facecolor="#f1f5f9",
            edgecolor="#e2e8f0",
            zorder=-1,
            clip_on=False,
        )
    )

    fig.savefig(output_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved to {output_dir / 'dashboard.png'}")


def generate_ground_track(frames, output_dir: Path) -> None:
    """Create a ground track plot showing the trajectory over Earth's surface."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    _apply_style(plt)

    lats, lons = _extract_ground_track(frames)
    if not lats:
        return
    alts = np.array([f.altitude_m / 1000 for f in frames])

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Continuous, altitude-coloured trajectory via a line collection (smoother
    # than a scatter and reads as a single flight path).
    pts = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidth=3.4, zorder=3, capstyle="round")
    lc.set_array(alts)
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, label="Altitude (km)", shrink=0.85, pad=0.02)
    cbar.outline.set_visible(False)

    # Launch + insertion markers with labels.
    ax.scatter([lons[0]], [lats[0]], marker="^", s=150, color=OK, edgecolor="white", linewidth=1.4, zorder=5)
    ax.annotate("  Launch (KSC)", (lons[0], lats[0]), fontsize=10, fontweight="bold", color=INK, va="center")
    ax.scatter([lons[-1]], [lats[-1]], marker="*", s=320, color=DANGER, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate(
        "  Insertion",
        (lons[-1], lats[-1]),
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="center",
    )

    # A little breathing room around the track.
    dlon = (max(lons) - min(lons)) or 1.0
    dlat = (max(lats) - min(lats)) or 1.0
    ax.set_xlim(min(lons) - 0.08 * dlon, max(lons) + 0.18 * dlon)
    ax.set_ylim(min(lats) - 0.12 * dlat, max(lats) + 0.12 * dlat)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Ascent Ground Track", fontsize=15)
    ax.set_facecolor("#f0f6ff")

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
