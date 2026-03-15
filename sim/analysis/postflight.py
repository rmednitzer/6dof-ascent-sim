"""Post-flight analysis: trajectory plots and mission summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim import config
from sim.telemetry.schemas import TelemetryFrame, MissionSummary


def generate_plots(
    frames: list[TelemetryFrame],
    summary: MissionSummary,
    output_dir: str = "output/plots",
) -> None:
    """Generate post-flight analysis plots.

    Args:
        frames: Internal telemetry frames.
        summary: Mission summary.
        output_dir: Directory for plot files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    times = [f.time_s for f in frames]
    if not times:
        print("No telemetry frames — skipping plots.")
        return

    # Altitude vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, [f.altitude_m / 1000 for f in frames])
    ax.axhline(config.TARGET_ALTITUDE_M / 1000, color="r", linestyle="--", label="Target")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "altitude_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Velocity vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, [f.velocity_mag_ms for f in frames])
    ax.axhline(config.TARGET_VELOCITY_MS, color="r", linestyle="--", label="Target")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Inertial Velocity vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "velocity_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Dynamic pressure vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    q_vals = [f.dynamic_pressure_pa / 1000 for f in frames]
    ax.plot(times, q_vals)
    ax.axhline(config.MAX_Q_PA / 1000, color="r", linestyle="--", label="Structural Limit")
    if q_vals:
        max_q_idx = int(np.argmax(q_vals))
        ax.axvline(times[max_q_idx], color="orange", linestyle=":", alpha=0.7,
                   label=f"Max-Q @ {times[max_q_idx]:.1f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dynamic Pressure (kPa)")
    ax.set_title("Dynamic Pressure vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "dynamic_pressure_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Axial G-load vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, [f.axial_g for f in frames])
    ax.axhline(config.MAX_AXIAL_G, color="r", linestyle="--", label="Structural Limit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Axial Acceleration (g)")
    ax.set_title("Axial G-Load vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "axial_g_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # EKF position uncertainty vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ekf_vals = [f.ekf_position_uncertainty_m for f in frames]
    ax.plot(times, ekf_vals)
    ax.axhline(config.FTS_COVARIANCE_LIMIT_M, color="r", linestyle="--", label="FTS Limit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Uncertainty (m)")
    ax.set_title("EKF Position Uncertainty vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if max(ekf_vals) > 100:
        ax.set_yscale("log")
    fig.savefig(out / "ekf_uncertainty_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Mass vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, [f.mass_kg for f in frames])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Vehicle Mass vs Time")
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "mass_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    xs = [f.position_eci_m[0] / 1000 for f in frames]
    ys = [f.position_eci_m[1] / 1000 for f in frames]
    zs = [f.position_eci_m[2] / 1000 for f in frames]
    ax3.plot(xs, ys, zs, linewidth=0.5)
    ax3.set_xlabel("X (km)")
    ax3.set_ylabel("Y (km)")
    ax3.set_zlabel("Z (km)")
    ax3.set_title("3D Trajectory (ECI)")
    fig.savefig(out / "trajectory_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {out}/")
