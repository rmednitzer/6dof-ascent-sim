#!/usr/bin/env python3
"""Export a compact telemetry bundle for the GitHub Pages site.

Reads the canonical example-run artefacts written by the simulation
(``output/telemetry_internal.json`` + ``output/mission_summary.json``) and
distils them into a small, columnar JSON that the static site
(``site/index.html``) loads to render interactive Plotly visualisations
(3-D trajectory, ground track, and the telemetry dashboard).

The internal-rate telemetry is ~40 MB (100 Hz over the whole ascent), far too
large to ship to a browser. This script:

  * decimates to a target sample count (a few-Hz stream is visually smooth),
  * keeps only the channels the page plots,
  * stores them column-wise and rounded (much smaller than the array-of-objects
    schema), and
  * precomputes the ground track (lat/lon) and the achieved orbit elements.

Usage::

    python examples/run_and_visualize.py        # produces output/*.json first
    python examples/export_web_telemetry.py      # then write site/data/telemetry.json
    python examples/export_web_telemetry.py --run # run the sim itself, then export

The output is deterministic for a given input, so the committed
``site/data/telemetry.json`` can be regenerated and diffed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Ensure the repo root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import config
from sim.core.reference_frames import ecef_to_lla, eci_to_ecef

# Target number of samples in the exported stream. ~900 points over a ~500 s
# ascent is ~2 Hz -- smooth for both the time-series plots and the 3-D path
# while keeping the payload around a couple hundred KB.
TARGET_POINTS = 900

# Health string -> compact integer code (legend shipped in the metadata).
_HEALTH_CODE = {"NOMINAL": 0, "WARNING": 1, "ALERT": 2, "CRITICAL": 3}


def _load_artifacts(output_dir: Path) -> tuple[list[dict], dict]:
    """Load the internal telemetry frames and mission summary dictionaries."""
    telem_path = output_dir / "telemetry_internal.json"
    summary_path = output_dir / "mission_summary.json"
    if not telem_path.exists() or not summary_path.exists():
        raise SystemExit(
            f"Missing {telem_path} / {summary_path}.\nRun `python examples/run_and_visualize.py` first, or pass --run."
        )
    with open(telem_path) as fh:
        frames = json.load(fh)
    with open(summary_path) as fh:
        summary = json.load(fh)
    return frames, summary


def _decimate(frames: list[dict], target: int) -> list[dict]:
    """Stride-decimate to ~*target* frames, always keeping the final frame."""
    n = len(frames)
    if n <= target:
        return list(frames)
    stride = max(1, n // target)
    sampled = frames[::stride]
    if sampled[-1] is not frames[-1]:
        sampled.append(frames[-1])
    return sampled


def _ground_track(frames: list[dict]) -> tuple[list[float], list[float]]:
    """Geodetic lat/lon (deg) for each frame from its ECI position."""
    lats: list[float] = []
    lons: list[float] = []
    for f in frames:
        pos_ecef = eci_to_ecef(np.asarray(f["position_eci_m"], dtype=float), f["time_s"])
        lat, lon, _ = ecef_to_lla(pos_ecef)
        lats.append(math.degrees(lat))
        lons.append(math.degrees(lon))
    return lats, lons


def _staging_time(frames: list[dict]) -> float | None:
    """Time (s) of the first stage-number transition, if any."""
    for i in range(1, len(frames)):
        if frames[i]["stage"] != frames[i - 1]["stage"]:
            return frames[i]["time_s"]
    return None


def _max_q_time(frames: list[dict]) -> float:
    """Time (s) of peak dynamic pressure over the full-resolution stream."""
    idx = int(np.argmax([f["dynamic_pressure_pa"] for f in frames]))
    return frames[idx]["time_s"]


def _orbit_elements(r_m: list[float], v_ms: list[float]) -> dict:
    """Keplerian apogee/perigee/inclination/period from a state vector."""
    mu = config.EARTH_MU
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))

    inclination_deg = math.degrees(math.acos(max(-1.0, min(1.0, h_vec[2] / h_mag))))
    energy = v_mag**2 / 2.0 - mu / r_mag
    a = -mu / (2.0 * energy)  # semi-major axis (m)
    e_vec = np.cross(v, h_vec) / mu - r / r_mag
    e = float(np.linalg.norm(e_vec))

    r_re = config.EARTH_RADIUS_M
    return {
        "apogee_km": (a * (1.0 + e) - r_re) / 1000.0,
        "perigee_km": (a * (1.0 - e) - r_re) / 1000.0,
        "inclination_deg": inclination_deg,
        "eccentricity": e,
        "period_min": 2.0 * math.pi * math.sqrt(a**3 / mu) / 60.0,
    }


def _projected_orbit(r_m: list[float], v_ms: list[float], n_points: int = 220) -> dict:
    """Forward-propagate the achieved orbit one revolution and return ECI km.

    Reuses the post-insertion :class:`OrbitPropagator` (J2-perturbed RK4) so the
    site can draw the *projected* trajectory the vehicle coasts onto after
    insertion. The path starts at the insertion state, so it joins the ascent
    arc seamlessly, and spans one orbital period (a near-closed loop).
    """
    from sim.core.state import VehicleState
    from sim.orbital.propagator import OrbitPropagator

    state = VehicleState(
        position_eci=np.asarray(r_m, dtype=float),
        velocity_eci=np.asarray(v_ms, dtype=float),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity_body=np.zeros(3),
        mass_kg=1000.0,
        time_s=0.0,
    )
    prop = OrbitPropagator(state)
    period_s = prop.state_to_elements().period_s
    if not math.isfinite(period_s) or period_s <= 0.0:
        return {"x_km": [], "y_km": [], "z_km": []}
    states = prop.propagate(duration_s=period_s, dt_s=period_s / n_points)
    return {
        "x_km": [round(float(s.position_eci[0]) / 1000.0, 1) for s in states],
        "y_km": [round(float(s.position_eci[1]) / 1000.0, 1) for s in states],
        "z_km": [round(float(s.position_eci[2]) / 1000.0, 1) for s in states],
    }


def build_bundle(frames: list[dict], summary: dict, target: int = TARGET_POINTS) -> dict:
    """Assemble the compact, columnar telemetry bundle for the web page."""
    sampled = _decimate(frames, target)
    lats, lons = _ground_track(sampled)

    def col(key: str, scale: float, ndigits: int) -> list[float]:
        return [round(f[key] * scale, ndigits) for f in sampled]

    series = {
        "t": col("time_s", 1.0, 2),
        "alt_km": col("altitude_m", 1e-3, 3),
        "vel_ms": col("velocity_mag_ms", 1.0, 1),
        "q_kpa": col("dynamic_pressure_pa", 1e-3, 3),
        "axial_g": col("axial_g", 1.0, 3),
        "lateral_g": col("lateral_g", 1.0, 4),
        "mass_t": col("mass_kg", 1e-3, 3),
        "throttle_pct": col("throttle", 100.0, 1),
        "mach": col("mach_number", 1.0, 3),
        "thrust_kn": col("thrust_n", 1e-3, 1),
        "ekf_m": col("ekf_position_uncertainty_m", 1.0, 2),
        "stage": [int(f["stage"]) for f in sampled],
        "health": [_HEALTH_CODE.get(f["health_status"], 0) for f in sampled],
        # ECI position in km for the 3-D view (1 km resolution is plenty).
        "x_km": [round(f["position_eci_m"][0] * 1e-3, 1) for f in sampled],
        "y_km": [round(f["position_eci_m"][1] * 1e-3, 1) for f in sampled],
        "z_km": [round(f["position_eci_m"][2] * 1e-3, 1) for f in sampled],
        "lat": [round(v, 4) for v in lats],
        "lon": [round(v, 4) for v in lons],
    }

    orbit = _orbit_elements(summary["final_position_eci_m"], summary["final_velocity_eci_ms"])

    meta = {
        "outcome": summary["outcome"],
        "final_time_s": round(summary["final_time_s"], 1),
        "final_altitude_km": round(summary["final_altitude_m"] / 1000.0, 1),
        "final_velocity_ms": round(summary["final_velocity_ms"], 1),
        "final_mass_kg": round(summary["final_mass_kg"], 0),
        "final_stage": summary["final_stage"],
        "peak_altitude_km": round(summary["peak_altitude_m"] / 1000.0, 1),
        "peak_velocity_ms": round(summary["peak_velocity_ms"], 1),
        "peak_q_kpa": round(summary["peak_dynamic_pressure_pa"] / 1000.0, 2),
        "peak_axial_g": round(summary["peak_axial_g"], 2),
        "peak_lateral_g": round(summary["peak_lateral_g"], 3),
        "peak_mach": round(summary["peak_mach_number"], 1),
        "fts_triggered": summary["fts_triggered"],
        "health_status_final": summary["health_status_final"],
        "boundary_violations": summary["total_boundary_violations"],
        "frames_internal": summary["total_frames_internal"],
        "telemetry_hash_sha256": summary["telemetry_hash_sha256"],
        "orbit": {k: round(v, 3) for k, v in orbit.items()},
        "targets": {
            "altitude_km": config.TARGET_ALTITUDE_M / 1000.0,
            "velocity_ms": config.TARGET_VELOCITY_MS,
            "inclination_deg": config.TARGET_INCLINATION_DEG,
        },
        "limits": {
            "max_q_kpa": config.MAX_Q_PA / 1000.0,
            "max_axial_g": config.MAX_AXIAL_G,
            "fts_covariance_m": config.FTS_COVARIANCE_LIMIT_M,
        },
        "events": {
            "max_q_s": round(_max_q_time(frames), 1),
            "staging_s": (round(s, 1) if (s := _staging_time(frames)) is not None else None),
        },
        "launch_site": {
            "name": "Kennedy Space Center",
            "lat": config.LAUNCH_LAT_DEG,
            "lon": config.LAUNCH_LON_DEG,
        },
        "earth_radius_km": config.EARTH_RADIUS_M / 1000.0,
        "n_points": len(sampled),
        "health_legend": {v: k for k, v in _HEALTH_CODE.items()},
    }

    projected = _projected_orbit(summary["final_position_eci_m"], summary["final_velocity_eci_ms"])

    return {"meta": meta, "series": series, "projected": projected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact telemetry JSON for the GitHub Pages site.")
    parser.add_argument("--run", action="store_true", help="Run the simulation first instead of reading output/.")
    parser.add_argument("--points", type=int, default=TARGET_POINTS, help="Approximate number of samples to export.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "site" / "data" / "telemetry.json",
        help="Destination JSON path.",
    )
    args = parser.parse_args()

    output_dir = Path("output")
    if args.run:
        from sim.main import run_simulation

        print("Running simulation ...")
        run_simulation()

    frames, summary = _load_artifacts(output_dir)
    bundle = build_bundle(frames, summary, target=args.points)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, separators=(",", ":")) + "\n", encoding="utf-8")

    size_kb = args.out.stat().st_size / 1024.0
    print(f"Wrote {args.out} ({bundle['meta']['n_points']} points, {size_kb:.1f} KB)")
    print(
        f"  outcome={bundle['meta']['outcome']}  apogee={bundle['meta']['orbit']['apogee_km']:.1f} km  "
        f"incl={bundle['meta']['orbit']['inclination_deg']:.2f} deg"
    )


if __name__ == "__main__":
    main()
