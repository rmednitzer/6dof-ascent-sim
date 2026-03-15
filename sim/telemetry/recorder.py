"""Telemetry capture system for the 6-DOF ascent simulation.

Records telemetry at two rates:
  - **Internal** (100 Hz) -- full-fidelity record for post-flight analysis.
  - **Downlink** (10 Hz) -- decimated stream mirroring a realistic TM link.

At simulation end, :meth:`TelemetryRecorder.write_output` persists both
streams and a mission summary to the ``output/`` directory.  The internal
telemetry JSON is hashed (SHA-256) and the digest is embedded in the summary
for integrity verification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sim.config import INTERNAL_HZ, TELEMETRY_HZ
from sim.telemetry.schemas import MissionSummary, TelemetryFrame


@dataclass
class TelemetryRecorder:
    """Dual-rate telemetry recorder.

    Attributes:
        internal_frames: Full list of frames captured at the internal rate.
        downlink_frames: Decimated list of frames captured at the downlink rate.
        output_dir: Filesystem directory for output artefacts.
    """

    internal_frames: list[TelemetryFrame] = field(default_factory=list)
    downlink_frames: list[TelemetryFrame] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Derived decimation ratio -- every Nth internal step produces a downlink frame.
    _decimation_ratio: int = field(init=False, repr=False)

    # Internal step counter used for decimation book-keeping.
    _step_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        """Compute the decimation ratio from the configured rates."""
        if TELEMETRY_HZ <= 0:
            raise ValueError("TELEMETRY_HZ must be positive")
        if INTERNAL_HZ <= 0:
            raise ValueError("INTERNAL_HZ must be positive")
        self._decimation_ratio = max(1, INTERNAL_HZ // TELEMETRY_HZ)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        true_state: Any,
        estimated_state: Any,
        health_monitor: Any,
        boundary_enforcer: Any,
        time_s: float,
        sim_context: dict[str, Any],
    ) -> None:
        """Capture a single telemetry frame from the current simulation step.

        This method is called once per internal physics step (100 Hz).  Every
        *decimation_ratio*-th call the frame is also appended to the downlink
        buffer.

        Args:
            true_state: The ground-truth :class:`VehicleState`.
            estimated_state: The EKF-estimated :class:`VehicleState` (used for
                position uncertainty).
            health_monitor: Object exposing ``status`` (str) for vehicle health.
            boundary_enforcer: Object exposing ``violation_count`` (int) and
                ``fts_triggered`` (bool).
            time_s: Current mission elapsed time (s).
            sim_context: Dictionary carrying per-step derived quantities.
                Expected keys include ``throttle``, ``thrust_n``,
                ``dynamic_pressure_pa``, ``mach_number``, ``axial_g``,
                ``lateral_g``, ``stage``, and
                ``ekf_position_uncertainty_m``.
        """
        frame = self._build_frame(
            true_state,
            estimated_state,
            health_monitor,
            boundary_enforcer,
            time_s,
            sim_context,
        )

        self.internal_frames.append(frame)
        self._step_count += 1

        if self._step_count % self._decimation_ratio == 0:
            self.downlink_frames.append(frame)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_output(
        self,
        outcome: str,
        true_state: Any,
        health_monitor: Any,
        boundary_enforcer: Any,
    ) -> MissionSummary:
        """Persist telemetry and mission summary to disk.

        Writes three files under :attr:`output_dir`:
          - ``telemetry_internal.json`` -- internal-rate frames.
          - ``telemetry_downlink.json`` -- downlink-rate frames.
          - ``mission_summary.json`` -- aggregate summary with integrity hash.

        Args:
            outcome: Terminal mission result string (e.g. ``"SUCCESS"``).
            true_state: Final ground-truth :class:`VehicleState`.
            health_monitor: Health monitor at simulation end.
            boundary_enforcer: Boundary enforcer at simulation end.

        Returns:
            The populated :class:`MissionSummary` instance.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -- Serialise frame lists ----------------------------------------
        internal_dicts = [f.to_dict() for f in self.internal_frames]
        downlink_dicts = [f.to_dict() for f in self.downlink_frames]

        internal_json = json.dumps(internal_dicts, indent=2)
        downlink_json = json.dumps(downlink_dicts, indent=2)

        internal_path = self.output_dir / "telemetry_internal.json"
        downlink_path = self.output_dir / "telemetry_downlink.json"
        summary_path = self.output_dir / "mission_summary.json"

        internal_path.write_text(internal_json, encoding="utf-8")
        downlink_path.write_text(downlink_json, encoding="utf-8")

        # -- Compute SHA-256 of internal telemetry ------------------------
        telemetry_hash = hashlib.sha256(internal_json.encode("utf-8")).hexdigest()

        # -- Build mission summary ----------------------------------------
        summary = self._build_summary(
            outcome=outcome,
            true_state=true_state,
            health_monitor=health_monitor,
            boundary_enforcer=boundary_enforcer,
            telemetry_hash=telemetry_hash,
        )

        summary_json = json.dumps(summary.to_dict(), indent=2)
        summary_path.write_text(summary_json, encoding="utf-8")

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_frame(
        true_state: Any,
        estimated_state: Any,
        health_monitor: Any,
        boundary_enforcer: Any,
        time_s: float,
        ctx: dict[str, Any],
    ) -> TelemetryFrame:
        """Assemble a :class:`TelemetryFrame` from simulation objects.

        Converts numpy arrays to plain Python lists so the frame is
        JSON-ready without further transformation.
        """
        pos = true_state.position_eci
        vel = true_state.velocity_eci
        quat = true_state.quaternion

        return TelemetryFrame(
            time_s=time_s,
            position_eci_m=pos.tolist() if hasattr(pos, "tolist") else list(pos),
            velocity_eci_ms=vel.tolist() if hasattr(vel, "tolist") else list(vel),
            altitude_m=true_state.altitude_m(),
            velocity_mag_ms=true_state.velocity_mag_ms(),
            quaternion=quat.tolist() if hasattr(quat, "tolist") else list(quat),
            mass_kg=true_state.mass_kg,
            throttle=ctx.get("throttle", 0.0),
            thrust_n=ctx.get("thrust_n", 0.0),
            dynamic_pressure_pa=ctx.get("dynamic_pressure_pa", 0.0),
            mach_number=ctx.get("mach_number", 0.0),
            axial_g=ctx.get("axial_g", 0.0),
            lateral_g=ctx.get("lateral_g", 0.0),
            stage=ctx.get("stage", 1),
            ekf_position_uncertainty_m=ctx.get("ekf_position_uncertainty_m", 0.0),
            health_status=getattr(health_monitor, "status", "NOMINAL"),
            boundary_violations=getattr(boundary_enforcer, "violation_count", 0),
            fts_triggered=getattr(boundary_enforcer, "fts_triggered", False),
        )

    def _build_summary(
        self,
        outcome: str,
        true_state: Any,
        health_monitor: Any,
        boundary_enforcer: Any,
        telemetry_hash: str,
    ) -> MissionSummary:
        """Compute peak values from recorded frames and populate a summary."""
        pos = true_state.position_eci
        vel = true_state.velocity_eci

        peak_alt = 0.0
        peak_vel = 0.0
        peak_q = 0.0
        peak_axial = 0.0
        peak_lateral = 0.0
        peak_mach = 0.0
        total_violations = 0

        for frame in self.internal_frames:
            if frame.altitude_m > peak_alt:
                peak_alt = frame.altitude_m
            if frame.velocity_mag_ms > peak_vel:
                peak_vel = frame.velocity_mag_ms
            if frame.dynamic_pressure_pa > peak_q:
                peak_q = frame.dynamic_pressure_pa
            if abs(frame.axial_g) > peak_axial:
                peak_axial = abs(frame.axial_g)
            if abs(frame.lateral_g) > peak_lateral:
                peak_lateral = abs(frame.lateral_g)
            if frame.mach_number > peak_mach:
                peak_mach = frame.mach_number
            total_violations += frame.boundary_violations

        return MissionSummary(
            outcome=outcome,
            final_time_s=true_state.time_s,
            final_altitude_m=true_state.altitude_m(),
            final_velocity_ms=true_state.velocity_mag_ms(),
            final_mass_kg=true_state.mass_kg,
            final_stage=getattr(true_state, "stage", self.internal_frames[-1].stage if self.internal_frames else 1),
            final_position_eci_m=pos.tolist() if hasattr(pos, "tolist") else list(pos),
            final_velocity_eci_ms=vel.tolist() if hasattr(vel, "tolist") else list(vel),
            peak_altitude_m=peak_alt,
            peak_velocity_ms=peak_vel,
            peak_dynamic_pressure_pa=peak_q,
            peak_axial_g=peak_axial,
            peak_lateral_g=peak_lateral,
            peak_mach_number=peak_mach,
            total_boundary_violations=total_violations,
            fts_triggered=getattr(boundary_enforcer, "fts_triggered", False),
            health_status_final=getattr(health_monitor, "status", "NOMINAL"),
            telemetry_hash_sha256=telemetry_hash,
            total_frames_internal=len(self.internal_frames),
            total_frames_downlink=len(self.downlink_frames),
        )
