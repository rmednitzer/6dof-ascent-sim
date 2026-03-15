"""Telemetry frame and mission summary schemas for the 6-DOF ascent simulation.

Defines the canonical data structures captured at each telemetry sample and
the aggregate mission summary written at simulation end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Health status enumeration (plain strings for JSON friendliness)
# ---------------------------------------------------------------------------
HEALTH_NOMINAL = "NOMINAL"
HEALTH_WARNING = "WARNING"
HEALTH_ALERT = "ALERT"
HEALTH_CRITICAL = "CRITICAL"

_VALID_HEALTH_STATUSES = {HEALTH_NOMINAL, HEALTH_WARNING, HEALTH_ALERT, HEALTH_CRITICAL}


@dataclass
class TelemetryFrame:
    """Single timestamped telemetry snapshot.

    Attributes:
        time_s: Mission elapsed time (s).
        position_eci_m: ECI position vector [x, y, z] (m).
        velocity_eci_ms: ECI velocity vector [vx, vy, vz] (m/s).
        altitude_m: Geodetic altitude above WGS-84 ellipsoid (m).
        velocity_mag_ms: Inertial speed magnitude (m/s).
        quaternion: Attitude quaternion [x, y, z, w] (scalar-last).
        mass_kg: Current total vehicle mass (kg).
        throttle: Commanded throttle setting [0.0, 1.0].
        thrust_n: Current net thrust magnitude (N).
        dynamic_pressure_pa: Dynamic pressure (Pa).
        mach_number: Free-stream Mach number.
        axial_g: Axial (longitudinal) acceleration in g-units.
        lateral_g: Lateral acceleration in g-units.
        stage: Active stage number (1-indexed).
        ekf_position_uncertainty_m: 1-sigma EKF position uncertainty (m).
        health_status: Vehicle health classification string.
        boundary_violations: Count of active boundary violations.
        fts_triggered: Whether the flight termination system has fired.
    """

    time_s: float = 0.0
    position_eci_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_eci_ms: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    altitude_m: float = 0.0
    velocity_mag_ms: float = 0.0
    quaternion: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    mass_kg: float = 0.0
    throttle: float = 0.0
    thrust_n: float = 0.0
    dynamic_pressure_pa: float = 0.0
    mach_number: float = 0.0
    axial_g: float = 0.0
    lateral_g: float = 0.0
    stage: int = 1
    ekf_position_uncertainty_m: float = 0.0
    health_status: str = HEALTH_NOMINAL
    boundary_violations: int = 0
    fts_triggered: bool = False

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of the frame."""
        return {
            "time_s": self.time_s,
            "position_eci_m": list(self.position_eci_m),
            "velocity_eci_ms": list(self.velocity_eci_ms),
            "altitude_m": self.altitude_m,
            "velocity_mag_ms": self.velocity_mag_ms,
            "quaternion": list(self.quaternion),
            "mass_kg": self.mass_kg,
            "throttle": self.throttle,
            "thrust_n": self.thrust_n,
            "dynamic_pressure_pa": self.dynamic_pressure_pa,
            "mach_number": self.mach_number,
            "axial_g": self.axial_g,
            "lateral_g": self.lateral_g,
            "stage": self.stage,
            "ekf_position_uncertainty_m": self.ekf_position_uncertainty_m,
            "health_status": self.health_status,
            "boundary_violations": self.boundary_violations,
            "fts_triggered": self.fts_triggered,
        }


@dataclass
class MissionSummary:
    """Aggregate summary of a completed simulation run.

    Attributes:
        outcome: Terminal mission result (e.g. ``"SUCCESS"``, ``"FTS_ABORT"``,
            ``"BOUNDARY_VIOLATION"``, ``"MAX_TIME_EXCEEDED"``).
        final_time_s: Simulation clock at termination (s).
        final_altitude_m: Geodetic altitude at termination (m).
        final_velocity_ms: Inertial speed at termination (m/s).
        final_mass_kg: Vehicle mass at termination (kg).
        final_stage: Active stage number at termination.
        final_position_eci_m: ECI position at termination (m).
        final_velocity_eci_ms: ECI velocity at termination (m/s).
        peak_altitude_m: Maximum altitude reached during flight (m).
        peak_velocity_ms: Maximum inertial speed reached (m/s).
        peak_dynamic_pressure_pa: Maximum dynamic pressure experienced (Pa).
        peak_axial_g: Maximum axial acceleration experienced (g).
        peak_lateral_g: Maximum lateral acceleration experienced (g).
        peak_mach_number: Maximum Mach number reached.
        total_boundary_violations: Cumulative boundary violation count.
        fts_triggered: Whether the flight termination system fired.
        health_status_final: Health classification at simulation end.
        telemetry_hash_sha256: SHA-256 hex digest of internal telemetry JSON.
        total_frames_internal: Number of internal-rate telemetry frames.
        total_frames_downlink: Number of downlink-rate telemetry frames.
    """

    outcome: str = ""
    final_time_s: float = 0.0
    final_altitude_m: float = 0.0
    final_velocity_ms: float = 0.0
    final_mass_kg: float = 0.0
    final_stage: int = 1
    final_position_eci_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    final_velocity_eci_ms: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    peak_altitude_m: float = 0.0
    peak_velocity_ms: float = 0.0
    peak_dynamic_pressure_pa: float = 0.0
    peak_axial_g: float = 0.0
    peak_lateral_g: float = 0.0
    peak_mach_number: float = 0.0
    total_boundary_violations: int = 0
    fts_triggered: bool = False
    health_status_final: str = HEALTH_NOMINAL
    telemetry_hash_sha256: str = ""
    total_frames_internal: int = 0
    total_frames_downlink: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of the mission summary."""
        return {
            "outcome": self.outcome,
            "final_time_s": self.final_time_s,
            "final_altitude_m": self.final_altitude_m,
            "final_velocity_ms": self.final_velocity_ms,
            "final_mass_kg": self.final_mass_kg,
            "final_stage": self.final_stage,
            "final_position_eci_m": list(self.final_position_eci_m),
            "final_velocity_eci_ms": list(self.final_velocity_eci_ms),
            "peak_altitude_m": self.peak_altitude_m,
            "peak_velocity_ms": self.peak_velocity_ms,
            "peak_dynamic_pressure_pa": self.peak_dynamic_pressure_pa,
            "peak_axial_g": self.peak_axial_g,
            "peak_lateral_g": self.peak_lateral_g,
            "peak_mach_number": self.peak_mach_number,
            "total_boundary_violations": self.total_boundary_violations,
            "fts_triggered": self.fts_triggered,
            "health_status_final": self.health_status_final,
            "telemetry_hash_sha256": self.telemetry_hash_sha256,
            "total_frames_internal": self.total_frames_internal,
            "total_frames_downlink": self.total_frames_downlink,
        }
