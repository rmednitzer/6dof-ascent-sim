"""Flight Termination System (FTS).

Evaluates every simulation timestep.  If *any* abort criterion is violated the
FTS triggers, latching irrevocably.  Once triggered the vehicle is commanded to
a safe state (engines off / destruct).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.config import (
    FTS_ATTITUDE_LIMIT_DEG,
    FTS_COVARIANCE_LIMIT_M,
    FTS_CROSSRANGE_LIMIT_M,
)
from sim.safety.boundary_enforcer import BoundaryEnforcer, BoundaryResult


@dataclass
class FTSState:
    """Snapshot captured at the moment FTS triggers.

    Attributes:
        fts_triggered: Latching flag — once True it never resets.
        trigger_time:  Monotonic timestamp of the trigger event.
        reason:        Human-readable description of the cause.
        snapshot:      Copy of the vehicle state at trigger time.
    """

    fts_triggered: bool = False
    trigger_time: float | None = None
    reason: str | None = None
    snapshot: dict[str, Any] = field(default_factory=dict)


class FlightTerminationSystem:
    """Autonomous flight-safety system evaluated every timestep.

    The FTS checks four independent criteria and triggers if **any** one of
    them is violated:

    1. Cross-range deviation exceeds ``FTS_CROSSRANGE_LIMIT_M``.
    2. Attitude error exceeds ``FTS_ATTITUDE_LIMIT_DEG``.
    3. EKF position uncertainty (1-sigma) exceeds ``FTS_COVARIANCE_LIMIT_M``.
    4. Structural limits exceeded (via :class:`BoundaryEnforcer`).

    Once triggered the latch cannot be reset.
    """

    def __init__(self, boundary_enforcer: BoundaryEnforcer) -> None:
        self.state = FTSState()
        self._boundary_enforcer = boundary_enforcer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def fts_triggered(self) -> bool:
        """Return True if the FTS has been triggered."""
        return self.state.fts_triggered

    def evaluate(
        self,
        position_ecef: np.ndarray,
        nominal_plane_normal: np.ndarray,
        nominal_plane_point: np.ndarray,
        q_actual: np.ndarray,
        q_desired: np.ndarray,
        ekf_pos_covariance: np.ndarray,
        axial_g: float,
        lateral_g: float,
        dynamic_pressure_pa: float,
        sim_time: float,
        altitude_m: float = 0.0,
    ) -> bool:
        """Run all FTS checks for the current timestep.

        Args:
            position_ecef:        Vehicle position in ECEF (m), shape (3,).
            nominal_plane_normal: Unit normal of the nominal trajectory plane
                                  in ECEF, shape (3,).
            nominal_plane_point:  A point on the nominal trajectory plane in
                                  ECEF (m), shape (3,).  Typically the launch
                                  site position.
            q_actual:             Actual attitude quaternion [w, x, y, z].
            q_desired:            Desired attitude quaternion [w, x, y, z].
            ekf_pos_covariance:   3x3 position covariance matrix (m²).
            axial_g:              Axial load factor (g).
            lateral_g:            Lateral load factor (g).
            dynamic_pressure_pa:  Dynamic pressure (Pa).
            sim_time:             Current simulation time (s).

        Returns:
            True if the FTS **triggers** on this call (or was already
            triggered on a previous call).
        """
        # Latched — once triggered, stay triggered.
        if self.state.fts_triggered:
            return True

        reasons: list[str] = []
        evidence: dict[str, Any] = {"sim_time": sim_time}

        # 1. Cross-range deviation (only checked below 100 km altitude)
        crossrange_m = self._compute_crossrange(
            position_ecef, nominal_plane_normal, nominal_plane_point
        )
        evidence["crossrange_m"] = crossrange_m
        if altitude_m < 100_000.0 and abs(crossrange_m) > FTS_CROSSRANGE_LIMIT_M:
            reasons.append(
                f"Cross-range deviation {crossrange_m:.1f} m exceeds "
                f"limit {FTS_CROSSRANGE_LIMIT_M:.1f} m"
            )

        # 2. Attitude error
        attitude_err_deg = self._compute_attitude_error(q_actual, q_desired)
        evidence["attitude_error_deg"] = attitude_err_deg
        if attitude_err_deg > FTS_ATTITUDE_LIMIT_DEG:
            reasons.append(
                f"Attitude error {attitude_err_deg:.2f} deg exceeds "
                f"limit {FTS_ATTITUDE_LIMIT_DEG:.1f} deg"
            )

        # 3. EKF position uncertainty (largest 1-sigma)
        pos_uncertainty_m = self._compute_position_uncertainty(ekf_pos_covariance)
        evidence["ekf_pos_uncertainty_m"] = pos_uncertainty_m
        if pos_uncertainty_m > FTS_COVARIANCE_LIMIT_M:
            reasons.append(
                f"EKF position uncertainty {pos_uncertainty_m:.1f} m exceeds "
                f"limit {FTS_COVARIANCE_LIMIT_M:.1f} m"
            )

        # 4. Structural limits
        struct_result: BoundaryResult = self._boundary_enforcer.check_structural_limits(
            axial_g, lateral_g, dynamic_pressure_pa
        )
        evidence["structural_check"] = struct_result.evidence
        if not struct_result.approved:
            reasons.append(
                f"Structural limit exceeded: {struct_result.violation_type}"
            )

        # --- Trigger decision ---
        if reasons:
            self._trigger(reasons, evidence, sim_time)
            return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_crossrange(
        position_ecef: np.ndarray,
        plane_normal: np.ndarray,
        plane_point: np.ndarray,
    ) -> float:
        """Signed perpendicular distance from *position_ecef* to the nominal
        trajectory plane.

        The nominal trajectory plane is defined by a point on the plane
        (``plane_point``) and its unit normal (``plane_normal``).  The
        cross-range deviation is simply the signed projection of the offset
        vector onto the normal.
        """
        offset = np.asarray(position_ecef) - np.asarray(plane_point)
        return float(np.dot(offset, np.asarray(plane_normal)))

    @staticmethod
    def _compute_attitude_error(
        q_actual: np.ndarray,
        q_desired: np.ndarray,
    ) -> float:
        """Angle (degrees) between two unit quaternions [w, x, y, z].

        Uses the inner-product formula:
            theta = 2 * arccos(|q_actual . q_desired|)
        """
        q_a = np.asarray(q_actual, dtype=np.float64)
        q_d = np.asarray(q_desired, dtype=np.float64)
        dot = np.clip(np.abs(np.dot(q_a, q_d)), 0.0, 1.0)
        return float(np.degrees(2.0 * np.arccos(dot)))

    @staticmethod
    def _compute_position_uncertainty(cov: np.ndarray) -> float:
        """Largest 1-sigma position uncertainty from a 3x3 covariance matrix.

        Returns the square root of the largest eigenvalue.
        """
        cov = np.asarray(cov, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(cov)
        return float(np.sqrt(np.max(np.abs(eigenvalues))))

    def _trigger(
        self,
        reasons: list[str],
        evidence: dict[str, Any],
        sim_time: float,
    ) -> None:
        """Latch the FTS and record the trigger event."""
        self.state.fts_triggered = True
        self.state.trigger_time = time.monotonic()
        self.state.reason = "; ".join(reasons)
        self.state.snapshot = {
            "sim_time": sim_time,
            "evidence": evidence,
            "reasons": list(reasons),
        }
