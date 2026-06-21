"""Flight Termination System (FTS).

Evaluates every simulation timestep.  If *any* abort criterion is violated the
FTS triggers, latching irrevocably.  Once triggered the vehicle is commanded to
a safe state (engines off / destruct).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim import config
from sim.config import (
    FTS_ATTITUDE_LIMIT_DEG,
    FTS_COVARIANCE_LIMIT_M,
    FTS_CROSSRANGE_LIMIT_M,
)
from sim.core.fast_math import max_sigma_3x3
from sim.core.reference_frames import body_to_eci
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
        # Attitude-criterion hysteresis (audit AD-19 mitigation): the sim_time
        # at which the thrust-axis error first crossed the limit in the current
        # run of violations, or None when the attitude is within limits. The
        # criterion only contributes an abort once the violation has persisted
        # for FTS_ATTITUDE_HYSTERESIS_S. Captured from config at construction so
        # the main loop and tests get a stable value (FTS limits are not
        # dispersed in Monte Carlo).
        self._attitude_hysteresis_s: float = config.FTS_ATTITUDE_HYSTERESIS_S
        self._attitude_violation_start_s: float | None = None

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
            q_actual:             Actual attitude quaternion [x, y, z, w].
            q_desired:            Desired attitude quaternion [x, y, z, w].
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
        crossrange_m = self._compute_crossrange(position_ecef, nominal_plane_normal, nominal_plane_point)
        evidence["crossrange_m"] = crossrange_m
        if altitude_m < 100_000.0 and abs(crossrange_m) > FTS_CROSSRANGE_LIMIT_M:
            reasons.append(f"Cross-range deviation {crossrange_m:.1f} m exceeds limit {FTS_CROSSRANGE_LIMIT_M:.1f} m")

        # 2. Attitude error — hysteresis-gated (audit AD-19). The thrust-axis
        # error must stay above the limit continuously for
        # FTS_ATTITUDE_HYSTERESIS_S before it contributes an abort; a single
        # marginal-frame excursion is debounced, a sustained loss of control
        # still trips. With hysteresis 0.0 this reduces to the original
        # instantaneous test (elapsed 0.0 >= 0.0 on the first violating frame).
        attitude_err_deg = self._compute_attitude_error(q_actual, q_desired)
        evidence["attitude_error_deg"] = attitude_err_deg
        if attitude_err_deg > FTS_ATTITUDE_LIMIT_DEG:
            if self._attitude_violation_start_s is None:
                self._attitude_violation_start_s = sim_time
            persisted_s = sim_time - self._attitude_violation_start_s
            evidence["attitude_violation_persisted_s"] = persisted_s
            if persisted_s >= self._attitude_hysteresis_s:
                reasons.append(
                    f"Attitude error {attitude_err_deg:.2f} deg exceeds limit "
                    f"{FTS_ATTITUDE_LIMIT_DEG:.1f} deg for {persisted_s:.2f} s"
                )
        else:
            # Within limits — reset the persistence clock.
            self._attitude_violation_start_s = None

        # 3. EKF position uncertainty (largest 1-sigma)
        pos_uncertainty_m = self._compute_position_uncertainty(ekf_pos_covariance)
        evidence["ekf_pos_uncertainty_m"] = pos_uncertainty_m
        if pos_uncertainty_m > FTS_COVARIANCE_LIMIT_M:
            reasons.append(
                f"EKF position uncertainty {pos_uncertainty_m:.1f} m exceeds limit {FTS_COVARIANCE_LIMIT_M:.1f} m"
            )

        # 4. Structural limits
        struct_result: BoundaryResult = self._boundary_enforcer.check_structural_limits(
            axial_g, lateral_g, dynamic_pressure_pa
        )
        evidence["structural_check"] = struct_result.evidence
        if not struct_result.approved:
            reasons.append(f"Structural limit exceeded: {struct_result.violation_type}")

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
        dx = position_ecef[0] - plane_point[0]
        dy = position_ecef[1] - plane_point[1]
        dz = position_ecef[2] - plane_point[2]
        return dx * plane_normal[0] + dy * plane_normal[1] + dz * plane_normal[2]

    @staticmethod
    def _compute_attitude_error(
        q_actual: np.ndarray,
        q_desired: np.ndarray,
    ) -> float:
        """Thrust-axis pointing error (degrees) — the angle between the
        commanded and actual body +X (thrust) axes in ECI.

        The previous implementation used the full 4-component quaternion
        inner product, which also counts rotation *about* the thrust
        axis (roll). This vehicle is axisymmetric with no roll control,
        so roll drifts freely — making the full-quaternion error grow to
        ~60 deg in benign flight while the thrust axis tracks to <5 deg.
        An FTS divergence monitor must track the loss-of-control
        quantity (where thrust points), not the irrelevant roll, so a
        tight, genuinely protective limit is meaningful.
        """
        body_x = np.array([1.0, 0.0, 0.0])
        a = body_to_eci(body_x, q_actual)
        d = body_to_eci(body_x, q_desired)
        na = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        nd = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if na < 1e-12 or nd < 1e-12:
            return 0.0
        cos_ang = (a[0] * d[0] + a[1] * d[1] + a[2] * d[2]) / (na * nd)
        if cos_ang > 1.0:
            cos_ang = 1.0
        elif cos_ang < -1.0:
            cos_ang = -1.0
        return math.degrees(math.acos(cos_ang))

    @staticmethod
    def _compute_position_uncertainty(cov: np.ndarray) -> float:
        """Largest 1-sigma position uncertainty from a 3x3 covariance matrix.

        Delegates to :func:`sim.core.fast_math.max_sigma_3x3`, which is also
        used by the health monitor so both safety consumers share one vetted
        implementation.
        """
        return max_sigma_3x3(cov)

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
