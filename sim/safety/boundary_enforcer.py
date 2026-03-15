"""Boundary enforcer — clamps and validates all actuator commands and structural loads.

Every command flowing to the vehicle passes through the BoundaryEnforcer so that
no single software fault can exceed physical or safety limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.config import (
    MAX_AXIAL_G,
    MAX_LATERAL_G,
    MAX_Q_PA,
    S1_THROTTLE_MIN,
    TVC_MAX_DEFLECTION_DEG,
    TVC_MAX_SLEW_RATE_DEG_S,
)


@dataclass
class BoundaryResult:
    """Result of a boundary-enforcement check.

    Attributes:
        approved:       True if the (possibly clamped) value is safe to use.
        value:          The approved value — may differ from the original command
                        when clamping was applied.
        was_clamped:    True if the value was modified to stay within limits.
        violation_type: Short tag describing the violation, or None when nominal.
        evidence:       Dictionary with supporting data and timestamps.
    """

    approved: bool
    value: Any
    was_clamped: bool = False
    violation_type: str | None = None
    evidence: dict = field(default_factory=dict)


class BoundaryEnforcer:
    """Validates and clamps actuator commands and structural loads.

    Tracks previous TVC commands for slew-rate limiting and maintains a
    running count of all detected violations.
    """

    def __init__(self) -> None:
        self._prev_pitch_deg: float = 0.0
        self._prev_yaw_deg: float = 0.0
        self.violation_count: int = 0

    # ------------------------------------------------------------------
    # Throttle
    # ------------------------------------------------------------------
    def validate_throttle(
        self,
        throttle_cmd: float,
        propellant_remaining_kg: float,
    ) -> BoundaryResult:
        """Clamp throttle to [0, 1], enforce minimum throttle when engine is
        running, and inhibit thrust when propellant is exhausted.

        Args:
            throttle_cmd:           Commanded throttle fraction (0–1).
            propellant_remaining_kg: Remaining usable propellant mass in kg.

        Returns:
            BoundaryResult with the approved throttle value.
        """
        ts = time.monotonic()
        was_clamped = False
        violation: str | None = None

        # No thrust with zero propellant.
        if propellant_remaining_kg <= 0.0:
            self.violation_count += 1
            return BoundaryResult(
                approved=True,
                value=0.0,
                was_clamped=(throttle_cmd != 0.0),
                violation_type="propellant_depleted",
                evidence={
                    "timestamp": ts,
                    "original_cmd": throttle_cmd,
                    "propellant_remaining_kg": propellant_remaining_kg,
                },
            )

        original = throttle_cmd

        # Hard clamp to [0, 1].
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
        if throttle_cmd != original:
            was_clamped = True
            violation = "throttle_out_of_range"

        # If the engine is running (throttle > 0) enforce minimum throttle.
        if 0.0 < throttle_cmd < S1_THROTTLE_MIN:
            throttle_cmd = S1_THROTTLE_MIN
            was_clamped = True
            violation = "below_min_throttle"

        if was_clamped:
            self.violation_count += 1

        return BoundaryResult(
            approved=True,
            value=throttle_cmd,
            was_clamped=was_clamped,
            violation_type=violation,
            evidence={
                "timestamp": ts,
                "original_cmd": original,
                "approved_cmd": throttle_cmd,
                "propellant_remaining_kg": propellant_remaining_kg,
            },
        )

    # ------------------------------------------------------------------
    # TVC gimbal
    # ------------------------------------------------------------------
    def validate_tvc(
        self,
        pitch_cmd_deg: float,
        yaw_cmd_deg: float,
        dt: float,
    ) -> BoundaryResult:
        """Enforce TVC deflection and slew-rate limits.

        The deflection is clamped to [-TVC_MAX_DEFLECTION_DEG,
        +TVC_MAX_DEFLECTION_DEG] in each axis, and the rate of change is
        limited to TVC_MAX_SLEW_RATE_DEG_S.

        Args:
            pitch_cmd_deg: Commanded pitch gimbal angle (deg).
            yaw_cmd_deg:   Commanded yaw gimbal angle (deg).
            dt:            Timestep since the last call (s).

        Returns:
            BoundaryResult whose *value* is a (pitch_deg, yaw_deg) tuple.
        """
        ts = time.monotonic()
        was_clamped = False
        violation: str | None = None
        original_pitch = pitch_cmd_deg
        original_yaw = yaw_cmd_deg

        # --- Deflection limits ---
        pitch_cmd_deg = float(np.clip(pitch_cmd_deg, -TVC_MAX_DEFLECTION_DEG, TVC_MAX_DEFLECTION_DEG))
        yaw_cmd_deg = float(np.clip(yaw_cmd_deg, -TVC_MAX_DEFLECTION_DEG, TVC_MAX_DEFLECTION_DEG))
        if pitch_cmd_deg != original_pitch or yaw_cmd_deg != original_yaw:
            was_clamped = True
            violation = "tvc_deflection_limit"

        # --- Slew-rate limits ---
        if dt > 0.0:
            max_delta = TVC_MAX_SLEW_RATE_DEG_S * dt

            pitch_delta = pitch_cmd_deg - self._prev_pitch_deg
            if abs(pitch_delta) > max_delta:
                pitch_cmd_deg = self._prev_pitch_deg + np.sign(pitch_delta) * max_delta
                was_clamped = True
                violation = "tvc_slew_rate_limit"

            yaw_delta = yaw_cmd_deg - self._prev_yaw_deg
            if abs(yaw_delta) > max_delta:
                yaw_cmd_deg = self._prev_yaw_deg + np.sign(yaw_delta) * max_delta
                was_clamped = True
                violation = "tvc_slew_rate_limit"

        # Update history for next call.
        self._prev_pitch_deg = float(pitch_cmd_deg)
        self._prev_yaw_deg = float(yaw_cmd_deg)

        if was_clamped:
            self.violation_count += 1

        return BoundaryResult(
            approved=True,
            value=(float(pitch_cmd_deg), float(yaw_cmd_deg)),
            was_clamped=was_clamped,
            violation_type=violation,
            evidence={
                "timestamp": ts,
                "original_pitch_deg": original_pitch,
                "original_yaw_deg": original_yaw,
                "approved_pitch_deg": float(pitch_cmd_deg),
                "approved_yaw_deg": float(yaw_cmd_deg),
            },
        )

    # ------------------------------------------------------------------
    # Staging
    # ------------------------------------------------------------------
    def validate_staging(
        self,
        current_thrust_fraction: float,
        staging_armed: bool,
    ) -> BoundaryResult:
        """Permit stage separation only when thrust is near zero and the
        staging sequencer has been armed.

        Args:
            current_thrust_fraction: Current thrust as a fraction of nominal (0–1).
            staging_armed:           Whether the staging sequencer is armed.

        Returns:
            BoundaryResult with approved=True if separation is allowed.
        """
        ts = time.monotonic()
        reasons: list[str] = []

        if current_thrust_fraction > 0.05:
            reasons.append("thrust_too_high")
        if not staging_armed:
            reasons.append("staging_not_armed")

        approved = len(reasons) == 0

        if not approved:
            self.violation_count += 1

        return BoundaryResult(
            approved=approved,
            value=approved,
            was_clamped=False,
            violation_type=";".join(reasons) if reasons else None,
            evidence={
                "timestamp": ts,
                "current_thrust_fraction": current_thrust_fraction,
                "staging_armed": staging_armed,
            },
        )

    # ------------------------------------------------------------------
    # Structural limits
    # ------------------------------------------------------------------
    def check_structural_limits(
        self,
        axial_g: float,
        lateral_g: float,
        dynamic_pressure_pa: float,
    ) -> BoundaryResult:
        """Check whether current loads are within structural limits.

        Args:
            axial_g:             Axial load factor (g).
            lateral_g:           Lateral load factor (g).
            dynamic_pressure_pa: Dynamic pressure (Pa).

        Returns:
            BoundaryResult with approved=False if any limit is exceeded.
            The *evidence* dict contains proximity fractions for each axis
            (1.0 = at limit, >1.0 = exceeding limit).
        """
        ts = time.monotonic()

        axial_frac = abs(axial_g) / MAX_AXIAL_G if MAX_AXIAL_G else 0.0
        lateral_frac = abs(lateral_g) / MAX_LATERAL_G if MAX_LATERAL_G else 0.0
        q_frac = dynamic_pressure_pa / MAX_Q_PA if MAX_Q_PA else 0.0

        violations: list[str] = []
        if axial_frac > 1.0:
            violations.append("axial_g_exceeded")
        if lateral_frac > 1.0:
            violations.append("lateral_g_exceeded")
        if q_frac > 1.0:
            violations.append("dynamic_pressure_exceeded")

        approved = len(violations) == 0

        if not approved:
            self.violation_count += 1

        return BoundaryResult(
            approved=approved,
            value=approved,
            was_clamped=False,
            violation_type=";".join(violations) if violations else None,
            evidence={
                "timestamp": ts,
                "axial_g": axial_g,
                "lateral_g": lateral_g,
                "dynamic_pressure_pa": dynamic_pressure_pa,
                "axial_g_fraction": axial_frac,
                "lateral_g_fraction": lateral_frac,
                "q_fraction": q_frac,
            },
        )
