"""Gain-scheduled PID attitude controller with TVC output.

Computes thrust vector control (TVC) gimbal deflection commands from
the attitude error between desired and estimated orientations.  Gains
are scheduled with dynamic pressure and vehicle mass to maintain
consistent closed-loop bandwidth across the flight envelope.

Reference:
    Greensite, B.V., "Analysis and Design of Space Vehicle Flight
    Control Systems", NASA CR-820, 1967.
    Frosch & Vallely, "Saturn AS-501/S-IC Flight Control System Design",
    JSR, Vol. 4, No. 8, 1967.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim import config
from sim.core.reference_frames import quaternion_conjugate, quaternion_multiply


@dataclass
class TVCCommand:
    """Thrust vector control deflection command.

    Attributes:
        pitch_deg: TVC pitch deflection in body frame (deg).
        yaw_deg: TVC yaw deflection in body frame (deg).
    """

    pitch_deg: float
    yaw_deg: float


class AttitudeController:
    """Gain-scheduled PID attitude controller producing TVC gimbal commands.

    Gains are scheduled as functions of dynamic pressure and vehicle mass:
    - At high dynamic pressure, gains are reduced to prevent structural
      overload (the aerodynamic torques provide additional stiffness).
    - As mass decreases, the moment of inertia drops; gains are reduced
      proportionally to maintain consistent bandwidth.

    The scheduling follows the standard approach:
        K_eff = K_base * (q_ref / max(q, q_min)) * (mass / mass_ref)

    This ensures roughly constant closed-loop natural frequency across
    the flight envelope.
    """

    def __init__(self) -> None:
        self._integral_pitch: float = 0.0
        self._integral_yaw: float = 0.0
        self._kp_base: float = config.CONTROL_KP
        self._kd_base: float = config.CONTROL_KD
        self._ki_base: float = config.CONTROL_KI
        self._int_limit_rad: float = np.radians(config.CONTROL_INTEGRATOR_LIMIT_DEG)
        self._gain_schedule_enabled: bool = config.CONTROL_GAIN_SCHEDULE_ENABLED

        # Current scheduled gains (updated each timestep)
        self._kp: float = self._kp_base
        self._kd: float = self._kd_base
        self._ki: float = self._ki_base

    def reset(self) -> None:
        """Reset integrator state (e.g. after staging)."""
        self._integral_pitch = 0.0
        self._integral_yaw = 0.0

    def _schedule_gains(self, dynamic_pressure_pa: float, mass_kg: float) -> None:
        """Update PID gains based on current flight conditions.

        The gain schedule ensures:
        - Gains decrease when q is high (aero loads provide stiffness)
        - Gains decrease as mass drops (lower inertia = same torque has more effect)
        - Below 100 Pa dynamic pressure, gains use the baseline values
          (exoatmospheric / low-speed regime)
        """
        if not self._gain_schedule_enabled:
            self._kp = self._kp_base
            self._kd = self._kd_base
            self._ki = self._ki_base
            return

        q_ref = config.CONTROL_Q_REF_PA
        mass_ref = config.CONTROL_MASS_REF_KG

        # Dynamic pressure scheduling:
        # - At high q: reduce gains (aero provides stiffness + avoid overload)
        # - In vacuum (q < 100 Pa): BOOST gains by 1.5x (no aero damping)
        # - At reference q: unity gain
        if dynamic_pressure_pa > 100.0:
            q_factor = q_ref / max(dynamic_pressure_pa, 100.0)
            q_factor = np.clip(q_factor, 0.3, 3.0)
        else:
            # Exoatmospheric: boost gains to compensate for lack of aero stiffness
            q_factor = 1.5

        # Mass scheduling: scale with sqrt of mass ratio (lower mass = lower inertia)
        mass_factor = np.sqrt(max(mass_kg, 1000.0) / mass_ref)
        mass_factor = np.clip(mass_factor, 0.5, 2.0)

        gain_scale = q_factor * mass_factor

        self._kp = self._kp_base * gain_scale
        self._kd = self._kd_base * gain_scale
        self._ki = self._ki_base * gain_scale

    def update(
        self,
        q_desired: np.ndarray,
        q_estimated: np.ndarray,
        omega_body: np.ndarray,
        dt: float,
        dynamic_pressure_pa: float = 0.0,
        mass_kg: float = 0.0,
    ) -> TVCCommand:
        """Compute TVC deflection commands.

        Args:
            q_desired: Desired attitude quaternion [x, y, z, w].
            q_estimated: Estimated (current) attitude quaternion [x, y, z, w].
            omega_body: Estimated body angular velocity (rad/s).
            dt: Control timestep (s).
            dynamic_pressure_pa: Current dynamic pressure (Pa) for gain scheduling.
            mass_kg: Current vehicle mass (kg) for gain scheduling.

        Returns:
            TVCCommand with pitch and yaw deflections (deg), clamped to
            actuator limits.
        """
        # Schedule gains based on flight conditions
        if mass_kg > 0.0:
            self._schedule_gains(dynamic_pressure_pa, mass_kg)

        # Error quaternion in BODY frame: q_err = conj(q_estimated) * q_desired
        q_err = quaternion_multiply(quaternion_conjugate(q_estimated), q_desired)

        # Ensure short-path rotation (w >= 0)
        if q_err[3] < 0.0:
            q_err = -q_err

        # Small-angle attitude error in body frame (rad)
        err_pitch = 2.0 * q_err[1]  # body Y axis -> pitch
        err_yaw = 2.0 * q_err[2]  # body Z axis -> yaw

        # Angular rate (derivative term)
        rate_pitch = omega_body[1]
        rate_yaw = omega_body[2]

        # Integrate error with anti-windup
        self._integral_pitch += err_pitch * dt
        self._integral_yaw += err_yaw * dt

        self._integral_pitch = np.clip(self._integral_pitch, -self._int_limit_rad, self._int_limit_rad)
        self._integral_yaw = np.clip(self._integral_yaw, -self._int_limit_rad, self._int_limit_rad)

        # PID law using scheduled gains
        cmd_pitch_rad = self._kp * err_pitch - self._kd * rate_pitch + self._ki * self._integral_pitch
        cmd_yaw_rad = -(self._kp * err_yaw - self._kd * rate_yaw + self._ki * self._integral_yaw)

        cmd_pitch_deg = np.degrees(cmd_pitch_rad)
        cmd_yaw_deg = np.degrees(cmd_yaw_rad)

        max_def = config.TVC_MAX_DEFLECTION_DEG
        cmd_pitch_deg = float(np.clip(cmd_pitch_deg, -max_def, max_def))
        cmd_yaw_deg = float(np.clip(cmd_yaw_deg, -max_def, max_def))

        return TVCCommand(pitch_deg=cmd_pitch_deg, yaw_deg=cmd_yaw_deg)
