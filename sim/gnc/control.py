"""PID attitude controller with TVC output.

Computes thrust vector control (TVC) gimbal deflection commands from
the attitude error between desired and estimated orientations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim import config
from sim.core.reference_frames import quaternion_multiply, quaternion_conjugate


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
    """PID attitude controller producing TVC gimbal commands.

    The controller operates on the vector part of the error quaternion,
    which for small angles approximates half the rotation vector.  The
    body-frame pitch (Y) and yaw (Z) components drive the two TVC axes.

    Gains are imported from ``sim.config``:
        - CONTROL_KP (proportional)
        - CONTROL_KD (derivative, uses body angular velocity)
        - CONTROL_KI (integral, with anti-windup)

    The integrator is clamped to ``CONTROL_INTEGRATOR_LIMIT_DEG`` on each
    axis to prevent windup.
    """

    def __init__(self) -> None:
        self._integral_pitch: float = 0.0
        self._integral_yaw: float = 0.0
        self._kp: float = config.CONTROL_KP
        self._kd: float = config.CONTROL_KD
        self._ki: float = config.CONTROL_KI
        self._int_limit_rad: float = np.radians(config.CONTROL_INTEGRATOR_LIMIT_DEG)

    def reset(self) -> None:
        """Reset integrator state (e.g. after staging)."""
        self._integral_pitch = 0.0
        self._integral_yaw = 0.0

    def update(
        self,
        q_desired: np.ndarray,
        q_estimated: np.ndarray,
        omega_body: np.ndarray,
        dt: float,
    ) -> TVCCommand:
        """Compute TVC deflection commands.

        Args:
            q_desired: Desired attitude quaternion [x, y, z, w].
            q_estimated: Estimated (current) attitude quaternion [x, y, z, w].
            omega_body: Estimated body angular velocity (rad/s).
            dt: Control timestep (s).

        Returns:
            TVCCommand with pitch and yaw deflections (deg), clamped to
            actuator limits.
        """
        # Error quaternion in BODY frame: q_err = conj(q_estimated) * q_desired
        # This gives the rotation from current to desired expressed in the
        # current body frame, so q_err[1] and q_err[2] are body Y/Z errors.
        q_err = quaternion_multiply(quaternion_conjugate(q_estimated), q_desired)

        # Ensure short-path rotation (w >= 0)
        if q_err[3] < 0.0:
            q_err = -q_err

        # Small-angle attitude error in body frame (rad)
        # For small errors: theta ~ 2 * [qx, qy, qz]
        err_pitch = 2.0 * q_err[1]   # body Y axis -> pitch
        err_yaw = 2.0 * q_err[2]     # body Z axis -> yaw

        # Angular rate (derivative term) — body Y and Z rates
        rate_pitch = omega_body[1]
        rate_yaw = omega_body[2]

        # Integrate error with anti-windup
        self._integral_pitch += err_pitch * dt
        self._integral_yaw += err_yaw * dt

        self._integral_pitch = np.clip(
            self._integral_pitch, -self._int_limit_rad, self._int_limit_rad
        )
        self._integral_yaw = np.clip(
            self._integral_yaw, -self._int_limit_rad, self._int_limit_rad
        )

        # PID law (in radians)
        # Pitch: positive error -> positive TVC pitch -> positive torque_y (correct)
        # Yaw: positive error -> need positive torque_z, but due to cross-product
        #   positive TVC yaw -> negative torque_z, so negate yaw command.
        # D-term: subtract rate to damp oscillations.
        cmd_pitch_rad = (
            self._kp * err_pitch
            - self._kd * rate_pitch
            + self._ki * self._integral_pitch
        )
        cmd_yaw_rad = -(
            self._kp * err_yaw
            - self._kd * rate_yaw
            + self._ki * self._integral_yaw
        )

        # Convert to degrees
        cmd_pitch_deg = np.degrees(cmd_pitch_rad)
        cmd_yaw_deg = np.degrees(cmd_yaw_rad)

        # Clamp to TVC actuator limits
        max_def = config.TVC_MAX_DEFLECTION_DEG
        cmd_pitch_deg = float(np.clip(cmd_pitch_deg, -max_def, max_def))
        cmd_yaw_deg = float(np.clip(cmd_yaw_deg, -max_def, max_def))

        return TVCCommand(pitch_deg=cmd_pitch_deg, yaw_deg=cmd_yaw_deg)
