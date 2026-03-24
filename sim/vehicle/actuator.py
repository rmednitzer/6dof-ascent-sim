"""Second-order TVC actuator dynamics model.

Models the hydraulic servoactuator that drives the engine gimbal as a
second-order linear system with rate and position limits:

    x_ddot + 2*zeta*omega_n*x_dot + omega_n^2*x = omega_n^2*x_cmd

This captures the finite bandwidth and phase lag of real actuators, which
is critical for control-structure interaction analysis and realistic
transient response simulation.

References:
    Wie, B., *Space Vehicle Dynamics and Control*, 2nd ed., Ch. 7.
    Greensite, B.V., "Analysis and Design of Space Vehicle Flight
    Control Systems", NASA CR-820, 1967.
"""

from __future__ import annotations

import math

from sim import config


class TVCActuator:
    """Second-order TVC actuator for a single axis (pitch or yaw).

    The actuator is modeled as a second-order system with:
    - Natural frequency omega_n (rad/s)
    - Damping ratio zeta
    - Position limit (max deflection)
    - Rate limit (max slew rate)

    Integration uses semi-implicit Euler for symplectic stability.
    """

    def __init__(self) -> None:
        self._omega_n: float = 2.0 * math.pi * config.TVC_ACTUATOR_NATURAL_FREQ_HZ
        self._zeta: float = config.TVC_ACTUATOR_DAMPING_RATIO
        self._max_pos_rad: float = math.radians(config.TVC_MAX_DEFLECTION_DEG)
        self._max_rate_rad: float = math.radians(config.TVC_MAX_SLEW_RATE_DEG_S)

        # State: position and rate (radians, rad/s)
        self._position: float = 0.0
        self._rate: float = 0.0

    @property
    def position_deg(self) -> float:
        """Current actuator position (deg)."""
        return math.degrees(self._position)

    @property
    def position_rad(self) -> float:
        """Current actuator position (rad)."""
        return self._position

    def update(self, command_deg: float, dt: float) -> float:
        """Advance actuator state by dt and return actual position.

        Args:
            command_deg: Commanded deflection angle (deg).
            dt: Timestep (s).

        Returns:
            Actual deflection angle after actuator dynamics (deg).
        """
        if not config.TVC_ACTUATOR_DYNAMICS_ENABLED:
            # Bypass: ideal actuator with only position limits
            clamped = max(-config.TVC_MAX_DEFLECTION_DEG, min(config.TVC_MAX_DEFLECTION_DEG, command_deg))
            self._position = math.radians(clamped)
            return clamped

        cmd_rad = math.radians(command_deg)
        cmd_rad = max(-self._max_pos_rad, min(self._max_pos_rad, cmd_rad))

        wn = self._omega_n
        zeta = self._zeta

        # Second-order dynamics: x_ddot = wn^2*(x_cmd - x) - 2*zeta*wn*x_dot
        accel = wn * wn * (cmd_rad - self._position) - 2.0 * zeta * wn * self._rate

        # Semi-implicit Euler (update rate first, then position)
        self._rate += accel * dt

        # Rate limiting
        self._rate = max(-self._max_rate_rad, min(self._max_rate_rad, self._rate))

        # Position update
        self._position += self._rate * dt

        # Position limiting (with rate zeroing at hard stops)
        if self._position > self._max_pos_rad:
            self._position = self._max_pos_rad
            self._rate = min(0.0, self._rate)
        elif self._position < -self._max_pos_rad:
            self._position = -self._max_pos_rad
            self._rate = max(0.0, self._rate)

        return math.degrees(self._position)


class TVCActuatorPair:
    """Paired pitch and yaw TVC actuators."""

    def __init__(self) -> None:
        self.pitch = TVCActuator()
        self.yaw = TVCActuator()

    def update(self, cmd_pitch_deg: float, cmd_yaw_deg: float, dt: float) -> tuple[float, float]:
        """Advance both actuators and return actual deflections.

        Args:
            cmd_pitch_deg: Commanded pitch deflection (deg).
            cmd_yaw_deg: Commanded yaw deflection (deg).
            dt: Timestep (s).

        Returns:
            (actual_pitch_deg, actual_yaw_deg) after actuator dynamics.
        """
        actual_pitch = self.pitch.update(cmd_pitch_deg, dt)
        actual_yaw = self.yaw.update(cmd_yaw_deg, dt)
        return actual_pitch, actual_yaw
