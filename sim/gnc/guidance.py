"""Three-phase ascent guidance law.

Phase 1 — Vertical rise (0 to VERTICAL_RISE_TIME_S):
    Hold pure vertical attitude, full throttle.

Phase 2 — Gravity turn (VERTICAL_RISE_TIME_S to MECO):
    Apply a small pitch kick, then steer thrust along the Earth-relative
    velocity vector (gravity turn).

Phase 3 — Terminal guidance (Stage 2):
    Simplified PEG / linear-tangent steering to achieve target orbit
    (altitude, velocity, flight path angle ~ 0 deg).
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np

from sim import config
from sim.core.reference_frames import (
    quaternion_from_axis_angle,
)
from sim.core.state import VehicleState


class GuidancePhase(IntEnum):
    """Guidance phase enumeration."""

    VERTICAL_RISE = 1
    GRAVITY_TURN = 2
    TERMINAL = 3


class GuidanceCommand:
    """Output of the guidance law."""

    def __init__(
        self,
        desired_quaternion: np.ndarray,
        throttle: float,
        phase: GuidancePhase,
    ) -> None:
        self.desired_quaternion = desired_quaternion
        self.throttle = throttle
        self.phase = phase


class GuidanceLaw:
    """Three-phase ascent guidance."""

    def __init__(self, meco_time_s: float = config.S1_BURN_TIME_S) -> None:
        self._meco_time_s = meco_time_s
        self._phase = GuidancePhase.VERTICAL_RISE
        self._pitch_kick_applied = False
        # Pre-compute launch azimuth direction in ECI at t=0
        self._launch_downrange_eci = self._compute_launch_downrange()

    def update(self, state: VehicleState) -> GuidanceCommand:
        """Compute guidance command for the current state.

        Args:
            state: Estimated vehicle state.

        Returns:
            GuidanceCommand with desired quaternion and throttle.
        """
        t = state.time_s

        if t < config.VERTICAL_RISE_TIME_S:
            self._phase = GuidancePhase.VERTICAL_RISE
            return self._vertical_rise(state)
        elif t < self._meco_time_s + 5.0:  # Continue gravity turn through staging
            self._phase = GuidancePhase.GRAVITY_TURN
            return self._gravity_turn(state)
        else:
            self._phase = GuidancePhase.TERMINAL
            return self._terminal_guidance(state)

    @property
    def phase(self) -> GuidancePhase:
        return self._phase

    def _vertical_rise(self, state: VehicleState) -> GuidanceCommand:
        """Hold thrust axis along local vertical."""
        desired_dir = self._local_up(state)
        q_des = self._quaternion_aligning_thrust(desired_dir)
        return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.VERTICAL_RISE)

    def _gravity_turn(self, state: VehicleState) -> GuidanceCommand:
        """Gravity turn: programmed pitch schedule blended with velocity tracking."""
        t = state.time_s
        up = self._local_up(state)

        # Compute Earth-relative velocity (subtract co-rotation)
        omega_earth = np.array([0.0, 0.0, config.EARTH_OMEGA])
        vel_earth_rel = state.velocity_eci - np.cross(omega_earth, state.position_eci)
        v_rel_mag = np.linalg.norm(vel_earth_rel)

        # Determine downrange direction: use launch azimuth initially,
        # blend toward velocity vector as horizontal speed builds
        launch_dr = self._launch_downrange_eci
        # Remove any component along up from launch downrange
        launch_dr_perp = launch_dr - np.dot(launch_dr, up) * up
        launch_dr_mag = np.linalg.norm(launch_dr_perp)
        if launch_dr_mag > 1e-6:
            launch_dr_perp /= launch_dr_mag
        else:
            launch_dr_perp = self._default_downrange(up)

        if v_rel_mag > 50.0:
            v_perp = vel_earth_rel - np.dot(vel_earth_rel, up) * up
            v_perp_mag = np.linalg.norm(v_perp)
            if v_perp_mag > 10.0:
                vel_dr = v_perp / v_perp_mag
                # Blend: use launch azimuth early, velocity direction later
                blend = min(1.0, v_perp_mag / 200.0)
                downrange = (1.0 - blend) * launch_dr_perp + blend * vel_dr
                downrange /= np.linalg.norm(downrange)
            else:
                downrange = launch_dr_perp
        else:
            downrange = launch_dr_perp

        elapsed = t - config.PITCH_KICK_TIME_S

        if elapsed < 0:
            desired_dir = up
        else:
            # Programmed pitch schedule: ramp from kick angle to ~85° at MECO
            # Using a quadratic ramp for smooth pitch-over
            burn_remaining = max(1.0, self._meco_time_s - config.PITCH_KICK_TIME_S)
            fraction = min(1.0, elapsed / burn_remaining)
            # Quadratic ramp: starts slow, accelerates
            target_pitch_at_meco_deg = 80.0
            programmed_pitch_deg = (
                config.PITCH_KICK_DEG + (target_pitch_at_meco_deg - config.PITCH_KICK_DEG) * fraction**1.5
            )
            programmed_pitch_rad = math.radians(min(programmed_pitch_deg, 89.0))

            # Also compute velocity-vector pitch for blending
            if v_rel_mag > 50.0:
                v_up = np.dot(vel_earth_rel / v_rel_mag, up)
                vel_pitch_from_vert = math.acos(np.clip(v_up, -1.0, 1.0))
            else:
                vel_pitch_from_vert = math.radians(config.PITCH_KICK_DEG)

            # Use the smaller of programmed and velocity-vector pitches
            # early on (safety), but allow programmed to lead later
            if fraction < 0.3:
                # Early: constrained to min of programmed and velocity
                pitch_from_vert = min(programmed_pitch_rad, vel_pitch_from_vert)
            else:
                # Later: follow programmed schedule (velocity will catch up)
                pitch_from_vert = programmed_pitch_rad

            desired_dir = up * math.cos(pitch_from_vert) + downrange * math.sin(pitch_from_vert)
            desired_dir /= np.linalg.norm(desired_dir)

        q_des = self._quaternion_aligning_thrust(desired_dir)
        return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.GRAVITY_TURN)

    def _terminal_guidance(self, state: VehicleState) -> GuidanceCommand:
        """Simplified PEG for terminal guidance to target orbit."""
        pos = state.position_eci
        vel = state.velocity_eci
        r = np.linalg.norm(pos)
        _v = np.linalg.norm(vel)  # noqa: F841 (reserved for future use)

        if r < 1.0:
            return GuidanceCommand(state.quaternion.copy(), 1.0, GuidancePhase.TERMINAL)

        r_hat = pos / r
        target_r = config.EARTH_RADIUS_M + config.TARGET_ALTITUDE_M

        # Radial and tangential velocity
        v_radial = np.dot(vel, r_hat)
        v_tangent_vec = vel - v_radial * r_hat
        v_tangent = np.linalg.norm(v_tangent_vec)

        if v_tangent > 1.0:
            t_hat = v_tangent_vec / v_tangent
        else:
            h = np.cross(pos, vel)
            t_hat = np.cross(h, pos)
            t_mag = np.linalg.norm(t_hat)
            t_hat = t_hat / t_mag if t_mag > 1e-6 else np.array([0.0, 1.0, 0.0])

        # Errors
        delta_r = target_r - r
        delta_v_tangent = config.TARGET_VELOCITY_MS - v_tangent

        # Estimate time-to-go
        a_thrust = config.S2_THRUST_VAC_N / max(state.mass_kg, 1.0)
        if a_thrust > 0.1 and delta_v_tangent > 0:
            t_go = max(1.0, delta_v_tangent / a_thrust)
        else:
            t_go = 10.0

        # Linear tangent steering: pitch from radial direction
        # Goal: bring v_radial to 0, achieve target altitude
        if t_go > 1.0:
            # Desired radial acceleration to zero v_radial over t_go
            # and reach target altitude
            a_radial_needed = -v_radial / t_go + 2 * (delta_r - v_radial * t_go) / (t_go**2)
            pitch_rad = math.atan2(a_radial_needed, a_thrust)
            pitch_rad = np.clip(pitch_rad, -math.pi / 3, math.pi / 3)
        else:
            # Near burnout: point tangential
            pitch_rad = 0.0

        # Desired direction: blend of radial and tangential
        desired_dir = r_hat * math.sin(pitch_rad) + t_hat * math.cos(pitch_rad)
        norm = np.linalg.norm(desired_dir)
        if norm > 1e-10:
            desired_dir /= norm

        q_des = self._quaternion_aligning_thrust(desired_dir)
        return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.TERMINAL)

    @staticmethod
    def _compute_launch_downrange() -> np.ndarray:
        """Compute the downrange direction in ECI at t=0 from launch azimuth."""
        lat_rad = math.radians(config.LAUNCH_LAT_DEG)
        lon_rad = math.radians(config.LAUNCH_LON_DEG)

        # Launch azimuth for target inclination: sin(az) = cos(inc)/cos(lat)
        cos_inc = math.cos(math.radians(config.TARGET_INCLINATION_DEG))
        cos_lat = math.cos(lat_rad)
        sin_az = min(1.0, cos_inc / max(cos_lat, 1e-10))
        cos_az = math.sqrt(max(0.0, 1.0 - sin_az**2))

        # North, East, Up unit vectors in ECEF at launch site
        sin_lat = math.sin(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        north_ecef = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        east_ecef = np.array([-sin_lon, cos_lon, 0.0])

        # Downrange in ECEF
        downrange_ecef = north_ecef * cos_az + east_ecef * sin_az

        # Convert to ECI at t=0 (ECEF = ECI at t=0)
        return downrange_ecef / np.linalg.norm(downrange_ecef)

    @staticmethod
    def _local_up(state: VehicleState) -> np.ndarray:
        """Unit vector from Earth center through vehicle."""
        r = np.linalg.norm(state.position_eci)
        if r < 1.0:
            return np.array([0.0, 0.0, 1.0])
        return state.position_eci / r

    @staticmethod
    def _default_downrange(up: np.ndarray) -> np.ndarray:
        """Compute a default downrange direction perpendicular to up."""
        arb = np.array([1.0, 0.0, 0.0])
        arb = arb - np.dot(arb, up) * up
        mag = np.linalg.norm(arb)
        if mag > 1e-6:
            return arb / mag
        return np.array([0.0, 1.0, 0.0])

    @staticmethod
    def _quaternion_aligning_thrust(desired_dir_eci: np.ndarray) -> np.ndarray:
        """Compute quaternion that rotates body +X to desired_dir_eci.

        Args:
            desired_dir_eci: Desired thrust direction in ECI (unit vector).

        Returns:
            Attitude quaternion [x, y, z, w].
        """
        body_x = np.array([1.0, 0.0, 0.0])
        d = desired_dir_eci / max(np.linalg.norm(desired_dir_eci), 1e-10)
        dot = np.dot(body_x, d)
        if dot > 0.99999:
            return np.array([0.0, 0.0, 0.0, 1.0])
        if dot < -0.99999:
            return np.array([0.0, 0.0, 1.0, 0.0])
        axis = np.cross(body_x, d)
        axis /= np.linalg.norm(axis)
        angle = math.acos(np.clip(dot, -1.0, 1.0))
        return quaternion_from_axis_angle(axis, angle)
