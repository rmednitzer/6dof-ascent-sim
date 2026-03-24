"""Three-phase ascent guidance law with Powered Explicit Guidance (PEG).

Phase 1 — Vertical rise (0 to VERTICAL_RISE_TIME_S):
    Hold pure vertical attitude, full throttle.

Phase 2 — Gravity turn (VERTICAL_RISE_TIME_S to MECO):
    Apply a small pitch kick, then steer thrust along the Earth-relative
    velocity vector (gravity turn).

Phase 3 — Terminal guidance (Stage 2):
    Powered Explicit Guidance (PEG) using the linear tangent steering law.
    This is the standard algorithm used on the Space Shuttle and many
    upper stages for precision orbit insertion.

References:
    Brand, Brown, Higgins, "Unified Powered Flight Guidance",
    NASA MSC Internal Note 73-FM-44, 1973.
    Jaggers, "An explicit solution to the exo-atmospheric powered flight
    guidance and trajectory optimization problem", JSR, 1977.
    Luidens & Miller, "Efficient numerical integration of gravity-turn
    trajectories", NASA TN D-3211, 1966.
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
    """Three-phase ascent guidance with PEG terminal phase."""

    def __init__(self, meco_time_s: float = config.S1_BURN_TIME_S) -> None:
        self._meco_time_s = meco_time_s
        self._phase = GuidancePhase.VERTICAL_RISE
        self._pitch_kick_applied = False
        self._launch_downrange_eci = self._compute_launch_downrange()

        # PEG state — A is initialized from current flight path on first call
        self._peg_initialized = False
        self._peg_A = 0.0  # Will be set from current flight direction
        self._peg_B = 0.0
        self._peg_T = 100.0
        self._peg_converged = False
        self._peg_last_update_t = -10.0
        self._peg_update_interval = 2.0  # Update PEG every 2 seconds

    def update(self, state: VehicleState) -> GuidanceCommand:
        """Compute guidance command for the current state."""
        t = state.time_s

        # Phase boundaries — start PEG early to maximize guidance efficiency
        peg_start = self._meco_time_s + 5.0  # PEG starts 5s after MECO
        blend_duration = 15.0  # 15s blending period from gravity turn to PEG

        if t < config.VERTICAL_RISE_TIME_S:
            self._phase = GuidancePhase.VERTICAL_RISE
            return self._vertical_rise(state)
        elif t < peg_start:
            self._phase = GuidancePhase.GRAVITY_TURN
            return self._gravity_turn(state)
        elif t < peg_start + blend_duration:
            # Smooth blend from gravity turn to PEG
            self._phase = GuidancePhase.TERMINAL
            blend_frac = (t - peg_start) / blend_duration
            gt_cmd = self._gravity_turn(state)
            peg_cmd = self._terminal_guidance_peg(state)
            # Slerp-like blend of desired directions
            gt_dir = self._quaternion_to_thrust_dir(gt_cmd.desired_quaternion)
            peg_dir = self._quaternion_to_thrust_dir(peg_cmd.desired_quaternion)
            blended_dir = (1.0 - blend_frac) * gt_dir + blend_frac * peg_dir
            norm = np.linalg.norm(blended_dir)
            if norm > 1e-10:
                blended_dir /= norm
            q_des = self._quaternion_aligning_thrust(blended_dir)
            return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.TERMINAL)
        else:
            self._phase = GuidancePhase.TERMINAL
            return self._terminal_guidance_peg(state)

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

        omega_earth = np.array([0.0, 0.0, config.EARTH_OMEGA])
        vel_earth_rel = state.velocity_eci - np.cross(omega_earth, state.position_eci)
        v_rel_mag = np.linalg.norm(vel_earth_rel)

        launch_dr = self._launch_downrange_eci
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
            burn_remaining = max(1.0, self._meco_time_s - config.PITCH_KICK_TIME_S)
            fraction = min(1.0, elapsed / burn_remaining)
            target_pitch_at_meco_deg = 85.0  # More aggressive for orbital efficiency
            programmed_pitch_deg = (
                config.PITCH_KICK_DEG + (target_pitch_at_meco_deg - config.PITCH_KICK_DEG) * fraction**1.5
            )
            programmed_pitch_rad = math.radians(min(programmed_pitch_deg, 89.0))

            if v_rel_mag > 50.0:
                v_up = np.dot(vel_earth_rel / v_rel_mag, up)
                vel_pitch_from_vert = math.acos(np.clip(v_up, -1.0, 1.0))
            else:
                vel_pitch_from_vert = math.radians(config.PITCH_KICK_DEG)

            if fraction < 0.3:
                pitch_from_vert = min(programmed_pitch_rad, vel_pitch_from_vert)
            else:
                pitch_from_vert = programmed_pitch_rad

            desired_dir = up * math.cos(pitch_from_vert) + downrange * math.sin(pitch_from_vert)
            desired_dir /= np.linalg.norm(desired_dir)

        q_des = self._quaternion_aligning_thrust(desired_dir)
        return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.GRAVITY_TURN)

    def _terminal_guidance_peg(self, state: VehicleState) -> GuidanceCommand:
        """Powered Explicit Guidance (PEG) for precision orbit insertion.

        Implements the linear tangent steering law:
            f_r(t) = A + B * (t - t0)
            f_h(t) = sqrt(1 - f_r^2)

        where f_r is the radial thrust fraction and f_h is the horizontal
        thrust fraction. A and B are computed iteratively to satisfy the
        terminal altitude and velocity constraints.
        """
        pos = state.position_eci
        vel = state.velocity_eci
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        t = state.time_s

        if r < 1.0 or v < 1.0:
            return GuidanceCommand(state.quaternion.copy(), 1.0, GuidancePhase.TERMINAL)

        r_hat = pos / r
        target_r = config.EARTH_RADIUS_M + config.TARGET_ALTITUDE_M

        # Radial and tangential decomposition
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

        # Thrust acceleration magnitude
        a_thrust = config.S2_THRUST_VAC_N / max(state.mass_kg, 1.0)

        # Exhaust velocity
        v_e = config.S2_ISP_VAC_S * config.G0

        # --- PEG iteration (update A, B, T periodically) ---
        if not self._peg_initialized:
            # Initialize A from current flight path angle to avoid discontinuity
            if v > 1.0:
                self._peg_A = v_radial / v  # Current radial fraction of velocity
            self._peg_B = 0.0
            self._peg_T = max(10.0, (config.TARGET_VELOCITY_MS - v_tangent) / max(a_thrust, 0.1))

        if t - self._peg_last_update_t >= self._peg_update_interval or not self._peg_initialized:
            self._peg_last_update_t = t
            self._update_peg_coefficients(
                r=r,
                v_radial=v_radial,
                v_tangent=v_tangent,
                target_r=target_r,
                target_v=config.TARGET_VELOCITY_MS,
                a_thrust=a_thrust,
                v_e=v_e,
                mass=state.mass_kg,
            )

        # --- Compute steering direction from linear tangent law ---
        # Time since PEG was last updated; A and B define the steering
        # f_r = sin(theta) where theta = atan(A + B*tau), tau measured from last update
        tau = 0.0  # At the update point
        f_r = self._peg_A + self._peg_B * tau

        # Clamp radial fraction — most thrust should go toward orbital velocity.
        # Once above target altitude, allow stronger downward steering to shed altitude.
        if r > target_r:
            f_r = np.clip(f_r, -0.5, 0.1)  # Above target: bias toward horizontal/down
        else:
            f_r = np.clip(f_r, -0.3, 0.4)  # Below target: allow some climb
        f_h = math.sqrt(max(0.0, 1.0 - f_r * f_r))

        desired_dir = r_hat * f_r + t_hat * f_h
        norm = np.linalg.norm(desired_dir)
        if norm > 1e-10:
            desired_dir /= norm

        q_des = self._quaternion_aligning_thrust(desired_dir)
        return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.TERMINAL)

    def _update_peg_coefficients(
        self,
        r: float,
        v_radial: float,
        v_tangent: float,
        target_r: float,
        target_v: float,
        a_thrust: float,
        v_e: float,
        mass: float,
    ) -> None:
        """Iteratively solve for PEG linear tangent coefficients A, B, T.

        Uses the PEG predictor-corrector iteration to find coefficients
        that satisfy terminal radius and velocity constraints simultaneously.

        The algorithm iterates on time-to-go (T) and the linear tangent
        parameters (A, B) until the predicted terminal conditions converge
        to the targets.
        """
        if a_thrust < 0.1:
            return

        # Gravitational acceleration at current altitude
        mu = config.EARTH_MU
        g_r_full = mu / (r * r)

        # Effective radial gravity: subtract centripetal acceleration (v_t^2/r)
        # This is the standard PEG formulation — the centripetal term is critical
        # for maintaining altitude during the burn.
        centripetal = v_tangent * v_tangent / r if r > 1.0 else 0.0
        g_r = g_r_full - centripetal

        # Target conditions
        r_target = target_r
        vr_target = 0.0  # Zero radial velocity at insertion (circular orbit)
        vt_target = target_v

        # Errors
        delta_r = r_target - r
        delta_vr = vr_target - v_radial
        delta_vt = vt_target - v_tangent

        T = self._peg_T if self._peg_initialized else max(10.0, delta_vt / max(a_thrust, 0.1))

        # PEG iteration (3 iterations for convergence)
        for _ in range(3):
            if T < 1.0:
                T = 1.0
                break

            # Thrust integrals (constant thrust approximation with mass depletion)
            # For constant-thrust rocket: tau = v_e / a_thrust
            tau = v_e / max(a_thrust, 0.1)

            if tau < T:
                # Mass ratio effects are significant
                # Integral quantities from the rocket equation
                # b0 = -v_e * ln(1 - T/tau)
                # b1 = b0*tau - v_e*T
                # c0 = b0*T - b1
                # c1 = c0*tau - v_e*T^2/2
                ratio = T / tau
                if ratio > 0.95:
                    ratio = 0.95  # Prevent singularity (can't burn all propellant)
                ln_term = -math.log(1.0 - ratio)
                b0 = v_e * ln_term
                b1 = b0 * tau - v_e * T
                c0 = b0 * T - b1
                c1 = c0 * tau - v_e * T * T / 2.0
            else:
                # Constant acceleration approximation (enough propellant)
                b0 = a_thrust * T
                b1 = a_thrust * T * T / 2.0
                c0 = a_thrust * T * T / 2.0
                c1 = a_thrust * T * T * T / 6.0

            if abs(b0) < 1e-6:
                break

            # Solve for A and B from terminal constraints
            # Using the simplified PEG equations:
            # delta_vr = b0*A + b1*B - g_r*T   (radial velocity)
            # delta_r  = c0*A + c1*B + v_radial*T - 0.5*g_r*T^2  (altitude)
            #
            # Rearranging:
            # b0*A + b1*B = delta_vr + g_r*T
            # c0*A + c1*B = delta_r - v_radial*T + 0.5*g_r*T^2

            rhs1 = delta_vr + g_r * T
            rhs2 = delta_r - v_radial * T + 0.5 * g_r * T * T

            det = b0 * c1 - b1 * c0
            if abs(det) < 1e-10:
                break

            A = (c1 * rhs1 - b1 * rhs2) / det
            B = (b0 * rhs2 - c0 * rhs1) / det

            # Update time-to-go from tangential velocity deficit
            if abs(a_thrust) > 0.1 and delta_vt > 0:
                # Use rocket equation for better T estimate
                mdot = config.S2_THRUST_VAC_N / (config.S2_ISP_VAC_S * config.G0)
                if mdot > 0 and mass > mdot * 1.0:
                    # Time to exhaust remaining propellant
                    T_max = (mass - config.S2_DRY_MASS_KG) / mdot
                    # Time to achieve tangential velocity (accounting for radial losses)
                    f_h = math.sqrt(max(0.1, 1.0 - min(A * A, 0.8)))
                    T_new = delta_vt / (a_thrust * f_h)
                    T_new = min(T_new, T_max)
                    T = 0.6 * T + 0.4 * max(1.0, T_new)

        # Store converged values
        try:
            self._peg_A = float(np.clip(A, -0.95, 0.95))
            self._peg_B = float(np.clip(B, -0.5, 0.5))
        except UnboundLocalError:
            pass  # Keep previous values if iteration didn't complete
        self._peg_T = max(1.0, T)
        self._peg_initialized = True

    @staticmethod
    def _quaternion_to_thrust_dir(q: np.ndarray) -> np.ndarray:
        """Extract thrust direction (body +X in ECI) from quaternion."""
        from sim.core.reference_frames import body_to_eci

        return body_to_eci(np.array([1.0, 0.0, 0.0]), q)

    @staticmethod
    def _compute_launch_downrange() -> np.ndarray:
        """Compute the downrange direction in ECI at t=0 from launch azimuth."""
        lat_rad = math.radians(config.LAUNCH_LAT_DEG)
        lon_rad = math.radians(config.LAUNCH_LON_DEG)

        cos_inc = math.cos(math.radians(config.TARGET_INCLINATION_DEG))
        cos_lat = math.cos(lat_rad)
        sin_az = min(1.0, cos_inc / max(cos_lat, 1e-10))
        cos_az = math.sqrt(max(0.0, 1.0 - sin_az**2))

        sin_lat = math.sin(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        north_ecef = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        east_ecef = np.array([-sin_lon, cos_lon, 0.0])

        downrange_ecef = north_ecef * cos_az + east_ecef * sin_az
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
        """Compute quaternion that rotates body +X to desired_dir_eci."""
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
