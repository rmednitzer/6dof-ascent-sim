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
from sim.core.fast_math import cross3, dot3, norm3
from sim.core.reference_frames import (
    lla_to_ecef,
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
        # Gravity-turn downrange uses the Earth-rotation-corrected (flight) azimuth
        # so the relative-velocity turn builds the right *inertial* plane (AD-17 a).
        self._launch_downrange_eci = self._compute_launch_downrange(rotation_correction=True)
        # Target inertial orbital-plane normal, fixed over the (short) ascent: the
        # plane through the launch site at the inertial azimuth for the target
        # inclination. The terminal phase steers to null velocity along this normal.
        downrange_inertial = self._compute_launch_downrange(rotation_correction=False)
        r_launch = lla_to_ecef(
            math.radians(config.LAUNCH_LAT_DEG), math.radians(config.LAUNCH_LON_DEG), config.LAUNCH_ALT_M
        )
        plane_normal = cross3(r_launch, downrange_inertial)
        self._target_plane_normal = plane_normal / norm3(plane_normal)

        # PEG state — A is initialized from current flight path on first call
        self._peg_initialized = False
        self._peg_A = 0.0  # Will be set from current flight direction
        self._peg_B = 0.0
        self._peg_T = 100.0
        self._peg_converged = False
        self._peg_last_update_t = -10.0
        self._peg_update_interval = 2.0  # Update PEG every 2 seconds

        # Commanded-direction slew limiter state
        self._prev_cmd_dir: np.ndarray | None = None
        self._prev_cmd_t: float | None = None

    def update(self, state: VehicleState) -> GuidanceCommand:
        """Compute guidance command for the current state."""
        t = state.time_s

        # Phase boundaries — start PEG early to maximize guidance efficiency
        peg_start = self._meco_time_s + 5.0  # PEG starts 5s after MECO
        blend_duration = 15.0  # 15s blending period from gravity turn to PEG

        if t < config.VERTICAL_RISE_TIME_S:
            self._phase = GuidancePhase.VERTICAL_RISE
            cmd = self._vertical_rise(state)
        elif t < peg_start:
            self._phase = GuidancePhase.GRAVITY_TURN
            cmd = self._gravity_turn(state)
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
            norm = norm3(blended_dir)
            if norm > 1e-10:
                blended_dir /= norm
            q_des = self._quaternion_aligning_thrust(blended_dir)
            cmd = GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.TERMINAL)
        else:
            self._phase = GuidancePhase.TERMINAL
            cmd = self._terminal_guidance_peg(state)

        return self._slew_limit(cmd, t)

    def _slew_limit(self, cmd: GuidanceCommand, t: float) -> GuidanceCommand:
        """Rate-limit the commanded thrust direction.

        Bounds the angular change of the commanded thrust axis to
        ``GUIDANCE_MAX_CMD_RATE_DEG_S`` so the attitude command stays
        physically trackable by the TVC. Without this, PEG can emit a
        step/jittery command that the actuator cannot follow, producing
        a large (spurious) actual-vs-commanded attitude error.
        """
        new_dir = self._quaternion_to_thrust_dir(cmd.desired_quaternion)
        n = norm3(new_dir)
        if n > 1e-10:
            new_dir = new_dir / n

        if self._prev_cmd_dir is None or self._prev_cmd_t is None:
            self._prev_cmd_dir = new_dir
            self._prev_cmd_t = t
            return cmd

        dt = t - self._prev_cmd_t
        self._prev_cmd_t = t
        if dt <= 0.0:
            self._prev_cmd_dir = new_dir
            return cmd

        prev = self._prev_cmd_dir
        cos_ang = float(np.clip(dot3(prev, new_dir), -1.0, 1.0))
        angle = math.acos(cos_ang)
        max_step = math.radians(config.GUIDANCE_MAX_CMD_RATE_DEG_S) * dt

        if angle <= max_step or angle < 1e-9:
            self._prev_cmd_dir = new_dir
            return cmd

        # Rotate `prev` toward `new_dir` by exactly max_step (vector slerp).
        axis = cross3(prev, new_dir)
        axis_mag = norm3(axis)
        if axis_mag < 1e-12:
            # prev and new_dir are (anti)parallel. The near-zero-angle
            # case already returned above, so this is the ~180 deg flip:
            # the cross product is degenerate but the command MUST still
            # be rate-limited (an instantaneous 180 deg jump is exactly
            # what the limiter exists to prevent). Any axis perpendicular
            # to prev rotates toward the antipode; pick a stable one.
            helper = np.array([1.0, 0.0, 0.0]) if abs(prev[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = cross3(prev, helper)
            axis_mag = norm3(axis)
            if axis_mag < 1e-12:
                self._prev_cmd_dir = new_dir
                return cmd
        axis /= axis_mag
        limited = (
            prev * math.cos(max_step)
            + cross3(axis, prev) * math.sin(max_step)
            + axis * dot3(axis, prev) * (1.0 - math.cos(max_step))
        )
        lm = norm3(limited)
        if lm > 1e-10:
            limited /= lm
        self._prev_cmd_dir = limited
        return GuidanceCommand(
            self._quaternion_aligning_thrust(limited),
            throttle=cmd.throttle,
            phase=cmd.phase,
        )

    @property
    def phase(self) -> GuidancePhase:
        return self._phase

    def _vertical_rise(self, state: VehicleState) -> GuidanceCommand:
        """Hold thrust axis along local vertical."""
        desired_dir = self._local_up(state)
        q_des = self._quaternion_aligning_thrust(desired_dir)
        return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.VERTICAL_RISE)

    def _gravity_turn(self, state: VehicleState) -> GuidanceCommand:
        """Gravity turn: thrust tracks the Earth-relative velocity vector
        (near-zero angle of attack), with the programmed pitch schedule
        acting only as an upper bound on pitch-over rate."""
        t = state.time_s
        up = self._local_up(state)

        # omega_earth = [0, 0, EARTH_OMEGA]; cross with position reduces to:
        #   omega × p = [-omega*p_y, omega*p_x, 0]
        p0, p1, _ = state.position_eci[0], state.position_eci[1], state.position_eci[2]
        omega_z = config.EARTH_OMEGA
        vel_earth_rel = state.velocity_eci - np.array([-omega_z * p1, omega_z * p0, 0.0])
        v_rel_mag = norm3(vel_earth_rel)

        launch_dr = self._launch_downrange_eci
        launch_dr_perp = launch_dr - dot3(launch_dr, up) * up
        launch_dr_mag = norm3(launch_dr_perp)
        if launch_dr_mag > 1e-6:
            launch_dr_perp /= launch_dr_mag
        else:
            launch_dr_perp = self._default_downrange(up)

        if v_rel_mag > 50.0:
            v_perp = vel_earth_rel - dot3(vel_earth_rel, up) * up
            v_perp_mag = norm3(v_perp)
            if v_perp_mag > 10.0:
                vel_dr = v_perp / v_perp_mag
                blend = min(1.0, v_perp_mag / 200.0)
                downrange = (1.0 - blend) * launch_dr_perp + blend * vel_dr
                downrange /= norm3(downrange)
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
                v_up = dot3(vel_earth_rel, up) / v_rel_mag
                if v_up > 1.0:
                    v_up = 1.0
                elif v_up < -1.0:
                    v_up = -1.0
                vel_pitch_from_vert = math.acos(v_up)
            else:
                vel_pitch_from_vert = math.radians(config.PITCH_KICK_DEG)

            # True gravity turn: follow the Earth-relative velocity vector
            # (zero AoA). The programmed schedule is only a ceiling so the
            # vehicle can never pitch over faster than the schedule allows
            # — it must not be flown open-loop (the old `else` branch did,
            # which lofted the trajectory and drove large AoA/lateral-G).
            pitch_from_vert = min(programmed_pitch_rad, vel_pitch_from_vert)

            desired_dir = up * math.cos(pitch_from_vert) + downrange * math.sin(pitch_from_vert)
            desired_dir /= norm3(desired_dir)

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
        r = norm3(pos)
        v = norm3(vel)
        t = state.time_s

        if r < 1.0 or v < 1.0:
            return GuidanceCommand(state.quaternion.copy(), 1.0, GuidancePhase.TERMINAL)

        r_hat = pos / r
        target_r = config.EARTH_RADIUS_M + config.TARGET_ALTITUDE_M

        # Radial and tangential decomposition
        v_radial = dot3(vel, r_hat)
        v_tangent_vec = vel - v_radial * r_hat
        v_tangent = norm3(v_tangent_vec)

        if v_tangent > 1.0:
            t_hat = v_tangent_vec / v_tangent
        else:
            h = cross3(pos, vel)
            t_hat = cross3(h, pos)
            t_mag = norm3(t_hat)
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
        # f_r(tau) = A + B*tau, with tau the time elapsed since the PEG
        # coefficients were last solved. Using tau=0 (the old code) made
        # B inert and degenerated PEG to a constant-pitch hold.
        tau = max(0.0, t - self._peg_last_update_t)
        f_r = self._peg_A + self._peg_B * tau

        # Safety clamp only: f_r is a direction cosine, |f_r| < 1 so that
        # f_h = sqrt(1 - f_r^2) is real. Do NOT clamp tighter than this —
        # the PEG solve owns the radial/horizontal split; clamping it to a
        # narrow band (as the old code did) discards the guidance solution.
        f_r = float(np.clip(f_r, -0.95, 0.95))
        f_h = math.sqrt(max(0.0, 1.0 - f_r * f_r))

        desired_dir = r_hat * f_r + t_hat * f_h
        norm = norm3(desired_dir)
        if norm > 1e-10:
            desired_dir /= norm

        desired_dir = self._apply_plane_steering(desired_dir, state)
        q_des = self._quaternion_aligning_thrust(desired_dir)
        return GuidanceCommand(q_des, throttle=1.0, phase=GuidancePhase.TERMINAL)

    def _apply_plane_steering(self, desired_dir: np.ndarray, state: VehicleState) -> np.ndarray:
        """Yaw the commanded thrust to null inertial out-of-plane velocity (AD-17).

        Rotates *desired_dir* toward the target orbital plane by an angle
        proportional to the out-of-plane velocity ``v · n`` (``n`` =
        ``_target_plane_normal``), clamped to ``GUIDANCE_MAX_YAW_DEG``. Driving
        that component to zero converges the achieved inclination to
        ``TARGET_INCLINATION_DEG`` without a separate plane-change burn.
        """
        n = self._target_plane_normal
        v_op = dot3(state.velocity_eci, n)  # inertial out-of-plane velocity (m/s)
        max_yaw = math.radians(config.GUIDANCE_MAX_YAW_DEG)
        yaw = float(np.clip(-config.GUIDANCE_PLANE_STEER_GAIN * v_op, -max_yaw, max_yaw))
        if abs(yaw) < 1e-6:
            return desired_dir
        # In-plane component of the commanded direction, then rotate by `yaw`
        # toward the plane normal (sign of yaw already steers against v_op).
        d_ip = desired_dir - dot3(desired_dir, n) * n
        d_ip_mag = norm3(d_ip)
        if d_ip_mag < 1e-6:
            return desired_dir
        d_ip /= d_ip_mag
        steered = d_ip * math.cos(yaw) + n * math.sin(yaw)
        s_mag = norm3(steered)
        return steered / s_mag if s_mag > 1e-10 else desired_dir

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

        # PEG iteration (3 iterations for convergence). Seed A/B from the last
        # converged coefficients so they are always bound: if the loop
        # degenerates (an early `break` before the solve), we keep the previous
        # values instead of catching UnboundLocalError after the fact (Q-05).
        A = self._peg_A
        B = self._peg_B
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
                # Effective burn time consistent with the clamped ratio. Using
                # the unclamped T in b1/c0/c1 while clamping only the ln term made
                # the thrust integrals mutually inconsistent, corrupting the B
                # steering coefficient (~40%) whenever T > tau (AD-10).
                t_eff = ratio * tau
                ln_term = -math.log(1.0 - ratio)
                b0 = v_e * ln_term
                b1 = b0 * tau - v_e * t_eff
                c0 = b0 * t_eff - b1
                c1 = c0 * tau - v_e * t_eff * t_eff / 2.0
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

        # Store converged values (A/B are always bound — seeded above, so a
        # degenerate iteration simply re-stores the previous clamped values).
        self._peg_A = float(np.clip(A, -0.95, 0.95))
        self._peg_B = float(np.clip(B, -0.5, 0.5))
        self._peg_T = max(1.0, T)
        self._peg_initialized = True

    @staticmethod
    def _quaternion_to_thrust_dir(q: np.ndarray) -> np.ndarray:
        """Extract thrust direction (body +X in ECI) from quaternion."""
        from sim.core.reference_frames import body_to_eci

        return body_to_eci(np.array([1.0, 0.0, 0.0]), q)

    @staticmethod
    def _compute_launch_downrange(rotation_correction: bool = False) -> np.ndarray:
        """Downrange direction in ECI at t=0 for the target inclination.

        The *inertial* azimuth (``sin Az = cos i / cos lat``) defines the target
        orbital plane. With ``rotation_correction`` the returned direction is the
        Earth-rotation-corrected *flight* azimuth — the ground-relative heading a
        gravity turn must fly so the resulting inertial velocity (flight velocity
        plus the launch-site eastward rotation ``ω R cos lat``) lands in that
        plane (AD-17). Without it, the raw inertial-azimuth direction is returned
        (used to fix the target-plane normal).
        """
        lat_rad = math.radians(config.LAUNCH_LAT_DEG)
        lon_rad = math.radians(config.LAUNCH_LON_DEG)

        cos_inc = math.cos(math.radians(config.TARGET_INCLINATION_DEG))
        cos_lat = math.cos(lat_rad)
        sin_az = min(1.0, cos_inc / max(cos_lat, 1e-10))
        cos_az = math.sqrt(max(0.0, 1.0 - sin_az**2))

        if rotation_correction:
            # Flight azimuth: subtract the launch-site eastward rotation velocity
            # from the required inertial eastward velocity (Az from north, toward east).
            v_orb = config.TARGET_VELOCITY_MS
            v_eqrot = config.EARTH_OMEGA * config.EARTH_RADIUS_M * cos_lat
            east_rel = v_orb * sin_az - v_eqrot
            north_rel = v_orb * cos_az
            az = math.atan2(east_rel, north_rel)
            sin_az, cos_az = math.sin(az), math.cos(az)

        sin_lat = math.sin(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        north_ecef = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        east_ecef = np.array([-sin_lon, cos_lon, 0.0])

        downrange_ecef = north_ecef * cos_az + east_ecef * sin_az
        return downrange_ecef / norm3(downrange_ecef)

    @staticmethod
    def _local_up(state: VehicleState) -> np.ndarray:
        """Unit vector from Earth center through vehicle."""
        r = norm3(state.position_eci)
        if r < 1.0:
            return np.array([0.0, 0.0, 1.0])
        return state.position_eci / r

    @staticmethod
    def _default_downrange(up: np.ndarray) -> np.ndarray:
        """Compute a default downrange direction perpendicular to up."""
        # arb = [1,0,0] - dot(arb, up) * up; dot([1,0,0], up) = up[0]
        arb = np.array([1.0 - up[0] * up[0], -up[0] * up[1], -up[0] * up[2]])
        mag = norm3(arb)
        if mag > 1e-6:
            return arb / mag
        return np.array([0.0, 1.0, 0.0])

    @staticmethod
    def _quaternion_aligning_thrust(desired_dir_eci: np.ndarray) -> np.ndarray:
        """Compute quaternion that rotates body +X to desired_dir_eci."""
        # body_x = [1,0,0], so dot(body_x, d) = d[0] and body_x × d = [0, -d[2], d[1]].
        d_mag = norm3(desired_dir_eci)
        inv = 1.0 / d_mag if d_mag > 1e-10 else 1e10
        d0 = desired_dir_eci[0] * inv
        d1 = desired_dir_eci[1] * inv
        d2 = desired_dir_eci[2] * inv
        if d0 > 0.99999:
            return np.array([0.0, 0.0, 0.0, 1.0])
        if d0 < -0.99999:
            return np.array([0.0, 0.0, 1.0, 0.0])
        ax1 = -d2
        ax2 = d1
        axis_mag = math.sqrt(ax1 * ax1 + ax2 * ax2)
        axis = np.array([0.0, ax1 / axis_mag, ax2 / axis_mag])
        if d0 > 1.0:
            d0 = 1.0
        elif d0 < -1.0:
            d0 = -1.0
        angle = math.acos(d0)
        return quaternion_from_axis_angle(axis, angle)
