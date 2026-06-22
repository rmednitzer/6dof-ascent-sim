"""Main simulation entry point — runs full ascent trajectory."""

from __future__ import annotations

import argparse
import logging
import math
import sys

import numpy as np

from sim import config
from sim.core.fast_math import cross3, dot3, norm3
from sim.core.integrator import StateDot, rk4_step
from sim.core.reference_frames import (
    body_to_eci,
    ecef_to_eci,
    ecef_to_lla,
    eci_to_ecef,
    lla_to_ecef,
    quaternion_derivative,
    quaternion_from_axis_angle,
)
from sim.core.state import VehicleState
from sim.dynamics.flex_body import FlexBody
from sim.dynamics.slosh import SloshModel
from sim.environment.atmosphere import atmosphere
from sim.environment.gravity import gravitational_acceleration
from sim.environment.wind import wind_velocity_eci
from sim.gnc.control import AttitudeController
from sim.gnc.guidance import GuidanceLaw
from sim.gnc.navigation import NavigationEKF
from sim.gnc.notch_filter import StructuralNotchFilter
from sim.gnc.sensors import SensorSuite
from sim.safety.boundary_enforcer import BoundaryEnforcer
from sim.safety.fts import FlightTerminationSystem
from sim.safety.health_monitor import HealthMonitor
from sim.telemetry.recorder import TelemetryRecorder
from sim.telemetry.schemas import MissionSummary
from sim.vehicle.actuator import TVCActuatorPair
from sim.vehicle.aerodynamics import AerodynamicsModel
from sim.vehicle.propulsion import EngineModel, thrust_at_pressure
from sim.vehicle.staging import StagingSequencer
from sim.vehicle.vehicle import Vehicle


def _init_state() -> VehicleState:
    """Initialize vehicle state on the launch pad."""
    lat_rad = math.radians(config.LAUNCH_LAT_DEG)
    lon_rad = math.radians(config.LAUNCH_LON_DEG)

    pos_ecef = lla_to_ecef(lat_rad, lon_rad, config.LAUNCH_ALT_M)
    pos_eci = ecef_to_eci(pos_ecef, 0.0)

    # Initial velocity = Earth rotation at launch site
    omega = np.array([0.0, 0.0, config.EARTH_OMEGA])
    vel_eci = cross3(omega, pos_eci)

    # Initial quaternion: vehicle thrust axis (+X body) pointing radially outward
    up_eci = pos_eci / norm3(pos_eci)
    body_x = np.array([1.0, 0.0, 0.0])
    dot = dot3(body_x, up_eci)
    if abs(dot - 1.0) < 1e-10:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    elif abs(dot + 1.0) < 1e-10:
        quat = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        axis = cross3(body_x, up_eci)
        axis /= norm3(axis)
        angle = math.acos(np.clip(dot, -1.0, 1.0))
        quat = quaternion_from_axis_angle(axis, angle)

    total_mass = config.S1_DRY_MASS_KG + config.S1_PROPELLANT_KG + config.S2_DRY_MASS_KG + config.S2_PROPELLANT_KG

    return VehicleState(
        position_eci=pos_eci,
        velocity_eci=vel_eci,
        quaternion=quat,
        angular_velocity_body=np.zeros(3),
        mass_kg=total_mass,
        time_s=0.0,
    )


def _is_orbital_insertion(state: VehicleState, stage_index: int):
    """Return ``(True, elements)`` iff *state* is a real LEO insertion.

    A genuine insertion requires the upper stage to be active
    (``stage_index >= 1``, i.e. stage 2), near-target altitude/velocity,
    a small flight-path angle, and — decisively — an osculating orbit
    that is bound, clears the sensible atmosphere, and is near-circular.
    The final orbit test is what prevents a steep sub-orbital arc
    (negative periapsis) from being misreported as SUCCESS. Keeping the
    stage gate inside this single predicate ensures the in-loop and
    end-of-sim paths apply an identical criterion.

    Returns ``(False, None)`` otherwise.
    """
    from sim.orbital.propagator import OrbitPropagator

    if stage_index < 1:
        return False, None

    alt_m = state.altitude_m()
    vel = state.velocity_mag_ms()
    if alt_m < config.INSERTION_MIN_ALTITUDE_FRAC * config.TARGET_ALTITUDE_M:
        return False, None
    if vel < config.INSERTION_MIN_VELOCITY_FRAC * config.TARGET_VELOCITY_MS:
        return False, None
    r = norm3(state.position_eci)
    if r < 1.0 or vel < 1.0:
        return False, None
    r_hat = state.position_eci / r
    v_hat = state.velocity_eci / vel
    fpa_deg = abs(math.degrees(math.asin(np.clip(dot3(r_hat, v_hat), -1.0, 1.0))))
    if fpa_deg > config.INSERTION_MAX_FPA_DEG:
        return False, None

    elements = OrbitPropagator(state).state_to_elements()
    if not elements.is_sustainable_leo(
        config.INSERTION_MIN_PERIAPSIS_ALT_M,
        config.INSERTION_MAX_ECCENTRICITY,
    ):
        return False, None
    return True, elements


def run_simulation(
    config_override: dict | None = None,
    quiet: bool = False,
    write_output: bool | None = None,
):
    """Run the complete ascent simulation.

    Args:
        config_override: Per-run parameter overrides (Monte Carlo). Applied via a
            context-local config (``config.override_context``) — no global
            mutation — so concurrent runs cannot interfere.
        quiet: Suppress progress output.
        write_output: Whether to run post-insertion orbit analysis and write
            telemetry files / plots. Defaults to ``True`` for a nominal run
            (``config_override is None``) and ``False`` for a Monte-Carlo run.

    Returns:
        MonteCarloResult with trajectory metrics.
    """
    dispersed_params = config_override or {}
    if write_output is None:
        write_output = config_override is None
    run_index = int(dispersed_params.get("_run_index", 0))

    with config.override_context(config_override):
        return _run_inner(quiet, write_output, run_index, dispersed_params)


def _run_inner(quiet: bool, write_output: bool, run_index: int, dispersed_params: dict):
    """Core simulation loop."""
    from sim.montecarlo.dispatcher import MonteCarloResult

    flex_enabled = config.FLEX_ENABLED
    slosh_enabled = config.SLOSH_ENABLED
    rng = np.random.default_rng(dispersed_params.get("_seed", 42))

    # --- Initialize ---
    true_state = _init_state()
    vehicle = Vehicle()
    s1_engine = EngineModel(vehicle.current_stage)
    s2_engine = EngineModel(vehicle.stages[1])
    s1_engine.ignite()  # Ignition at T-0

    staging = StagingSequencer(vehicle, s1_engine, s2_engine)
    aero = AerodynamicsModel()
    aero.reset()
    guidance = GuidanceLaw()
    sensors = SensorSuite(rng=rng)
    ekf = NavigationEKF(true_state)
    controller = AttitudeController()
    enforcer = BoundaryEnforcer()
    fts = FlightTerminationSystem(enforcer)
    health_monitor = HealthMonitor()
    recorder = TelemetryRecorder()
    tvc_actuators = TVCActuatorPair()

    flex_body = FlexBody() if flex_enabled else None
    # Structural notch filter on the control-loop rate feedback (AD-04); only
    # active when the flex model is live. Scheduled on the modal frequencies.
    flex_notch = (
        StructuralNotchFilter(flex_body.n_modes, config.INTERNAL_HZ, config.FLEX_NOTCH_Q)
        if (flex_body is not None and config.FLEX_NOTCH_ENABLED)
        else None
    )
    # Notch-filtered measured pitch rate fed to the controller (one-step lag,
    # i.e. the sense -> compute -> actuate latency of a real flight computer).
    flex_control_pitch_rate = float(true_state.angular_velocity_body[1])
    slosh_model = SloshModel() if slosh_enabled else None

    # Nominal trajectory plane for FTS cross-range check
    pos_ecef_init = eci_to_ecef(true_state.position_eci, 0.0)
    lat_rad = math.radians(config.LAUNCH_LAT_DEG)
    lon_rad = math.radians(config.LAUNCH_LON_DEG)

    # Compute launch azimuth for target inclination:
    # sin(az) = cos(inc) / cos(lat)
    cos_inc = math.cos(math.radians(config.TARGET_INCLINATION_DEG))
    cos_lat = math.cos(lat_rad)
    sin_az = min(1.0, cos_inc / max(cos_lat, 1e-10))
    launch_azimuth_rad = math.asin(sin_az)

    # North and East unit vectors in ECEF at launch site
    sin_lat = math.sin(lat_rad)
    north_ecef = np.array(
        [
            -sin_lat * math.cos(lon_rad),
            -sin_lat * math.sin(lon_rad),
            cos_lat,
        ]
    )
    east_ecef = np.array([-math.sin(lon_rad), math.cos(lon_rad), 0.0])

    # Downrange direction in ECEF (along launch azimuth)
    downrange_ecef = north_ecef * math.cos(launch_azimuth_rad) + east_ecef * math.sin(launch_azimuth_rad)
    # Up direction in ECEF
    up_ecef = pos_ecef_init / norm3(pos_ecef_init)

    # The nominal trajectory plane contains up and downrange.
    # The plane normal is perpendicular to both (cross-range direction).
    nominal_plane_normal = cross3(downrange_ecef, up_ecef)
    nominal_plane_normal /= norm3(nominal_plane_normal)
    nominal_plane_point = pos_ecef_init.copy()

    # Tracking variables
    outcome = "TIMEOUT"
    fts_trigger_time = None
    peak_q = 0.0
    peak_axial_g = 0.0
    peak_ekf_uncertainty = 0.0
    peak_attitude_est_error_deg = 0.0
    last_print_time = -10.0

    dt = config.DT
    num_steps = int(config.T_MAX / dt)
    current_engine = s1_engine
    density_scale = config.ATMO_DENSITY_SCALE

    # Non-gravitational (specific-force) acceleration in ECI from the
    # previous step, fed to the IMU this step. A strapdown accelerometer
    # senses (thrust + aero + ...) / m, never gravity. The one-step
    # (~10 ms) lag is realistic IMU latency and well within EKF tolerance.
    prev_specific_force_eci = np.zeros(3)

    for step_i in range(num_steps):
        t = step_i * dt
        true_state.time_s = t

        # --- Environment ---
        pos_ecef = eci_to_ecef(true_state.position_eci, t)
        lat, lon, alt_geo = ecef_to_lla(pos_ecef)
        alt_m = max(0.0, alt_geo)
        atmo = atmosphere(alt_m, density_scale=density_scale)
        rho = atmo.density_kg_m3
        pressure = atmo.pressure_pa
        speed_of_sound = atmo.speed_of_sound_ms
        grav_eci = gravitational_acceleration(true_state.position_eci)
        wind_eci = wind_velocity_eci(true_state.position_eci, t, rng=rng)

        # --- Aerodynamics (wind-relative velocity and Mach) ---
        vel_rel = true_state.velocity_eci - wind_eci
        vel_rel_mag = norm3(vel_rel)
        mach = vel_rel_mag / max(speed_of_sound, 1.0)
        q_pa = 0.5 * rho * vel_rel_mag * vel_rel_mag
        peak_q = max(peak_q, q_pa)

        # --- Guidance ---
        estimated_state = ekf.estimated_state()
        estimated_state.time_s = t
        # Attitude authority (ADR 0020): with USE_ESTIMATED_ATTITUDE off, guidance
        # and the controller run on the true attitude (the error-state EKF still
        # estimates attitude in parallel, for validation/telemetry); with it on,
        # they consume the EKF's estimated attitude.
        if not config.USE_ESTIMATED_ATTITUDE:
            estimated_state.quaternion = true_state.quaternion.copy()
        guidance_cmd = guidance.update(estimated_state)

        # --- Engine update (get thrust/mdot for this step) ---
        if staging.is_complete:
            current_engine = s2_engine
        else:
            current_engine = s1_engine

        # Throttle from guidance, with max-q and G-limit management
        throttle_raw = guidance_cmd.throttle

        # Predict full-throttle thrust at current ambient pressure
        full_thrust = thrust_at_pressure(current_engine._stage, pressure)

        # Max-Q throttle management: reduce throttle to keep q below structural limit
        if q_pa > 0.70 * config.MAX_Q_PA:
            q_margin = (config.MAX_Q_PA - q_pa) / (0.30 * config.MAX_Q_PA)
            q_throttle = max(config.S1_THROTTLE_MIN, min(1.0, q_margin))
            throttle_raw = min(throttle_raw, q_throttle)

        # G-limit throttle management: reduce throttle to stay under max axial G
        predicted_axial_g = full_thrust / max(true_state.mass_kg, 1.0) / config.G0
        if predicted_axial_g > 0.90 * config.MAX_AXIAL_G:
            g_throttle = (0.90 * config.MAX_AXIAL_G * config.G0 * true_state.mass_kg) / max(full_thrust, 1.0)
            throttle_raw = min(throttle_raw, max(config.S1_THROTTLE_MIN, g_throttle))

        # --- Boundary enforcement: throttle ---
        throttle_result = enforcer.validate_throttle(throttle_raw, vehicle.propellant_remaining())
        approved_throttle = throttle_result.value

        # Set throttle on engine
        current_engine.set_throttle(approved_throttle)
        thrust_n, mdot = current_engine.update(dt, pressure)
        # The engine model does not track propellant; the vehicle ledger
        # does. If this step would burn more than remains, the engine can
        # only fire for the fraction of the step the propellant lasts.
        # Scale BOTH thrust and mass flow by that fraction so specific
        # impulse stays physical — applying full thrust while capping only
        # the mass burned would be a spurious delta-v boost on the
        # depletion tick (and could shift stage-end insertion outcomes).
        propellant_avail = vehicle.propellant_remaining()
        if mdot > 0.0:
            if propellant_avail <= 0.0:
                thrust_n = 0.0
                mdot = 0.0
            else:
                step_demand = mdot * dt
                if step_demand > propellant_avail:
                    burn_frac = propellant_avail / step_demand
                    thrust_n *= burn_frac
                    mdot *= burn_frac

        # --- Control (gain-scheduled) ---
        # When the flex model is live, the controller's rate feedback is the
        # measured body rate (rigid + structural bending sensed at the IMU)
        # passed through the structural notch filter, rather than the clean true
        # rate. This is the control-structure interaction the notch tames; a
        # naive coupling without it flutters and FTS-aborts (AD-04).
        if flex_body is not None:
            omega_for_control = np.array(
                [
                    true_state.angular_velocity_body[0],
                    flex_control_pitch_rate,
                    true_state.angular_velocity_body[2],
                ]
            )
        else:
            omega_for_control = true_state.angular_velocity_body
        tvc_cmd = controller.update(
            guidance_cmd.desired_quaternion,
            estimated_state.quaternion,
            omega_for_control,
            dt,
            dynamic_pressure_pa=q_pa,
            mass_kg=true_state.mass_kg,
        )

        # --- Boundary enforcement: TVC ---
        tvc_result = enforcer.validate_tvc(tvc_cmd.pitch_deg, tvc_cmd.yaw_deg, dt)
        enforced_pitch, enforced_yaw = tvc_result.value

        # --- TVC actuator dynamics (second-order response) ---
        approved_tvc_pitch, approved_tvc_yaw = tvc_actuators.update(enforced_pitch, enforced_yaw, dt)

        # --- Compute accelerations for structural check ---
        axial_g = thrust_n / max(true_state.mass_kg, 1.0) / config.G0
        sin_tvc_pitch = math.sin(math.radians(approved_tvc_pitch))
        sin_tvc_yaw = math.sin(math.radians(approved_tvc_yaw))
        tvc_lateral_force = thrust_n * (abs(sin_tvc_pitch) + abs(sin_tvc_yaw))
        lateral_g = tvc_lateral_force / max(true_state.mass_kg, 1.0) / config.G0
        peak_axial_g = max(peak_axial_g, axial_g)

        enforcer.check_structural_limits(axial_g, lateral_g, q_pa)

        # --- Sensor measurements ---
        # IMU senses specific force (non-gravitational accel), not gravity.
        imu_meas, gps_meas, baro_meas, star_meas, ground_meas = sensors.update(true_state, prev_specific_force_eci, dt)

        # --- Navigation (EKF) ---
        # The error-state EKF estimates attitude itself, propagating the nominal
        # quaternion from the measured gyro (ADR 0020) — it is no longer handed
        # the true attitude. It still consumes the raw IMU (no flex
        # contamination: coning/sculling cross-products would amplify structural
        # vibration that is not true rigid-body rotation).
        ekf.set_mass(true_state.mass_kg)
        ekf.predict(imu_meas, grav_eci, dt)

        # NOTE (AD-04): the flex model couples into the loop through the *control*
        # rate feedback (see the flex-body update below), tamed by a frequency-
        # scheduled structural notch. The EKF deliberately stays on the clean
        # IMU: its coning/sculling cross-products would amplify structural
        # vibration that is not true rigid-body rotation.
        if gps_meas is not None:
            ekf.update_gps(gps_meas)
        if baro_meas is not None:
            ekf.update_baro(baro_meas)
        # Star tracker keeps attitude observable above the COCOM GPS ceiling
        # (ADR 0020): it bounds the attitude error through the GPS-denied upper
        # stage, so the attitude→velocity→position covariance coupling does not
        # trip the FTS limit.
        if star_meas is not None:
            ekf.update_star_tracker(star_meas)
        # Ground-station ranging is independent of GPS (the vehicle is tracked, not
        # self-locating), so it keeps bounding EKF *position* covariance through the
        # COCOM GPS-denied coast — where it would otherwise grow unbounded and, for
        # the most-lofted dispersions, trip the FTS covariance limit (ADR 0023).
        for ground_range in ground_meas:
            ekf.update_ground_range(ground_range)

        ekf_uncertainty = ekf.position_uncertainty_m()
        peak_ekf_uncertainty = max(peak_ekf_uncertainty, ekf_uncertainty)

        # Attitude-estimation error (estimated vs true), for validation/telemetry.
        dot_q = abs(float(np.dot(ekf.quaternion, true_state.quaternion)))
        attitude_est_error_deg = math.degrees(2.0 * math.acos(min(1.0, dot_q)))
        peak_attitude_est_error_deg = max(peak_attitude_est_error_deg, attitude_est_error_deg)

        # --- FTS check ---
        fts_triggered = fts.evaluate(
            position_ecef=pos_ecef,
            nominal_plane_normal=nominal_plane_normal,
            nominal_plane_point=nominal_plane_point,
            q_actual=true_state.quaternion,
            q_desired=guidance_cmd.desired_quaternion,
            ekf_pos_covariance=ekf.covariance[0:3, 0:3],
            axial_g=axial_g,
            lateral_g=lateral_g,
            dynamic_pressure_pa=q_pa,
            sim_time=t,
            altitude_m=alt_geo,
        )
        if fts_triggered:
            outcome = "FTS_ABORT"
            fts_trigger_time = t
            if not quiet:
                print(f"  FTS TRIGGERED at t={t:.2f}s: {fts.state.reason}")
            break

        # --- Staging ---
        staging_event = staging.update(dt)
        if staging_event:
            if not quiet:
                print(f"  [{t:.1f}s] {staging_event}")
            # Reset controller integrators on staging transitions
            controller.reset()

        # --- Health monitoring ---
        health_monitor.update(
            ekf_pos_covariance=ekf.covariance[0:3, 0:3],
            dynamic_pressure_pa=q_pa,
            propellant_remaining_kg=vehicle.propellant_remaining(),
            propellant_initial_kg=vehicle.current_stage.propellant,
        )

        # --- Flex body (live control-structure interaction, AD-04) ---
        if flex_body is not None:
            # Excite the bending modes with this step's TVC lateral force, using
            # a realistic generalised modal mass (the default 1.0 kg would make
            # the modal response physically enormous).
            tvc_force = thrust_n * sin_tvc_pitch
            flex_body.update(
                dt,
                tvc_force,
                vehicle.propellant_fraction(),
                modal_mass_kg=config.FLEX_MODAL_MASS_KG,
            )
            # The IMU gyro senses rigid rate + bending rate. Feed that measured
            # pitch rate to the controller next step, after notching out the
            # modal content so the loop does not chase the structure.
            flex_rate = flex_body.total_bending_rate_at_imu()
            measured_pitch_rate = float(true_state.angular_velocity_body[1]) + flex_rate
            if flex_notch is not None:
                mode_freqs = flex_body.modal_frequencies_hz(vehicle.propellant_fraction())
                flex_control_pitch_rate = flex_notch.process(measured_pitch_rate, mode_freqs)
            else:
                flex_control_pitch_rate = measured_pitch_rate

        # --- Slosh ---
        slosh_force_body = np.zeros(3)
        slosh_torque_body = np.zeros(3)
        if slosh_model is not None:
            # Lateral acceleration at tank
            lat_accel = tvc_lateral_force / max(true_state.mass_kg, 1.0)
            forces, torques = slosh_model.update(
                dt,
                lat_accel,
                vehicle.propellant_remaining(),
                vehicle.propellant_fraction(),
            )
            # Sum across tanks, place in body Y axis
            slosh_force_body[1] = float(np.sum(forces))
            slosh_torque_body[2] = float(np.sum(torques))

        # --- Aerodynamic forces and moments (full 6-DOF) ---
        # Compute center of mass offset from nose for aero moment arm
        # CoM moves aft as propellant burns: interpolate from ~40% to ~25% of length
        prop_frac = vehicle.propellant_fraction()
        com_from_nose = config.VEHICLE_LENGTH_M * (0.25 + 0.20 * prop_frac)

        aero_result = aero.compute_aero_forces(
            vel_rel_eci=vel_rel,
            quaternion=true_state.quaternion,
            omega_body=true_state.angular_velocity_body,
            rho=rho,
            speed_of_sound=speed_of_sound,
            com_offset_from_nose=com_from_nose,
        )

        # --- Physics: compute forces and integrate ---
        # Thrust vector in body frame with TVC deflection
        pitch_rad = math.radians(approved_tvc_pitch)
        yaw_rad = math.radians(approved_tvc_yaw)
        cos_pitch_rad = math.cos(pitch_rad)
        cos_yaw_rad = math.cos(yaw_rad)
        thrust_body = np.array(
            [
                thrust_n * cos_pitch_rad * cos_yaw_rad,
                thrust_n * sin_tvc_yaw,
                thrust_n * sin_tvc_pitch,
            ]
        )
        thrust_eci = body_to_eci(thrust_body, true_state.quaternion)
        slosh_force_eci = body_to_eci(slosh_force_body, true_state.quaternion)

        # Normal aerodynamic force (body frame -> ECI)
        aero_normal_eci = body_to_eci(aero_result.normal_force_body, true_state.quaternion)

        # TVC torques
        moment_arm = config.VEHICLE_LENGTH_M * 0.45
        tvc_torque = np.array(
            [
                0.0,
                moment_arm * thrust_n * sin_tvc_pitch,
                -moment_arm * thrust_n * sin_tvc_yaw,
            ]
        )
        # Total torque: TVC + slosh + aerodynamic moments (normal force + pitch damping)
        total_torque = tvc_torque + slosh_torque_body + aero_result.aero_moment_body

        # Moment of inertia (composite body model)
        # Better than pure cylinder: accounts for propellant distribution
        radius = math.sqrt(config.REFERENCE_AREA_M2 / math.pi)
        length = config.VEHICLE_LENGTH_M
        # Dry structure (thin-walled cylinder)
        dry_mass = vehicle.current_stage.dry_mass
        if vehicle.stage_index == 0:
            dry_mass += config.S2_DRY_MASS_KG + vehicle._propellant_remaining[1]
        prop_mass = vehicle.propellant_remaining()
        # Parallel axis theorem: propellant concentrated in tanks offset from CG
        I_dry = dry_mass * (radius**2 / 4.0 + length**2 / 12.0)
        # Propellant as distributed mass (shorter effective length for tank section)
        tank_length = length * 0.5
        I_prop = prop_mass * (radius**2 / 4.0 + tank_length**2 / 12.0)
        inertia = max(100.0, I_dry + I_prop)

        # Mass flow — single source of truth. Never integrate away more
        # propellant than the vehicle actually has, so the RK4 state mass
        # and the vehicle propellant ledger cannot diverge (which would
        # let mass_kg fall below dry mass and inflate acceleration).
        propellant_avail = vehicle.propellant_remaining()
        mass_consumed = min(mdot * dt, propellant_avail) if mdot > 0.0 else 0.0
        actual_mdot = -mass_consumed / dt if dt > 0.0 else 0.0

        # Non-gravitational specific force this step (for next step's IMU).
        total_force_eci = thrust_eci + aero_result.drag_force_eci + aero_normal_eci + slosh_force_eci
        prev_specific_force_eci = total_force_eci / max(true_state.mass_kg, 1.0)

        # Create derivatives closure. The total translational force and the
        # angular acceleration are zero-order-held across the four RK4 sub-stages
        # (ADR 0003), so precompute them ONCE instead of re-summing the four
        # force vectors and re-dividing torque/inertia on every sub-stage call.
        # Reuses total_force_eci already summed above for the IMU; bit-identical
        # to the previous per-call computation (same operands, same order).
        _grav_eci = grav_eci
        _total_force_eci = total_force_eci
        _angular_accel = total_torque / inertia
        _mass_rate = actual_mdot

        def derivatives_fn(  # noqa: B023
            t_eval: float,
            s: VehicleState,
        ) -> StateDot:
            accel = _grav_eci + _total_force_eci / max(s.mass_kg, 1.0)
            quat_dot = quaternion_derivative(s.quaternion, s.angular_velocity_body)
            return StateDot(
                velocity_eci=s.velocity_eci,
                acceleration_eci=accel,
                quaternion_dot=quat_dot,
                angular_acceleration_body=_angular_accel,
                mass_rate_kg_s=_mass_rate,
            )

        true_state = rk4_step(true_state, derivatives_fn, dt)

        # Update vehicle mass tracking (same amount removed from RK4 state)
        if mass_consumed > 0.0:
            vehicle.consume_propellant(mass_consumed)

        # --- Telemetry ---
        sim_context = {
            "throttle": approved_throttle,
            "thrust_n": thrust_n,
            "dynamic_pressure_pa": q_pa,
            "mach_number": mach,
            "axial_g": axial_g,
            "lateral_g": lateral_g,
            "stage": vehicle.stage_index + 1,
            "ekf_position_uncertainty_m": ekf_uncertainty,
        }
        recorder.record(
            true_state=true_state,
            estimated_state=estimated_state,
            health_monitor=health_monitor,
            boundary_enforcer=enforcer,
            time_s=t,
            sim_context=sim_context,
            fts=fts,
        )

        # --- Progress ---
        if not quiet and t - last_print_time >= 10.0:
            last_print_time = t
            print(
                f"  t={t:6.1f}s | alt={true_state.altitude_m() / 1000:7.1f} km | "
                f"v={true_state.velocity_mag_ms():7.1f} m/s | "
                f"m={true_state.mass_kg:8.0f} kg | "
                f"q={q_pa / 1000:5.1f} kPa | stg={vehicle.stage_index + 1}"
            )

        # --- Insertion check: SUCCESS only for a genuine, sustainable orbit ---
        inserted, _ = _is_orbital_insertion(true_state, vehicle.stage_index)
        if inserted:
            outcome = "SUCCESS"
            if not quiet:
                print(f"  ORBITAL INSERTION at t={t:.1f}s!")
            break

    # End-of-sim check — apply the identical orbit-validity test (same
    # predicate, including the stage gate). A run that times out is only
    # SUCCESS if it actually reached orbit.
    if outcome == "TIMEOUT":
        inserted, _ = _is_orbital_insertion(true_state, vehicle.stage_index)
        if inserted:
            outcome = "SUCCESS"

    # Compute flight path angle
    final_fpa = 0.0
    r_norm = norm3(true_state.position_eci)
    if r_norm > 0 and true_state.velocity_mag_ms() > 0:
        r_hat = true_state.position_eci / r_norm
        v_hat = true_state.velocity_eci / true_state.velocity_mag_ms()
        final_fpa = math.degrees(math.asin(np.clip(dot3(r_hat, v_hat), -1.0, 1.0)))

    # --- Post-insertion orbit analysis ---
    orbit_elements_dict = None
    if outcome == "SUCCESS" and write_output:
        try:
            from sim.orbital.maneuvers import total_correction_budget
            from sim.orbital.propagator import OrbitPropagator

            propagator = OrbitPropagator(true_state)
            elements = propagator.state_to_elements()
            corr_dv = total_correction_budget(
                elements,
                config.TARGET_ALTITUDE_M,
                config.TARGET_INCLINATION_DEG,
            )
            orbit_elements_dict = {
                "semi_major_axis_km": elements.semi_major_axis_m / 1000,
                "eccentricity": elements.eccentricity,
                "inclination_deg": elements.inclination_deg,
                "apoapsis_alt_km": elements.apoapsis_alt_km,
                "periapsis_alt_km": elements.periapsis_alt_km,
                "period_min": elements.period_s / 60,
                "correction_dv_ms": corr_dv,
            }
        except Exception:
            if not quiet:
                logging.exception("Orbit analysis error")

    # --- Write telemetry / plots (output runs only) ---
    if write_output:
        summary = recorder.write_output(
            outcome=outcome,
            true_state=true_state,
            health_monitor=health_monitor,
            boundary_enforcer=enforcer,
            fts=fts,
        )
        if not quiet:
            _print_summary(summary, orbit_elements_dict)
            print(f"Peak attitude est error (EKF vs truth): {peak_attitude_est_error_deg:.3f} deg")
            # Generate plots
            try:
                from sim.analysis.postflight import generate_plots

                generate_plots(recorder.internal_frames, summary)
            except Exception:
                logging.exception("Plot generation error")

    # Return MonteCarloResult

    return MonteCarloResult(
        run_index=run_index,
        seed=int(dispersed_params.get("_seed", 0)),
        outcome=outcome,
        dispersed_params={k: v for k, v in dispersed_params.items() if not k.startswith("_")},
        insertion_altitude_m=true_state.altitude_m() if outcome == "SUCCESS" else None,
        insertion_velocity_ms=true_state.velocity_mag_ms() if outcome == "SUCCESS" else None,
        insertion_fpa_deg=final_fpa if outcome == "SUCCESS" else None,
        peak_q_pa=peak_q,
        peak_axial_g=peak_axial_g,
        peak_ekf_uncertainty_m=peak_ekf_uncertainty,
        boundary_clamp_count=enforcer.violation_count,
        fts_trigger_time_s=fts_trigger_time,
        total_time_s=true_state.time_s,
    )


def _print_summary(summary: MissionSummary, orbit: dict | None) -> None:
    """Print mission summary to stdout."""
    print()
    print("Mission Summary")
    print("=" * 40)
    print(f"Outcome: {summary.outcome}")
    print(f"Final altitude: {summary.final_altitude_m / 1000:.1f} km")
    print(f"Final velocity: {summary.final_velocity_ms:.1f} m/s")
    print(
        f"Peak dynamic pressure: {summary.peak_dynamic_pressure_pa:.0f} Pa "
        f"({summary.peak_dynamic_pressure_pa / config.MAX_Q_PA * 100:.1f}% of limit)"
    )
    print(
        f"Peak axial G: {summary.peak_axial_g:.2f} g ({summary.peak_axial_g / config.MAX_AXIAL_G * 100:.1f}% of limit)"
    )
    print(f"Boundary violations: {summary.total_boundary_violations}")
    print(f"FTS triggered: {summary.fts_triggered}")
    print(f"Total sim time: {summary.final_time_s:.1f} s")
    print(f"Telemetry hash (SHA-256): {summary.telemetry_hash_sha256}")

    if orbit:
        print()
        print("Orbit Characterization:")
        print(f"  Semi-major axis: {orbit['semi_major_axis_km']:.1f} km")
        print(f"  Eccentricity: {orbit['eccentricity']:.4f}")
        print(f"  Inclination: {orbit['inclination_deg']:.2f} deg")
        print(f"  Apoapsis: {orbit['apoapsis_alt_km']:.1f} km")
        print(f"  Periapsis: {orbit['periapsis_alt_km']:.1f} km")
        print(f"  Period: {orbit['period_min']:.1f} min")
        print(f"  Correction dv: {orbit['correction_dv_ms']:.1f} m/s")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="6-DOF Launch Vehicle Ascent Simulation")
    parser.add_argument("--no-flex", action="store_true", help="Disable flex body model")
    parser.add_argument("--no-slosh", action="store_true", help="Disable propellant slosh model")
    args = parser.parse_args()

    overrides: dict = {}
    if args.no_flex:
        overrides["FLEX_ENABLED"] = False
    if args.no_slosh:
        overrides["SLOSH_ENABLED"] = False

    print("6-DOF Ascent Simulation")
    print("=" * 40)
    print(f"Target: {config.TARGET_ALTITUDE_M / 1000:.0f} km, {config.TARGET_INCLINATION_DEG} deg inc")
    print(f"Flex body: {'OFF' if args.no_flex else 'ON'}")
    print(f"Slosh: {'OFF' if args.no_slosh else 'ON'}")
    print()

    # Overrides go through the context-local config; keep the detailed output path.
    result = run_simulation(config_override=overrides or None, write_output=True)

    if result is not None:
        if result.outcome == "SUCCESS":
            sys.exit(0)
        elif result.outcome == "FTS_ABORT":
            sys.exit(1)
        else:
            sys.exit(2)


if __name__ == "__main__":
    main()
