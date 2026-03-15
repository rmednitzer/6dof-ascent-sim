"""Main simulation entry point — runs full ascent trajectory."""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from sim import config
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
from sim.gnc.sensors import SensorSuite
from sim.safety.boundary_enforcer import BoundaryEnforcer
from sim.safety.fts import FlightTerminationSystem
from sim.safety.health_monitor import HealthMonitor
from sim.telemetry.recorder import TelemetryRecorder
from sim.telemetry.schemas import MissionSummary
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
    vel_eci = np.cross(omega, pos_eci)

    # Initial quaternion: vehicle thrust axis (+X body) pointing radially outward
    up_eci = pos_eci / np.linalg.norm(pos_eci)
    body_x = np.array([1.0, 0.0, 0.0])
    dot = np.dot(body_x, up_eci)
    if abs(dot - 1.0) < 1e-10:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    elif abs(dot + 1.0) < 1e-10:
        quat = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        axis = np.cross(body_x, up_eci)
        axis /= np.linalg.norm(axis)
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


def _save_config() -> dict:
    """Save config values that might be overridden."""
    keys = [
        "S1_THRUST_VAC_N",
        "S1_ISP_VAC_S",
        "S2_THRUST_VAC_N",
        "S2_ISP_VAC_S",
        "S1_PROPELLANT_KG",
        "CD_SCALE_FACTOR",
        "ATMO_DENSITY_SCALE",
        "WIND_SPEED_MS",
        "WIND_DIRECTION_DEG",
        "IMU_ACCEL_BIAS_MPS2",
        "IMU_GYRO_BIAS_RADS",
        "GPS_POS_NOISE_M",
        "S1_DRY_MASS_KG",
        "FLEX_ENABLED",
        "SLOSH_ENABLED",
    ]
    return {k: getattr(config, k) for k in keys if hasattr(config, k)}


def _restore_config(saved: dict) -> None:
    """Restore config values."""
    for key, value in saved.items():
        if hasattr(config, key):
            setattr(config, key, value)


def _apply_overrides(overrides: dict | None) -> dict:
    """Apply config overrides, return the overrides dict."""
    if overrides is None:
        return {}
    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    return overrides


def run_simulation(
    config_override: dict | None = None,
    quiet: bool = False,
):
    """Run the complete ascent simulation.

    Args:
        config_override: Config parameter overrides for Monte Carlo.
        quiet: Suppress progress output.

    Returns:
        MonteCarloResult with trajectory metrics.
    """

    saved_config = _save_config()
    dispersed_params = _apply_overrides(config_override)
    is_mc = config_override is not None
    run_index = int(dispersed_params.get("_run_index", 0)) if is_mc else 0

    try:
        return _run_inner(quiet, is_mc, run_index, dispersed_params)
    finally:
        if is_mc:
            _restore_config(saved_config)


def _run_inner(quiet: bool, is_mc: bool, run_index: int, dispersed_params: dict):
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
    guidance = GuidanceLaw()
    sensors = SensorSuite(rng=rng)
    ekf = NavigationEKF(true_state)
    controller = AttitudeController()
    enforcer = BoundaryEnforcer()
    fts = FlightTerminationSystem(enforcer)
    health_monitor = HealthMonitor()
    recorder = TelemetryRecorder()

    flex_body = FlexBody() if flex_enabled else None
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
    up_ecef = pos_ecef_init / np.linalg.norm(pos_ecef_init)

    # The nominal trajectory plane contains up and downrange.
    # The plane normal is perpendicular to both (cross-range direction).
    nominal_plane_normal = np.cross(downrange_ecef, up_ecef)
    nominal_plane_normal /= np.linalg.norm(nominal_plane_normal)
    nominal_plane_point = pos_ecef_init.copy()

    # Tracking variables
    outcome = "TIMEOUT"
    fts_trigger_time = None
    peak_q = 0.0
    peak_axial_g = 0.0
    peak_ekf_uncertainty = 0.0
    last_print_time = -10.0
    _sensor_degradation_count = 0  # noqa: F841

    dt = config.DT
    num_steps = int(config.T_MAX / dt)
    current_engine = s1_engine

    for step_i in range(num_steps):
        t = step_i * dt
        true_state.time_s = t

        # --- Environment ---
        pos_ecef = eci_to_ecef(true_state.position_eci, t)
        lat, lon, alt_geo = ecef_to_lla(pos_ecef)
        alt_m = max(0.0, alt_geo)
        atmo = atmosphere(alt_m)
        rho = atmo.density_kg_m3
        pressure = atmo.pressure_pa
        _temperature = atmo.temperature_k  # noqa: F841
        speed_of_sound = atmo.speed_of_sound_ms
        grav_eci = gravitational_acceleration(true_state.position_eci)
        wind_eci = wind_velocity_eci(true_state.position_eci, t, rng=rng)

        # --- Aerodynamics ---
        vel_rel = true_state.velocity_eci - wind_eci
        vel_rel_mag = float(np.linalg.norm(vel_rel))
        mach = vel_rel_mag / max(speed_of_sound, 1.0)
        drag_force_eci = aero.compute_drag(vel_rel, rho, speed_of_sound)
        q_pa = aero.current_q
        peak_q = max(peak_q, q_pa)

        # --- Guidance ---
        estimated_state = ekf.estimated_state()
        estimated_state.time_s = t
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
        # Zero thrust/mdot if propellant depleted (engine model doesn't track propellant)
        if vehicle.propellant_remaining() <= 0.0:
            thrust_n = 0.0
            mdot = 0.0

        # --- Control ---
        tvc_cmd = controller.update(
            guidance_cmd.desired_quaternion,
            estimated_state.quaternion,
            true_state.angular_velocity_body,
            dt,
        )

        # --- Boundary enforcement: TVC ---
        tvc_result = enforcer.validate_tvc(tvc_cmd.pitch_deg, tvc_cmd.yaw_deg, dt)
        approved_tvc_pitch, approved_tvc_yaw = tvc_result.value

        # --- Compute accelerations for structural check ---
        axial_g = thrust_n / max(true_state.mass_kg, 1.0) / config.G0
        tvc_lateral_force = thrust_n * (
            abs(math.sin(math.radians(approved_tvc_pitch))) + abs(math.sin(math.radians(approved_tvc_yaw)))
        )
        lateral_g = tvc_lateral_force / max(true_state.mass_kg, 1.0) / config.G0
        peak_axial_g = max(peak_axial_g, axial_g)

        enforcer.check_structural_limits(axial_g, lateral_g, q_pa)

        # --- Sensor measurements ---
        imu_meas, gps_meas, baro_meas = sensors.update(true_state, grav_eci, dt)

        # Add flex body bending rate to IMU gyro
        if flex_body is not None:
            flex_rate = flex_body.total_bending_rate_at_imu()
            # The sensor suite already captured a measurement, but the flex
            # contribution should be in the true angular velocity seen by IMU.
            # We'll add it to the gyro measurement post-hoc.
            imu_meas.gyro_body_rads = imu_meas.gyro_body_rads + np.array([0.0, flex_rate, 0.0])

        # --- Navigation (EKF) ---
        ekf.set_attitude(true_state.quaternion, true_state.angular_velocity_body)
        ekf.set_mass(true_state.mass_kg)
        ekf.predict(imu_meas, grav_eci, dt)
        if gps_meas is not None:
            ekf.update_gps(gps_meas)
        if baro_meas is not None:
            ekf.update_baro(baro_meas)

        ekf_uncertainty = ekf.position_uncertainty_m()
        peak_ekf_uncertainty = max(peak_ekf_uncertainty, ekf_uncertainty)

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

        # --- Flex body ---
        if flex_body is not None:
            # TVC lateral force for flex excitation
            tvc_force = thrust_n * math.sin(math.radians(approved_tvc_pitch))
            flex_body.update(dt, tvc_force, vehicle.propellant_fraction())

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

        # --- Physics: compute forces and integrate ---
        # Thrust vector in body frame with TVC deflection
        # Convention: TVC pitch deflects thrust in body Z (creates torque about Y)
        #             TVC yaw deflects thrust in body Y (creates torque about Z)
        pitch_rad = math.radians(approved_tvc_pitch)
        yaw_rad = math.radians(approved_tvc_yaw)
        thrust_body = np.array(
            [
                thrust_n * math.cos(pitch_rad) * math.cos(yaw_rad),
                thrust_n * math.sin(yaw_rad),
                thrust_n * math.sin(pitch_rad),
            ]
        )
        thrust_eci = body_to_eci(thrust_body, true_state.quaternion)
        slosh_force_eci = body_to_eci(slosh_force_body, true_state.quaternion)

        # Torques in body frame from TVC
        # Engine is behind CG: r = [-arm, 0, 0]
        # Torque = r x F:
        #   pitch: F_z = T*sin(pitch_rad) -> torque_y = -(-arm)*F_z = arm*T*sin(pitch_rad)
        #   yaw:   F_y = T*sin(yaw_rad)   -> torque_z = (-arm)*F_y = -arm*T*sin(yaw_rad)
        # Sign convention: positive TVC pitch -> positive torque_y -> correct pitch-up
        moment_arm = config.VEHICLE_LENGTH_M * 0.45
        tvc_torque = np.array(
            [
                0.0,
                moment_arm * thrust_n * math.sin(pitch_rad),
                -moment_arm * thrust_n * math.sin(yaw_rad),
            ]
        )
        total_torque = tvc_torque + slosh_torque_body

        # Moment of inertia (cylindrical approximation)
        radius = math.sqrt(config.REFERENCE_AREA_M2 / math.pi)
        length = config.VEHICLE_LENGTH_M
        inertia = max(100.0, true_state.mass_kg * (radius**2 / 4 + length**2 / 12))

        # Mass flow
        actual_mdot = -mdot if vehicle.propellant_remaining() > 0 else 0.0

        # Create derivatives closure
        _thrust_eci = thrust_eci
        _grav_eci = grav_eci
        _drag_eci = drag_force_eci
        _slosh_eci = slosh_force_eci
        _torque = total_torque
        _inertia = inertia
        _mass_rate = actual_mdot
        _mass = true_state.mass_kg

        def derivatives_fn(  # noqa: B023
            t_eval: float,
            s: VehicleState,
        ) -> StateDot:
            total_force = _thrust_eci + _drag_eci + _slosh_eci
            accel = _grav_eci + total_force / max(s.mass_kg, 1.0)
            angular_accel = _torque / _inertia
            quat_dot = quaternion_derivative(s.quaternion, s.angular_velocity_body)
            return StateDot(
                velocity_eci=s.velocity_eci,
                acceleration_eci=accel,
                quaternion_dot=quat_dot,
                angular_acceleration_body=angular_accel,
                mass_rate_kg_s=_mass_rate,
            )

        true_state = rk4_step(true_state, derivatives_fn, dt)

        # Update vehicle mass tracking
        if mdot > 0:
            vehicle.consume_propellant(mdot * dt)

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

        # --- Insertion check ---
        alt_km = true_state.altitude_m() / 1000
        vel = true_state.velocity_mag_ms()
        if (
            alt_km > config.TARGET_ALTITUDE_M / 1000 * 0.90
            and vel > config.TARGET_VELOCITY_MS * 0.95
            and vehicle.stage_index >= 1
        ):
            r_hat = true_state.position_eci / max(np.linalg.norm(true_state.position_eci), 1.0)
            v_hat = true_state.velocity_eci / max(vel, 1.0)
            fpa = math.asin(np.clip(np.dot(r_hat, v_hat), -1.0, 1.0))
            if abs(math.degrees(fpa)) < 5.0:
                outcome = "SUCCESS"
                if not quiet:
                    print(f"  ORBITAL INSERTION at t={t:.1f}s!")
                break

    # End-of-sim fallback check
    if outcome == "TIMEOUT":
        alt_km = true_state.altitude_m() / 1000
        vel = true_state.velocity_mag_ms()
        if alt_km > 200 and vel > 7000:
            outcome = "SUCCESS"

    # Compute flight path angle
    final_fpa = 0.0
    r_norm = np.linalg.norm(true_state.position_eci)
    if r_norm > 0 and true_state.velocity_mag_ms() > 0:
        r_hat = true_state.position_eci / r_norm
        v_hat = true_state.velocity_eci / true_state.velocity_mag_ms()
        final_fpa = math.degrees(math.asin(np.clip(np.dot(r_hat, v_hat), -1.0, 1.0)))

    # --- Post-insertion orbit analysis ---
    orbit_elements_dict = None
    if outcome == "SUCCESS" and not is_mc:
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
        except Exception as e:
            if not quiet:
                print(f"  Orbit analysis error: {e}")

    # --- Write telemetry (non-MC only) ---
    if not is_mc:
        summary = recorder.write_output(
            outcome=outcome,
            true_state=true_state,
            health_monitor=health_monitor,
            boundary_enforcer=enforcer,
        )
        if not quiet:
            _print_summary(summary, orbit_elements_dict)
            # Generate plots
            try:
                from sim.analysis.postflight import generate_plots

                generate_plots(recorder.internal_frames, summary)
            except Exception as e:
                print(f"  Plot generation error: {e}")

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
    parser = argparse.ArgumentParser(description="6-DOF Launch Vehicle Ascent Simulation")
    parser.add_argument("--no-flex", action="store_true", help="Disable flex body model")
    parser.add_argument("--no-slosh", action="store_true", help="Disable propellant slosh model")
    args = parser.parse_args()

    if args.no_flex:
        config.FLEX_ENABLED = False
    if args.no_slosh:
        config.SLOSH_ENABLED = False

    print("6-DOF Ascent Simulation")
    print("=" * 40)
    print(f"Target: {config.TARGET_ALTITUDE_M / 1000:.0f} km, {config.TARGET_INCLINATION_DEG} deg inc")
    print(f"Flex body: {'ON' if config.FLEX_ENABLED else 'OFF'}")
    print(f"Slosh: {'ON' if config.SLOSH_ENABLED else 'OFF'}")
    print()

    result = run_simulation()

    if result is not None:
        if result.outcome == "SUCCESS":
            sys.exit(0)
        elif result.outcome == "FTS_ABORT":
            sys.exit(1)
        else:
            sys.exit(2)


if __name__ == "__main__":
    main()
