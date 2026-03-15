# System Architecture

## Module Hierarchy

```
sim/
  config.py                   # Single source of truth for all parameters
  main.py                     # Entry point, simulation loop, CLI
  core/
    state.py                  # VehicleState dataclass (6-DOF state vector)
    integrator.py             # RK4 integrator with injected derivatives
    reference_frames.py       # ECI/ECEF/NED/Body frame transforms, quaternion math
  environment/
    atmosphere.py             # US Standard Atmosphere 1976
    gravity.py                # WGS84 + J2 gravity
    wind.py                   # Altitude-dependent wind + Gaussian gusts
  vehicle/
    vehicle.py                # Stage configs, mass bookkeeping
    aerodynamics.py           # Mach-dependent Cd, drag force computation
    propulsion.py             # Engine model with throttle, ignition/shutdown transients
    staging.py                # Staging state machine (NOMINAL -> TAIL_OFF -> COAST -> SEP -> S2_IGN)
  dynamics/
    flex_body.py              # First-N lateral bending modes (damped oscillators)
    slosh.py                  # Pendulum-analogy propellant slosh
  gnc/
    guidance.py               # Three-phase guidance law
    control.py                # PID attitude controller with TVC output
    navigation.py             # 12-state EKF (pos, vel, accel bias, gyro bias)
    sensors.py                # IMU, GPS, barometer sensor models
  safety/
    boundary_enforcer.py      # Clamps throttle, TVC deflection, slew rate; checks structural limits
    fts.py                    # Flight Termination System (crossrange, attitude, EKF, structural)
    health_monitor.py         # Multi-channel health assessment (EKF, q, propellant, sensors, engine)
  telemetry/
    recorder.py               # Dual-rate recorder (100 Hz internal, 10 Hz downlink)
    schemas.py                # TelemetryFrame and MissionSummary dataclasses
  orbital/
    propagator.py             # Cartesian-to-Keplerian conversion, J2-perturbed orbit propagation
    maneuvers.py              # Correction delta-v budget
    decay.py                  # Orbital decay analysis
  montecarlo/
    dispatcher.py             # Multiprocessing campaign dispatcher
    dispersions.py            # Parameter dispersion definitions (Gaussian, uniform, truncated)
    statistics.py             # Post-campaign statistics and plots
  analysis/
    postflight.py             # Plot generation (altitude, velocity, q, G-load, EKF, mass, 3D)
```

## Data Flow (Per Timestep)

Each of the 100 Hz physics steps in `sim/main.py` follows this pipeline:

```
Environment (gravity, atmosphere, wind)
    |
    v
Aerodynamics (Mach -> Cd -> drag force)
    |
    v
Guidance (3-phase law -> desired quaternion + throttle)
    |
    v
Control (PID attitude error -> TVC pitch/yaw commands)
    |
    v
Boundary Enforcer (clamp throttle, TVC deflection, slew rate)
    |
    v
FTS Evaluation (crossrange, attitude error, EKF uncertainty, structural)
    |
    v
Physics (sum forces/torques in ECI, compute angular acceleration)
    |
    v
RK4 Integration (advance state by dt)
    |
    v
Telemetry Recording
```

## State Vector

Defined in `sim/core/state.py` as `VehicleState`:

| Field                    | Type         | Frame  | Units   |
|--------------------------|--------------|--------|---------|
| `position_eci`           | `ndarray(3)` | ECI    | m       |
| `velocity_eci`           | `ndarray(3)` | ECI    | m/s     |
| `quaternion`             | `ndarray(4)` | --     | [x,y,z,w] (scalar-last) |
| `angular_velocity_body`  | `ndarray(3)` | Body   | rad/s   |
| `mass_kg`                | `float`      | --     | kg      |
| `time_s`                 | `float`      | --     | s       |

The state derivative (`StateDot` in `sim/core/integrator.py`) contains velocity, acceleration, quaternion derivative, angular acceleration, and mass flow rate.

## Frame Conventions

- **ECI (Earth-Centered Inertial)**: Primary integration frame. Z-axis aligned with Earth spin axis. ECEF and ECI coincide at t=0.
- **ECEF**: Rotates with Earth at `EARTH_OMEGA`. Used for FTS crossrange checks and geodetic conversions.
- **Body**: +X axis is the thrust axis (nose direction). TVC pitch deflects thrust in body Z; TVC yaw deflects in body Y.
- **Quaternion**: Scalar-last convention `[x, y, z, w]`. `quat_to_dcm()` returns the DCM that rotates ECI vectors to body frame. `body_to_eci()` uses `DCM^T`.

## Integration

RK4 with fixed timestep `DT = 0.01 s` (100 Hz). The integrator (`sim/core/integrator.py`) is physics-agnostic: the main loop constructs a `derivatives_fn` closure that captures all forces and torques, and passes it to `rk4_step()`. After each step, the quaternion is re-normalized and a NaN/Inf check runs.

## GNC Architecture

### Guidance (`sim/gnc/guidance.py`)

Three sequential phases:

1. **Vertical Rise** (0 to `VERTICAL_RISE_TIME_S`): Hold thrust axis along local vertical, full throttle.
2. **Gravity Turn** (`VERTICAL_RISE_TIME_S` to MECO + 5 s): Programmed pitch schedule (quadratic ramp from `PITCH_KICK_DEG` to ~80 deg), blended with Earth-relative velocity vector tracking.
3. **Terminal** (Stage 2): Simplified PEG / linear-tangent steering toward target altitude and velocity. Computes desired radial acceleration to zero out radial velocity and reach `TARGET_ALTITUDE_M`.

Output: `GuidanceCommand` containing a desired attitude quaternion and throttle fraction.

### Control (`sim/gnc/control.py`)

PID attitude controller operating on the vector part of the error quaternion (small-angle approximation: `theta ~ 2 * [qx, qy, qz]`). Body Y and Z components drive two TVC axes. Gains: `CONTROL_KP`, `CONTROL_KD`, `CONTROL_KI`. Anti-windup clamp at `CONTROL_INTEGRATOR_LIMIT_DEG`. Integrators reset on staging. Output is clamped to `TVC_MAX_DEFLECTION_DEG`.

### Navigation (`sim/gnc/navigation.py`)

12-state EKF estimating `[pos(3), vel(3), accel_bias(3), gyro_bias(3)]` in ECI. Predict at 100 Hz via strapdown IMU mechanization. Updates:
- GPS at `GPS_UPDATE_HZ` (1 Hz), available below 60 km (COCOM limit)
- Barometer at `BARO_UPDATE_HZ` (10 Hz), available below 40 km

Innovation gate rejects measurements exceeding `EKF_RESIDUAL_SIGMA_THRESHOLD` sigma. Covariance update uses Joseph form. **Attitude is not estimated by the EKF** -- it is passed in directly via `set_attitude()` from the true state.

## Structural Dynamics

- **Flex body** (`sim/dynamics/flex_body.py`): First 3 lateral bending modes modeled as damped harmonic oscillators. Frequencies interpolate between full/empty propellant values. TVC force excites modes at the engine station; bending rates are sensed at the IMU station and added to gyro measurements.
- **Slosh** (`sim/dynamics/slosh.py`): Pendulum-analogy model. Slosh mass is `SLOSH_MASS_FRACTION` of remaining propellant. Frequency interpolates with fill level. Produces lateral forces and torques on the vehicle body.

Both can be toggled via `FLEX_ENABLED` / `SLOSH_ENABLED` in config or CLI flags `--no-flex` / `--no-slosh`.
