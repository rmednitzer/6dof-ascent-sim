# Modeling Assumptions and Simplifications

> **Fidelity disclaimer:** This simulation is an educational and research tool.
> It is **not suitable for flight certification, launch licensing, or
> safety-of-flight decisions** without independent verification and validation
> (V&V) against higher-fidelity models and flight test data. The simplifications
> documented below introduce modelling errors that may not bound worst-case
> flight conditions. Any use of this simulation in a regulatory or certification
> context must be independently assessed.

## Earth Model

- **WGS84 ellipsoid** with semi-major axis 6,378,137 m and flattening 1/298.257223563 (`sim/config.py`).
- **Zonal harmonics J2–J6** (`EARTH_J2 = 1.08262668e-3` and `EARTH_J3`–`EARTH_J6`, EGM96 values), computed as the exact analytic gradient of the geopotential and verified against a finite-difference gradient to ~1e-10 (relative) across all latitudes (`sim/environment/gravity.py`; `tests/test_fidelity_fixes.py`). No tesseral/sectoral harmonics. (A previous hand-coded form silently zeroed the odd zonals J3/J5 via an extra `1/r` factor; see audit AD-03/AD-11.)
- **Constant rotation rate** `EARTH_OMEGA = 7.2921150e-5 rad/s`. No precession, nutation, or polar motion. ECI and ECEF coincide at t=0.

## Atmosphere

- **US Standard Atmosphere 1976** (`sim/environment/atmosphere.py`): seven lapse-rate layers from 0 to 86 km, evaluated in geopotential altitude (geometric→geopotential conversion applied); a piecewise-exponential thermosphere model from 86 to 1000 km; vacuum above 1000 km.
- **No latitude, season, or day/night variation**. The model is purely altitude-dependent.
- Density is scaled by `ATMO_DENSITY_SCALE` (nominal 1.0) for Monte Carlo dispersion.
- Speed of sound computed from ideal-gas relation with gamma = 1.4. Above 86 km, temperature is held at the 86 km value (real thermosphere heats up, but density is negligible).

## Propulsion

- **Thrust** interpolates linearly between sea-level and vacuum with ambient pressure (`sim/vehicle/propulsion.py`): `F(p) = F_vac - (F_vac - F_sl) * (p / p_sl)`, modelling nozzle back-pressure.
- **Mass flow is pump-limited and constant at a fixed throttle**, independent of altitude: `mdot = throttle * F_vac / (Isp_vac * g0)`. The effective specific impulse `Isp(p) = F(p) / (mdot * g0)` therefore emerges consistently. (A previous version interpolated thrust and Isp independently, letting `mdot` drift ~1.35% sea-level→vacuum so propellant burn was not conserved; see audit AD-12.)
- **Ignition and shutdown transients** modeled as 0.5 s linear ramps.
- **No combustion instability**, nozzle erosion, or mixture ratio effects.
- **Stage 2 operates in vacuum only**: `thrust_sl = thrust_vac` and `isp_sl = isp_vac` (`sim/vehicle/vehicle.py`, STAGE_2 definition).
- **Stage 2 carries a performance margin via Isp** (`S2_ISP_VAC_S` = 356 s, ADR 0021): raised from 348 s (+2.3%) so the upper stage reaches orbit with margin. The earlier value inserted with almost no reserve, so sub-1-sigma adverse propulsion/drag dispersions left the stage just short of orbit — the dominant driver of the Monte-Carlo abort rate (N-01). Margin was added through Isp rather than propellant because the nominal trajectory is chaotically sensitive to liftoff mass (a propellant increase flips it between SUCCESS and abort across small mass steps), whereas raising Isp leaves liftoff mass and the nominal trajectory unchanged.
- No propellant sloshing effect on the feed system.

## Aerodynamics

- **Drag-only model** (`sim/vehicle/aerodynamics.py`): Mach-dependent Cd interpolated from a 9-point table via cubic spline. Scaled by `CD_SCALE_FACTOR` for Monte Carlo.
- **No angle-of-attack effects**: Cd depends only on Mach number, not on alpha or beta. No lift, side force, or aerodynamic moments.
- **No base drag or plume interaction** modeling.
- Center of pressure is fixed at `COP_OFFSET_FROM_NOSE_M = 12.0 m` (no Mach dependency).
- Reference area is the constant cross-section `REFERENCE_AREA_M2 = 10.52 m^2`.

## Structural / Inertia

- **Cylindrical moment of inertia approximation** (`sim/main.py`):
  ```
  I = mass * (radius^2 / 4 + length^2 / 12)
  ```
  where radius is derived from `REFERENCE_AREA_M2` and length is `VEHICLE_LENGTH_M`. This is a uniform-density solid cylinder approximation -- actual launch vehicles have non-uniform mass distributions.
- **Moment of inertia is scalar** (isotropic). Roll, pitch, and yaw share one inertia value. No products of inertia.
- Minimum inertia clamped to 100 kg-m^2 to avoid numerical issues.

## Vehicle Dynamics

- **Rigid body** with two optional dynamic sub-models:
  - **Flex body** (`sim/dynamics/flex_body.py`): First 3 lateral bending modes as damped harmonic oscillators. Semi-implicit Euler integration. Modal frequencies interpolate linearly between full- and empty-propellant values. **Live in the control loop** (audit AD-04): the bending rate sensed at the IMU is added to the controller's measured pitch-rate feedback (one-step sense->compute->actuate lag), and a **frequency-scheduled structural notch filter** (`sim/gnc/notch_filter.py`), centred on the propellant-varying modal frequencies, suppresses the lightly-damped modes so the loop does not flutter. Disabling the notch (`FLEX_NOTCH_ENABLED = False`) reproduces the control-structure instability — a TVC limit cycle that floods the actuator with clamp events. The flex model is a sensor/control-path effect only: it applies no back-reaction force to the rigid body. Generalised modal mass (`FLEX_MODAL_MASS_KG`) is a tuning parameter because the mode-shape normalisation is arbitrary.
  - **Propellant slosh** (`sim/dynamics/slosh.py`): Single-tank pendulum analogy. 30% of remaining propellant participates in slosh (`SLOSH_MASS_FRACTION`). Frequency interpolates with fill level. Damping ratio 0.03 (baffled tank). Semi-implicit Euler integration.
  - **TVC actuator** (`sim/vehicle/actuator.py`): second-order servo (25 Hz bandwidth, ζ=0.7) integrated on adaptive sub-steps so the response stays numerically stable at the 100 Hz outer rate (a single semi-implicit Euler step was divergent and limit-cycled; audit AD-02).
- **No structural flexibility coupling** between flex modes and slosh.
- **Instantaneous stage separation** (mass drop). Coast duration is 1.0 s between tail-off and separation.
- **RK4 force hold** (`sim/main.py`): the translational forces, torques and gravity are evaluated once per 100 Hz step and held constant across the four RK4 sub-stages (only the kinematic states vary per sub-stage). Net dynamics accuracy is therefore effectively first-order within a step; the 100 Hz rate keeps the per-step error small for this trajectory but the "RK4" label refers to the kinematic integration, not the force model.

## Navigation

- **Attitude is estimated by the EKF** (`sim/gnc/navigation.py`, ADR 0020). The filter is an error-state / multiplicative EKF: it carries a nominal attitude quaternion propagated from the *measured* gyro plus a 3-DOF multiplicative attitude error in its covariance. Below the GPS ceiling attitude is observable through the specific-force/velocity coupling (`δv̇ = -[f_n]_x δθ`); above it the **star tracker** observes attitude directly (all three axes, including roll). Accelerometer and gyro biases are co-estimated. The `USE_ESTIMATED_ATTITUDE` flag (overridable; **default `True`** since Stage 2) selects whether guidance, the controller, and the FTS consume the *estimated* attitude — the closed-loop default — or the true attitude (with the estimator still running in parallel for validation/telemetry).
- The 15-error-state EKF estimates attitude (3), position (3), velocity (3), accelerometer bias (3), and gyro bias (3). Biases are modeled as random walks.
- **GPS available below 60 km only** (`config.GPS_AVAILABILITY_CEILING_M`, the COCOM export limit; set `+∞` to model an ITAR-cleared receiver). **Barometer available below 40 km only.** Above those, the upper stage is GPS/baro-denied and the **star tracker** (`config.STAR_TRACKER_*`, ADR 0020) keeps attitude observable: it is usable only above the sensible atmosphere (~100 km) and below an image-smear slew rate — the upper-stage regime. Without it, gyro-only attitude dead-reckoning through the ~200 s GPS-denied coast diverges (>20°) and inflates the position covariance past the FTS limit. Position/velocity still dead-reckon through that phase, so the peak EKF position uncertainty is ≈1.7 km (well under the 10 km FTS limit).
- Innovation gating uses a chi-square (NIS) consistency test: a measurement is rejected when its normalised innovation squared `yᵀ S⁻¹ y` exceeds the chi-square quantile at `EKF_INNOVATION_GATE_P = 0.9973` for its dimension (Mahalanobis distance over the full innovation covariance). At one DOF this matches the previous 3-sigma intent; for multi-component GPS it is the principled joint test rather than a per-axis magnitude check. A non-finite innovation/covariance is rejected as a counted fault (`measurement_rejections`). Gating can still reject valid measurements during rapid maneuvering.

## Sensors

- **IMU** (`sim/gnc/sensors.py`): 100 Hz. Gaussian noise + random-walk bias on each axis. No quantization, scale factor errors, or cross-axis coupling.
- **GPS**: 1 Hz. Gaussian position and velocity noise. No multipath, ionospheric delays, or satellite geometry effects.
- **Barometer**: 10 Hz. Gaussian altitude noise. No hysteresis or lag.
- Sensor noise parameters are all configurable in `sim/config.py` and dispersed in Monte Carlo.

## Wind

- **Simple altitude profile** (`sim/environment/wind.py`): surface wind ramps linearly to 2x at 12 km (crude jet stream), then decays to zero at 30 km.
- **Gaussian gusts**: zero-mean, `WIND_GUST_SIGMA_MS` standard deviation, scaled by the same altitude profile. Gusts are independent each timestep (white noise, no temporal correlation / Dryden model).
- **Co-rotating atmosphere**: wind velocity in ECI includes Earth-rotation so a vehicle at rest on the pad sees near-zero airspeed.
- Wind direction uses meteorological convention (direction wind comes from).

## Guidance

- **No powered explicit guidance (PEG) for Stage 1**. Gravity turn uses a programmed quadratic pitch schedule blended with velocity-vector tracking.
- **Terminal guidance** is a simplified linear-tangent steering law, not a full iterative PEG. It computes a desired pitch from radial/tangential velocity errors with a time-to-go estimate.
- **No roll program** or roll steering. Only pitch and yaw are controlled.
- **Launch azimuth uses the inertial relation** `sin(Az) = cos(i)/cos(lat)` without the Earth-rotation correction, and the gravity turn follows the Earth-relative velocity with no active out-of-plane (yaw) plane-steering. The achieved inclination therefore falls short of the 51.6 deg target (~45 deg), which inflates the reported insertion correction-dv. This is an **accepted simplification** (audit AD-17): the standard azimuth correction and active yaw out-of-plane nulling were both prototyped and *do* reach the target (~51.4 deg; correction-dv ~995 -> ~266 m/s), but they systematically regress Monte-Carlo robustness because the PEG terminal phase already rides the 25 deg FTS attitude limit under dispersions (every dispersed abort is a marginal ~25 deg thrust-axis-error hit near insertion). Closing AD-17 is therefore coupled to hardening that PEG terminal attitude margin and is deferred.

## Orbit

- **Insertion criteria** (`sim/main.py`, `sim/orbital/propagator.py`): altitude > 95% and velocity > 97% of target, stage >= 2, flight path angle < 5 deg, **and** the osculating orbit must be bound, near-circular (e < 0.05), with periapsis above the sensible atmosphere (> 140 km). A steep sub-orbital arc therefore cannot be misreported as SUCCESS.
- Post-insertion orbit analysis uses J2-perturbed Cowell's method (`sim/orbital/propagator.py`). No drag, solar radiation pressure, or third-body perturbations.
