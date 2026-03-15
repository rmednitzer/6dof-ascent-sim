# Runbook

## Installation

Requires Python >= 3.11.

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

Dependencies: `numpy >= 1.24`, `scipy >= 1.10`, `matplotlib >= 3.7`.

## Basic Run

```bash
python -m sim.main
```

This runs a single nominal ascent simulation targeting 400 km circular LEO at 51.6 deg inclination (ISS orbit). Progress is printed every 10 seconds of simulation time.

### CLI Options

| Flag | Effect |
|------|--------|
| `--no-flex` | Disable structural flex body model (sets `FLEX_ENABLED = False`) |
| `--no-slosh` | Disable propellant slosh model (sets `SLOSH_ENABLED = False`) |

Example:

```bash
python -m sim.main --no-flex --no-slosh
```

## Output

A single run produces files in the `output/` directory:

| File | Description |
|------|-------------|
| `output/telemetry_internal.json` | Full 100 Hz telemetry (every physics step) |
| `output/telemetry_downlink.json` | Decimated 10 Hz telemetry (simulated TM link) |
| `output/mission_summary.json` | Aggregate metrics, peak values, SHA-256 hash of internal telemetry |
| `output/plots/*.png` | Post-flight analysis plots |

### Generated Plots

- `altitude_vs_time.png` -- Altitude with target overlay
- `velocity_vs_time.png` -- Inertial velocity with target overlay
- `dynamic_pressure_vs_time.png` -- Q with structural limit and max-Q marker
- `axial_g_vs_time.png` -- Axial G-load with structural limit
- `ekf_uncertainty_vs_time.png` -- EKF position uncertainty with FTS limit
- `mass_vs_time.png` -- Vehicle mass (shows staging discontinuity)
- `trajectory_3d.png` -- 3D trajectory in ECI frame

## Monte Carlo

Run a Monte Carlo campaign using the dispatcher:

```python
from sim.montecarlo.dispatcher import MonteCarloDispatcher

dispatcher = MonteCarloDispatcher(num_runs=100, seed=42)
results = dispatcher.execute(workers=4)
```

Or from the command line:

```bash
python -m sim.montecarlo.dispatcher --runs 100 --seed 42 --workers 4
```

Monte Carlo output:
- `output/montecarlo_results.json` -- Per-run results (outcome, insertion state, peak loads, boundary clamp count)
- Console summary with success rate, insertion accuracy statistics, and limit proximity

### Dispersed Parameters

Defined in `sim/montecarlo/dispersions.py` (`DEFAULT_DISPERSIONS`):

| Parameter | Distribution | 1-sigma or Bounds |
|-----------|-------------|-------------------|
| `S1_THRUST_VAC_N` | Gaussian | 76,070 N |
| `S1_ISP_VAC_S` | Gaussian | 3.11 s |
| `S2_THRUST_VAC_N` | Gaussian | 9,810 N |
| `S2_ISP_VAC_S` | Gaussian | 3.48 s |
| `S1_PROPELLANT_KG` | Gaussian | 395.7 kg |
| `CD_SCALE_FACTOR` | Truncated Gaussian | sigma=0.10, bounds [0.7, 1.3] |
| `ATMO_DENSITY_SCALE` | Truncated Gaussian | sigma=0.05, bounds [0.8, 1.2] |
| `WIND_SPEED_MS` | Truncated Gaussian | sigma=15.0, bounds [0, 50] |
| `WIND_DIRECTION_DEG` | Uniform | [0, 360] |
| `IMU_ACCEL_BIAS_MPS2` | Gaussian | 0.002 m/s^2 |
| `IMU_GYRO_BIAS_RADS` | Gaussian | 0.0002 rad/s |
| `GPS_POS_NOISE_M` | Truncated Gaussian | sigma=2.0, bounds [1, 15] |
| `S1_DRY_MASS_KG` | Gaussian | 222 kg |

## Configuration

All simulation parameters live in `sim/config.py`. Key parameter groups:

- **Orbital target**: `TARGET_ALTITUDE_M`, `TARGET_INCLINATION_DEG`, `TARGET_VELOCITY_MS`
- **Earth model**: `EARTH_RADIUS_M`, `EARTH_MU`, `EARTH_J2`, `EARTH_OMEGA`
- **Simulation**: `DT` (timestep), `T_MAX` (max sim time), `TELEMETRY_HZ`, `INTERNAL_HZ`
- **Launch site**: `LAUNCH_LAT_DEG`, `LAUNCH_LON_DEG`
- **Stage 1/2**: Mass, thrust, Isp, burn time for each stage
- **Aerodynamics**: `CD_TABLE_MACH`, `CD_TABLE_VALUE`, `REFERENCE_AREA_M2`
- **Structural limits**: `MAX_Q_PA` (35 kPa), `MAX_AXIAL_G` (6.0), `MAX_LATERAL_G` (0.5)
- **TVC limits**: `TVC_MAX_DEFLECTION_DEG` (5.0), `TVC_MAX_SLEW_RATE_DEG_S` (10.0)
- **EKF/Sensors**: Noise levels, bias instabilities, update rates, innovation gate threshold
- **FTS**: `FTS_CROSSRANGE_LIMIT_M`, `FTS_ATTITUDE_LIMIT_DEG`, `FTS_COVARIANCE_LIMIT_M`
- **Flex/Slosh**: Mode frequencies, damping ratios, enable flags
- **Guidance**: `PITCH_KICK_DEG`, `VERTICAL_RISE_TIME_S`
- **Control gains**: `CONTROL_KP`, `CONTROL_KD`, `CONTROL_KI`, `CONTROL_INTEGRATOR_LIMIT_DEG`
- **Monte Carlo**: `MC_NUM_RUNS`, `MC_SEED`, `MC_WORKERS`

Parameters can be overridden at runtime via the `config_override` dict passed to `run_simulation()`.

## Interpreting Results

### Outcomes

| Outcome | Meaning | Exit Code |
|---------|---------|-----------|
| `SUCCESS` | Orbital insertion achieved (altitude > 90% target, velocity > 95% target, FPA < 5 deg, stage >= 2) | 0 |
| `FTS_ABORT` | Flight Termination System triggered (crossrange, attitude, EKF uncertainty, or structural limit violated) | 1 |
| `TIMEOUT` | Simulation reached `T_MAX` (600 s) without insertion or FTS trigger | 2 |

A `TIMEOUT` outcome is automatically upgraded to `SUCCESS` if the vehicle has altitude > 200 km and velocity > 7000 m/s at the end.

### Mission Summary Fields

Key fields in `output/mission_summary.json`:
- `outcome` -- Terminal result
- `final_altitude_m`, `final_velocity_ms` -- State at simulation end
- `peak_dynamic_pressure_pa` -- Compare against `MAX_Q_PA` (35,000 Pa)
- `peak_axial_g` -- Compare against `MAX_AXIAL_G` (6.0)
- `total_boundary_violations` -- Number of times the boundary enforcer clamped a command
- `fts_triggered` -- Whether the FTS fired
- `telemetry_hash_sha256` -- Integrity hash of internal telemetry JSON

### Monte Carlo Statistics

The `statistics.py` module reports:
- Success/FTS/timeout rates
- Insertion accuracy (altitude, velocity, flight path angle) with mean, std, 3-sigma bounds
- Limit proximity (peak Q and peak G as % of structural limits, P99 values)
- Boundary clamp counts (mean, max)

## Running Tests

```bash
pytest tests/
```

Test infrastructure is in the `tests/` directory. The project uses pytest (installed with `pip install -e ".[dev]"`).
