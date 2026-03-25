# 6-DOF Launch Vehicle Ascent Simulation

[![CI](https://github.com/rmednitzer/6dof-ascent-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/rmednitzer/6dof-ascent-sim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A high-fidelity six-degree-of-freedom simulation of a two-stage orbital launch vehicle from ignition through LEO insertion.

## Features

- **6-DOF rigid body dynamics** with quaternion attitude representation (scalar-last `[x,y,z,w]`)
- **RK4 integration** at 100 Hz fixed timestep
- **WGS84 gravity model** with J2 oblateness perturbation
- **US Standard Atmosphere 1976** with altitude-dependent wind and gusts
- **Two-stage propulsion** with pressure-dependent thrust/Isp interpolation
- **Stage separation state machine** with safety interlocks
- **Three-phase guidance**: vertical rise, programmed gravity turn, PEG terminal guidance
- **PID attitude controller** producing TVC gimbal commands
- **12-state Extended Kalman Filter** (position, velocity, accel bias, gyro bias)
- **Boundary enforcement** on all actuator commands and structural loads
- **Flight Termination System** with autonomous abort criteria
- **Structural dynamics**: flex body bending modes + propellant slosh (pendulum analogy)
- **Monte Carlo** dispersion analysis with multiprocessing
- **Telemetry recording** with SHA-256 integrity hashing
- **Post-flight analysis** with trajectory plots and orbit characterization

## Quick Start

```bash
pip install -e .
python -m sim.main
```

## CLI Options

```bash
python -m sim.main                 # Full simulation with flex + slosh
python -m sim.main --no-flex       # Disable flex body model
python -m sim.main --no-slosh      # Disable propellant slosh model
```

## Example: Run with Visualization

The `examples/` directory contains a standalone script that runs a nominal ascent and generates a visualization dashboard, ground track plot, and text summary:

```bash
python examples/run_and_visualize.py
python examples/run_and_visualize.py --no-flex --no-slosh
```

This produces three files in `examples/output/`:

| File | Description |
|---|---|
| `dashboard.png` | 8-panel mission dashboard (altitude, velocity, dynamic pressure, G-load, mass, throttle, EKF uncertainty, Mach) |
| `ground_track.png` | Latitude/longitude trajectory colored by altitude |
| `mission_summary.txt` | Plain-text report with key metrics and safety margins |

## Output

- `output/telemetry.json` — Complete telemetry timeline
- `output/summary.json` — Mission summary with key metrics
- `output/plots/` — Trajectory visualization plots

## Configuration

All simulation parameters are in `sim/config.py` — single source of truth for orbital targets, vehicle specs, environment models, GNC gains, safety limits, and Monte Carlo settings.

## Development

```bash
pip install -e ".[dev]"        # Install with dev dependencies
pre-commit install              # Set up pre-commit hooks
```

## Tests

```bash
pytest tests/ -v                           # Run tests
pytest tests/ -v --cov=sim --cov-report=term-missing  # With coverage
```

## Project Structure

```
sim/
├── config.py              # All simulation parameters
├── main.py                # Main simulation loop
├── core/
│   ├── state.py           # VehicleState dataclass
│   ├── integrator.py      # RK4 integrator
│   └── reference_frames.py # ECI/ECEF/NED/Body transforms
├── environment/
│   ├── gravity.py         # WGS84 + J2 gravity
│   ├── atmosphere.py      # US Standard Atmosphere 1976
│   └── wind.py            # Wind profile + gusts
├── vehicle/
│   ├── vehicle.py         # Stage configuration + mass tracking
│   ├── propulsion.py      # Engine model with transients
│   ├── aerodynamics.py    # Mach-dependent drag
│   └── staging.py         # Separation state machine
├── dynamics/
│   ├── flex_body.py       # Bending mode dynamics
│   └── slosh.py           # Propellant slosh model
├── gnc/
│   ├── guidance.py        # Three-phase ascent guidance
│   ├── control.py         # PID attitude controller + TVC
│   ├── sensors.py         # IMU/GPS/Baro sensor models
│   └── navigation.py      # Extended Kalman Filter
├── safety/
│   ├── boundary_enforcer.py # Command validation + clamping
│   ├── fts.py             # Flight Termination System
│   └── health_monitor.py  # Subsystem health tracking
├── telemetry/
│   ├── schemas.py         # TelemetryFrame + MissionSummary
│   └── recorder.py        # Telemetry recording + output
├── orbital/
│   ├── propagator.py      # Orbit elements + propagation
│   ├── maneuvers.py       # Correction budget estimation
│   └── decay.py           # Orbit decay estimation
├── montecarlo/
│   ├── dispersions.py     # Parameter dispersion definitions
│   ├── dispatcher.py      # Parallel run management
│   └── statistics.py      # Result analysis + plots
└── analysis/
    └── postflight.py      # Post-flight trajectory plots
examples/
└── run_and_visualize.py   # Example run with dashboard visualization
```

## Documentation

- [Architecture](docs/architecture.md) — System design and data flow
- [Assumptions](docs/assumptions.md) — Modeling assumptions and simplifications
- [STPA Analysis](docs/stpa-analysis.md) — Safety analysis
- [Runbook](docs/runbook.md) — Operating procedures

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.

## License

See [LICENSE](LICENSE) for details.
