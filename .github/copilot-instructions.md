# Copilot Instructions — 6-DOF Ascent Simulation

## Project

6-DOF launch vehicle ascent simulation (Python 3.11+). Simulates a two-stage orbital vehicle from ignition through LEO insertion with full GNC, safety systems, and telemetry.

## Build & Run

```bash
pip install -e ".[dev]"       # Install with dev dependencies
python -m sim.main             # Run simulation
python -m sim.main --no-flex   # Disable flex body model
python -m sim.main --no-slosh  # Disable slosh model
```

## Test & Lint

```bash
pytest tests/ -v               # Run all tests
ruff check .                   # Lint
ruff format --check .          # Format check
ruff format .                  # Auto-format
```

## Architecture

- `sim/config.py` — Single source of truth for all parameters (~100 constants)
- `sim/main.py` — Main simulation loop (100 Hz RK4 integration)
- `sim/core/` — State dataclass, RK4 integrator, reference frame transforms
- `sim/environment/` — Atmosphere, gravity (WGS84+J2), wind models
- `sim/vehicle/` — Propulsion, aerodynamics, staging state machine
- `sim/dynamics/` — Flex body bending modes, propellant slosh
- `sim/gnc/` — Guidance (3-phase), PID control + TVC, 12-state EKF, sensors
- `sim/safety/` — Boundary enforcement, FTS abort, health monitoring
- `sim/telemetry/` — Recording schemas, dual-rate recorder
- `sim/orbital/` — Orbit propagation, maneuvers, decay analysis
- `sim/montecarlo/` — Parallel dispatcher, dispersions, statistics
- `sim/analysis/` — Post-flight trajectory plots
- `tests/` — pytest suite (10 modules)

## Conventions

- Quaternions use scalar-last `[x, y, z, w]` convention
- All units SI (meters, seconds, kilograms, radians)
- Ruff linter with 120 char line length
- Safety-critical code uses explicit nested ifs (no collapsing with `and`) for readability
- Type hints encouraged on public APIs
- NumPy for all vector/matrix operations
