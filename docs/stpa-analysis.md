# STPA Safety Analysis

Systems-Theoretic Process Analysis (STPA) for the 6-DOF ascent simulation safety architecture.

## 1. Losses

| ID  | Loss                                                      |
|-----|-----------------------------------------------------------|
| L-1 | Loss of vehicle (structural breakup, uncontrolled flight) |
| L-2 | Loss of mission (failure to reach target orbit)           |
| L-3 | Ground impact in populated area (range safety violation)  |

## 2. Hazards

| ID  | Hazard                           | Related Losses |
|-----|----------------------------------|----------------|
| H-1 | Loss of attitude control         | L-1, L-2       |
| H-2 | Structural overload (q, G)       | L-1            |
| H-3 | Trajectory deviation from corridor | L-3          |
| H-4 | Navigation divergence            | L-2, L-3       |
| H-5 | Premature or failed staging      | L-1, L-2       |

## 3. Unsafe Control Actions

### 3.1 TVC Deflection

| UCA | Description | Hazard |
|-----|-------------|--------|
| UCA-1 | TVC commands deflection exceeding `TVC_MAX_DEFLECTION_DEG` (5 deg) | H-1, H-2 |
| UCA-2 | TVC slew rate exceeds `TVC_MAX_SLEW_RATE_DEG_S` (10 deg/s), exciting structural flex modes | H-1, H-2 |
| UCA-3 | TVC commands full deflection during max-Q, causing excessive lateral G | H-2 |
| UCA-4 | Controller integrator windup produces saturated commands after sustained error | H-1 |

### 3.2 Throttle

| UCA | Description | Hazard |
|-----|-------------|--------|
| UCA-5 | Throttle commanded above 1.0 or below `S1_THROTTLE_MIN` (0.4) while engine is running | H-2 |
| UCA-6 | Full throttle maintained through max-Q region, exceeding `MAX_Q_PA` (35 kPa) | H-2 |
| UCA-7 | Full throttle at low vehicle mass, exceeding `MAX_AXIAL_G` (6.0 g) | H-2 |
| UCA-8 | Throttle commanded when propellant is depleted | H-5 |

### 3.3 Staging

| UCA | Description | Hazard |
|-----|-------------|--------|
| UCA-9  | Stage separation commanded while S1 engine thrust > 5% of rated (interlock violation) | H-1, H-5 |
| UCA-10 | Staging not triggered despite propellant depletion | H-2, H-5 |
| UCA-11 | S2 ignition fails or is delayed beyond the 0.5 s ramp window | H-2 |

### 3.4 Guidance

| UCA | Description | Hazard |
|-----|-------------|--------|
| UCA-12 | Guidance mode transitions at wrong time (e.g., terminal guidance during Stage 1) | H-1, H-3 |
| UCA-13 | Guidance commands attitude that would violate structural limits during max-Q | H-2 |
| UCA-14 | Pitch kick too aggressive, causing loss of gravity-turn stability | H-1, H-3 |

## 4. Causal Scenarios

| Scenario | Description | UCAs |
|----------|-------------|------|
| CS-1 | **Sensor failure**: IMU bias drift exceeds model, EKF position estimate diverges, guidance steers based on wrong state | UCA-12, UCA-13 |
| CS-2 | **EKF divergence**: Innovation gate rejects valid GPS measurements, filter covariance grows unbounded | UCA-12 |
| CS-3 | **Guidance mode confusion**: Time-based phase transitions trigger incorrectly if clock or MECO time is wrong | UCA-12 |
| CS-4 | **Flex-control coupling**: TVC commands at bending-mode frequencies amplify structural oscillations, gyro sees flex rate as rigid-body rate | UCA-1, UCA-2 |
| CS-5 | **Slosh-control coupling**: Pendulum slosh frequency near control bandwidth, lateral oscillations grow | UCA-3 |
| CS-6 | **Wind shear at max-Q**: Sudden gust increases angle of attack, combined with high q produces excessive lateral loads | UCA-6 |
| CS-7 | **Staging interlock bypass**: Software fault allows separation while S1 still thrusting | UCA-9 |
| CS-8 | **Propellant depletion mistrack**: Vehicle mass bookkeeping error causes throttle command to depleted stage | UCA-8, UCA-10 |

## 5. Safety Constraints and Enforcement

### 5.1 Boundary Enforcer (`sim/safety/boundary_enforcer.py`)

The boundary enforcer sits between the GNC outputs and the physics engine. Every actuator command passes through it.

| Constraint | Enforcement | Config Parameter |
|------------|-------------|------------------|
| Throttle in [0, 1] | Hard clamp | -- |
| Minimum throttle when engine running | Clamp to `S1_THROTTLE_MIN` | `S1_THROTTLE_MIN = 0.4` |
| No thrust on depleted propellant | Force throttle to 0 | -- |
| TVC deflection limit | Clamp to +/- `TVC_MAX_DEFLECTION_DEG` | `TVC_MAX_DEFLECTION_DEG = 5.0` |
| TVC slew rate limit | Rate-limit change per timestep | `TVC_MAX_SLEW_RATE_DEG_S = 10.0` |
| Staging interlock | Block separation if thrust > 5% | `THRUST_INTERLOCK_FRACTION = 0.05` |
| Structural limits | Check axial G, lateral G, dynamic pressure against limits | `MAX_AXIAL_G`, `MAX_LATERAL_G`, `MAX_Q_PA` |

The enforcer tracks a cumulative `violation_count` for telemetry and post-flight analysis.

### 5.2 Flight Termination System (`sim/safety/fts.py`)

The FTS evaluates four independent abort criteria every timestep. If any is violated, it latches irrevocably.

| Criterion | Threshold | Config Parameter |
|-----------|-----------|------------------|
| Cross-range deviation (below 100 km altitude) | `FTS_CROSSRANGE_LIMIT_M = 200,000 m` | `FTS_CROSSRANGE_LIMIT_M` |
| Attitude error (angle between actual and desired quaternion) | `FTS_ATTITUDE_LIMIT_DEG = 90 deg` | `FTS_ATTITUDE_LIMIT_DEG` |
| EKF position uncertainty (largest 1-sigma eigenvalue) | `FTS_COVARIANCE_LIMIT_M = 10,000 m` | `FTS_COVARIANCE_LIMIT_M` |
| Structural limit exceeded | Via `BoundaryEnforcer.check_structural_limits()` | `MAX_AXIAL_G`, `MAX_LATERAL_G`, `MAX_Q_PA` |

Cross-range is computed as the signed distance from the vehicle's ECEF position to the nominal trajectory plane (defined by the launch site position and the launch azimuth).

### 5.3 Health Monitor (`sim/safety/health_monitor.py`)

Continuous multi-channel assessment with four severity levels: NOMINAL, WARNING, ALERT, CRITICAL.

| Channel | WARNING | ALERT | CRITICAL |
|---------|---------|-------|----------|
| EKF covariance | 50% of FTS limit | 80% of FTS limit | 100% |
| Dynamic pressure | 80% of `MAX_Q_PA` | 95% | > 100% |
| Propellant margin | < 5% remaining | -- | 0% remaining |
| Sensor status | 1 sensor degraded | 2+ degraded | -- |
| Engine health | 5% thrust deviation | 10% deviation | 20% deviation |

## 6. Mitigations Implemented

| Mitigation | Addresses | Implementation |
|------------|-----------|----------------|
| **Anti-windup** | UCA-4 | Controller integrator clamped to `CONTROL_INTEGRATOR_LIMIT_DEG = 2.0 deg` per axis (`sim/gnc/control.py`) |
| **TVC slew-rate limiting** | UCA-2, CS-4 | Boundary enforcer limits rate of change to `TVC_MAX_SLEW_RATE_DEG_S` per timestep |
| **Max-Q throttle management** | UCA-6, CS-6 | Main loop reduces throttle when q > 70% of `MAX_Q_PA`; linear reduction to `S1_THROTTLE_MIN` (`sim/main.py`) |
| **G-limit throttle management** | UCA-7 | Main loop reduces throttle when predicted axial G > 90% of `MAX_AXIAL_G` (`sim/main.py`) |
| **Staging safety interlock** | UCA-9, CS-7 | `StagingSequencer` checks `effective_throttle <= 0.05` before allowing separation; boundary enforcer also gates on thrust fraction and arming state (`sim/vehicle/staging.py`) |
| **Controller reset on staging** | UCA-4 | Integrator state zeroed when staging event occurs (`sim/main.py`) |
| **Innovation gating** | CS-2 | EKF rejects measurements with residuals > 3 sigma (`sim/gnc/navigation.py`) |
| **Propellant depletion guard** | UCA-8, CS-8 | Boundary enforcer forces throttle to 0 when propellant remaining <= 0 |
| **Latching FTS** | H-1 through H-4 | Once triggered, FTS cannot be reset. Simulation terminates with `FTS_ABORT` outcome |
| **NaN/Inf guard** | All | Integrator checks all state components for NaN/Inf after each RK4 step, raises `RuntimeError` on detection |
