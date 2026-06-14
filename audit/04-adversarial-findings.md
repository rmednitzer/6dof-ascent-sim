# Audit 04 — Adversarial Physics / Numerics / GNC Findings Register

Date: 2026-06-14
Branch: `claude/amazing-lamport-tn1h3b`
Scope: an **adversarial** re-audit targeting the areas the 2026-06-12 audit
(`audit/02-security-findings.md`, `audit/03-final-report.md`) deliberately did
**not** touch — it "deliberately changed no physics". This pass therefore went
after numerics, physics, frame transforms, GNC, and concurrency.

## Method

Four read-only auditor agents (core/env, vehicle/dynamics, GNC,
safety/orbital/telemetry/MC) each ran isolated numerical repros. **Every finding
below was then independently re-verified in this session with its own repro**;
where an agent's analysis was wrong in its specifics, the corrected, verified
result is recorded (see the "agent corrections" notes). Baseline at audit start:
207 tests passing, 55% coverage, ruff clean, nominal end-to-end run = SUCCESS
(insertion at t=488.2 s).

Severity: critical / high / medium / low / info.
Status: **fixed** (this branch) / recorded (documented here + `BACKLOG.md`).
A repro for each is reproducible from the commands quoted in "Evidence".

## Findings register

| ID | Sev | Title | File:line | Status |
|----|-----|-------|-----------|--------|
| AD-01 | critical | Truncated-Gaussian dispersions collapse to constants (2 of 4 frozen) | `sim/montecarlo/dispersions.py:60-72,93-103` | **fixed** |
| AD-18 | high | IMU bias-instability dispersed as unbounded Gaussian → ~52% of MC runs crash (`scale < 0`) | `sim/montecarlo/dispersions.py` (sensors), `sim/gnc/sensors.py:119-120` | **fixed** |
| AD-02 | high | Second-order TVC actuator numerically unstable at 100 Hz (limit cycle) | `sim/vehicle/actuator.py` | **fixed** |
| AD-03 | medium | J3 zonal-gravity term suppressed ~1/r by an extra factor (effectively zero) | `sim/environment/gravity.py` | **fixed** |
| AD-04 | medium | Flex-body model is inert — gyro coupling written after EKF reads it | `sim/main.py` | **fixed** (live + notch, ADR 0012) |
| AD-05 | medium | Gain-schedule 2× discontinuity in all PID gains at q = 100 Pa | `sim/gnc/control.py` | **fixed** |
| AD-06 | medium | `fts_triggered` always False in telemetry/summary (wrong source object) | `sim/telemetry/recorder.py` | **fixed** |
| AD-07 | medium | `total_correction_budget` underestimates ~50% for elliptical orbit | `sim/orbital/maneuvers.py` | **fixed** |
| AD-08 | medium | `validate_throttle` counts every post-burnout coast tick as a violation | `sim/safety/boundary_enforcer.py` | **fixed** |
| AD-09 | medium | Staging SEPARATION abort gets stuck in an infinite ABORT loop | `sim/vehicle/staging.py` | **fixed** |
| AD-10 | medium | PEG coefficients use unclamped `T` after clamping `ratio` (inconsistent B) | `sim/gnc/guidance.py` | **fixed** |
| AD-11 | low | J5 zonal-gravity term suppressed ~1/r (same defect as J3) | `sim/environment/gravity.py` | **fixed** |
| AD-12 | low | Propulsion `mdot` not conserved across ambient pressure (1.35% at fixed throttle) | `sim/vehicle/propulsion.py` | **fixed** |
| AD-13 | low | `cop_com_margin` returns inverted static-stability polarity (unused) | `sim/vehicle/aerodynamics.py` | **fixed** |
| AD-14 | low | `compute_statistics([])` crashes on empty results (`np.max([])`) | `sim/montecarlo/statistics.py` | **fixed** |
| AD-15 | low | Downlink telemetry omits the t = 0 frame (pre-increment off-by-one) | `sim/telemetry/recorder.py` | **fixed** |
| AD-16 | low | `eci_to_ned` ignores ECI→ECEF rotation + transport term (unused) | `sim/core/reference_frames.py` | **fixed** |
| AD-17 | info | Launch azimuth ignores Earth-rotation contribution to inclination | `sim/gnc/guidance.py` | documented (accepted simplification) |
| AD-19 | medium | PEG terminal phase rides the 25° FTS attitude limit under dispersions (found via AD-17) | `sim/gnc/guidance.py` | recorded |

**Update 2026-06-14 (fidelity pass):** AD-02, AD-03, AD-05–AD-16 were fixed on
branch `claude/amazing-lamport-tn1h3b`, each with a regression test in
`tests/test_fidelity_fixes.py` and validated against an authoritative source or a
known-good numerical check (e.g. gravity vs the geopotential gradient to ~1e-10).
An end-to-end pipeline test (`tests/test_e2e_simulation.py`) was added. **AD-04**
was first attempted and reverted (a direct coupling FTS-aborts without a
structural notch); it is now **fixed** in a follow-up pass — the flex mode is
live in the control loop, stabilised by a frequency-scheduled structural notch
(see the AD-04 note below and ADR 0012). **AD-17** was investigated in the same
pass and is **documented as an accepted simplification**: the corrections work
but regress Monte-Carlo robustness through a pre-existing PEG terminal
attitude-margin fragility (now logged as AD-19); see the AD-17 note below.

## Details (verified evidence)

### AD-01 (critical, FIXED) — truncated-Gaussian dispersions collapse to constants

`sample_dispersion` drew a zero-mean offset `val = N(0, sigma)` and clipped it to
the dispersion's **absolute** `bounds` before `generate_dispersed_config` added
the nominal. Because the offsets are tiny relative to the bounds, the clip pinned
every draw to a bound:

- `CD_SCALE_FACTOR` (sigma 0.10, bounds (0.7, 1.3), nominal 1.0): 20 000 draws →
  **min = max = mean = 1.7000, std = 0, unique = 1**. Every run used CD = 1.7.
- `ATMO_DENSITY_SCALE` (sigma 0.05, bounds (0.8, 1.2)): every run used **1.8**.
- `WIND_SPEED_MS`, `GPS_POS_NOISE_M`: lower tail removed, biased high.

So **two of four** truncated dispersions had zero spread and the headline Monte
Carlo dispersion analysis never explored drag or atmospheric density at all.
(Agent correction: the auditor agent cited bounds `(0.8, 2.0)` and called the
clip "inert"; the actual bounds are `(0.7, 1.3)` and the true effect is a frozen
constant — verified by the draw histogram above.)

Fix (this branch): truncation now applies to the **final** value
(`np.clip(nominal + offset, low, high)`); the offset clip was removed. The RNG
draw sequence is unchanged (still one `normal()` per parameter), so seeding and
determinism are preserved — only the (previously degenerate) values change.
Post-fix verification: CD min/max = 0.70/1.30, std 0.10; ATMO 0.80/1.20, std
0.05; wind 0–50; GPS 1–15 — all within bounds **and** varied. Regression test:
`tests/test_dispersions.py::TestTruncatedGaussian::test_not_degenerate`.

This also corrected the `montecarlo/` package from **0% coverage** to tested.
The SLURM HPC backend added in this branch runs these same dispersions, so it
inherits the fix; its determinism guarantee (sharded == local for a seed) was
verified after the fix.

### AD-18 (high, FIXED) — IMU bias dispersion crashes ~half of all MC runs

`IMU_ACCEL_BIAS_MPS2` and `IMU_GYRO_BIAS_RADS` were dispersed as **unbounded**
Gaussians (`nominal + N(0, sigma)`), but the sensor model consumes them as the
**standard deviation** of a Gaussian draw (`sensors.py:119-120`,
`rng.normal(0, config.IMU_*_BIAS_* * sqrt(dt))`), which requires a non-negative
scale. Verified over 20 000 seeds: `IMU_ACCEL_BIAS_MPS2` is negative 31.1% of the
time and `IMU_GYRO_BIAS_RADS` 30.8%, so **52.5% of Monte Carlo runs** raise
`ValueError: scale < 0` and are recorded as `ERROR`. Found by executing a
generated sbatch worker script end-to-end (the unit tests passed because they use
a stubbed simulation). Pre-existing and independent of the SLURM backend (the
local dispatcher has the same failure rate).

Fix (this branch): make both terms `truncated_gaussian` with strictly positive
bounds — mirroring `GPS_POS_NOISE_M`, which is *already* a truncated scale
parameter for exactly this reason. RNG draw count is unchanged, so determinism
holds; the nominal run is unaffected (it applies no dispersions). Post-fix: 0%
negative over 8 000 seeds (`tests/test_dispersions.py::TestScaleParametersNonNegative`).
A complementary hardening (clamp the scale to `max(0, ·)` in `sensors.py`) is
recommended so a future custom dispersion cannot reintroduce the crash; left to
a focused PR as it touches the sensor model.

### AD-02 (high) — TVC actuator second-order integrator is unstable at 100 Hz

`actuator.py` integrates `x'' + 2ζωₙx' + ωₙ²x = ωₙ²u` with semi-implicit Euler
at `dt = 0.01 s`, but `ωₙ = 2π·25 = 157.08 rad/s` ⇒ `ωₙ·dt = 1.571`. The update
matrix has `|eigenvalues| = [3.059, 0.392]` (>1 ⇒ unstable). The rate limiter
(`±20 °/s`) prevents blow-up but converts the divergence into a **permanent
limit cycle**: a constant 1.0° command oscillates `1.000 ↔ 0.800°` forever
(peak-to-peak 0.2°, never settles over 200 steps). Sub-stepping ×10
(`ωₙ·dt_sub = 0.157`) settles to 1.000° in ~14 steps.

Impact: the documented "second-order actuator response" is non-physical — it
injects a step-rate chatter into every TVC command. The mission still succeeds
because the controller consumes the actuator's clamped *position* (±5°) and the
vehicle inertia averages the chatter, but the model output is wrong.
Fix: sub-step the ODE so `ωₙ·dt_sub ≲ 0.3`, or use the exact ZOH discrete
state-space `Φ = expm(A·dt)`. Changes nominal telemetry ⇒ regenerate example
outputs.

### AD-03 (medium) / AD-11 (low) — J3 and J5 gravity terms suppressed by ~1/r

`c_xy_j3`, `c_z_j3`, `c_xy_j5`, `c_z_j5` each carry an extra `* r_inv` that the
dimensionless J2/J4/J6 factors do not. Since all factors multiply the same
`-(mu/r²)(x/r)` prefactor, the J3/J5 factors must be dimensionless; the extra
`1/r` makes them ~`1/r ≈ 1.45e-7` too small. Term-isolated repro (J3-only code
output vs an independent numerical gradient of the J3-only geopotential):

| lat | code |a_J3| | numerical |a_J3| | ratio |
|-----|-------------|------------------|-------|
| 28.5° | 5.17e-12 | 3.04e-05 | 1.7e-7 (≈1/r) |
| 89° | 1.73e-11 | 6.80e-05 | 2.5e-7 |

So J3 (and J5) are effectively zero; the model advertises J2–J6 but delivers only
the even zonals. **This overturns the 2026-06-12 "investigated and dismissed"
J3 finding** (`audit/02-security-findings.md`): that check compared the *total*
gravity to a numerical gradient, where J3 is ~1e-6 of the signal — at the
reported finite-difference floor — so it could not distinguish a correct J3 from
a near-zero one. The coefficient (`21.0`) is fine; the `* r_inv` is the bug.
Trajectory impact is small (~1 m over a 500 s ascent; larger N/S asymmetry at
high latitude) but it is a real dimensional error nullifying a documented term.
Fix: delete `* r_inv` from `gravity.py:75-77` and `:88-89`.

### AD-04 (medium) — the flex-body model is inert

`main.py` adds the flex bending rate to the gyro **after** `ekf.predict` has
already consumed that measurement (lines 392 then 397-399), and the controller
uses `true_state.angular_velocity_body` (line ~356), never the gyro. Nothing
reads `imu_meas.gyro_body_rads` again before it is overwritten next step, and
flex contributes no force/torque to the dynamics. Proof: a 40 s / 4000-frame run
produces an **identical telemetry SHA-256** with flex on vs off
(`1f6c9a15…b80baa` both). So the entire flex subsystem has zero effect on the
simulation. (Agent correction: the auditor claimed flex *corrupts the EKF and
trips the FTS*; the identical-hash result disproves that — the contamination
never reaches a consumer.) A latent contributor: `main.py:449` calls
`flex_body.update(...)` without `modal_mass_kg`, so the default `1.0 kg` would
make the bending rates physically enormous **if** the path were ever made live.
Fix needs both: feed the gyro contamination before the EKF/controller read it
(or into the rate feedback) **and** pass a realistic modal mass — a coupled
change that alters all outputs.

**Attempted and reverted (2026-06-14).** This pass implemented the fix: a
physical modal mass (1e5 kg) and the bending rate fed into the controller's rate
feedback (one-step lag), the intended control-structure interaction. Result: the
controller chases the 1.2 Hz bending mode and saturates the TVC slew limit nearly
every step (boundary violations 245 → ~16 000), and at the full duration the
vehicle **FTS-aborts at 1.2 km** — a real flutter/limit-cycle instability.
Adding a first-order structural low-pass either still destabilised (cutoff high)
or attenuated flex back to inert (cutoff low). The proper fix is a
**frequency-scheduled structural notch filter** with gain/phase-margin analysis
(scheduled on the propellant-varying modal frequency) — a control-design task in
its own right. AD-04 is therefore left **open** and the flex path stays inert
(with this finding now empirically strengthened: naive coupling is unstable, not
merely ineffective).

**Resolved (2026-06-14, follow-up).** The flex mode is now live in the control
loop and stabilised by the frequency-scheduled structural notch the previous note
called for (`sim/gnc/notch_filter.py`, ADR 0012). The controller's pitch-rate
feedback is the measured rate (rigid + bending at the IMU, one-step lag) passed
through a cascade of RBJ notch biquads centred on the propellant-varying modal
frequencies; the excitation uses a realistic generalised modal mass
(`FLEX_MODAL_MASS_KG`). Evidence: (1) flex-on telemetry now differs from flex-off
(the inert signature is gone — hash `f592…` → `9cd2…`); (2) disabling the notch
(`FLEX_NOTCH_ENABLED=False`) reproduces the flutter — TVC clamp events jump ~200×
over a truncated ascent (`tests/test_e2e_simulation.py`); (3) it is
robustness-neutral — a 24-seed dispersed sweep matches the baseline exactly
(20/24, same aborts). The notch frequency response is unit-tested
(`tests/test_notch_filter.py`: full rejection at the modal frequency, unity at
DC/Nyquist).

### AD-05 (medium) — gain-schedule discontinuity at q = 100 Pa

`_schedule_gains` uses `q_factor = clamp(q_ref/q, 0.3, 3.0)` for `q > 100 Pa`,
else a flat `1.5`. Just above 100 Pa, `q_ref/q = 10000/100 = 100 → clamp 3.0`;
at/below 100 Pa it is `1.5`. So crossing 100 Pa steps **all three PID gains by
2×** (1.5 → 3.0), both on the way up (late gravity turn) and on the way down
(leaving the atmosphere). The docstring's stated "1.5× vacuum boost" disagrees
with the 3.0× the branch actually produces near 100 Pa. Fix: make `q_factor`
continuous across 100 Pa (e.g. cap the ratio branch at the 1.5 boost value, or
blend), so there is no step.

### AD-06 (medium) — `fts_triggered` is always False in telemetry

`recorder._build_frame` and `_build_summary` read
`getattr(boundary_enforcer, "fts_triggered", False)`, but `BoundaryEnforcer` has
no such attribute (`hasattr → False`); the `FlightTerminationSystem` that owns
`fts_triggered` is never passed to the recorder. Every frame and the
SHA-256-hashed mission summary therefore report `fts_triggered = false` even
after an abort. (Same class of bug as the prior Q-01 `health_status` defect.) The
Monte Carlo `fts_trigger_time` field is computed separately in the loop and is
correct; only the recorder's copy is wrong. Fix: thread the FTS instance into
`record`/`write_output` and read `fts.fts_triggered`.

### AD-07 (medium) — `total_correction_budget` underestimates elliptical orbits

The altitude term uses `r_achieved = achieved.semi_major_axis_m` as the Hohmann
departure radius while the eccentricity term circularizes at **periapsis**. When
SMA ≈ target these are inconsistent: for peri = 142 km, apo = 658 km
(SMA-alt = 400 km, target = 400 km), the code returns `dv_altitude = 0` and a
total of **147.4 m/s**, whereas a consistent "circularize at periapsis, then
Hohmann 142→400 km" costs **297.7 m/s** — a 50% underestimate. The docstring
acknowledges a "first-order budget", but the SMA-vs-periapsis inconsistency makes
it under-count whenever the achieved orbit is elliptical. Fix: use the periapsis
radius as the Hohmann reference (`EARTH_RADIUS_M + periapsis_alt`).

### AD-08 (medium) — `validate_throttle` inflates the violation count while coasting

When `propellant_remaining_kg <= 0` the method increments `violation_count`
unconditionally and returns, even for a legitimate `throttle_cmd = 0.0` (engine
commanded off during coast). 4000 coast calls at 100 Hz → 4000 false violations.
`total_boundary_violations` in telemetry/summary becomes dominated by normal
post-burnout coasting and can no longer flag real clamp events. (Not seen in the
nominal run, which inserts before any coast, but present whenever a run coasts.)
Fix: only count a violation when `throttle_cmd > 0` with no propellant.

### AD-09 (medium) — staging SEPARATION abort is a stuck infinite loop

In `StagingPhase.SEPARATION`, if `_safe_to_separate()` is false the method
`return`s the ABORT event **without** changing `_phase`/`_phase_elapsed` or
shutting the engine. The next `update()` re-enters SEPARATION, the interlock
fails again, and ABORT is emitted forever — no recovery, no S2 ignition. Verified:
forcing SEPARATION with an active engine yields 10/10 ABORT events with phase
unchanged. Off-nominal path (nominal flight reaches SEPARATION only after thrust
has decayed), but a single stuck-thrust fault is unrecoverable. Fix needs a
maintainer decision on recovery semantics (re-enter TAIL_OFF and force shutdown,
vs latch a fault) — recorded rather than guessed.

### AD-10 (medium) — PEG uses unclamped `T` after clamping `ratio`

In `_update_peg_coefficients`, when `T > tau` the code clamps `ratio = T/tau` to
0.95 but keeps the unclamped `T` in `b1`, `c0`, `c1`, `rhs1`, `rhs2`. The
implied effective burn time (`0.95·tau`) is inconsistent with `T`, corrupting the
`B` steering coefficient (~40% error in the agent's tau = 300 s, T = 340 s case);
inner iterations partially recover `A` but not `B`, biasing the radial steering
during early terminal guidance. This is distinct from the known Q-05
(`except UnboundLocalError`) PEG item. Fix: use `T_eff = ratio*tau` consistently,
or hold previous coefficients when `T ≥ 0.95·tau`.

### AD-12 (low) — propulsion mdot not conserved across ambient pressure

`thrust_at_pressure` and `isp_at_pressure` interpolate thrust and Isp
independently with the same pressure fraction; since their SL→vac fractional
changes differ (10.5% vs 9.3%), `mdot = thrust/(Isp·g0)` drifts: SL = 2461 kg/s,
vac = 2494 kg/s (1.35%). At fixed throttle the propellant flow should be ~constant
(only thrust changes with back-pressure). ~1.4% propellant-budget error over a
long SL burn. Fix: derive one interpolation from the other so `F/(Isp·g0)` is
constant.

### AD-13 (low) — `cop_com_margin` inverted polarity (unused)

`cop_com_margin = com_offset_from_nose - cop_offset_from_nose` with docstring
"Positive = statically stable", but static stability requires CoP **aft** of CoM
(`cop_from_nose > com_from_nose`), i.e. the expression is negative when stable.
The force model in `compute_aero_forces` is correct; only this helper's sign is
backwards. No callers outside the module (verified by grep), so no runtime impact
— but an API correctness trap. Fix: flip the subtraction or correct the docstring.

### AD-14 (low) — `compute_statistics([])` crashes

`np.max([])`/`np.mean([])` on the empty `peak_q`/`clamps` arrays raise
`ValueError: zero-size array to reduction…`; the `p99` lines are guarded by
`if n > 0` but the sibling `mean`/`max` are not. A campaign with zero results
crashes summary generation. Fix: extend the `n > 0` guard to all reductions.

### AD-15 (low) — downlink telemetry omits the t = 0 frame

`record()` appends the frame, increments `_step_count`, then checks
`_step_count % ratio == 0`. At step 0 the count is already 1, so the first
downlink frame is emitted at step 9 (t = 0.09 s); the t = 0 ignition state is
absent from the downlink stream (internal stream is fine). Fix: test the modulo
before incrementing (or use a 0-based counter).

### AD-16 (low) — `eci_to_ned` ignores the ECI→ECEF rotation (unused)

`eci_to_ned` applies the ECEF-defined NED rotation directly to an ECI velocity
without first rotating ECI→ECEF (and ignores the `ω×r` transport term), giving a
~121 m/s error at t = 300 s for a 7.8 km/s velocity. No callers (verified by
grep), so latent/dead. Fix: add `time_s`, rotate the velocity to ECEF (minus
transport term) before the NED projection.

### AD-17 (info) — launch azimuth ignores Earth-rotation contribution

`main.py`/`guidance.py` set `sin(az) = cos(inc)/cos(lat)` — the inertial azimuth
— without correcting for the launch-site eastward velocity (`ω·R·cos(lat)`). The
nominal run targets 51.6° but achieves 45.28° inclination, contributing to the
~995 m/s "correction dv" reported at insertion. A standard simplification, not a
code defect, but a targeting-fidelity gap worth noting.

**Investigated and documented as an accepted simplification (2026-06-14).** Three
fixes were prototyped: (a) the rotating-frame azimuth correction
`atan2(v·sinAz_in − ωR cos lat, v·cosAz_in)` → 45.28°→46.56°, dv ~995→910; (b)
holding the target inertial plane → 46.80°; (c) adding active out-of-plane
velocity nulling (yaw steering) → **51.4°, correction-dv ~266 m/s** — i.e. the
target is reachable, with peak loads unchanged. **But all three systematically
regress Monte-Carlo robustness**: a 48-seed dispersed sweep drops 39→35 SUCCESS
for (a) alone (the new aborts are a strict superset of the baseline aborts), and
worse for (c). The cause is **AD-19**: the PEG terminal phase already rides the
25° FTS attitude limit under dispersions (every dispersed abort observed is a
marginal 25.0x° thrust-axis-error hit near insertion), so any trajectory-plane
change tips more marginal seeds over. AD-17 is therefore **accepted as a
simplification** and the launch-azimuth code is left at the inertial relation;
re-attempt once AD-19 (PEG terminal attitude-margin hardening) lands.

### AD-19 (medium) — PEG terminal phase rides the 25° FTS attitude limit

Found during the AD-17 investigation. Under Monte-Carlo dispersions ~19% of runs
FTS-abort, and **every observed dispersed abort is a thrust-axis (attitude) error
of 25.0x° just over the `FTS_ATTITUDE_LIMIT_DEG = 25°` limit, late in the S2 burn
near insertion** — i.e. the PEG terminal transient operates right at the abort
threshold with no margin (nominal flight tracks to <5°, so the limit is correct;
the dispersed terminal transient is the problem). This is the dominant
Monte-Carlo failure mode and it blocks AD-17 (any trajectory change perturbs the
marginal population). Fix needs a PEG terminal-guidance hardening pass (smoother
gravity-turn→PEG handover, command-rate vs low-propellant control authority near
cutoff) — a dedicated control-design task, recorded rather than guessed.

## Positive controls (verified clean)

- **EKF (`gnc/navigation.py`) and sensors (`gnc/sensors.py`)**: 38 isolated
  repros covering the F-matrix Jacobian, Van Loan process noise Q, Joseph-form
  covariance update, Kalman gain, per-component innovation gating, quaternion
  normalization, IMU specific-force projection, bias random-walk scaling,
  GPS/baro availability, and shared-RNG cross-sensor correlation — all consistent
  with reference derivations. No findings.
- **Integrator / atmosphere / J2 / J4 / J6 / reference-frame round-trips**:
  no new defects (J2/J4/J6 verified against the numerical geopotential gradient;
  only the odd zonals J3/J5 carry the `r_inv` bug).
- The **SLURM HPC backend** added in this branch reproduces the local dispatcher
  exactly for a given seed (round-trip determinism test), i.e. it neither adds
  nor hides any of the above.

## Disposition

**AD-01** and **AD-18** were fixed here (both with regression tests) because
they are intrinsic to the Monte Carlo subsystem this branch scales out: a
cluster campaign over frozen dispersions (AD-01) or with ~half the runs crashing
(AD-18) would be a broken feature. Both fixes live in `montecarlo/dispersions.py`,
preserve the RNG draw sequence (so determinism holds), and do not touch the
nominal run. Every other finding changes nominal physics/telemetry outputs (and
would require regenerating committed example artifacts) or needs a maintainer
design decision; following the discipline of the prior audit ("changed no
physics"), they are **recorded** here and in `BACKLOG.md` with verified repros
and concrete fixes, for separate focused PRs.
