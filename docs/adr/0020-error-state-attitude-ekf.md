# 0020. Error-state attitude EKF + star-tracker attitude aiding

- Status: accepted
- Date: 2026-06-22
- Deciders: GNC eng (cross-repo lessons pass, `audit/05` X-11)
- Related: ADR 0013 (NIS gate, retained), ADR 0002 (scalar-last quaternion),
  ADR 0016/0018 (config schema / context-local override)

> **Stage 2 update (2026-06-22).** `USE_ESTIMATED_ATTITUDE` now defaults to
> `True` (and is an overridable config parameter): guidance, the controller, and
> the FTS fly on the EKF's estimated attitude. Gated by a paired Monte-Carlo
> campaign over 24 dispersed seeds run on both authorities — **true attitude and
> estimated attitude scored an identical 16/24 success rate.** Closing the loop
> introduced two FTS false trips (seeds 56, 63) and two false saves (43, 44) —
> symmetric, net-neutral, within Monte-Carlo noise — and shifted insertion by
> ~2.4 km on average (≈0.6 % of orbit). The absolute abort rate (~33 %) is
> dominated by the `IMU_*_BIAS` dispersions (drawn up to ~7× nominal), which
> inflate the EKF position covariance past the 10 km FTS limit during the
> GPS-denied coast; that mechanism is **identical for both authorities** and is a
> pre-existing robustness property, not a consequence of using the estimate. See
> BACKLOG for the follow-up (revisit IMU-dispersion realism / FTS covariance
> limit / GPS reacquisition above 60 km). Conclusion: the estimate is good enough
> to fly the loop with no systematic regression.

## Context and Problem Statement

The previous 12-state EKF (`sim/gnc/navigation.py`) estimated
`[pos, vel, accel_bias, gyro_bias]` but **did not estimate attitude** — the true
quaternion was handed to it every step via `set_attitude()`. Its position/velocity
dead-reckoning therefore propagated on a *perfect* attitude, and the nominal
mission only closed because of that. The cross-repo audit (`audit/05`, the PX4
ECL/EKF2 lesson) flagged this as the central navigation-realism gap: a
self-contained estimator must determine attitude from the IMU and aiding sensors,
and pay the resulting uncertainty.

Two coupled facts drive this ADR:

1. **Attitude observability.** With a strapdown IMU, attitude is observable only
   through an external reference. GPS gives it indirectly via the
   specific-force/velocity coupling `δv̇ = -[f_n]_x δθ` (a wrong attitude rotates
   the sensed force into a velocity error GPS can see) — but only on the two axes
   perpendicular to the specific force, never roll about it, and only while GPS is
   available.
2. **The GPS model cuts out at the COCOM ceiling (60 km).** For an orbital
   trajectory the upper stage then flies the entire ~200 s coast/burn GPS-denied.
   A self-contained filter dead-reckoning attitude on a MEMS-grade gyro through
   that window diverges badly: measured nominal-run attitude error reached
   **23.5°**, and the attitude→velocity→position covariance coupling inflated the
   EKF position uncertainty past the **10 km FTS limit**, aborting the mission at
   t≈336 s. (The 12-state filter never saw this because it was fed truth.)

The realistic fix is the avionics a real upper stage actually carries: a **star
tracker** that measures inertial attitude directly. It complements GPS — usable
above the sensible atmosphere at low slew rates (the GPS-denied regime) — and
observes *all three* attitude axes, including roll.

## Decision

**1. Replace the 12-state filter with a 15-error-state multiplicative EKF**
(Sola 2017; Groves 2013 Ch. 14; Titterton & Weston; cross-checked against PX4
ECL/EKF2). The filter carries a nominal attitude quaternion propagated from the
*measured* gyro (Savage two-sample coning/sculling) plus a 3-DOF multiplicative
attitude error `δθ` in a 15×15 covariance `[δp, δv, δb_a, δb_g, δθ]`. Corrections
apply additively to pos/vel/bias and multiplicatively to the quaternion with an
error-state reset; covariance uses the Joseph form; the NIS innovation gate
(ADR 0013) is retained. `set_attitude()` is removed.

**2. Add a star-tracker attitude reference; keep the COCOM GPS ceiling.** A new
`StarTracker` sensor measures the inertial attitude quaternion with
arcsecond-class noise (`STAR_TRACKER_NOISE_ARCSEC`), available only above the
sensible atmosphere (`STAR_TRACKER_MIN_ALT_M`) and below an image-smear slew rate
(`STAR_TRACKER_MAX_RATE_RADS`) — i.e. exactly the upper-stage regime where GPS is
COCOM-denied. The EKF gains a 3-DOF attitude update (`update_star_tracker`) that
observes `δθ` directly. The GPS ceiling stays at 60 km
(`GPS_AVAILABILITY_CEILING_M`, configurable; set `+inf` for a cleared receiver).
Below 60 km GPS aids; above it the star tracker keeps attitude bounded.

**3. Two-stage rollout via `USE_ESTIMATED_ATTITUDE`.** A config flag selects
whether guidance, the controller, and the FTS consume the *estimated* attitude
(full closed-loop realism) or the true attitude while the estimator still runs in
parallel for validation/telemetry.
- **Stage 1 (this change):** flag `False`. The estimator is validated *in
  isolation* and the nominal mission is re-validated on the true attitude — a
  low-risk landing that proves the filter without yet trusting it in the loop.
- **Stage 2 (follow-up):** flag `True` by default — the estimate drives the loop —
  gated by Monte-Carlo abort-rate and FTS false-trip analysis.

## Validation

- **Error-state transition Jacobian** — `_error_state_transition` is checked
  against a central-difference Jacobian of the nominal one-step propagation
  (`tests/test_ekf.py::TestErrorStateTransition`); the dominant couplings
  (`F[δv,δθ] = -[f_n]_x dt`, `F[δv,δb_a] = -R dt`, `F[δθ,δb_g] = -R dt`) match to
  <1e-6.
- **Attitude observability/convergence** — an injected 10° error is driven below
  2° by GPS aiding (in-motion alignment); a star-tracker update corrects a yaw
  *and* a roll error (roll being unobservable to GPS) to <10% in a few updates.
- **Filter consistency** — the mean GPS NIS over many updates sits inside the
  χ²(6) consistency band (Bar-Shalom NIS test).
- **Star-tracker sensor** — availability gating (atmosphere / slew rate) and
  arcsecond noise magnitude verified.
- **System** — the nominal run still reaches the same orbit (SUCCESS, 487.9 s);
  the one move of note is `peak_ekf_uncertainty_m` ≈ 982 m → ≈ 1700 m (attitude
  is now estimated, not truth-fed, so the GPS-denied position dead-reckoning
  covariance is a little larger — still far below the 10 km FTS limit). A
  Stage-2 preview with the estimate in the loop also succeeds (peak ≈ 1660 m, no
  FTS trip). Golden baseline and `examples/output/` artifacts regenerated. Full
  `pytest`, `ruff`, and `pyright` pass.

## Consequences

- **Positive:** the EKF is now a genuine, self-contained navigation filter with
  estimated attitude and IMU-bias observability — the audit's central nav gap is
  closed, and the trajectory no longer depends on truth attitude being injected.
- **Positive:** the sensor suite matches real upper-stage avionics (GPS below the
  COCOM ceiling + star tracker above it); attitude — including roll — stays
  observable for the whole flight without weakening the GPS realism.
- **Positive:** GPS availability and the star-tracker parameters are explicit,
  documented config, not buried magic numbers.
- **Cost:** a new sensor and measurement model, plus its availability/noise
  characterisation, are added and maintained. The star tracker is modelled
  simply (no sun/Earth-keep-out exclusions or lost-in-space acquisition lag);
  those are recorded as possible future fidelity.

## Notes / Evidence

Cross-repo rationale: `audit/05-cross-repo-lessons.md` (X-11). Filter math:
`sim/gnc/navigation.py` module docstring. Decisive experiment: with the 60 km
GPS cutoff and no attitude reference the self-estimating nominal run FTS-aborts at
t≈336 s (attitude error 23.5°, covariance 10 km); adding the star tracker keeps
attitude bounded and the run succeeds with peak position uncertainty ≈ 1700 m.
