# 0012. Live flex/control coupling stabilised by a frequency-scheduled structural notch

- Status: accepted
- Date: 2026-06-14
- Deciders: fidelity/model-quality pass (AD-04 follow-up)

## Context and Problem Statement

The flex-body model (`sim/dynamics/flex_body.py`) computes the first three
lateral bending-mode rates sensed at the IMU, but the subsystem was **inert**
(AD-04): the bending rate was added to the gyro *after* the EKF and controller
had already read it, and the controller used the *true* body rate, so flex had
zero effect on the trajectory (flex-on and flex-off produced byte-identical
telemetry).

A naive fix — feeding the bending rate straight into the controller's rate
feedback — destabilises the vehicle: the controller chases the lightly-damped
(`ζ = 0.01`) ~1.2 Hz first bending mode, gimbals the TVC at that frequency,
re-excites the structure, and the loop flutters into a limit cycle (the classic
"tail-wags-dog" control-structure interaction). The 2026-06-14 pass verified
this empirically — a direct coupling FTS-aborts the vehicle.

## Decision

Make the flex coupling **live** and stabilise it with a **frequency-scheduled
structural notch filter** (`sim/gnc/notch_filter.py`):

1. The controller's pitch-rate feedback is the *measured* rate — rigid body rate
   plus the bending rate sensed at the IMU — with a one-step lag (the real
   sense -> compute -> actuate latency), instead of the clean true rate.
2. That measured rate passes through a cascade of second-order notch biquads
   (one per bending mode, RBJ cookbook design, bilinear-discretised at 100 Hz),
   each centred on the mode's **current** natural frequency, which drifts upward
   as propellant drains. The notch has unity gain at DC and Nyquist and a
   transmission zero at the modal frequency, so the rigid-body control band
   (~0.3 Hz) passes through while the structural modes are rejected.
3. The flex oscillators are driven with a realistic generalised modal mass
   (`FLEX_MODAL_MASS_KG`); the previous default of 1.0 kg made the modal
   response physically enormous.

The EKF deliberately stays on the clean IMU (its coning/sculling cross-products
would amplify structural vibration that is not true rigid-body rotation).

## Consequences

- Positive: **flex is now live and verified.** Flex-on telemetry differs from
  flex-off (the AD-04 inert signature is gone). Disabling the notch
  (`FLEX_NOTCH_ENABLED = False`) reproduces the instability — TVC clamp events
  jump ~200x over a truncated ascent — proving both that the coupling is real
  and that the notch is what tames it.
- Positive: **robustness-neutral.** With the notch, the nominal still inserts to
  LEO and a 24-seed dispersed sweep matches the pre-change baseline exactly
  (same successes, same aborts), because a correctly-designed notch renders the
  structure transparent to control.
- Neutral: the notch makes flex's effect on the *trajectory* small by design;
  the live coupling is demonstrated through the notch-off contrast and the
  changed telemetry hash, not through a large nominal trajectory change.
- Cost: a 3-section biquad cascade on the 100 Hz control path (negligible), plus
  two new tunables (`FLEX_MODAL_MASS_KG`, `FLEX_NOTCH_Q`). The generalised modal
  mass is a tuning parameter, not a measured property (mode-shape normalisation
  is arbitrary).
- Regression tests: `tests/test_notch_filter.py` (filter frequency response) and
  `tests/test_e2e_simulation.py::TestFlexControlStructureInteraction`.

## Reference

Wie, *Space Vehicle Dynamics and Control*, 2nd ed., Ch. 7 (structural filtering
and control-structure interaction); Robert Bristow-Johnson, "Cookbook formulae
for audio EQ biquad filter coefficients." Finding and evidence:
`audit/04-adversarial-findings.md` (AD-04).
