# 0011. Zonal-harmonic gravity via the verified analytic geopotential gradient

- Status: accepted
- Date: 2026-06-14
- Deciders: fidelity/model-quality pass

## Context and Problem Statement

The gravity model (`sim/environment/gravity.py`) applies zonal harmonics J2–J6.
The terms were hand-coded per harmonic. An adversarial re-audit (AD-03/AD-11)
found the odd zonals J3 and J5 carried an extra `1/r` factor and a sign error,
which suppressed them by ~7 orders of magnitude (effectively zero) and flipped
their sign at high latitude. The 2026-06-12 audit had *dismissed* a J3 concern
because its check compared the *total* acceleration to a numerical gradient,
where J3/J5 are ~1e-6 of the signal — at the finite-difference noise floor — so a
broken odd zonal was invisible.

## Decision

Replace the per-term hand-coded coefficients with a single closed form derived as
the exact analytic gradient of the perturbing geopotential
`U_n = -(mu/r) J_n (R_e/r)^n P_n(sinφ)`:

```
a_xy = (mu J_n (R_e/r)^n / r^2) * (x,y / r) * [(n+1) P_n(s) + s P_n'(s)]
a_z  = (mu J_n (R_e/r)^n / r^2) * [(n+1) s P_n(s) - (1 - s^2) P_n'(s)]
```

with `s = z/r` and exact Legendre polynomials `P_n`, `P_n'` (hard-coded for
n = 2..6, so the form is non-singular at the poles). One formula handles every
zonal term (even and odd) consistently.

## Consequences

- Positive: **verified correct.** A term-isolated and full-model comparison
  against a finite-difference gradient of the geopotential matches to ~1e-10
  (relative) at latitudes 0–89°, vs ~3–6e-6 for the old code. J3/J5 now
  contribute their true ~1e-5 / ~3e-6 m/s² instead of ~zero. Regression test:
  `tests/test_fidelity_fixes.py::TestGravityZonalHarmonics`.
- Positive: one consistent derivation removes the per-term transcription risk
  that produced the J3/J5 defect.
- Neutral: trajectory impact is small (~1 m over a 500 s ascent; larger N/S
  asymmetry at high latitude), so the nominal example outputs change only
  slightly.
- Cost: a 5-iteration loop per call (the function is on the 100 Hz hot path, 4×
  per RK4 step); measured overhead is negligible against the rest of the step.

## Reference

Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §8.6;
Montenbruck & Gill, *Satellite Orbits*, §3.2. Validation method and evidence:
`audit/04-adversarial-findings.md` (AD-03/AD-11).
