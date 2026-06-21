# 0013. Chi-square (NIS) innovation gating in the navigation EKF

- Status: accepted
- Date: 2026-06-21
- Deciders: GNC eng (cross-repo lessons pass, `audit/05`)

## Context and Problem Statement

The 12-state EKF (`sim/gnc/navigation.py`) fuses GPS (6 components) and
barometer (1 component) measurements. The original innovation gate rejected a
measurement if **any single component** of the residual exceeded
`EKF_RESIDUAL_SIGMA_THRESHOLD = 3.0` times that component's standard deviation
(`√S[i,i]`). This uses only the diagonal of the innovation covariance `S`, so it
ignores the cross-covariance between, e.g., position and velocity components,
and applies a per-axis magnitude test rather than a joint consistency test. It is
also dimension-blind: the same per-component 3σ is far stricter for a 6-component
GPS update than for a 1-component baro update.

PX4's EKF2 (`audit/05`, finding A1.2) gates on the **normalised innovation
squared** (NIS), the standard estimator-consistency statistic.

## Decision

Gate on the Mahalanobis distance `NIS = yᵀ S⁻¹ y`, which is chi-square
distributed with `len(y)` degrees of freedom under the filter-consistency
hypothesis. Reject the measurement when `NIS` exceeds the chi-square quantile at
`EKF_INNOVATION_GATE_P` (default `0.9973`), computed via `scipy.stats.chi2.ppf`
and cached per `(dim, p)` pair. Additionally, reject any non-finite innovation or
innovation covariance as a counted fault (`measurement_rejections`) instead of
letting a NaN reach the state (defence in depth; the integrator also guards).
`EKF_RESIDUAL_SIGMA_THRESHOLD` is removed.

## Consequences

- Positive: a single principled, dimension-aware gate that accounts for the full
  innovation covariance; the 1-DOF case (`P = 0.9973`) reproduces the previous
  3σ intent exactly (`chi2.ppf(0.9973, 1) = 9.0 = 3²`), so the baro gate is
  unchanged.
- Positive: non-finite measurements are now an explicit, counted rejection.
- Positive: `last_nis` and `measurement_rejections` are exposed for Monte-Carlo
  health analysis (no telemetry-schema change).
- Neutral: adds a `scipy.stats` import to the navigation module (SciPy is already
  a project dependency); `chi2.ppf` is cached, and GPS/baro update rates are low
  (≤10 Hz), so cost is negligible.
- Risk: a different accept/reject decision on the seeded nominal run would change
  nominal telemetry. Verified after the change: the nominal end-to-end run is
  unchanged (committed example artifacts did not need regeneration). [If a future
  tuning of `EKF_INNOVATION_GATE_P` changes nominal telemetry, regenerate the
  example outputs per `BACKLOG` D-06.]

## Notes / Evidence

`sim/gnc/navigation.py:_apply_update` and `_nis_gate_threshold`; config
`EKF_INNOVATION_GATE_P`. Regression tests in `tests/test_ekf.py`
(`TestEKFInnovationGate`): an in-family measurement is accepted, a gross outlier
is rejected and counted, a non-finite measurement is rejected, and a zero-DOF
edge is handled. Cross-repo rationale: `audit/05-cross-repo-lessons.md` §A1.2.
