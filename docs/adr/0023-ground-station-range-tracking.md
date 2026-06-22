# 0023. Ground-station range tracking for the GPS-denied coast (N-01)

- Status: accepted
- Date: 2026-06-22
- Deciders: GNC eng (BACKLOG N-01)
- Related: ADR 0020 (COCOM GPS ceiling + star tracker — kept), ADR 0021/0022
  (earlier N-01 fixes)

## Context and Problem Statement

After the S2 Isp margin (ADR 0021) and the realistic IMU dispersion (ADR 0022),
the dispersed estimated-attitude campaign sat at 23/24. The lone residual
(seed 56) was a per-seed diagnosis dead end for the earlier fixes: its EKF
position covariance reached the 10 km FTS limit **even with a nominal IMU**, because
its high-drag trajectory lengthens the GPS-denied coast.

The root cause is structural, not a bad sensor draw: above the COCOM GPS ceiling
(60 km) the COTS receiver is denied, the barometer is gone above 40 km, and the
star tracker aids *attitude only*. So for the entire ~300 s upper-stage coast there
is **no position aiding at all** — position and velocity dead-reckon on the IMU and
the covariance grows monotonically (nominal peak ≈1.7 km; the longest/most-lofted
dispersions reach the 10 km limit). ADR 0020 already flagged this as the open
follow-up. Neither a propellant nor a dispersion tweak addresses it; the coast
needs a position-fixing measurement.

## Decision

**Add a ground-station range-tracking network** (`sim/gnc/sensors.py`
`GroundStation`, `config.GROUND_STATIONS`). Each station ranges the vehicle as a
radar / transponder target and the EKF gains a nonlinear slant-range update
(`update_ground_range`).

Crucially this is **independent of GPS**: the vehicle is a *tracked* target, not a
self-locating receiver, so ground tracking is **not bound by the COCOM limit** and
keeps aiding through the GPS-denied coast. This preserves ADR 0020's deliberate
COCOM realism (the onboard GPS receiver still cuts out at 60 km) rather than
reverting it — and it mirrors how real range-safety nav actually works (the Range
Safety position solution is the ground tracking solution).

Model:
- Stations at fixed geodetic locations along the Eastern-Range north-east track —
  KSC (launch site) + Bermuda — covering the coast by line of sight.
- Each contributes a slant-range measurement (1-sigma `GROUND_RANGE_NOISE_M`
  = 30 m, coarser than GPS) at `GROUND_TRACK_UPDATE_HZ` (5 Hz) while the vehicle is
  above `GROUND_TRACK_ELEV_MASK_DEG` (5°).
- Measurement model `ρ = |r − r_station|`; Jacobian = the line-of-sight unit
  vector on the position-error block. A single range constrains the line of sight;
  the two stations' differing look-angles (and the line-of-sight sweep as the
  vehicle moves) multilaterate the full 3-D position.

## Validation

- **Covariance bounded** — nominal peak EKF position uncertainty **1.7 km →
  0.52 km**; across the dispersed campaign it stays ≈0.3–0.8 km (was up to 14 km).
- **Abort rate** — fresh 24-seed estimated-attitude campaign: **4.2% → 0%**
  (23/24 → **24/24**); seed 56 recovers (14.1 km → 0.57 km).
- **Tests** — `tests/test_ekf.py`: the range update reduces position covariance
  and drives the estimate to the measured range; station visibility (tracks an
  overhead vehicle, no track below the horizon mask).
- **Nominal** — still SUCCESS; golden re-baselined for the lower coast covariance
  (and the small trajectory shift from guidance consuming a sharper position
  estimate); `examples/output/` regenerated. Full `pytest`, `ruff`, `pyright` pass.

## Consequences

- **Positive:** EKF position covariance is bounded for the whole flight, the way a
  real launch range tracks the vehicle — closing the N-01 residual and the
  position-aiding gap ADR 0020 left open. Dispersed abort rate reaches 0/24.
- **Positive:** ADR 0020's GPS/COCOM realism is preserved — ground tracking is a
  separate, independent system, not a relaxed GPS ceiling.
- **Cost:** a new sensor + nonlinear EKF update to maintain. Station geometry is
  modelled simply (geocentric elevation mask, no terrain/atmospheric refraction,
  no range-rate or angle measurements, perfect station ephemeris); those are
  possible future fidelity. Coverage assumes the chosen stations see the
  north-east track; a very different launch azimuth would want a different station
  set.

## Notes / Evidence

N-01 is now fully resolved across ADR 0021 (Isp margin), 0022 (IMU dispersion),
and 0023 (ground tracking): dispersed abort rate 33% → 0%. The decisive A/B was
seed 56 reaching 10 km with a nominal IMU — isolating the unbounded-coast
mechanism from the sensor-draw one — which range aiding bounds to ≈0.6 km.
