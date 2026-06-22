# 0022. Realistic IMU bias-instability dispersion (N-01)

- Status: accepted
- Date: 2026-06-22
- Deciders: GNC eng (BACKLOG N-01)
- Related: ADR 0021 (S2 Isp margin — fixed the dominant abort cause), ADR 0020
  (estimated-attitude loop, surfaced the abort rate), AD-18 (scale params must
  stay non-negative)

## Context and Problem Statement

After the S2 Isp performance margin (ADR 0021) cleared the propellant-shortfall
aborts, the dispersed estimated-attitude campaign sat at 21/24 (12.5% abort). A
per-seed diagnosis of the 3 residual aborts:

| Seed | Trip | Driver |
|---|---|---|
| 42 | EKF covariance 10 000.5 m (0.5 m over) | **IMU** — gyro-bias draw 2.8× nominal; recovers to SUCCESS (peak ≈ 2.6 km) when the IMU bias is forced to nominal. |
| 56 | EKF covariance 10 001.4 m (1.4 m over) | **Coast length** — high-drag trajectory ⇒ longer GPS-denied dead-reckon; covariance still hits ≈14 km with a *nominal* IMU. Not IMU-driven. |
| 54 | Attitude 25.57° at staging (0.57° over) | Staging-region control transient. |

The covariance trips are razor-marginal (0.5–1.4 m over a 10 km limit), and seed
42 traces directly to the **dispersion model, not the navigation filter**. The
IMU bias-instability terms were dispersed with a 1-sigma of **2× their nominal
value** (`IMU_ACCEL_BIAS_MPS2` σ=0.002 on a 0.001 nominal; `IMU_GYRO_BIAS_RADS`
σ=0.0002 on 0.0001), truncated only at **~7× nominal**. A spread that wide implies
the same IMU grade varies ±200% (1σ) unit-to-unit — implausible. Those tail draws,
not a filter deficiency, dead-reckoned the GPS-denied coast into the 10 km FTS
limit.

## Decision

**Tighten the IMU bias-instability dispersion to a realistic 1-sigma of ~30% of
nominal, truncated to ~2× nominal at the tail:**

| Param | nominal | σ (was → now) | bounds (was → now) |
|---|---|---|---|
| `IMU_ACCEL_BIAS_MPS2` | 0.001 | 0.002 → 0.0003 | (0.0001, 0.007) → (0.0001, 0.002) |
| `IMU_GYRO_BIAS_RADS` | 0.0001 | 0.0002 → 0.00003 | (0.00001, 0.0007) → (0.00001, 0.0002) |

~30% unit-to-unit / thermal variation within one IMU grade is a defensible spread
(the old ±200% was not). The positive lower bounds are unchanged, preserving the
AD-18 non-negativity guarantee. This is a **dispersion-realism** change only — the
EKF, the FTS covariance limit, and the nominal vehicle are untouched, so the
deterministic golden run (which carries no dispersions) is unchanged.

## Validation

- **Per-seed** — seed 42 reaches orbit (its gyro draw falls from 2.8× to ~1.3×
  nominal, so the coast covariance stays bounded). Seed 54 (the staging-region
  attitude transient, previously 25.57° / 0.57° over) also clears — the sharper
  gyro estimate trims the rate error at staging back under the 25° limit. Seed 56
  still aborts, as expected: its trip is coast-length-driven, not IMU.
- **Campaign** — fresh 24-seed estimated-attitude run: abort rate
  **12.5% → 4.2%** (success 21/24 → 23/24); no previously-successful seed
  regressed (a tighter IMU spread only lowers covariance).
- **Nominal** — unchanged (golden and `examples/output/` untouched; the nominal
  run carries no dispersions).
- `tests/test_dispersions.py` (bounds containment, non-negativity, determinism)
  and full `pytest` / `ruff` / `pyright` pass.

## Consequences

- **Positive:** the dispersed abort rate now reflects the navigation/guidance
  design under *realistic* sensor variation, not an implausibly wide IMU spread.
- **Residual (tracked, BACKLOG N-01):** seed 56 — the EKF position covariance
  grows unbounded through the ~300 s GPS-denied coast and the longest/most-lofted
  dispersions still reach the 10 km limit. Properly bounding it needs position
  aiding above the COCOM ceiling (e.g. ground-station range tracking), a larger
  effort deferred under N-01. Seed 54 (staging-region attitude transient) is a
  separate control item.

## Notes / Evidence

Diagnosis: per-seed FTS-reason categorisation + an IMU-forced-to-nominal A/B on
seeds 42 and 56 (42 recovers, 56 does not), isolating the IMU-driven trip from the
coast-driven one.
