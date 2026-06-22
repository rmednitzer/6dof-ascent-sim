# 0025. Post-separation ullage-settling coast (cold staging)

- Status: accepted
- Date: 2026-06-22
- Deciders: GNC / vehicle eng
- Related: ADR 0021 (staging propellant margin), AD-09 (separation interlock)

## Context and Problem Statement

A review of the staging coast timing found the sequence was
`TAIL_OFF (1 s) → COAST (1 s) → SEPARATION → S2_IGNITION`, i.e.:

- **MECO → separation = 2.0 s** (tail-off + coast) — reasonable; real two-stage
  vehicles take ~2–4 s to let S1 thrust decay before parting (Falcon 9 ≈ 3 s), and
  the 5 %-thrust interlock correctly gates separation.
- **separation → S2 ignition = 0.0 s** — S2 was commanded at the *instant* of
  separation. This vehicle stages **cold** (it coasts unpowered before parting), and
  a cold-staged upper stage normally coasts **~2–5 s after** separation before SES
  so the spent stage clears (no recontact / plume impingement) and the
  upper-stage propellant settles at the tank outlet before the turbopump spins up
  (Falcon 9 SES ≈ 3 s after sep). Lighting at the separation instant skips both,
  and putting the only coast *before* the mass drop is backwards versus practice.

The sim models neither ullage settling nor recontact (point-mass propellant
ledger), so this was a **timeline-fidelity** gap rather than a functional bug — its
only modelled effect is a few seconds of gravity loss — but the 0 s post-sep coast
was unrealistic.

## Decision

Insert a **`SETTLING` phase** (`POST_SEP_COAST_DURATION = 3 s`) between
`SEPARATION` and `S2_IGNITION`, and defer `s2_engine.ignite()` to the *end* of
that coast. The pre-separation tail-off + coast (~2 s) is unchanged. The sequence
is now:

```
TAIL_OFF (1 s) → COAST (1 s) → SEPARATION → SETTLING (3 s) → S2_IGNITION (0.5 s)
```

So MECO → separation = 2 s, separation → S2 ignition = 3 s, MECO → S2 thrust ≈
5.5 s — in line with real cold-staging practice.

## Validation

- **Timeline** — nominal run: MECO 164.6 s, separation 166.6 s (+2.0 s), S2
  ignition 169.6 s (+3.0 s), S2 thrust 170.1 s. A new
  `tests/test_fidelity_fixes.py::TestPostSeparationCoast` pins that S2 is not
  ignited at separation and lights only after the settling coast.
- **Nominal** — still SUCCESS; the ~3 s coast lofts the staging arc slightly and
  pushes insertion ~2.5 s later (fpa 0.70°→1.12°, t 493.8→496.3 s). Golden
  re-baselined; `examples/output/` regenerated.
- **Robustness** — 24-seed dispersed campaign holds **24/24** (no regression).
- Full `pytest`, `ruff`, `pyright` pass.

## Consequences

- **Positive:** the staging timeline matches real cold-staging — a settling/
  clearance coast after separation, not an instantaneous relight — and the coast
  phases are correctly ordered (pre-sep tail-off, post-sep settling).
- **Cost / limitation:** the underlying settling and recontact physics are still
  not modelled; `SETTLING` is a timeline coast (the vehicle flies unpowered),
  not a propellant-settling simulation. The 3 s value is a representative
  cold-stage figure, not tuned to a specific vehicle.

## Notes / Evidence

Found via a direct coast-timing review. The change is purely in the staging state
machine (`sim/vehicle/staging.py`); S2 thrust still applies only once the sequence
completes, so deferring `ignite()` to the end of `SETTLING` preserves the existing
S2 start-up ramp behaviour, merely inserting the coast ahead of it.
