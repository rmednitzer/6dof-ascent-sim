# 0024. Inclination targeting — azimuth correction + yaw plane-steering (AD-17)

- Status: accepted
- Date: 2026-06-22
- Deciders: GNC eng (audit AD-17 / AD-19)
- Related: ADR 0021 + 0023 (N-01 — cleared the AD-19 blocker), ADR 0014 (FTS
  attitude limit)

## Context and Problem Statement

The ascent reached only **~45° inclination against the 51.6° target** (AD-17). Two
gaps: (1) the launch azimuth used the inertial relation `sin Az = cos i / cos lat`
with no Earth-rotation correction, and (2) nothing actively held the target
inertial orbital plane — the gravity turn and PEG just follow the current velocity
direction. The 6.6° miss inflated the post-insertion plane-change budget to
~995 m/s.

The fix was **prototyped during the original audit and reached 51.4°**, but was
**deferred**: it "systematically regressed Monte-Carlo robustness" because of
**AD-19** — the PEG terminal phase rode the 25° FTS attitude limit under
dispersions (~19% of runs FTS-aborted on marginal ~25° thrust-axis hits near
insertion), so any trajectory-plane change tipped more marginal seeds over. AD-17
was explicitly recorded as "re-attempt once AD-19 (PEG terminal attitude-margin
hardening) lands."

**AD-19 has since been resolved by the N-01 work**: ADR 0021 (S2 Isp margin —
removed the propellant-shortfall terminal tumbles that *were* the 25° hits) and
ADR 0023 (ground-station tracking — bounded coast covariance). The dispersed abort
rate is now 0/24 with the terminal phase off the limit. So AD-17 is unblocked.

## Decision

Add inclination targeting in `sim/gnc/guidance.py`:

1. **Earth-rotation azimuth correction.** The gravity-turn downrange now uses the
   *flight* azimuth `atan2(v·sin Az_in − ωR cos lat, v·cos Az_in)` — the
   ground-relative heading whose inertial velocity (flight velocity + the
   launch-site eastward rotation) lands in the target plane.
2. **Target inertial-plane normal.** Computed once from the inertial azimuth and
   the launch position; fixed over the short ascent.
3. **Terminal yaw out-of-plane steering.** The S2 terminal phase rotates the
   commanded thrust to null the inertial out-of-plane velocity `v·n`, by an angle
   proportional to it (`GUIDANCE_PLANE_STEER_GAIN`) and clamped to
   `GUIDANCE_MAX_YAW_DEG` (12°). Steering is terminal-phase only (vacuum), so it
   adds no atmospheric side loads. The clamp keeps the *commanded* attitude
   trackable, so the FTS sees the small tracking error, not the commanded yaw.

`insertion_inclination_deg` is now a first-class `MonteCarloResult` field so the
golden can pin it.

## Validation

- **Inclination 44.98° → 51.04°** (target 51.6°). The ~0.5° residual to the
  51.53° target plane (the 0.07° from 51.6° is geodetic-vs-geocentric latitude) is
  the out-of-plane velocity not fully nulled by the insertion-detection epoch;
  increasing the gain/clamp does not close it (the burn ends first).
- **Correction-dv 995 → 74 m/s** — the on-orbit plane-change budget all but
  vanishes.
- **Peak loads unchanged** (peak-q 91%, axial-g 5.40) — steering is exo-atmospheric.
- **Robustness holds — 24/24 (0% abort)**, the decisive test: the trajectory-plane
  change no longer tips marginal seeds over, confirming the AD-19 blocker is gone.
- **Tests** — `tests/test_guidance.py`: the rotation correction steers more
  northerly; the target-plane normal matches the target inclination; the yaw
  steering opposes out-of-plane velocity (and is a no-op in-plane). Golden
  re-baselined with the inclination pinned; `examples/output/` regenerated. Full
  `pytest`, `ruff`, `pyright` pass.

## Consequences

- **Positive:** AD-17 is closed — the vehicle reaches the target inclination, and
  inclination is now a tracked, golden-pinned output. The reported correction-dv
  reflects real residual targeting error, not a known azimuth bias.
- **Positive:** demonstrates the N-01 margin work paid a second dividend — AD-19
  was the stated blocker, and with it gone the previously-regressing fix lands
  cleanly.
- **Cost / limitations:** the steering is a simple clamped-proportional law applied
  only in the terminal phase, leaving a ~0.5° / 74 m/s residual; tightening it
  further (earlier/gravity-turn steering, or targeting the osculating inclination
  directly) is possible future fidelity. Station/azimuth assume the north-east
  track.

## Notes / Evidence

Audit: `audit/04-adversarial-findings.md` AD-17 (now resolved) and AD-19 (resolved
via N-01). Decisive result: the same azimuth + yaw plane-steering that dropped
39→35 SUCCESS in the original 48-seed sweep now holds 24/24, because the N-01 work
moved the terminal phase off the 25° FTS limit.
