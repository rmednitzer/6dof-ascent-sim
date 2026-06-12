# 0002. Scalar-last `[x, y, z, w]` quaternion convention

- Status: accepted
- Date: 2026-06-12 (backfilled)
- Deciders: original authors

## Context and Problem Statement

Quaternions appear throughout attitude representation, control, navigation, FTS,
and telemetry. Two conventions are common: scalar-first `[w, x, y, z]` (used by
some texts and `scipy`-adjacent code) and scalar-last `[x, y, z, w]` (used by
`scipy.spatial.transform.Rotation`). Mixing them silently corrupts rotations.

## Decision

Use scalar-last `[x, y, z, w]` everywhere. All quaternion helpers in
`sim/core/reference_frames.py` unpack `x, y, z, w = q`, and the convention is
stated in `CLAUDE.md`, `README.md`, `docs/architecture.md`, and
`sim/core/state.py`.

## Consequences

- Positive: matches the most widely used Python rotation library, easing future
  interop.
- Negative: requires discipline in docstrings. The audit found one drifted
  docstring (`sim/safety/fts.py`, finding D-01) that mislabeled the order as
  `[w, x, y, z]`; it was corrected. A lint/test cannot easily catch prose drift,
  so reviewers must watch for it.

## Notes / Evidence

Verified this session: `reference_frames.quat_to_dcm/body_to_eci/...` all unpack
scalar-last; `grep` across the tree shows scalar-last in every authored doc and
docstring after the D-01 fix.
