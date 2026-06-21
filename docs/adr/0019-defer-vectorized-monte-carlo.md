# 0019. Defer vectorized Monte Carlo (X-14) — spike evidence

- Status: accepted (decision: defer, evidence recorded)
- Date: 2026-06-21
- Deciders: Sim eng (cross-repo lessons pass, `audit/05` X-14)
- Related: ADR 0018 (unblocks this), ADR 0017 (same profile-first / defer
  discipline), ADR 0010 (SLURM HPC backend)

## Context and Problem Statement

`audit/05` X-14 proposed running Monte-Carlo trajectories batched over a leading
dimension (numpy/GPU) instead of one-per-process, to remove per-trajectory
Python overhead. ADR 0018 (context-local config) removed the global mutation that
made this impossible. A time-boxed spike was run to decide go/no-go.

## Spike result

`spike_vectorized_mc.py` integrates N trajectories of a representative
translational core (J2 gravity + exponential-atmosphere drag, RK4) two ways and
compares throughput:

| Path | Throughput | |
|------|-----------:|--|
| Vectorized batch (N over a leading dim) | ~626 traj/s | |
| Sequential Python loop (one at a time) | ~2.8 traj/s | |
| **Vectorization speedup** | **~220×** | vs 1 core; ~55× vs a 4-core multiprocessing pool, on this fully-vectorizable core |

So the ceiling is **high**, not low — this is the opposite of the numba finding
(ADR 0017). Vectorizing the parts that *can* vectorize is very fast.

## Decision

**Defer.** Despite the attractive ceiling, do not pursue vectorized MC now:

1. **Cost.** Capturing the speedup requires making the *entire* per-step stack
   batch-aware — the force model (gravity, the Mach-spline aero, wind),
   propulsion, the 12-state EKF (per-trajectory 12×12 linear algebra), the
   sensors, and the GNC — a multi-week rewrite of safety-tested code. The spike
   covers only the translational force math.
2. **Control flow caps the gain.** Trajectories terminate at *different* steps
   (FTS abort, insertion detection) and branch differently (staging FSM, max-q /
   G throttle limits). A batched/SIMD approach must run all N to the maximum
   duration behind per-row masks, wasting compute on already-terminated
   trajectories and eroding the 220× ceiling.
3. **Throughput at scale is already solved horizontally.** The multiprocessing
   dispatcher (ADR-0004) scales near-linearly across cores, and the SLURM HPC
   backend (ADR-0010) scales across a cluster — both without rewriting the
   physics. The problem X-14 targets is largely already addressed.

## Recommendation (when to revisit)

Pursue only if **single-node** MC throughput becomes a proven, recurring
bottleneck not met by multiprocessing + SLURM. At that point prefer a dedicated,
staged **GPU/torch batched-sim backend** (a single-threaded CPU-numpy batch
already shows 220×; a GPU would dwarf that at campaign scale) rather than an
incremental CPU change. The prerequisite — per-run config with no global mutation
(ADR 0018) — is already in place, so the door is open whenever the need is real.

## Consequences

- Positive: no large, high-risk rewrite undertaken on the strength of a ceiling
  number; the 220× evidence is recorded so the decision is revisitable with data.
- Positive: closes the Tier-2 X-14 question with evidence (go/no-go = defer),
  mirroring the discipline of ADR 0017.
- The spike script is committed as reproducible evidence; it is not part of the
  product (no test/CI gate), like a one-off measurement.

## Notes / Evidence

`spike_vectorized_mc.py` (run: `python spike_vectorized_mc.py`). Cross-repo
rationale: `audit/05-cross-repo-lessons.md` §B2 / §G (X-14). Baseline throughput
context: `benchmark.py --full` (~1,300 steps/s/process).
