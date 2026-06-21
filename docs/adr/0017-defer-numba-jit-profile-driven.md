# 0017. Evaluate numba-JIT for the hot loop; defer (profile-driven)

- Status: accepted
- Date: 2026-06-21
- Deciders: Sim eng (cross-repo lessons pass, `audit/05` X-13)

## Context and Problem Statement

`audit/05` X-13 (from genesis-world §B4 / quadrants §C4) proposed numba-JIT'ing
the RK4 hot loop plus scratch-buffer reuse for a large speedup. quadrants §C4
also stresses *profiling before optimizing*. There was no end-to-end performance
measurement in the repo (`benchmark.py` only micro-benchmarked the flex/slosh
models), so the premise was untested.

## Decision

Profile first, then act on the evidence.

A `cProfile` of a representative powered-ascent slice shows **no dominant hot
spot**: the RK4 integration path (`rk4_step` + `_apply_state_dot` +
`derivatives_fn`) is **~8 %** of runtime. The remaining cost is spread across the
EKF `predict` (the single largest leaf), the force model (gravity J2–J6, the
Mach-spline aero, wind), the reference-frame transforms, and **~460 k small
`numpy.array` allocations** per 120 s of flight. The integrator itself is already
allocation-conscious.

Therefore:

1. **Do not** add numba as a hard dependency. JIT'ing only the integrator buys
   ~8 % for a heavy LLVM/llvmlite dependency; capturing a real speedup would
   require JIT'ing the entire force model + EKF (a large, high-risk refactor of
   safety-relevant code that calls into SciPy).
2. **Add an end-to-end benchmark harness** (`benchmark.py --full` / `--profile`):
   wall-time, µs/step, steps/s, and a `cProfile` breakdown — so optimisation
   targets are always chosen from evidence and perf regressions are visible.
3. **Apply one safe, bit-identical micro-optimisation** the profile supports:
   hoist the zero-order-held total force and angular acceleration out of the
   per-sub-stage RK4 closure (they were re-summed/re-divided on each of the four
   sub-stage calls). Verified bit-identical — the nominal telemetry SHA-256 and
   committed example artifacts are unchanged.

The real performance lever is **X-14** (batched / vectorised Monte Carlo over a
leading run dimension), which removes per-trajectory Python/allocation overhead
wholesale; numba-JIT of the force model remains a possible future effort, but
only behind a profile that justifies it.

## Consequences

- Positive: performance is now measurable and trackable; no heavy dependency was
  added on the strength of an unverified "10–50×" expectation; the micro-opt is
  provably behaviour-preserving (golden trajectory unchanged).
- Negative / deferred: single-run speed is essentially unchanged (the micro-opt
  is small). Meaningful gains wait on X-14 or a dedicated, profile-justified
  force-model JIT.

## Notes / Evidence

`benchmark.py` (`--full`, `--profile`); micro-opt in `sim/main.py` (derivatives
closure). Smoke test: `tests/test_benchmark.py`. Profile method:
`python benchmark.py --profile --t-max 120`. Cross-repo rationale:
`audit/05-cross-repo-lessons.md` §B4 / §C4 / §G (X-13, X-14).
