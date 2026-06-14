# 0010. SLURM HPC Monte Carlo via job array + shard/collect

- Status: accepted (experimental)
- Date: 2026-06-14
- Deciders: this audit/feature session

## Context and Problem Statement

The local Monte Carlo dispatcher (ADR 0004) parallelises a campaign across the
cores of a single machine with `multiprocessing.Pool`. A full nominal run takes
~26 s wall-clock, so the default `MC_NUM_RUNS = 1000` campaign is ~1.8 h on a
4-core box and scales no further than one node. Dispersion studies want
10³–10⁵ runs, which needs a cluster. The target HPC environment runs SLURM,
the dominant batch scheduler.

The workload is embarrassingly parallel and each run is already independent and
deterministically seeded, so the question is purely *how to distribute and
re-aggregate* without changing numerical results.

## Decision

Add an **experimental** SLURM backend (`sim/montecarlo/hpc.py`) using a SLURM
**job array** with a **shard/collect** model, rather than MPI or a third-party
distributed framework (Dask/Ray):

1. **Partition** the `N` runs into `M = ceil(N / runs_per_task)` contiguous
   chunks, one per array task (`#SBATCH --array=0-(M-1)`).
2. Each array task **regenerates the full deterministic config list** and runs
   only its slice, writing `shard_<task_id>.json` to a shared filesystem. The
   config list is a pure function of `(num_runs, seed, dispersions)` — run `i`
   always gets `seed + i` — so a task needs no shipped inputs and shards are
   reproducible.
3. A dependent **collect** job (`--dependency=afterok:<array_job_id>`)
   aggregates the shards into `montecarlo_results.json` + statistics, failing
   loudly on any missing run unless `--allow-incomplete` is passed.

Submission is a **dry run by default**: scripts are always written, and
`sbatch` is invoked only when SLURM is detected *and* `--submit` is given.
Cluster placement (partition, account, walltime, memory) lives in a `SlurmConfig`
dataclass, deliberately *not* in `sim/config.py`, which remains the source of
truth for simulation physics (ADR 0001); only the campaign-level
`MC_RUNS_PER_TASK` and `MC_HPC_OUTPUT_DIR` defaults are added there.

## Consequences

- Positive: scales to as many nodes as the array allows; reuses the existing
  dispatcher and `MonteCarloResult` unchanged; **bit-for-bit identical** to a
  local run for the same base seed (proven by a round-trip test).
- Positive: no new runtime dependency. SLURM is optional; the whole workflow
  (partition, script generation, shard/collect) is unit-tested without a
  cluster by monkeypatching the simulation. Lifts `montecarlo/` from 0% coverage.
- Positive: minimal new attack surface. `sbatch`/`squeue` are invoked with a
  fixed argv list and `shell=False`; the only shell expansion embedded in a
  generated script is `${SLURM_ARRAY_TASK_ID}`. (The prior audit noted "no
  subprocess" as a positive control; this is the one controlled exception, and
  it is gated behind explicit `--submit`.)
- Negative: shards are JSON on a shared filesystem, so very large campaigns
  create many small files. Acceptable at 10³–10⁵; a future revision could switch
  to a chunked binary/Parquet store.
- Negative: it inherits the global-config-override fragility of ADR 0004 (Q-02)
  via the same `run_simulation` path; ADR 0009 would fix both.

## Alternatives considered

- **MPI (`mpi4py`)**: adds a heavyweight dependency and a tighter coupling than
  an embarrassingly-parallel sweep needs; harder to restart partial campaigns.
- **Dask/Ray**: capable, but a large dependency and a second scheduler layered
  on top of SLURM; overkill for independent runs.
- **One SLURM job per run** (`N` jobs instead of `M` array tasks): swamps the
  scheduler at high `N`; the array with `runs_per_task` batching is the standard
  HPC idiom and supports `%K` concurrency throttling.

## Notes / Evidence

Determinism verified by `tests/test_hpc_slurm.py::TestShardCollectRoundTrip`:
running every array task and collecting yields `asdict`-equal results to the
local dispatcher's config list for the same seed. Partitioning is proven to
cover every run index exactly once. See `docs/hpc-slurm.md` for operations.
