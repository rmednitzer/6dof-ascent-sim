# Running Monte Carlo on a SLURM HPC cluster (experimental)

This guide covers `sim/montecarlo/hpc.py`, which distributes a Monte Carlo
ascent campaign across a SLURM cluster using a **job array**. It produces
results **identical** to the local dispatcher (`python -m sim.montecarlo.dispatcher`)
for the same base seed — sharding only changes *where* runs execute, never
*what* is computed.

> **Status: experimental.** The API and on-disk shard format
> (`schema_version = 1`) may change. Submission is gated behind `--submit` and
> degrades to a script-only dry run where SLURM is absent.

## Model

```
                 partition N runs into M chunks (runs_per_task each)
   submit  ──────────────────────────────────────────────────────────►
                 │
                 ▼
      ┌──────────────────── SLURM job array (--array=0-(M-1)) ───────────────────┐
      │  task 0            task 1            task 2        ...        task M-1     │
      │  runs [0,R)        runs [R,2R)       runs [2R,3R)             runs [.,N)  │
      │  shard_00000.json  shard_00001.json shard_00002.json         shard_*.json│
      └────────────────────────────────────────────────────────────────────────┘
                 │  (--dependency=afterok)
                 ▼
              collect  ──►  montecarlo_results.json  +  statistics summary
```

Each array task **regenerates the full deterministic run list** and executes
only its slice. Run `i` always uses seed `base_seed + i`, so no run inputs are
shipped between nodes and any task/shard is independently reproducible.

## Quick start

### 1. Dry run (no cluster needed) — generate the scripts

```bash
python -m sim.montecarlo.hpc submit \
    --runs 5000 --runs-per-task 100 --seed 42 \
    --output-dir /scratch/$USER/mc_run1 \
    --partition cpu --account myproj --time 00:30:00 --mem 2G
```

This writes `mc-ascent_array.sbatch` and `mc-ascent_collect.sbatch` into the
output directory and prints the exact `sbatch` commands to run. Nothing is
submitted. Inspect the scripts, then submit by hand or re-run with `--submit`.

### 2. Submit on a SLURM login node

```bash
python -m sim.montecarlo.hpc submit \
    --runs 5000 --runs-per-task 100 --seed 42 \
    --output-dir /scratch/$USER/mc_run1 \
    --partition cpu --account myproj --time 00:30:00 \
    --cpus-per-task 4 --max-concurrent 50 \
    --preamble "module load python/3.11" \
    --preamble "source /scratch/$USER/venv/bin/activate" \
    --submit
```

`--submit` submits the array job and a dependent collect job that runs only
`afterok` the array succeeds. `--max-concurrent K` caps simultaneously running
array tasks (`--array=0-(M-1)%K`). `--preamble` lines (repeatable) run before
the worker — use them to load modules and activate the environment that has
this package installed.

### 3. Check progress

```bash
python -m sim.montecarlo.hpc status            # squeue --me
squeue --me                                    # equivalently
```

### 4. Collect manually (if you skipped the dependent job)

```bash
python -m sim.montecarlo.hpc collect \
    --runs 5000 --runs-per-task 100 --seed 42 \
    --output-dir /scratch/$USER/mc_run1
```

`collect` fails loudly if any run index is missing (e.g. a task hit its
walltime), listing the gaps so you can re-run just those array indices:

```bash
sbatch --array=17,42 mc-ascent_array.sbatch    # re-run only the failed tasks
```

Use `--allow-incomplete` to aggregate a partial campaign anyway.

## Per-task parallelism

Set `--cpus-per-task N` (SLURM) together with `--local-workers N` (worker) to
use a node's cores *within* each array task via `multiprocessing`. Trade-off:
fewer, fatter tasks reduce scheduler load; more, thinner tasks improve
backfill. A good starting point is `runs_per_task` ≈ a few × `cpus_per_task`.

## Reproducibility and sizing

- **Determinism.** `(num_runs, seed, dispersions)` fully determines every run.
  The same three values on a laptop and on the cluster give identical results;
  verified in `tests/test_hpc_slurm.py::TestShardCollectRoundTrip`.
- **Sizing.** A nominal run is ~26 s. Pick `runs_per_task` so a task finishes
  comfortably inside `--time`. Example: at ~26 s/run, `--runs-per-task 100`
  serially is ~43 min → request `--time 01:00:00`, or set `--cpus-per-task 4
  --local-workers 4` to bring it to ~11 min.

## Output layout

```
<output-dir>/
├── mc-ascent_array.sbatch      # generated array (worker) script
├── mc-ascent_collect.sbatch    # generated dependent collect script
├── slurm_logs/                 # %x_%A_%a.out / .err per array task
├── shard_00000.json            # one shard per array task (atomic write)
├── shard_00001.json
├── ...
└── montecarlo_results.json     # written by `collect` (same schema as local)
```

`montecarlo_results.json` is identical in schema to the local dispatcher's
output, so existing `sim.montecarlo.statistics` analysis/plots apply unchanged.

## Programmatic API

```python
from sim.montecarlo.hpc import CampaignSpec, SlurmConfig, submit_campaign, collect_shards

spec = CampaignSpec(num_runs=5000, seed=42, runs_per_task=100,
                    output_dir="/scratch/me/mc_run1")
slurm = SlurmConfig(partition="cpu", account="myproj", time_limit="00:30:00",
                    cpus_per_task=4, max_concurrent=50,
                    preamble=("module load python/3.11", "source venv/bin/activate"))

plan = submit_campaign(spec, slurm, submit=True)   # submit=False => dry run
print(plan.array_job_id, plan.collect_job_id)

results = collect_shards(spec)                      # after the array finishes
```

## Security note

Submission shells out to `sbatch`/`squeue` with a fixed argv list and
`shell=False`; there is no command-injection surface. The only shell expansion
embedded in a generated script is `${SLURM_ARRAY_TASK_ID}`. Submission happens
only when SLURM is detected **and** `--submit` is passed, so importing or
dry-running the module never touches the scheduler.
