"""Experimental: distribute a Monte Carlo campaign across a SLURM HPC cluster.

This module extends the local :class:`~sim.montecarlo.dispatcher.MonteCarloDispatcher`
(which parallelises across the cores of a single machine) to a SLURM cluster
using a **job array**. A campaign of ``N`` runs is partitioned into ``M`` chunks,
one per array task. Each array task regenerates the *full* deterministic run
configuration list and executes only its own slice, writing a JSON **shard**.
A dependent *collect* step aggregates the shards into the combined results and
statistics — exactly what the local dispatcher would have produced.

Why this works (determinism)
----------------------------
The local dispatcher assigns run ``i`` the seed ``base_seed + i`` and derives
that run's dispersed configuration from a generator seeded with it
(:meth:`MonteCarloDispatcher.generate_run_configs`). The mapping
``run_index -> (seed, dispersed_config)`` is therefore a pure function of the
campaign ``(num_runs, seed, dispersions)`` and is independent of *where* the run
executes. A sharded campaign reconstructs the same list on every task and slices
it, so the aggregated output is bit-for-bit identical to a single-machine run
with the same base seed. No run inputs need to be shipped between nodes.

Operational model
-----------------
``submit``    -- partition the runs, write an ``sbatch`` array script (and a
                dependent collect script), and optionally submit them. The
                default is a *dry run*: scripts are written and the commands
                that would execute are printed, so the workflow is usable and
                testable on a machine without SLURM.
``run-task``  -- the per-array-task worker (invoked inside the SLURM job). Reads
                ``$SLURM_ARRAY_TASK_ID``, runs its slice, writes ``shard_<id>.json``.
``collect``   -- aggregate all shards into ``montecarlo_results.json`` plus the
                statistics summary; fail loudly if any run is missing.
``status``    -- thin ``squeue`` wrapper (no-op message if SLURM is absent).

This is an **experimental** feature. Submission shells out to ``sbatch`` only
when SLURM is detected *and* ``--submit`` is passed; arguments are passed as an
argv list (never through a shell), so there is no command-injection surface.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from sim import config
from sim.montecarlo.dispatcher import MonteCarloResult, _run_single
from sim.montecarlo.dispersions import DEFAULT_DISPERSIONS, Dispersion

logger = logging.getLogger(__name__)

SHARD_SCHEMA_VERSION = 1
SHARD_GLOB = "shard_*.json"


# --------------------------------------------------------------------------- #
# Run partitioning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunChunk:
    """A contiguous slice of run indices assigned to one SLURM array task.

    Attributes:
        task_id: Zero-based array task index (the ``SLURM_ARRAY_TASK_ID``).
        start: First run index (inclusive).
        stop: One past the last run index (exclusive).
    """

    task_id: int
    start: int
    stop: int

    @property
    def count(self) -> int:
        """Number of runs in this chunk."""
        return self.stop - self.start

    @property
    def run_indices(self) -> range:
        """The run indices this task is responsible for."""
        return range(self.start, self.stop)


def partition_runs(num_runs: int, runs_per_task: int) -> list[RunChunk]:
    """Split ``num_runs`` runs into contiguous chunks of at most ``runs_per_task``.

    Every run index in ``[0, num_runs)`` appears in exactly one chunk; the final
    chunk absorbs the remainder. The number of chunks equals the SLURM array
    size and is ``ceil(num_runs / runs_per_task)``.

    Args:
        num_runs: Total number of Monte Carlo runs (must be >= 1).
        runs_per_task: Maximum runs per array task (must be >= 1).

    Returns:
        Ordered list of :class:`RunChunk`, ``task_id`` running ``0..M-1``.

    Raises:
        ValueError: If either argument is not a positive integer.
    """
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    if runs_per_task < 1:
        raise ValueError(f"runs_per_task must be >= 1, got {runs_per_task}")

    chunks: list[RunChunk] = []
    for task_id, start in enumerate(range(0, num_runs, runs_per_task)):
        stop = min(start + runs_per_task, num_runs)
        chunks.append(RunChunk(task_id=task_id, start=start, stop=stop))
    return chunks


# --------------------------------------------------------------------------- #
# Campaign + cluster configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CampaignSpec:
    """Defines *what* to simulate — the campaign is fully reproducible from this.

    These are the only parameters that affect numerical results. The
    ``(num_runs, seed, dispersions)`` triple plus the simulation code uniquely
    determine every run, which is what lets independent tasks reproduce their
    slices. Cluster placement lives separately in :class:`SlurmConfig`.

    Attributes:
        num_runs: Total Monte Carlo runs in the campaign.
        seed: Base random seed; run ``i`` uses ``seed + i``.
        runs_per_task: Runs executed per SLURM array task.
        output_dir: Directory for shards and aggregated outputs (shared FS).
        local_workers: Worker processes *within* one array task (use the
            task's allocated CPUs). ``1`` runs the slice serially.
        no_flex: Disable the flex-body model (mirrors ``main.py --no-flex``).
        no_slosh: Disable the slosh model (mirrors ``main.py --no-slosh``).
    """

    num_runs: int = config.MC_NUM_RUNS
    seed: int = config.MC_SEED
    runs_per_task: int = config.MC_RUNS_PER_TASK
    output_dir: str = config.MC_HPC_OUTPUT_DIR
    local_workers: int = 1
    no_flex: bool = False
    no_slosh: bool = False

    def __post_init__(self) -> None:
        # Validate eagerly so a bad spec fails at submit time, not on the node.
        partition_runs(self.num_runs, self.runs_per_task)
        if self.local_workers < 1:
            raise ValueError(f"local_workers must be >= 1, got {self.local_workers}")

    @property
    def chunks(self) -> list[RunChunk]:
        """The full partition of this campaign into array tasks."""
        return partition_runs(self.num_runs, self.runs_per_task)

    @property
    def num_tasks(self) -> int:
        """SLURM array size (number of array tasks)."""
        return len(self.chunks)


@dataclass(frozen=True)
class SlurmConfig:
    """Cluster placement and resource directives for the ``sbatch`` scripts.

    These are site-specific and deliberately kept out of ``sim/config.py``
    (which is the single source of truth for *simulation physics*, not
    infrastructure). Defaults are conservative and partition-agnostic; set
    ``partition``/``account`` for your site.

    Attributes:
        job_name: SLURM job name.
        partition: Partition/queue name (``--partition``). ``None`` omits it.
        account: Accounting/allocation (``--account``). ``None`` omits it.
        qos: Quality-of-service (``--qos``). ``None`` omits it.
        time_limit: Wall-clock limit per array task (``HH:MM:SS``).
        mem: Memory per task (e.g. ``"2G"``).
        cpus_per_task: CPUs per array task (set ``CampaignSpec.local_workers``
            to use them).
        max_concurrent: Throttle simultaneously-running array tasks (``%K``).
        log_subdir: Sub-directory (under the output dir) for ``.out``/``.err``.
        python_executable: Interpreter the job runs (defaults to this one).
        preamble: Shell lines emitted before the worker (``module load ...``,
            ``source .../activate``), in order.
        mail_type / mail_user: Optional SLURM email notifications.
        extra_directives: Raw extra ``#SBATCH`` directives, e.g.
            ``("--constraint=ib", "--exclusive")``.
    """

    job_name: str = "mc-ascent"
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    time_limit: str = "01:00:00"
    mem: str = "2G"
    cpus_per_task: int = 1
    max_concurrent: int | None = None
    log_subdir: str = "slurm_logs"
    python_executable: str = field(default_factory=lambda: sys.executable or "python3")
    preamble: tuple[str, ...] = ()
    mail_type: str | None = None
    mail_user: str | None = None
    extra_directives: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmissionPlan:
    """Result of preparing (and possibly submitting) a campaign.

    Attributes:
        array_script: Path to the written array (worker) sbatch script.
        collect_script: Path to the written dependent collect sbatch script.
        array_command: argv that submits (or would submit) the array job.
        collect_command_template: human-readable collect submission, with the
            array job id substituted only after a real submission.
        submitted: Whether jobs were actually handed to ``sbatch``.
        array_job_id: SLURM job id of the array job, if submitted.
        collect_job_id: SLURM job id of the collect job, if submitted.
    """

    array_script: Path
    collect_script: Path
    array_command: list[str]
    collect_command_template: list[str]
    submitted: bool = False
    array_job_id: str | None = None
    collect_job_id: str | None = None


class IncompleteCampaignError(RuntimeError):
    """Raised by :func:`collect_shards` when run results are missing."""

    def __init__(self, missing: list[int], found: int, expected: int):
        self.missing = missing
        self.found = found
        self.expected = expected
        preview = missing[:10]
        more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        super().__init__(
            f"Incomplete campaign: {found}/{expected} runs collected; "
            f"missing run indices {preview}{more}. "
            f"Re-run the failed array tasks or pass allow_incomplete=True."
        )


# --------------------------------------------------------------------------- #
# SLURM detection
# --------------------------------------------------------------------------- #
def slurm_available() -> bool:
    """Return True iff the ``sbatch`` submission command is on PATH."""
    return shutil.which("sbatch") is not None


# --------------------------------------------------------------------------- #
# sbatch script generation
# --------------------------------------------------------------------------- #
def _worker_argv(spec: CampaignSpec, task_id_expr: str) -> list[str]:
    """Build the ``run-task`` argv (the interpreter is prepended by the caller).

    ``task_id_expr`` is inserted verbatim so a shell expansion such as
    ``${SLURM_ARRAY_TASK_ID}`` survives quoting in :func:`_format_command`.
    """
    args = [
        "-m",
        "sim.montecarlo.hpc",
        "run-task",
        "--task-id",
        task_id_expr,
        "--num-runs",
        str(spec.num_runs),
        "--runs-per-task",
        str(spec.runs_per_task),
        "--seed",
        str(spec.seed),
        "--output-dir",
        str(spec.output_dir),
        "--local-workers",
        str(spec.local_workers),
    ]
    if spec.no_flex:
        args.append("--no-flex")
    if spec.no_slosh:
        args.append("--no-slosh")
    return args


def _collect_argv(spec: CampaignSpec) -> list[str]:
    """Build the ``collect`` argv (interpreter prepended by the caller)."""
    return [
        "-m",
        "sim.montecarlo.hpc",
        "collect",
        "--num-runs",
        str(spec.num_runs),
        "--runs-per-task",
        str(spec.runs_per_task),
        "--seed",
        str(spec.seed),
        "--output-dir",
        str(spec.output_dir),
    ]


def _sbatch_header(spec: CampaignSpec, slurm: SlurmConfig, *, array: bool) -> list[str]:
    """Common ``#SBATCH`` directive lines for the array and collect scripts."""
    out_dir = Path(spec.output_dir)
    log_dir = out_dir / slurm.log_subdir
    suffix = "array" if array else "collect"
    lines = ["#!/bin/bash", f"#SBATCH --job-name={slurm.job_name}-{suffix}"]
    if array:
        array_spec = f"0-{spec.num_tasks - 1}"
        if slurm.max_concurrent is not None:
            array_spec += f"%{slurm.max_concurrent}"
        lines.append(f"#SBATCH --array={array_spec}")
        lines.append(f"#SBATCH --output={log_dir}/%x_%A_%a.out")
        lines.append(f"#SBATCH --error={log_dir}/%x_%A_%a.err")
    else:
        lines.append(f"#SBATCH --output={log_dir}/%x_%j.out")
        lines.append(f"#SBATCH --error={log_dir}/%x_%j.err")
    lines.append(f"#SBATCH --time={slurm.time_limit}")
    lines.append(f"#SBATCH --mem={slurm.mem}")
    lines.append(f"#SBATCH --cpus-per-task={slurm.cpus_per_task if array else 1}")
    lines.append("#SBATCH --nodes=1")
    if slurm.partition:
        lines.append(f"#SBATCH --partition={slurm.partition}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    if slurm.qos:
        lines.append(f"#SBATCH --qos={slurm.qos}")
    if slurm.mail_type:
        lines.append(f"#SBATCH --mail-type={slurm.mail_type}")
    if slurm.mail_user:
        lines.append(f"#SBATCH --mail-user={slurm.mail_user}")
    for directive in slurm.extra_directives:
        lines.append(f"#SBATCH {directive}")
    return lines


def _format_command(python: str, args: list[str]) -> str:
    """Quote a worker/collect command for embedding in a shell script.

    Every token is shell-quoted *except* a bare ``${SLURM_ARRAY_TASK_ID}``,
    which must survive as a shell expansion.
    """
    parts = [shlex.quote(python)]
    for a in args:
        if a == "${SLURM_ARRAY_TASK_ID}":
            parts.append(a)
        else:
            parts.append(shlex.quote(a))
    return " ".join(parts)


def generate_sbatch_script(spec: CampaignSpec, slurm: SlurmConfig) -> str:
    """Render the array (worker) ``sbatch`` script as a string."""
    lines = _sbatch_header(spec, slurm, array=True)
    lines += ["", "set -euo pipefail", *slurm.preamble, ""]
    cmd = _format_command(slurm.python_executable, _worker_argv(spec, "${SLURM_ARRAY_TASK_ID}"))
    lines.append(cmd)
    return "\n".join(lines) + "\n"


def generate_collect_script(spec: CampaignSpec, slurm: SlurmConfig) -> str:
    """Render the dependent ``collect`` ``sbatch`` script as a string."""
    lines = _sbatch_header(spec, slurm, array=False)
    lines += ["", "set -euo pipefail", *slurm.preamble, ""]
    cmd = _format_command(slurm.python_executable, _collect_argv(spec))
    lines.append(cmd)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Worker: run one array task and write a shard
# --------------------------------------------------------------------------- #
def shard_path(output_dir: str | Path, task_id: int) -> Path:
    """Deterministic path of the shard written by ``task_id``."""
    return Path(output_dir) / f"shard_{task_id:05d}.json"


def _run_chunk(chunk: RunChunk, spec: CampaignSpec, dispersions: list[Dispersion]) -> list[dict]:
    """Execute every run in ``chunk`` and return MonteCarloResult dicts.

    Reconstructs the full deterministic config list (cheap — RNG draws only,
    no simulation) and slices out this task's runs, guaranteeing the seeds and
    dispersed parameters match a single-machine campaign exactly.
    """
    from sim.montecarlo.dispatcher import MonteCarloDispatcher

    dispatcher = MonteCarloDispatcher(num_runs=spec.num_runs, dispersions=dispersions, seed=spec.seed)
    all_configs = dispatcher.generate_run_configs()
    my_configs = all_configs[chunk.start : chunk.stop]

    if spec.local_workers > 1 and len(my_configs) > 1:
        import multiprocessing

        with multiprocessing.Pool(processes=min(spec.local_workers, len(my_configs))) as pool:
            return list(pool.imap(_run_single, my_configs))
    return [_run_single(cfg) for cfg in my_configs]


def run_task(
    task_id: int,
    spec: CampaignSpec,
    *,
    dispersions: list[Dispersion] | None = None,
) -> Path:
    """Run the chunk owned by ``task_id`` and write its shard. Returns the path.

    Raises:
        IndexError: If ``task_id`` is outside ``[0, num_tasks)``.
    """
    chunks = spec.chunks
    if not 0 <= task_id < len(chunks):
        raise IndexError(f"task_id {task_id} out of range for {len(chunks)} array tasks")
    chunk = chunks[task_id]
    disp = dispersions if dispersions is not None else DEFAULT_DISPERSIONS

    logger.info(
        "Array task %d: running runs [%d, %d) (%d runs)",
        task_id,
        chunk.start,
        chunk.stop,
        chunk.count,
    )
    result_dicts = _run_chunk(chunk, spec, disp)

    out = Path(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "task_id": task_id,
        "start": chunk.start,
        "stop": chunk.stop,
        "num_runs": spec.num_runs,
        "seed": spec.seed,
        "results": result_dicts,
    }
    # Atomic write: write to a temp file then rename, so a collect running
    # concurrently never reads a half-written shard.
    dest = shard_path(out, task_id)
    tmp = dest.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp, dest)
    logger.info("Array task %d: wrote %d results to %s", task_id, len(result_dicts), dest)
    return dest


# --------------------------------------------------------------------------- #
# Collect: aggregate shards
# --------------------------------------------------------------------------- #
def collect_shards(spec: CampaignSpec, *, allow_incomplete: bool = False) -> list[MonteCarloResult]:
    """Aggregate all shards in ``output_dir`` into an ordered result list.

    Args:
        spec: Campaign spec (defines the expected run indices).
        allow_incomplete: If False (default), raise
            :class:`IncompleteCampaignError` when any run is missing.

    Returns:
        Results ordered by ``run_index``.
    """
    out = Path(spec.output_dir)
    by_index: dict[int, MonteCarloResult] = {}
    shard_files = sorted(out.glob(SHARD_GLOB))
    if not shard_files:
        raise FileNotFoundError(f"No shards ({SHARD_GLOB}) found in {out}")

    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        for rec in data.get("results", []):
            result = MonteCarloResult(**rec)
            # Later shards win on duplicate indices (idempotent re-runs of a task).
            by_index[result.run_index] = result

    expected = set(range(spec.num_runs))
    missing = sorted(expected - set(by_index))
    if missing and not allow_incomplete:
        raise IncompleteCampaignError(missing, found=len(by_index), expected=spec.num_runs)

    return [by_index[i] for i in sorted(by_index)]


def write_aggregate(
    results: list[MonteCarloResult],
    output_dir: str | Path,
    filename: str = "montecarlo_results.json",
) -> Path:
    """Write aggregated results to ``output_dir/filename`` and return the path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / filename
    with open(dest, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    return dest


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #
def _parse_job_id(sbatch_stdout: str) -> str | None:
    """Extract the job id from ``sbatch`` output (``Submitted batch job 12345``)."""
    for token in sbatch_stdout.split():
        if token.isdigit():
            return token
    return None


def submit_campaign(
    spec: CampaignSpec,
    slurm: SlurmConfig,
    *,
    submit: bool = False,
    with_collect: bool = True,
) -> SubmissionPlan:
    """Write the sbatch scripts and, if requested and possible, submit them.

    Behaviour:
      * Always writes ``<output_dir>/<job_name>_array.sbatch`` and the collect
        script, plus creates the log directory.
      * If ``submit`` is False (default) -> dry run: returns the plan with the
        commands that *would* run; nothing is handed to SLURM.
      * If ``submit`` is True but SLURM is absent -> raises ``RuntimeError``.
      * If ``submit`` is True and SLURM is present -> submits the array job,
        then (optionally) the collect job with
        ``--dependency=afterok:<array_job_id>``.

    Returns:
        A :class:`SubmissionPlan` describing what was written/submitted.
    """
    out = Path(spec.output_dir)
    (out / slurm.log_subdir).mkdir(parents=True, exist_ok=True)

    array_script = out / f"{slurm.job_name}_array.sbatch"
    collect_script = out / f"{slurm.job_name}_collect.sbatch"
    array_script.write_text(generate_sbatch_script(spec, slurm))
    collect_script.write_text(generate_collect_script(spec, slurm))

    array_command = ["sbatch", str(array_script)]
    collect_command_template = [
        "sbatch",
        "--dependency=afterok:<ARRAY_JOB_ID>",
        str(collect_script),
    ]

    plan = SubmissionPlan(
        array_script=array_script,
        collect_script=collect_script,
        array_command=array_command,
        collect_command_template=collect_command_template,
    )

    if not submit:
        return plan

    if not slurm_available():
        raise RuntimeError(
            "submit=True but 'sbatch' is not on PATH. Run on a SLURM login node, "
            "or use a dry run (submit=False) to generate the scripts."
        )

    array_proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user-controlled binary
        array_command, capture_output=True, text=True, check=True
    )
    array_job_id = _parse_job_id(array_proc.stdout)
    collect_job_id = None
    if with_collect and array_job_id is not None:
        collect_command = [
            "sbatch",
            f"--dependency=afterok:{array_job_id}",
            str(collect_script),
        ]
        collect_proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            collect_command, capture_output=True, text=True, check=True
        )
        collect_job_id = _parse_job_id(collect_proc.stdout)

    return replace(
        plan,
        submitted=True,
        array_job_id=array_job_id,
        collect_job_id=collect_job_id,
    )


def squeue_status(job_name: str | None = None) -> str:
    """Return ``squeue`` output for the current user (or a message if absent)."""
    if shutil.which("squeue") is None:
        return "SLURM not available (squeue not on PATH)."
    cmd = ["squeue", "--me"]
    if job_name:
        cmd += ["--name", job_name]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.stdout or proc.stderr


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_campaign_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--runs", type=int, default=config.MC_NUM_RUNS, help="Total Monte Carlo runs")
    p.add_argument("--seed", type=int, default=config.MC_SEED, help="Base random seed")
    p.add_argument(
        "--runs-per-task",
        type=int,
        default=config.MC_RUNS_PER_TASK,
        help="Runs per SLURM array task",
    )
    p.add_argument(
        "--output-dir",
        default=config.MC_HPC_OUTPUT_DIR,
        help="Shared directory for shards and aggregated output",
    )
    p.add_argument("--no-flex", action="store_true", help="Disable flex body model")
    p.add_argument("--no-slosh", action="store_true", help="Disable slosh model")


def _spec_from_args(args: argparse.Namespace, *, local_workers: int = 1) -> CampaignSpec:
    return CampaignSpec(
        num_runs=args.runs,
        seed=args.seed,
        runs_per_task=args.runs_per_task,
        output_dir=args.output_dir,
        local_workers=local_workers,
        no_flex=getattr(args, "no_flex", False),
        no_slosh=getattr(args, "no_slosh", False),
    )


def _cmd_submit(args: argparse.Namespace) -> int:
    spec = _spec_from_args(args)
    slurm = SlurmConfig(
        job_name=args.job_name,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        time_limit=args.time,
        mem=args.mem,
        cpus_per_task=args.cpus_per_task,
        max_concurrent=args.max_concurrent,
        preamble=tuple(args.preamble or ()),
    )
    plan = submit_campaign(spec, slurm, submit=args.submit, with_collect=not args.no_collect)
    print(
        f"Campaign: {spec.num_runs} runs, {spec.num_tasks} array tasks "
        f"({spec.runs_per_task}/task), base seed {spec.seed}"
    )
    print(f"Array script:   {plan.array_script}")
    print(f"Collect script: {plan.collect_script}")
    if plan.submitted:
        print(f"Submitted array job {plan.array_job_id}; collect job {plan.collect_job_id}")
    else:
        reason = "" if slurm_available() else "  (SLURM not detected — scripts only)"
        print(f"Dry run — nothing submitted.{reason}")
        print("To submit:")
        print(f"  {shlex.join(plan.array_command)}")
        print(f"  {shlex.join(plan.collect_command_template)}  # after the array job id is known")
    return 0


def _cmd_run_task(args: argparse.Namespace) -> int:
    # SLURM exports SLURM_ARRAY_TASK_ID; --task-id overrides for manual runs/tests.
    task_id = args.task_id
    if task_id is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            raise SystemExit("No --task-id and SLURM_ARRAY_TASK_ID is unset.")
        task_id = int(env)
    spec = _spec_from_args(args, local_workers=args.local_workers)
    dest = run_task(task_id, spec)
    print(f"Task {task_id}: shard written to {dest}")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from sim.montecarlo.statistics import print_summary

    spec = _spec_from_args(args)
    results = collect_shards(spec, allow_incomplete=args.allow_incomplete)
    dest = write_aggregate(results, spec.output_dir)
    print(f"Collected {len(results)} runs -> {dest}")
    print_summary(results)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    print(squeue_status(args.job_name))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m sim.montecarlo.hpc``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Experimental: run a Monte Carlo ascent campaign on a SLURM cluster.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Generate sbatch scripts and (optionally) submit")
    _add_campaign_args(p_submit)
    p_submit.add_argument("--job-name", default="mc-ascent")
    p_submit.add_argument("--partition", default=None)
    p_submit.add_argument("--account", default=None)
    p_submit.add_argument("--qos", default=None)
    p_submit.add_argument("--time", default="01:00:00", help="Wall time per task (HH:MM:SS)")
    p_submit.add_argument("--mem", default="2G")
    p_submit.add_argument("--cpus-per-task", type=int, default=1)
    p_submit.add_argument("--max-concurrent", type=int, default=None, help="Throttle (%%K) array tasks")
    p_submit.add_argument("--preamble", action="append", help="Shell line(s) before the worker (repeatable)")
    p_submit.add_argument("--no-collect", action="store_true", help="Do not chain a collect job")
    p_submit.add_argument("--submit", action="store_true", help="Actually submit (requires SLURM)")
    p_submit.set_defaults(func=_cmd_submit)

    p_task = sub.add_parser("run-task", help="Worker: run one array task and write a shard")
    _add_campaign_args(p_task)
    p_task.add_argument("--task-id", type=int, default=None, help="Array task id (default: $SLURM_ARRAY_TASK_ID)")
    p_task.add_argument("--local-workers", type=int, default=1, help="Worker processes within this task")
    p_task.set_defaults(func=_cmd_run_task)

    p_collect = sub.add_parser("collect", help="Aggregate shards into combined results + statistics")
    _add_campaign_args(p_collect)
    p_collect.add_argument("--allow-incomplete", action="store_true", help="Do not fail on missing runs")
    p_collect.set_defaults(func=_cmd_collect)

    p_status = sub.add_parser("status", help="Show squeue status for your jobs")
    p_status.add_argument("--job-name", default=None)
    p_status.set_defaults(func=_cmd_status)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
