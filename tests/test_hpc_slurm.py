"""Tests for the experimental SLURM HPC Monte Carlo feature (sim/montecarlo/hpc.py).

These tests never run the real simulation: ``sim.main.run_simulation`` is
monkeypatched with a fast deterministic stand-in (it is imported lazily inside
``_run_single``, so patching the attribute is sufficient). That keeps the suite
fast while still exercising the full partition -> shard -> collect pipeline and,
crucially, the determinism contract that makes sharding equivalent to a local
run.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from sim import config
from sim.montecarlo import hpc
from sim.montecarlo.dispatcher import MonteCarloDispatcher, MonteCarloResult


def _fake_run_simulation(config_override=None, quiet=True):
    """Deterministic, fast stand-in for run_simulation keyed on the run's seed.

    The metrics deliberately depend on ``_seed`` and ``_run_index`` so that an
    incorrectly seeded or mis-ordered shard pipeline produces a detectable
    mismatch against the local dispatcher's config list.
    """
    co = config_override or {}
    seed = int(co.get("_seed", 0))
    idx = int(co.get("_run_index", 0))
    return MonteCarloResult(
        run_index=idx,
        seed=seed,
        outcome="SUCCESS" if seed % 7 else "FTS_ABORT",
        dispersed_params={k: v for k, v in co.items() if not k.startswith("_")},
        insertion_altitude_m=400_000.0 + seed,
        insertion_velocity_ms=7670.0 + (seed % 10),
        insertion_fpa_deg=0.1,
        peak_q_pa=30_000.0 + idx,
        peak_axial_g=5.0,
        peak_ekf_uncertainty_m=3.0,
        boundary_clamp_count=idx,
        fts_trigger_time_s=None,
        total_time_s=480.0,
    )


@pytest.fixture
def fake_sim(monkeypatch):
    """Patch the lazily-imported run_simulation with the fast stand-in."""
    import sim.main

    monkeypatch.setattr(sim.main, "run_simulation", _fake_run_simulation)
    return _fake_run_simulation


class TestPartitionRuns:
    """partition_runs must cover every run index exactly once."""

    def test_even_division(self):
        chunks = hpc.partition_runs(1000, 50)
        assert len(chunks) == 20
        assert all(c.count == 50 for c in chunks)

    def test_remainder_in_last_chunk(self):
        chunks = hpc.partition_runs(1003, 50)
        assert len(chunks) == 21
        assert chunks[-1].start == 1000
        assert chunks[-1].stop == 1003
        assert chunks[-1].count == 3

    def test_covers_all_indices_exactly_once(self):
        seen = [i for c in hpc.partition_runs(257, 16) for i in c.run_indices]
        assert seen == list(range(257))

    def test_task_ids_are_contiguous_from_zero(self):
        chunks = hpc.partition_runs(95, 10)
        assert [c.task_id for c in chunks] == list(range(len(chunks)))

    def test_single_run(self):
        chunks = hpc.partition_runs(1, 50)
        assert len(chunks) == 1
        assert chunks[0].count == 1

    def test_runs_per_task_larger_than_runs(self):
        chunks = hpc.partition_runs(5, 50)
        assert len(chunks) == 1
        assert chunks[0].count == 5

    @pytest.mark.parametrize("num_runs,rpt", [(0, 10), (-1, 10), (10, 0), (10, -3)])
    def test_invalid_inputs_raise(self, num_runs, rpt):
        with pytest.raises(ValueError):
            hpc.partition_runs(num_runs, rpt)


class TestCampaignSpec:
    """CampaignSpec validation and derived quantities."""

    def test_defaults_from_config(self):
        spec = hpc.CampaignSpec()
        assert spec.num_runs == config.MC_NUM_RUNS
        assert spec.seed == config.MC_SEED
        assert spec.runs_per_task == config.MC_RUNS_PER_TASK
        assert spec.output_dir == config.MC_HPC_OUTPUT_DIR

    def test_num_tasks(self):
        assert hpc.CampaignSpec(num_runs=1003, runs_per_task=50).num_tasks == 21

    def test_invalid_num_runs_rejected(self):
        with pytest.raises(ValueError):
            hpc.CampaignSpec(num_runs=0)

    def test_invalid_local_workers_rejected(self):
        with pytest.raises(ValueError):
            hpc.CampaignSpec(local_workers=0)


class TestSbatchGeneration:
    """sbatch script rendering."""

    def _spec_slurm(self):
        spec = hpc.CampaignSpec(num_runs=1003, seed=7, runs_per_task=50, output_dir="output/mc")
        slurm = hpc.SlurmConfig(
            job_name="camp",
            partition="cpu",
            account="proj",
            qos="normal",
            max_concurrent=8,
            cpus_per_task=4,
            preamble=("module load python", "source venv/bin/activate"),
            extra_directives=("--constraint=ib",),
        )
        return spec, slurm

    def test_array_range_and_throttle(self):
        spec, slurm = self._spec_slurm()
        script = hpc.generate_sbatch_script(spec, slurm)
        assert "#SBATCH --array=0-20%8" in script

    def test_task_id_expansion_unquoted(self):
        spec, slurm = self._spec_slurm()
        script = hpc.generate_sbatch_script(spec, slurm)
        # Must survive as a live shell expansion, not be quoted into a literal.
        assert "--task-id ${SLURM_ARRAY_TASK_ID}" in script

    def test_directives_present(self):
        spec, slurm = self._spec_slurm()
        script = hpc.generate_sbatch_script(spec, slurm)
        for token in (
            "#SBATCH --partition=cpu",
            "#SBATCH --account=proj",
            "#SBATCH --qos=normal",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --constraint=ib",
            "module load python",
            "set -euo pipefail",
        ):
            assert token in script, token

    def test_worker_command_carries_campaign(self):
        spec, slurm = self._spec_slurm()
        script = hpc.generate_sbatch_script(spec, slurm)
        assert "run-task" in script
        # The flag must match the CLI parser (which defines --runs, not --num-runs).
        assert "--runs 1003" in script
        assert "--num-runs" not in script
        assert "--seed 7" in script
        assert "--runs-per-task 50" in script

    def test_generated_commands_are_accepted_by_the_cli(self):
        """Round-trip guard: the argv embedded in the sbatch scripts must parse
        through the real CLI, or the array/collect jobs fail on the cluster with
        'unrecognized arguments'. (This is what the flag-name bug evaded.)"""
        spec = hpc.CampaignSpec(num_runs=10, seed=3, runs_per_task=4, output_dir="output/mc", no_flex=True)
        parser = hpc._build_parser()
        # Strip the interpreter prefix ("-m", "sim.montecarlo.hpc"); a concrete
        # task id stands in for ${SLURM_ARRAY_TASK_ID}.
        worker = hpc._worker_argv(spec, "0")[2:]
        collect = hpc._collect_argv(spec)[2:]
        parser.parse_args(worker)  # raises SystemExit on an unknown flag
        parser.parse_args(collect)

    def test_optional_directives_omitted_when_unset(self):
        spec = hpc.CampaignSpec(num_runs=10, runs_per_task=5)
        script = hpc.generate_sbatch_script(spec, hpc.SlurmConfig())
        assert "--partition" not in script
        assert "--account" not in script
        assert "%" not in script.split("--array=")[1].splitlines()[0]  # no throttle

    def test_collect_is_not_an_array_job(self):
        spec, slurm = self._spec_slurm()
        cs = hpc.generate_collect_script(spec, slurm)
        assert "--array" not in cs
        assert "collect" in cs
        assert "--cpus-per-task=1" in cs  # collect is single-task

    def test_no_flex_no_slosh_passthrough(self):
        spec = hpc.CampaignSpec(num_runs=10, runs_per_task=5, no_flex=True, no_slosh=True)
        script = hpc.generate_sbatch_script(spec, hpc.SlurmConfig())
        assert "--no-flex" in script
        assert "--no-slosh" in script


class TestSlurmDetectionAndParsing:
    def test_slurm_available_false_without_sbatch(self, monkeypatch):
        monkeypatch.setattr(hpc.shutil, "which", lambda _: None)
        assert hpc.slurm_available() is False

    def test_slurm_available_true_with_sbatch(self, monkeypatch):
        monkeypatch.setattr(hpc.shutil, "which", lambda _: "/usr/bin/sbatch")
        assert hpc.slurm_available() is True

    def test_parse_job_id(self):
        assert hpc._parse_job_id("Submitted batch job 123456") == "123456"
        assert hpc._parse_job_id("Submitted batch job 99 on cluster") == "99"
        assert hpc._parse_job_id("error: invalid") is None


class TestShardCollectRoundTrip:
    """The determinism contract: sharded collect == local dispatcher run."""

    def test_round_trip_matches_local(self, fake_sim, tmp_path):
        spec = hpc.CampaignSpec(num_runs=17, seed=42, runs_per_task=5, output_dir=str(tmp_path))
        assert spec.num_tasks == 4
        for chunk in spec.chunks:
            hpc.run_task(chunk.task_id, spec)

        collected = hpc.collect_shards(spec)
        assert len(collected) == 17
        assert [r.run_index for r in collected] == list(range(17))

        disp = MonteCarloDispatcher(num_runs=17, seed=42)
        truth = [_fake_run_simulation(co) for (_, _, co) in disp.generate_run_configs()]
        assert [asdict(a) for a in collected] == [asdict(b) for b in truth]

    def test_shard_payload_metadata(self, fake_sim, tmp_path):
        import json

        spec = hpc.CampaignSpec(num_runs=10, seed=1, runs_per_task=4, output_dir=str(tmp_path))
        dest = hpc.run_task(0, spec)
        payload = json.loads(dest.read_text())
        assert payload["schema_version"] == hpc.SHARD_SCHEMA_VERSION
        assert payload["task_id"] == 0
        assert payload["start"] == 0 and payload["stop"] == 4
        assert len(payload["results"]) == 4

    def test_out_of_range_task_id_raises(self, fake_sim, tmp_path):
        spec = hpc.CampaignSpec(num_runs=10, seed=1, runs_per_task=5, output_dir=str(tmp_path))
        with pytest.raises(IndexError):
            hpc.run_task(99, spec)

    def test_duplicate_task_rerun_is_idempotent(self, fake_sim, tmp_path):
        spec = hpc.CampaignSpec(num_runs=10, seed=1, runs_per_task=5, output_dir=str(tmp_path))
        for chunk in spec.chunks:
            hpc.run_task(chunk.task_id, spec)
        hpc.run_task(0, spec)  # re-run task 0
        collected = hpc.collect_shards(spec)
        assert len(collected) == 10  # no duplicates after dedupe by run_index


class TestCollectErrors:
    def test_missing_shards_raise(self, fake_sim, tmp_path):
        spec = hpc.CampaignSpec(num_runs=15, seed=1, runs_per_task=5, output_dir=str(tmp_path))
        hpc.run_task(0, spec)  # only first 5 runs
        with pytest.raises(hpc.IncompleteCampaignError) as exc:
            hpc.collect_shards(spec)
        assert exc.value.found == 5
        assert exc.value.expected == 15
        assert 5 in exc.value.missing

    def test_allow_incomplete_returns_partial(self, fake_sim, tmp_path):
        spec = hpc.CampaignSpec(num_runs=15, seed=1, runs_per_task=5, output_dir=str(tmp_path))
        hpc.run_task(0, spec)
        partial = hpc.collect_shards(spec, allow_incomplete=True)
        assert len(partial) == 5

    def test_no_shards_raises_filenotfound(self, tmp_path):
        spec = hpc.CampaignSpec(num_runs=10, runs_per_task=5, output_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            hpc.collect_shards(spec)

    def test_mismatched_campaign_shard_rejected(self, fake_sim, tmp_path):
        """A shard from a different campaign (seed) in the same dir must not be
        silently merged into the aggregate."""
        spec_a = hpc.CampaignSpec(num_runs=10, seed=1, runs_per_task=5, output_dir=str(tmp_path))
        hpc.run_task(0, spec_a)
        spec_b = hpc.CampaignSpec(num_runs=10, seed=2, runs_per_task=5, output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="does not match"):
            hpc.collect_shards(spec_b)


class TestSubmission:
    def test_dry_run_writes_scripts_without_submitting(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hpc.shutil, "which", lambda _: None)
        spec = hpc.CampaignSpec(num_runs=20, runs_per_task=5, output_dir=str(tmp_path))
        plan = hpc.submit_campaign(spec, hpc.SlurmConfig(), submit=False)
        assert plan.array_script.exists()
        assert plan.collect_script.exists()
        assert plan.submitted is False
        assert plan.array_job_id is None
        assert (tmp_path / "slurm_logs").is_dir()

    def test_submit_without_slurm_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hpc.shutil, "which", lambda _: None)
        spec = hpc.CampaignSpec(num_runs=20, runs_per_task=5, output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="sbatch"):
            hpc.submit_campaign(spec, hpc.SlurmConfig(), submit=True)

    def test_submit_invokes_sbatch_and_chains_collect(self, tmp_path, monkeypatch):
        calls = []

        class _Proc:
            def __init__(self, stdout):
                self.stdout = stdout
                self.returncode = 0

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # First call submits the array, second the collect job.
            jobid = "1001" if len(calls) == 1 else "1002"
            return _Proc(f"Submitted batch job {jobid}")

        monkeypatch.setattr(hpc.shutil, "which", lambda _: "/usr/bin/sbatch")
        monkeypatch.setattr(hpc.subprocess, "run", fake_run)

        spec = hpc.CampaignSpec(num_runs=20, runs_per_task=5, output_dir=str(tmp_path))
        plan = hpc.submit_campaign(spec, hpc.SlurmConfig(), submit=True)
        assert plan.submitted is True
        assert plan.array_job_id == "1001"
        assert plan.collect_job_id == "1002"
        # The collect submission must depend on the array job id.
        assert any("--dependency=afterok:1001" in tok for tok in calls[1])


class TestCli:
    """End-to-end CLI: run-task then collect through main(argv)."""

    def test_run_task_then_collect(self, fake_sim, tmp_path, capsys):
        common = [
            "--runs",
            "12",
            "--seed",
            "5",
            "--runs-per-task",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
        for task_id in range(3):
            rc = hpc.main(["run-task", "--task-id", str(task_id), *common])
            assert rc == 0
        rc = hpc.main(["collect", *common])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Collected 12 runs" in out
        assert "Monte Carlo Summary (N=12)" in out

    def test_run_task_reads_env_task_id(self, fake_sim, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
        rc = hpc.main(
            ["run-task", "--runs", "12", "--seed", "5", "--runs-per-task", "4", "--output-dir", str(tmp_path)]
        )
        assert rc == 0
        assert hpc.shard_path(tmp_path, 1).exists()

    def test_submit_dry_run_cli(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(hpc.shutil, "which", lambda _: None)
        rc = hpc.main(
            ["submit", "--runs", "20", "--runs-per-task", "5", "--output-dir", str(tmp_path), "--partition", "cpu"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert (tmp_path / "mc-ascent_array.sbatch").exists()
