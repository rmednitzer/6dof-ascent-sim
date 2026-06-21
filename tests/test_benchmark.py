"""Smoke test for the performance benchmark harness (ADR 0017).

Keeps the end-to-end benchmark importable and runnable so it cannot bit-rot.
Uses a tiny duration so it stays fast in CI.
"""

from __future__ import annotations

import benchmark


def test_full_run_benchmark_runs(capsys):
    """The end-to-end benchmark executes and reports throughput."""
    benchmark.benchmark_full_run(t_max=3.0, repeats=1)
    out = capsys.readouterr().out
    assert "steps/s" in out


def test_profile_runs(capsys):
    """The cProfile path executes and prints a breakdown."""
    benchmark.profile_full_run(t_max=3.0, top=5)
    out = capsys.readouterr().out
    assert "function calls" in out
