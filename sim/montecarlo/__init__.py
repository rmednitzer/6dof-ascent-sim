"""Monte Carlo simulation framework.

Local execution lives in :mod:`sim.montecarlo.dispatcher` (parallel across one
machine's cores). The experimental SLURM HPC backend
(:mod:`sim.montecarlo.hpc`) distributes the same campaign across a cluster job
array, producing identical results for a given base seed.

Import the public API directly from the submodules, e.g.
``from sim.montecarlo.hpc import CampaignSpec, submit_campaign``. (The package
intentionally does not re-export submodule symbols, so ``python -m
sim.montecarlo.hpc`` / ``... .dispatcher`` run without a runpy double-import
warning.)
"""
