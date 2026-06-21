"""Tests for the Monte-Carlo override/dispersion validation schema (ADR 0016).

Covers the validation layer added as step 1 of ADR-0009: typo/bounds rejection
on overrides, fail-fast on dispersions targeting unknown parameters, and the
single-sourced overridable set that closes the Q-02 key-list drift.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from sim.config_schema import (
    OVERRIDABLE_PARAM_NAMES,
    validate_dispersions,
    validate_overrides,
)
from sim.montecarlo.dispersions import (
    DEFAULT_DISPERSIONS,
    Dispersion,
    generate_dispersed_config,
)


class TestValidateOverrides:
    def test_valid_override_passes(self):
        validate_overrides({"CD_SCALE_FACTOR": 1.1, "FLEX_ENABLED": False})

    def test_internal_keys_ignored(self):
        validate_overrides({"_seed": 7, "_run_index": 3, "CD_SCALE_FACTOR": 1.0})

    def test_empty_override_ok(self):
        validate_overrides({})

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError):
            validate_overrides({"CD_SCALE_FACTORR": 1.0})  # typo

    def test_non_positive_scale_rejected(self):
        # A non-positive sensor-noise scale is the AD-18 failure class.
        with pytest.raises(ValidationError):
            validate_overrides({"IMU_ACCEL_BIAS_MPS2": -0.001})

    def test_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            validate_overrides({"CD_SCALE_FACTOR": 1000.0})

    def test_wrong_type_rejected(self):
        with pytest.raises(ValidationError):
            validate_overrides({"CD_SCALE_FACTOR": "not-a-number"})


class TestValidateDispersions:
    def test_default_dispersions_valid(self):
        validate_dispersions(DEFAULT_DISPERSIONS)

    def test_typo_dispersion_rejected(self):
        bad = [Dispersion("CD_SCALE_FACTORR", "truncated_gaussian", sigma=0.1, bounds=(0.7, 1.3))]
        with pytest.raises(ValueError, match="unknown"):
            validate_dispersions(bad)

    def test_generated_default_overrides_validate(self):
        # Every override the shipped dispersions can produce must pass the gate.
        for seed in range(200):
            override = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(seed))
            validate_overrides(override)


class TestOverridableSetCoversDispersions:
    """Guards Q-02: the save/restore set (now derived from the schema) must
    cover every dispersed parameter, else an in-process override could leak."""

    def test_all_dispersed_params_are_overridable(self):
        names = set(OVERRIDABLE_PARAM_NAMES)
        for d in DEFAULT_DISPERSIONS:
            assert d.parameter in names, f"{d.parameter} is dispersed but not in OVERRIDABLE_PARAM_NAMES"

    def test_save_config_derives_from_schema(self):
        from sim.main import _save_config

        assert set(_save_config()) == set(OVERRIDABLE_PARAM_NAMES)


class TestDispatcherValidatesCampaign:
    def test_dispatcher_rejects_unknown_dispersion(self):
        from sim.montecarlo.dispatcher import MonteCarloDispatcher

        bad = [Dispersion("NOT_A_PARAM", "gaussian", sigma=1.0)]
        with pytest.raises(ValueError, match="unknown"):
            MonteCarloDispatcher(num_runs=4, dispersions=bad)
