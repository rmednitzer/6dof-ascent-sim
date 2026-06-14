"""Regression tests for Monte Carlo parameter dispersions (sim/montecarlo/dispersions.py).

Guards the truncation fix: previously ``truncated_gaussian`` clipped the
zero-mean *offset* against the parameter's *absolute* bounds, which collapsed
e.g. ``CD_SCALE_FACTOR`` and ``ATMO_DENSITY_SCALE`` to a single constant value
(zero dispersion) and biased the others. Truncation must instead apply to the
final value (nominal + offset).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config
from sim.montecarlo.dispersions import (
    DEFAULT_DISPERSIONS,
    Dispersion,
    generate_dispersed_config,
    sample_dispersion,
)

_TRUNCATED = [d for d in DEFAULT_DISPERSIONS if d.distribution == "truncated_gaussian"]


def _draw(param: str, n: int = 6000) -> np.ndarray:
    vals = [generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(i))[param] for i in range(n)]
    return np.array(vals)


class TestTruncatedGaussian:
    @pytest.mark.parametrize("disp", _TRUNCATED, ids=[d.parameter for d in _TRUNCATED])
    def test_values_within_bounds(self, disp: Dispersion):
        a = _draw(disp.parameter)
        low, high = disp.bounds
        assert a.min() >= low - 1e-9, f"{disp.parameter} below lower bound"
        assert a.max() <= high + 1e-9, f"{disp.parameter} above upper bound"

    @pytest.mark.parametrize("disp", _TRUNCATED, ids=[d.parameter for d in _TRUNCATED])
    def test_not_degenerate(self, disp: Dispersion):
        """The core regression: dispersions must actually disperse."""
        a = _draw(disp.parameter)
        assert a.std() > 0.0, f"{disp.parameter} has zero spread (collapsed to a constant!)"
        assert len(np.unique(a)) > 100, f"{disp.parameter} takes too few distinct values"

    def test_cd_scale_not_constant(self):
        """Explicit guard for the exact prior bug (CD pinned to 1.7)."""
        a = _draw("CD_SCALE_FACTOR")
        assert a.std() > 0.01
        assert not np.allclose(a, a[0]), "CD_SCALE_FACTOR collapsed to a constant"

    def test_symmetric_truncation_stays_centered(self):
        """For a parameter whose bounds are symmetric about nominal, the mean
        should remain near nominal (truncation is symmetric)."""
        for param in ("CD_SCALE_FACTOR", "ATMO_DENSITY_SCALE"):
            a = _draw(param)
            nominal = getattr(config, param)
            assert abs(a.mean() - nominal) < 0.02, f"{param} mean drifted from nominal"


class TestGaussianAndUniform:
    def test_gaussian_is_nominal_plus_noise(self):
        a = np.array(
            [
                generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(i))["S1_THRUST_VAC_N"]
                for i in range(4000)
            ]
        )
        # mean ~ nominal, std ~ sigma (76_070)
        assert abs(a.mean() - config.S1_THRUST_VAC_N) < 5_000
        assert abs(a.std() - 76_070) < 5_000

    def test_uniform_in_bounds(self):
        d = Dispersion("WIND_DIRECTION_DEG", "uniform", bounds=(0, 360))
        vals = np.array([sample_dispersion(d, np.random.default_rng(i)) for i in range(2000)])
        assert vals.min() >= 0.0
        assert vals.max() <= 360.0

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            sample_dispersion(Dispersion("X", "weibull", sigma=1.0), np.random.default_rng(0))


class TestScaleParametersNonNegative:
    """Parameters consumed as a Gaussian standard deviation (RNG scale) in the
    sensor model must never be dispersed negative, or runs crash with
    'scale < 0' (regression for AD-18)."""

    @pytest.mark.parametrize("param", ["IMU_ACCEL_BIAS_MPS2", "IMU_GYRO_BIAS_RADS", "GPS_POS_NOISE_M"])
    def test_scale_never_negative(self, param):
        a = _draw(param, n=8000)
        assert a.min() > 0.0, f"{param} dispersed to a non-positive scale (min={a.min()})"


class TestDeterminism:
    def test_same_seed_same_config(self):
        a = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(123))
        b = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(123))
        assert a == b

    def test_different_seed_differs(self):
        a = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(1))
        b = generate_dispersed_config(DEFAULT_DISPERSIONS, np.random.default_rng(2))
        assert a != b
