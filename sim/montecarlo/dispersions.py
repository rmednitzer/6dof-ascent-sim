"""Monte Carlo parameter dispersion definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dispersion:
    """Definition of a parameter dispersion for Monte Carlo analysis.

    Attributes:
        parameter: Config parameter name.
        distribution: One of 'gaussian', 'uniform', 'truncated_gaussian'.
        sigma: 1-sigma value for Gaussian distributions.
        bounds: (min, max) for uniform or truncation limits.
    """

    parameter: str
    distribution: str
    sigma: float | None = None
    bounds: tuple[float, float] | None = None


DEFAULT_DISPERSIONS = [
    # Propulsion
    Dispersion("S1_THRUST_VAC_N", "gaussian", sigma=76_070, bounds=None),
    Dispersion("S1_ISP_VAC_S", "gaussian", sigma=3.11, bounds=None),
    Dispersion("S2_THRUST_VAC_N", "gaussian", sigma=9_810, bounds=None),
    Dispersion("S2_ISP_VAC_S", "gaussian", sigma=3.48, bounds=None),
    Dispersion("S1_PROPELLANT_KG", "gaussian", sigma=395.7, bounds=None),
    # Aerodynamics
    Dispersion("CD_SCALE_FACTOR", "truncated_gaussian", sigma=0.10, bounds=(0.7, 1.3)),
    # Atmosphere
    Dispersion("ATMO_DENSITY_SCALE", "truncated_gaussian", sigma=0.05, bounds=(0.8, 1.2)),
    # Wind
    Dispersion("WIND_SPEED_MS", "truncated_gaussian", sigma=15.0, bounds=(0, 50)),
    Dispersion("WIND_DIRECTION_DEG", "uniform", sigma=None, bounds=(0, 360)),
    # Sensors
    Dispersion("IMU_ACCEL_BIAS_MPS2", "gaussian", sigma=0.002, bounds=None),
    Dispersion("IMU_GYRO_BIAS_RADS", "gaussian", sigma=0.0002, bounds=None),
    Dispersion("GPS_POS_NOISE_M", "truncated_gaussian", sigma=2.0, bounds=(1, 15)),
    # Mass
    Dispersion("S1_DRY_MASS_KG", "gaussian", sigma=222, bounds=None),
]


def sample_dispersion(dispersion: Dispersion, rng: np.random.Generator) -> float:
    """Sample a single dispersion value.

    Returns the zero-mean *offset* for ``gaussian`` and ``truncated_gaussian``
    (the caller adds it to the nominal), or the absolute drawn value for
    ``uniform``.

    Truncation of a ``truncated_gaussian`` to its ``bounds`` is deliberately
    *not* applied here: ``bounds`` are absolute limits on the resulting
    parameter, not on the zero-mean offset, so clamping must happen against
    ``nominal + offset`` in :func:`generate_dispersed_config`. (Clamping the
    offset here is what previously collapsed e.g. ``CD_SCALE_FACTOR`` to a
    constant, because every small offset fell below the absolute lower bound.)

    Args:
        dispersion: Dispersion definition.
        rng: Numpy random generator.

    Returns:
        Sampled parameter offset (for Gaussian/truncated) or absolute value
        (for uniform).
    """
    if dispersion.distribution == "gaussian":
        return float(rng.normal(0.0, dispersion.sigma))
    elif dispersion.distribution == "uniform":
        low, high = dispersion.bounds
        return float(rng.uniform(low, high))
    elif dispersion.distribution == "truncated_gaussian":
        return float(rng.normal(0.0, dispersion.sigma))
    else:
        raise ValueError(f"Unknown distribution: {dispersion.distribution}")


def generate_dispersed_config(
    dispersions: list[Dispersion],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Generate a config override dictionary from dispersions.

    For Gaussian dispersions, the sampled value is added to the nominal.
    For uniform dispersions, the sampled value replaces the nominal.

    Args:
        dispersions: List of dispersion definitions.
        rng: Numpy random generator.

    Returns:
        Dictionary mapping parameter names to dispersed values.
    """
    from sim import config

    overrides: dict[str, float] = {}
    for d in dispersions:
        nominal = getattr(config, d.parameter, None)
        if nominal is None:
            continue
        sample = sample_dispersion(d, rng)
        if d.distribution == "uniform":
            overrides[d.parameter] = sample
        else:
            value = nominal + sample
            # Truncated Gaussian: clamp the FINAL parameter value to the
            # absolute bounds (these constrain the parameter, e.g. drag scale
            # in [0.7, 1.3], not the zero-mean offset). A plain "gaussian" has
            # no bounds and is left unconstrained.
            if d.distribution == "truncated_gaussian" and d.bounds is not None:
                low, high = d.bounds
                value = float(np.clip(value, low, high))
            overrides[d.parameter] = value
    return overrides
