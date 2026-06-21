"""Validation schema for Monte-Carlo override / dispersion parameters.

This is step 1 of ADR-0009 ("replace mutable-global config override with
explicit parameter passing"): a *validation layer* over today's
``sim.config``. It does **not** change how the simulation reads configuration
(modules still use ``from sim import config``); it adds a single, typed,
bounded declaration of *which* parameters may be overridden per Monte-Carlo run
and validates dispersion definitions and override dicts against it.

Two concrete wins:

* **Catch typos / bad values before a campaign.** ``generate_dispersed_config``
  silently skips a dispersion whose ``parameter`` is not a real config
  attribute (``getattr(..., None)``), so a mis-spelled parameter name would
  quietly disable that dispersion. :func:`validate_dispersions` rejects it, and
  :func:`validate_overrides` range-checks the drawn values (e.g. a non-positive
  IMU bias scale, the AD-18 failure class).
* **Single-source the overridable set.** :data:`OVERRIDABLE_PARAM_NAMES` is the
  one place the overridable parameters are declared; ``main._save_config`` reads
  it instead of a hand-maintained key list (closing the Q-02 drift).

Parameter *values* still live solely in ``sim.config`` (the single source of
truth, ADR-0001) — each field below defaults from it; this schema only adds the
type, bounds, and units.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sim import config


class OverridableParams(BaseModel):
    """Typed, bounded schema of the per-run overridable parameters.

    ``extra="forbid"`` makes an unknown key (a typo) a hard validation error.
    All fields have defaults sourced from ``sim.config`` so a partial override
    dict validates against this model directly.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Propulsion ---
    S1_THRUST_VAC_N: float = Field(
        default=config.S1_THRUST_VAC_N, gt=0.0, le=5e7, description="Stage-1 vacuum thrust (N)"
    )
    S1_ISP_VAC_S: float = Field(default=config.S1_ISP_VAC_S, gt=0.0, le=1000.0, description="Stage-1 vacuum Isp (s)")
    S2_THRUST_VAC_N: float = Field(
        default=config.S2_THRUST_VAC_N, gt=0.0, le=5e7, description="Stage-2 vacuum thrust (N)"
    )
    S2_ISP_VAC_S: float = Field(default=config.S2_ISP_VAC_S, gt=0.0, le=1000.0, description="Stage-2 vacuum Isp (s)")
    S1_PROPELLANT_KG: float = Field(
        default=config.S1_PROPELLANT_KG, gt=0.0, le=1e7, description="Stage-1 propellant (kg)"
    )
    S1_DRY_MASS_KG: float = Field(default=config.S1_DRY_MASS_KG, gt=0.0, le=1e6, description="Stage-1 dry mass (kg)")

    # --- Aerodynamics / atmosphere / wind ---
    CD_SCALE_FACTOR: float = Field(
        default=config.CD_SCALE_FACTOR, gt=0.0, le=10.0, description="Drag-coefficient multiplier (-)"
    )
    ATMO_DENSITY_SCALE: float = Field(
        default=config.ATMO_DENSITY_SCALE, ge=0.0, le=10.0, description="Atmospheric-density multiplier (-)"
    )
    WIND_SPEED_MS: float = Field(default=config.WIND_SPEED_MS, ge=0.0, le=200.0, description="Mean wind speed (m/s)")
    WIND_DIRECTION_DEG: float = Field(
        default=config.WIND_DIRECTION_DEG, ge=0.0, le=360.0, description="Wind direction, met. convention (deg)"
    )

    # --- Sensors (scale parameters: must be strictly positive — AD-18) ---
    IMU_ACCEL_BIAS_MPS2: float = Field(
        default=config.IMU_ACCEL_BIAS_MPS2, gt=0.0, le=1.0, description="Accel bias-instability RMS (m/s^2)"
    )
    IMU_GYRO_BIAS_RADS: float = Field(
        default=config.IMU_GYRO_BIAS_RADS, gt=0.0, le=1.0, description="Gyro bias-instability RMS (rad/s)"
    )
    GPS_POS_NOISE_M: float = Field(
        default=config.GPS_POS_NOISE_M, gt=0.0, le=1000.0, description="GPS position noise 1-sigma (m)"
    )

    # --- Structural-dynamics model toggles / tunables ---
    FLEX_ENABLED: bool = Field(default=config.FLEX_ENABLED, description="Enable the flex-body model")
    FLEX_NOTCH_ENABLED: bool = Field(
        default=config.FLEX_NOTCH_ENABLED, description="Enable the structural notch filter"
    )
    FLEX_NOTCH_Q: float = Field(
        default=config.FLEX_NOTCH_Q, gt=0.0, le=100.0, description="Structural notch quality factor (-)"
    )
    FLEX_MODAL_MASS_KG: float = Field(
        default=config.FLEX_MODAL_MASS_KG, gt=0.0, le=1e9, description="Generalised modal mass (kg)"
    )
    SLOSH_ENABLED: bool = Field(default=config.SLOSH_ENABLED, description="Enable the propellant-slosh model")


# The single declaration of the overridable parameter set. Consumed by
# main._save_config so the save/restore key list cannot drift (Q-02).
OVERRIDABLE_PARAM_NAMES: tuple[str, ...] = tuple(OverridableParams.model_fields)


def validate_overrides(override: Mapping[str, Any]) -> None:
    """Validate a Monte-Carlo override dict.

    Raises ``pydantic.ValidationError`` if any (non-internal) key is not an
    overridable parameter or any value is the wrong type or out of bounds.
    Internal bookkeeping keys (leading underscore, e.g. ``_seed``,
    ``_run_index``) are ignored.

    Args:
        override: Parameter-name -> value mapping for a single run.
    """
    params = {k: v for k, v in override.items() if not k.startswith("_")}
    OverridableParams(**params)


def validate_dispersions(dispersions: Iterable[Any]) -> None:
    """Ensure every dispersion targets a known, overridable parameter.

    ``generate_dispersed_config`` silently skips a dispersion whose
    ``parameter`` is not a real config attribute, so a typo'd name would
    disable that dispersion with no error. Fail fast instead.

    Args:
        dispersions: Iterable of ``Dispersion`` (anything with a ``parameter``).

    Raises:
        ValueError: if any dispersion targets an unknown / non-overridable name.
    """
    known = set(OVERRIDABLE_PARAM_NAMES)
    unknown = sorted({d.parameter for d in dispersions if d.parameter not in known})
    if unknown:
        raise ValueError(
            f"Dispersion(s) target unknown / non-overridable parameter(s): {unknown}. "
            f"Overridable parameters: {sorted(known)}"
        )
