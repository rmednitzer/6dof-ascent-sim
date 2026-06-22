"""Validation schema and default values for the overridable / dispersion parameters.

Step 2 of ADR-0009 (ADR 0018): this module is the single source of truth for the
*overridable* parameters — the ones a Monte-Carlo run may disperse. It declares
their types, bounds, units, and default values. ``sim.config`` resolves
``config.<NAME>`` for these names from a context-local instance of
:class:`OverridableParams` (see ``sim/config.py``), so per-run overrides no
longer mutate global state.

The *fixed* physical/model constants (Earth model, tables, safety limits, control
gains, …) remain plain module globals in ``sim.config``. This module must not
import ``sim.config`` (``sim.config`` imports it), so the defaults below are
literals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OverridableParams(BaseModel):
    """Typed, bounded schema + defaults for the per-run overridable parameters.

    ``extra="forbid"`` makes an unknown key (a typo) a hard validation error.
    All fields have defaults, so a partial override dict validates directly and
    a default instance supplies the nominal configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Propulsion ---
    S1_THRUST_VAC_N: float = Field(default=7_607_000.0, gt=0.0, le=5e7, description="Stage-1 vacuum thrust (N)")
    S1_ISP_VAC_S: float = Field(default=311.0, gt=0.0, le=1000.0, description="Stage-1 vacuum Isp (s)")
    S2_THRUST_VAC_N: float = Field(default=981_000.0, gt=0.0, le=5e7, description="Stage-2 vacuum thrust (N)")
    # Raised 348 -> 356 s (+2.3%) for performance margin (ADR 0021 / BACKLOG N-01):
    # the higher-Isp upper stage reaches orbit with margin so adverse dispersions
    # no longer deplete just short of insertion. Unlike a propellant increase it
    # leaves liftoff mass (and the fragile nominal trajectory) unchanged.
    S2_ISP_VAC_S: float = Field(default=356.0, gt=0.0, le=1000.0, description="Stage-2 vacuum Isp (s)")
    S1_PROPELLANT_KG: float = Field(default=395_700.0, gt=0.0, le=1e7, description="Stage-1 propellant (kg)")
    S1_DRY_MASS_KG: float = Field(default=22_200.0, gt=0.0, le=1e6, description="Stage-1 dry mass (kg)")

    # --- Aerodynamics / atmosphere / wind ---
    CD_SCALE_FACTOR: float = Field(default=1.0, gt=0.0, le=10.0, description="Drag-coefficient multiplier (-)")
    ATMO_DENSITY_SCALE: float = Field(default=1.0, ge=0.0, le=10.0, description="Atmospheric-density multiplier (-)")
    WIND_SPEED_MS: float = Field(default=10.0, ge=0.0, le=200.0, description="Mean wind speed (m/s)")
    WIND_DIRECTION_DEG: float = Field(
        default=270.0, ge=0.0, le=360.0, description="Wind direction, met. convention (deg)"
    )

    # --- Sensors (scale parameters: must be strictly positive — AD-18) ---
    IMU_ACCEL_BIAS_MPS2: float = Field(default=0.001, gt=0.0, le=1.0, description="Accel bias-instability RMS (m/s^2)")
    IMU_GYRO_BIAS_RADS: float = Field(default=0.0001, gt=0.0, le=1.0, description="Gyro bias-instability RMS (rad/s)")
    GPS_POS_NOISE_M: float = Field(default=5.0, gt=0.0, le=1000.0, description="GPS position noise 1-sigma (m)")

    # --- Structural-dynamics model toggles / tunables ---
    FLEX_ENABLED: bool = Field(default=True, description="Enable the flex-body model")
    FLEX_NOTCH_ENABLED: bool = Field(default=True, description="Enable the structural notch filter")
    FLEX_NOTCH_Q: float = Field(default=2.0, gt=0.0, le=100.0, description="Structural notch quality factor (-)")
    FLEX_MODAL_MASS_KG: float = Field(default=1_000_000.0, gt=0.0, le=1e9, description="Generalised modal mass (kg)")
    SLOSH_ENABLED: bool = Field(default=True, description="Enable the propellant-slosh model")

    # --- Navigation / GNC (ADR 0020) ---
    # When True (Stage 2 default), guidance, the attitude controller, and the FTS
    # consume the error-state EKF's *estimated* attitude; when False they use the
    # true attitude while the estimator still runs in parallel for validation.
    # Overridable so a Monte-Carlo / regression run can compare both authorities.
    USE_ESTIMATED_ATTITUDE: bool = Field(default=True, description="Close the loop on the EKF's estimated attitude")


# The single declaration of the overridable parameter set. Consumed by
# sim.config (context-local resolution) and the Monte-Carlo dispatcher.
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
