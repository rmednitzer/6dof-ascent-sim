"""Vehicle configuration and mass bookkeeping.

Provides a ``StageConfig`` dataclass for per-stage parameters and a ``Vehicle``
class that tracks the current stage, remaining propellant, and exposes
convenience properties for the guidance / dynamics loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sim import config


# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageConfig:
    """Immutable description of a single propulsive stage."""

    dry_mass: float        # kg — structural / inert mass
    propellant: float      # kg — total usable propellant
    thrust_vac: float      # N  — vacuum thrust
    thrust_sl: float       # N  — sea-level thrust
    isp_vac: float         # s  — vacuum specific impulse
    isp_sl: float          # s  — sea-level specific impulse
    burn_time: float       # s  — nominal burn duration
    throttle_min: float    # –  — minimum throttle fraction [0, 1]


# ---------------------------------------------------------------------------
# Default stage table built from sim.config
# ---------------------------------------------------------------------------

STAGE_1 = StageConfig(
    dry_mass=config.S1_DRY_MASS_KG,
    propellant=config.S1_PROPELLANT_KG,
    thrust_vac=config.S1_THRUST_VAC_N,
    thrust_sl=config.S1_THRUST_SL_N,
    isp_vac=config.S1_ISP_VAC_S,
    isp_sl=config.S1_ISP_SL_S,
    burn_time=config.S1_BURN_TIME_S,
    throttle_min=config.S1_THROTTLE_MIN,
)

STAGE_2 = StageConfig(
    dry_mass=config.S2_DRY_MASS_KG,
    propellant=config.S2_PROPELLANT_KG,
    thrust_vac=config.S2_THRUST_VAC_N,
    thrust_sl=config.S2_THRUST_VAC_N,       # S2 operates in vacuum — SL ≈ vac
    isp_vac=config.S2_ISP_VAC_S,
    isp_sl=config.S2_ISP_VAC_S,             # Same assumption
    burn_time=config.S2_BURN_TIME_S,
    throttle_min=0.4,                        # Conservative default
)


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------

class Vehicle:
    """Tracks the composite launch-vehicle state across staging events.

    Parameters
    ----------
    stages : list[StageConfig], optional
        Ordered list of stages (index 0 = first to fire).  Defaults to
        ``[STAGE_1, STAGE_2]``.
    """

    def __init__(self, stages: List[StageConfig] | None = None) -> None:
        self.stages: List[StageConfig] = list(stages or [STAGE_1, STAGE_2])
        self._stage_index: int = 0

        # Propellant remaining in *each* stage (allows partial burns).
        self._propellant_remaining: List[float] = [
            s.propellant for s in self.stages
        ]

    # -- Stage selection -----------------------------------------------------

    @property
    def stage_index(self) -> int:
        """Zero-based index of the currently active stage."""
        return self._stage_index

    @property
    def current_stage(self) -> StageConfig:
        """Configuration of the currently active stage."""
        return self.stages[self._stage_index]

    @property
    def has_next_stage(self) -> bool:
        """True if there is another stage after the current one."""
        return self._stage_index < len(self.stages) - 1

    def advance_stage(self) -> None:
        """Advance to the next stage.

        Raises
        ------
        RuntimeError
            If no further stages remain.
        """
        if not self.has_next_stage:
            raise RuntimeError("No further stages available.")
        self._stage_index += 1

    # -- Mass bookkeeping ----------------------------------------------------

    def total_mass(self) -> float:
        """Total vehicle mass (kg) including all remaining stages."""
        mass = 0.0
        for i in range(self._stage_index, len(self.stages)):
            mass += self.stages[i].dry_mass + self._propellant_remaining[i]
        return mass

    def propellant_remaining(self) -> float:
        """Propellant remaining in the *current* stage (kg)."""
        return self._propellant_remaining[self._stage_index]

    def propellant_fraction(self) -> float:
        """Fraction of propellant remaining in the current stage [0, 1]."""
        total = self.current_stage.propellant
        if total <= 0.0:
            return 0.0
        return self._propellant_remaining[self._stage_index] / total

    def consume_propellant(self, dm: float) -> float:
        """Consume *dm* kg of propellant from the current stage.

        Returns the mass actually consumed (may be less than *dm* if the
        stage is nearly empty).
        """
        idx = self._stage_index
        available = self._propellant_remaining[idx]
        consumed = min(dm, available)
        self._propellant_remaining[idx] = available - consumed
        return consumed

    def thrust_fraction(self) -> float:
        """Ratio of current thrust-to-weight vs. initial stage TWR.

        Useful as a quick health metric for telemetry.  Returns 0.0 if the
        stage has no propellant.
        """
        if self.propellant_remaining() <= 0.0:
            return 0.0
        initial_mass = self.current_stage.dry_mass + self.current_stage.propellant
        return initial_mass / self.total_mass()
