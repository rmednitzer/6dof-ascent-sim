"""Stage-separation sequencer.

Implements a state-machine that monitors propellant depletion and drives
the separation sequence:

    NOMINAL → TAIL_OFF (1 s) → COAST (1 s) → SEPARATION → S2_IGNITION (0.5 s ramp)

A safety interlock prevents separation while effective throttle exceeds 5 %.
"""

from __future__ import annotations

from enum import Enum, auto

from sim.vehicle.propulsion import EngineModel
from sim.vehicle.vehicle import Vehicle

# ---------------------------------------------------------------------------
# Sequence timing
# ---------------------------------------------------------------------------
TAIL_OFF_DURATION: float = 1.0  # Engine tail-off (s)
COAST_DURATION: float = 1.0  # Unpowered coast (s)
S2_IGNITION_RAMP: float = 0.5  # S2 engine start-up ramp (s)

# Safety interlock threshold — fraction of rated thrust
THRUST_INTERLOCK_FRACTION: float = 0.05


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class StagingPhase(Enum):
    """Staging-sequencer states."""

    NOMINAL = auto()  # Burning on current stage
    TAIL_OFF = auto()  # S1 engine shutting down
    COAST = auto()  # Unpowered gap between stages
    SEPARATION = auto()  # Mass drop (instantaneous)
    S2_IGNITION = auto()  # S2 engine ramping up
    COMPLETE = auto()  # Separation finished — normal ops on next stage


class StagingSequencer:
    """Monitors propellant and orchestrates stage separation.

    Parameters
    ----------
    vehicle : Vehicle
        The vehicle being staged.
    s1_engine : EngineModel
        Engine model for the currently active (first) stage.
    s2_engine : EngineModel
        Engine model for the next stage.
    propellant_threshold_kg : float, optional
        Propellant remaining (kg) below which separation is triggered.
        Default: 0.0 (trigger on full depletion).
    """

    def __init__(
        self,
        vehicle: Vehicle,
        s1_engine: EngineModel,
        s2_engine: EngineModel,
        propellant_threshold_kg: float = 0.0,
    ) -> None:
        self._vehicle: Vehicle = vehicle
        self._s1_engine: EngineModel = s1_engine
        self._s2_engine: EngineModel = s2_engine
        self._threshold: float = propellant_threshold_kg

        self._phase: StagingPhase = StagingPhase.NOMINAL
        self._phase_elapsed: float = 0.0
        self._separation_complete: bool = False

    # -- Read-only properties ------------------------------------------------

    @property
    def phase(self) -> StagingPhase:
        """Current staging phase."""
        return self._phase

    @property
    def is_complete(self) -> bool:
        """True once the full separation sequence has finished."""
        return self._separation_complete

    # -- Safety interlock ----------------------------------------------------

    def _safe_to_separate(self) -> bool:
        """Return True if S1 thrust is below the interlock threshold."""
        return self._s1_engine.effective_throttle <= THRUST_INTERLOCK_FRACTION

    # -- Update --------------------------------------------------------------

    def update(self, dt: float) -> str | None:
        """Advance the staging state machine by *dt* seconds.

        Parameters
        ----------
        dt : float
            Simulation timestep (s).

        Returns
        -------
        event : str or None
            A human-readable event string when a phase transition occurs,
            or ``None`` during steady-state.
        """
        if self._separation_complete:
            return None

        event: str | None = None

        # ------------------------------------------------------------------
        if self._phase is StagingPhase.NOMINAL:
            # Check for propellant depletion
            if self._vehicle.propellant_remaining() <= self._threshold:
                self._phase = StagingPhase.TAIL_OFF
                self._phase_elapsed = 0.0
                self._s1_engine.shutdown()
                event = "STAGING: S1 propellant depleted — tail-off initiated"

        # ------------------------------------------------------------------
        elif self._phase is StagingPhase.TAIL_OFF:
            self._phase_elapsed += dt
            if self._phase_elapsed >= TAIL_OFF_DURATION:
                # Verify interlock before proceeding
                if self._safe_to_separate():
                    self._phase = StagingPhase.COAST
                    self._phase_elapsed = 0.0
                    event = "STAGING: tail-off complete — coasting"
                # else: stay in TAIL_OFF until thrust decays

        # ------------------------------------------------------------------
        elif self._phase is StagingPhase.COAST:
            self._phase_elapsed += dt
            if self._phase_elapsed >= COAST_DURATION:
                self._phase = StagingPhase.SEPARATION
                self._phase_elapsed = 0.0
                event = "STAGING: coast complete — separation commanded"

        # ------------------------------------------------------------------
        elif self._phase is StagingPhase.SEPARATION:
            # Safety check one more time
            if not self._safe_to_separate():
                event = "STAGING: ABORT — thrust above interlock at separation"
                return event

            # Drop the spent stage
            self._vehicle.advance_stage()
            self._phase = StagingPhase.S2_IGNITION
            self._phase_elapsed = 0.0
            self._s2_engine.ignite()
            event = "STAGING: separation — S2 ignition commanded"

        # ------------------------------------------------------------------
        elif self._phase is StagingPhase.S2_IGNITION:
            self._phase_elapsed += dt
            if self._phase_elapsed >= S2_IGNITION_RAMP:
                self._phase = StagingPhase.COMPLETE
                self._separation_complete = True
                event = "STAGING: S2 ignition ramp complete — nominal ops"

        return event
