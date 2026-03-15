"""Engine / propulsion model.

Models thrust and mass-flow variation with ambient pressure, throttle
commands, and ignition / shutdown transients.
"""

from __future__ import annotations

from sim import config
from sim.vehicle.vehicle import StageConfig

# ---------------------------------------------------------------------------
# Physical constants (module-level for convenience)
# ---------------------------------------------------------------------------
G0: float = config.G0          # 9.80665 m/s^2
P_SL: float = config.P_SL      # 101325.0 Pa

# Transient ramp duration (seconds) for ignition and shutdown
IGNITION_RAMP_TIME: float = 0.5
SHUTDOWN_RAMP_TIME: float = 0.5


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _pressure_fraction(p_ambient: float) -> float:
    """Normalised ambient pressure in [0, 1]."""
    return max(0.0, min(1.0, p_ambient / P_SL))


def thrust_at_pressure(stage: StageConfig, p_ambient: float) -> float:
    """Compute thrust (N) for a given ambient pressure.

    The model linearly interpolates between sea-level and vacuum values::

        F = F_vac - (F_vac - F_sl) * (p / p_sl)

    Parameters
    ----------
    stage : StageConfig
        Active stage configuration.
    p_ambient : float
        Ambient static pressure (Pa).

    Returns
    -------
    float
        Thrust (N).
    """
    pf: float = _pressure_fraction(p_ambient)
    return stage.thrust_vac - (stage.thrust_vac - stage.thrust_sl) * pf


def isp_at_pressure(stage: StageConfig, p_ambient: float) -> float:
    """Compute specific impulse (s) for a given ambient pressure.

    Same linear interpolation as thrust::

        Isp = Isp_vac - (Isp_vac - Isp_sl) * (p / p_sl)

    Parameters
    ----------
    stage : StageConfig
        Active stage configuration.
    p_ambient : float
        Ambient static pressure (Pa).

    Returns
    -------
    float
        Specific impulse (s).
    """
    pf: float = _pressure_fraction(p_ambient)
    return stage.isp_vac - (stage.isp_vac - stage.isp_sl) * pf


def mass_flow_rate(thrust: float, isp: float) -> float:
    """Mass flow rate (kg/s).

    .. math:: \\dot{m} = F / (I_{sp} \\, g_0)

    Returns 0.0 if *isp* is non-positive to avoid division by zero.
    """
    if isp <= 0.0:
        return 0.0
    return thrust / (isp * G0)


# ---------------------------------------------------------------------------
# Engine model (stateful)
# ---------------------------------------------------------------------------

class EngineModel:
    """Single-engine model with throttle, ignition, and shutdown transients.

    Parameters
    ----------
    stage : StageConfig
        Stage that owns this engine (or engine cluster).
    """

    def __init__(self, stage: StageConfig) -> None:
        self._stage: StageConfig = stage

        # Current commanded throttle [throttle_min, 1.0]
        self._throttle_command: float = 1.0

        # Engine on/off state
        self._ignited: bool = False
        self._shutting_down: bool = False

        # Transient tracking (elapsed time since event start)
        self._transient_elapsed: float = 0.0
        self._transient_duration: float = 0.0
        self._transient_start_level: float = 0.0
        self._transient_end_level: float = 0.0

        # Effective throttle after transient envelope [0, 1]
        self._effective_throttle: float = 0.0

    # -- Commands ------------------------------------------------------------

    def ignite(self) -> None:
        """Command engine ignition (linear ramp from 0 to commanded throttle)."""
        if self._ignited and not self._shutting_down:
            return  # already running
        self._ignited = True
        self._shutting_down = False
        self._transient_elapsed = 0.0
        self._transient_duration = IGNITION_RAMP_TIME
        self._transient_start_level = 0.0
        self._transient_end_level = self._throttle_command

    def shutdown(self) -> None:
        """Command engine shutdown (linear ramp from current throttle to 0)."""
        if not self._ignited:
            return
        self._shutting_down = True
        self._transient_elapsed = 0.0
        self._transient_duration = SHUTDOWN_RAMP_TIME
        self._transient_start_level = self._effective_throttle
        self._transient_end_level = 0.0

    def set_throttle(self, command: float) -> None:
        """Set the throttle command, clamped to ``[throttle_min, 1.0]``.

        Parameters
        ----------
        command : float
            Desired throttle setting.
        """
        tmin: float = self._stage.throttle_min
        self._throttle_command = max(tmin, min(1.0, command))

    # -- State query ---------------------------------------------------------

    @property
    def is_ignited(self) -> bool:
        """True while the engine is firing (including transients)."""
        return self._ignited

    @property
    def effective_throttle(self) -> float:
        """Current effective throttle after transient envelope [0, 1]."""
        return self._effective_throttle

    # -- Update --------------------------------------------------------------

    def update(self, dt: float, p_ambient: float) -> tuple[float, float]:
        """Advance the engine model by *dt* seconds.

        Parameters
        ----------
        dt : float
            Timestep (s).
        p_ambient : float
            Ambient static pressure (Pa).

        Returns
        -------
        thrust : float
            Net thrust (N) at this instant.
        mdot : float
            Mass flow rate (kg/s) at this instant.
        """
        if not self._ignited:
            self._effective_throttle = 0.0
            return 0.0, 0.0

        # --- transient ramp ---
        if self._transient_duration > 0.0:
            self._transient_elapsed += dt
            frac = min(1.0, self._transient_elapsed / self._transient_duration)
            level = (
                self._transient_start_level
                + (self._transient_end_level - self._transient_start_level) * frac
            )
            self._effective_throttle = max(0.0, min(1.0, level))

            # Transient complete
            if frac >= 1.0:
                self._transient_duration = 0.0
                if self._shutting_down:
                    self._ignited = False
                    self._shutting_down = False
                    self._effective_throttle = 0.0
                    return 0.0, 0.0
        else:
            # Steady-state: follow throttle command
            self._effective_throttle = self._throttle_command

        # --- Thrust & mass flow ---
        f_full: float = thrust_at_pressure(self._stage, p_ambient)
        isp: float = isp_at_pressure(self._stage, p_ambient)

        thrust: float = f_full * self._effective_throttle
        mdot: float = mass_flow_rate(thrust, isp)

        return thrust, mdot
