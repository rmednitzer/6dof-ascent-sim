"""Aerodynamic force model.

Computes drag as a function of velocity, atmospheric density, and Mach
number.  Cd is interpolated from the table in ``sim.config`` and scaled by
``CD_SCALE_FACTOR`` (dispersed in Monte-Carlo runs).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from sim import config

# ---------------------------------------------------------------------------
# Pre-build the Cd interpolator (cubic spline, clamped outside table range)
# ---------------------------------------------------------------------------
_cd_interp: interp1d = interp1d(
    config.CD_TABLE_MACH,
    config.CD_TABLE_VALUE,
    kind="cubic",
    bounds_error=False,
    fill_value=(config.CD_TABLE_VALUE[0], config.CD_TABLE_VALUE[-1]),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mach_number(v: float, speed_of_sound: float) -> float:
    """Compute the Mach number.

    Parameters
    ----------
    v : float
        Airspeed magnitude (m/s).
    speed_of_sound : float
        Local speed of sound (m/s).  Must be positive.

    Returns
    -------
    float
        Mach number (dimensionless).
    """
    if speed_of_sound <= 0.0:
        return 0.0
    return v / speed_of_sound


def drag_coefficient(mach: float) -> float:
    """Interpolate Cd from the config table, applying the scale factor.

    Parameters
    ----------
    mach : float
        Current Mach number.

    Returns
    -------
    float
        Scaled drag coefficient.
    """
    cd_base: float = float(_cd_interp(mach))
    return cd_base * config.CD_SCALE_FACTOR


def dynamic_pressure(rho: float, v: float) -> float:
    """Dynamic pressure *q* (Pa).

    Parameters
    ----------
    rho : float
        Atmospheric density (kg/m^3).
    v : float
        Airspeed magnitude (m/s).

    Returns
    -------
    float
        Dynamic pressure (Pa).
    """
    return 0.5 * rho * v * v


# ---------------------------------------------------------------------------
# Main aerodynamic model
# ---------------------------------------------------------------------------


class AerodynamicsModel:
    """Stateful aerodynamics model that tracks max-q and stability metrics.

    Parameters
    ----------
    reference_area : float, optional
        Reference cross-section area (m^2).  Defaults to ``REFERENCE_AREA_M2``
        from config.
    vehicle_length : float, optional
        Total vehicle length (m).  Used for CoP / CoM stability calculations.
    cop_offset_from_nose : float, optional
        Distance from vehicle nose to centre of pressure (m).
    """

    def __init__(
        self,
        reference_area: float = config.REFERENCE_AREA_M2,
        vehicle_length: float = config.VEHICLE_LENGTH_M,
        cop_offset_from_nose: float = config.COP_OFFSET_FROM_NOSE_M,
    ) -> None:
        self.reference_area: float = reference_area
        self.vehicle_length: float = vehicle_length
        self.cop_offset_from_nose: float = cop_offset_from_nose

        # Tracking
        self._max_q: float = 0.0
        self._current_q: float = 0.0

    # -- Public interface ----------------------------------------------------

    def compute_drag(
        self,
        velocity_body: NDArray[np.float64],
        rho: float,
        speed_of_sound: float,
    ) -> NDArray[np.float64]:
        """Compute the aerodynamic drag force vector.

        The drag force opposes the velocity vector.

        Parameters
        ----------
        velocity_body : ndarray, shape (3,)
            Velocity of the vehicle relative to the atmosphere (m/s), in the
            body or inertial frame (the caller is responsible for consistency).
        rho : float
            Atmospheric density (kg/m^3).
        speed_of_sound : float
            Local speed of sound (m/s).

        Returns
        -------
        ndarray, shape (3,)
            Drag force vector (N), opposing *velocity_body*.
        """
        v_mag: float = float(np.linalg.norm(velocity_body))
        if v_mag < 1.0e-6:
            return np.zeros(3)

        mach: float = mach_number(v_mag, speed_of_sound)
        cd: float = drag_coefficient(mach)
        q: float = dynamic_pressure(rho, v_mag)

        # Update max-q tracking
        self._current_q = q
        if q > self._max_q:
            self._max_q = q

        # Drag magnitude and direction (opposing velocity)
        f_drag_mag: float = q * cd * self.reference_area
        drag_unit: NDArray[np.float64] = -velocity_body / v_mag
        return f_drag_mag * drag_unit

    # -- Stability metrics ---------------------------------------------------

    def cop_com_margin(self, com_offset_from_nose: float) -> float:
        """Static stability margin.

        Parameters
        ----------
        com_offset_from_nose : float
            Distance from vehicle nose to centre of mass (m).

        Returns
        -------
        float
            Signed margin (m).  Positive means CoP is forward of CoM
            (statically stable).
        """
        # CoP forward of CoM => stable if cop_offset < com_offset
        # (both measured from nose, so *smaller* offset = more forward)
        return com_offset_from_nose - self.cop_offset_from_nose

    def max_q_fraction(self) -> float:
        """Fraction of structural max-q limit currently being experienced.

        Returns
        -------
        float
            ``current_q / MAX_Q_PA``.  Values above 1.0 indicate the
            structural limit is exceeded.
        """
        if config.MAX_Q_PA <= 0.0:
            return 0.0
        return self._current_q / config.MAX_Q_PA

    @property
    def max_q_experienced(self) -> float:
        """Highest dynamic pressure seen so far (Pa)."""
        return self._max_q

    @property
    def current_q(self) -> float:
        """Most recently computed dynamic pressure (Pa)."""
        return self._current_q
