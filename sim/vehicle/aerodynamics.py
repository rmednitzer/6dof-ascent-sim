"""Aerodynamic force and moment model.

Computes axial drag, normal (side) force, and aerodynamic pitching moment
as functions of velocity, atmospheric density, Mach number, and angle of
attack.  Cd and CN_alpha are interpolated from tables in ``sim.config``.

The normal force model follows slender-body theory (Barrowman, 1967) and
the pitch-damping term follows the standard aerodynamic derivative
formulation (Nelson, *Flight Stability and Automatic Control*, 1998).

References:
    Barrowman, J.S., "The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles", 1967.
    Nelson, R.C., *Flight Stability and Automatic Control*, 2nd ed., 1998.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from sim import config
from sim.core.fast_math import norm3

# ---------------------------------------------------------------------------
# Pre-build cubic splines (clamped outside table range).
# ``CubicSpline.__call__`` is significantly faster than ``interp1d(kind="cubic")``
# because it operates directly on pre-computed piecewise polynomial
# coefficients without allocating intermediate arrays on every call.
# ---------------------------------------------------------------------------
_CD_TABLE_MACH = np.asarray(config.CD_TABLE_MACH, dtype=float)
_CD_TABLE_VALUE = np.asarray(config.CD_TABLE_VALUE, dtype=float)
_CN_ALPHA_TABLE_MACH = np.asarray(config.CN_ALPHA_TABLE_MACH, dtype=float)
_CN_ALPHA_TABLE_VALUE = np.asarray(config.CN_ALPHA_TABLE_VALUE, dtype=float)

_cd_spline = CubicSpline(_CD_TABLE_MACH, _CD_TABLE_VALUE, bc_type="not-a-knot", extrapolate=False)
_cn_alpha_spline = CubicSpline(_CN_ALPHA_TABLE_MACH, _CN_ALPHA_TABLE_VALUE, bc_type="not-a-knot", extrapolate=False)

_CD_MACH_LO: float = float(_CD_TABLE_MACH[0])
_CD_MACH_HI: float = float(_CD_TABLE_MACH[-1])
_CD_VAL_LO: float = float(_CD_TABLE_VALUE[0])
_CD_VAL_HI: float = float(_CD_TABLE_VALUE[-1])

_CN_MACH_LO: float = float(_CN_ALPHA_TABLE_MACH[0])
_CN_MACH_HI: float = float(_CN_ALPHA_TABLE_MACH[-1])
_CN_VAL_LO: float = float(_CN_ALPHA_TABLE_VALUE[0])
_CN_VAL_HI: float = float(_CN_ALPHA_TABLE_VALUE[-1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mach_number(v: float, speed_of_sound: float) -> float:
    """Compute the Mach number."""
    if speed_of_sound <= 0.0:
        return 0.0
    return v / speed_of_sound


def drag_coefficient(mach: float) -> float:
    """Interpolate Cd from the config table, applying the scale factor.

    The cubic spline is clamped to the table endpoints for out-of-range
    inputs, matching the behaviour of the original ``interp1d`` wrapper.
    """
    if mach <= _CD_MACH_LO:
        cd_base = _CD_VAL_LO
    elif mach >= _CD_MACH_HI:
        cd_base = _CD_VAL_HI
    else:
        cd_base = float(_cd_spline(mach))
    return cd_base * config.CD_SCALE_FACTOR


def normal_force_coefficient_slope(mach: float) -> float:
    """Interpolate CN_alpha (per radian) from the config table."""
    if mach <= _CN_MACH_LO:
        return _CN_VAL_LO
    if mach >= _CN_MACH_HI:
        return _CN_VAL_HI
    return float(_cn_alpha_spline(mach))


def dynamic_pressure(rho: float, v: float) -> float:
    """Dynamic pressure *q* (Pa)."""
    return 0.5 * rho * v * v


# ---------------------------------------------------------------------------
# Aerodynamic result container
# ---------------------------------------------------------------------------


class AeroForces:
    """Container for aerodynamic forces and moments."""

    __slots__ = ("drag_force_eci", "normal_force_body", "aero_moment_body", "alpha_rad")

    def __init__(
        self,
        drag_force_eci: NDArray[np.float64],
        normal_force_body: NDArray[np.float64],
        aero_moment_body: NDArray[np.float64],
        alpha_rad: float,
    ) -> None:
        self.drag_force_eci = drag_force_eci
        self.normal_force_body = normal_force_body
        self.aero_moment_body = aero_moment_body
        self.alpha_rad = alpha_rad


# ---------------------------------------------------------------------------
# Main aerodynamic model
# ---------------------------------------------------------------------------


class AerodynamicsModel:
    """Stateful aerodynamics model with drag, normal force, and pitch damping.

    Computes:
    - Axial drag (opposes velocity, Mach-dependent Cd)
    - Normal force from angle of attack (CN_alpha * alpha * q * S_ref)
    - Restoring/destabilizing pitching moment from CoP-CoM offset
    - Pitch damping moment (Cmq)

    Parameters
    ----------
    reference_area : float, optional
        Reference cross-section area (m^2).
    vehicle_length : float, optional
        Total vehicle length (m).
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

    def reset(self) -> None:
        """Reset tracking state for a new simulation run."""
        self._max_q = 0.0
        self._current_q = 0.0

    # -- Public interface ----------------------------------------------------

    def compute_drag(
        self,
        velocity_body: NDArray[np.float64],
        rho: float,
        speed_of_sound: float,
    ) -> NDArray[np.float64]:
        """Compute the aerodynamic drag force vector (backward compatible).

        Parameters
        ----------
        velocity_body : ndarray, shape (3,)
            Velocity of the vehicle relative to the atmosphere (m/s).
        rho : float
            Atmospheric density (kg/m^3).
        speed_of_sound : float
            Local speed of sound (m/s).

        Returns
        -------
        ndarray, shape (3,)
            Drag force vector (N), opposing *velocity_body*.
        """
        v_mag: float = norm3(velocity_body)
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

    def compute_aero_forces(
        self,
        vel_rel_eci: NDArray[np.float64],
        quaternion: NDArray[np.float64],
        omega_body: NDArray[np.float64],
        rho: float,
        speed_of_sound: float,
        com_offset_from_nose: float,
    ) -> AeroForces:
        """Compute full aerodynamic forces and moments.

        Includes axial drag, normal force from angle of attack, restoring
        moment from CoP-CoM offset, and pitch-damping moment.

        Parameters
        ----------
        vel_rel_eci : ndarray, shape (3,)
            Atmosphere-relative velocity in ECI (m/s).
        quaternion : ndarray, shape (4,)
            Attitude quaternion [x, y, z, w].
        omega_body : ndarray, shape (3,)
            Body angular velocity (rad/s).
        rho : float
            Atmospheric density (kg/m^3).
        speed_of_sound : float
            Local speed of sound (m/s).
        com_offset_from_nose : float
            Distance from nose to center of mass (m).

        Returns
        -------
        AeroForces
            Container with drag, normal force, moment, and alpha.
        """
        from sim.core.reference_frames import eci_to_body

        v_mag: float = norm3(vel_rel_eci)
        if v_mag < 1.0e-6 or rho < 1.0e-15:
            return AeroForces(
                drag_force_eci=np.zeros(3),
                normal_force_body=np.zeros(3),
                aero_moment_body=np.zeros(3),
                alpha_rad=0.0,
            )

        mach: float = mach_number(v_mag, speed_of_sound)
        cd: float = drag_coefficient(mach)
        q: float = dynamic_pressure(rho, v_mag)

        self._current_q = q
        if q > self._max_q:
            self._max_q = q

        # --- Axial drag in ECI (opposes velocity) ---
        v_hat_eci = vel_rel_eci / v_mag
        f_drag_eci = -q * cd * self.reference_area * v_hat_eci

        # --- Transform velocity to body frame for AoA computation ---
        vel_body = eci_to_body(vel_rel_eci, quaternion)
        vb_mag = norm3(vel_body)
        if vb_mag < 1.0e-6:
            return AeroForces(
                drag_force_eci=f_drag_eci,
                normal_force_body=np.zeros(3),
                aero_moment_body=np.zeros(3),
                alpha_rad=0.0,
            )

        # Angle of attack: angle between body +X axis and velocity in body frame
        # Body X is the vehicle's longitudinal axis (thrust direction)
        cos_alpha = vel_body[0] / vb_mag
        if cos_alpha > 1.0:
            cos_alpha = 1.0
        elif cos_alpha < -1.0:
            cos_alpha = -1.0
        # Full [0, pi] range. The previous abs() folded backward flight
        # (cos_alpha < 0, AoA > 90deg) into [0, 90deg], understating AoA
        # and normal force exactly when the vehicle is tumbling/aborting.
        alpha = math.acos(cos_alpha)

        # Normal force direction: component of velocity perpendicular to body X
        # in the body frame
        vp1, vp2 = vel_body[1], vel_body[2]
        vel_perp_mag = math.sqrt(vp1 * vp1 + vp2 * vp2)

        normal_force_body = np.zeros(3)
        aero_moment_body = np.zeros(3)

        if vel_perp_mag > 1.0e-6 and alpha > 1.0e-6:
            # Normal force direction opposes the lateral velocity component
            inv_mag = 1.0 / vel_perp_mag
            n_hat = np.array([0.0, -vp1 * inv_mag, -vp2 * inv_mag])

            # CN_alpha from Mach-dependent table (per radian)
            cn_alpha = normal_force_coefficient_slope(mach)

            # Normal force magnitude: F_N = CN_alpha * alpha * q * S_ref
            f_normal_mag = cn_alpha * alpha * q * self.reference_area
            normal_force_body = f_normal_mag * n_hat

            # --- Restoring moment from CoP-CoM offset ---
            # Moment arm: distance from CoM to CoP (positive = CoP forward of CoM = stable)
            # Both measured from nose, so moment_arm = CoM_offset - CoP_offset
            # Positive moment_arm → restoring moment (statically stable)
            cop_com_arm = com_offset_from_nose - self.cop_offset_from_nose

            # Pitching moment about CoM: M = F_N * arm
            # Cross body-x = [1,0,0] with n_hat = [0, nh1, nh2] simplifies to
            # [0, -nh2, nh1]; then normalise.
            nh1, nh2 = n_hat[1], n_hat[2]
            t_mag = math.sqrt(nh1 * nh1 + nh2 * nh2)
            if t_mag > 1.0e-10:
                inv_t = 1.0 / t_mag
                axis1 = -nh2 * inv_t
                axis2 = nh1 * inv_t
                scale = cop_com_arm * f_normal_mag
                aero_moment_body[1] += scale * axis1
                aero_moment_body[2] += scale * axis2

        # --- Pitch damping moment (Cmq) ---
        # M_damp = Cmq * (q * S_ref * L) * (omega * L / (2V))
        # This provides aerodynamic damping of angular rates
        cmq = config.CMQ_PITCH_DAMPING
        if v_mag > 10.0:
            L = self.vehicle_length
            qSL = q * self.reference_area * L
            damping_factor = L / (2.0 * v_mag)
            # Apply to pitch (Y) and yaw (Z) body rates
            aero_moment_body[1] += cmq * qSL * omega_body[1] * damping_factor
            aero_moment_body[2] += cmq * qSL * omega_body[2] * damping_factor

        return AeroForces(
            drag_force_eci=f_drag_eci,
            normal_force_body=normal_force_body,
            aero_moment_body=aero_moment_body,
            alpha_rad=alpha,
        )

    # -- Stability metrics ---------------------------------------------------

    def cop_com_margin(self, com_offset_from_nose: float) -> float:
        """Static stability margin (m). Positive = statically stable."""
        return com_offset_from_nose - self.cop_offset_from_nose

    def max_q_fraction(self) -> float:
        """Fraction of structural max-q limit currently experienced."""
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
