"""RK4 integrator — knows nothing about physics, only integrates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sim.core.state import VehicleState


class StateDot:
    """Time derivatives of the vehicle state."""

    __slots__ = (
        "velocity_eci",
        "acceleration_eci",
        "quaternion_dot",
        "angular_acceleration_body",
        "mass_rate_kg_s",
    )

    def __init__(
        self,
        velocity_eci: np.ndarray | None = None,
        acceleration_eci: np.ndarray | None = None,
        quaternion_dot: np.ndarray | None = None,
        angular_acceleration_body: np.ndarray | None = None,
        mass_rate_kg_s: float = 0.0,
    ):
        self.velocity_eci = velocity_eci if velocity_eci is not None else np.zeros(3)
        self.acceleration_eci = acceleration_eci if acceleration_eci is not None else np.zeros(3)
        self.quaternion_dot = quaternion_dot if quaternion_dot is not None else np.zeros(4)
        self.angular_acceleration_body = (
            angular_acceleration_body if angular_acceleration_body is not None else np.zeros(3)
        )
        self.mass_rate_kg_s = mass_rate_kg_s

    def scale(self, factor: float) -> StateDot:
        """Return a new StateDot scaled by factor."""
        return StateDot(
            velocity_eci=self.velocity_eci * factor,
            acceleration_eci=self.acceleration_eci * factor,
            quaternion_dot=self.quaternion_dot * factor,
            angular_acceleration_body=self.angular_acceleration_body * factor,
            mass_rate_kg_s=self.mass_rate_kg_s * factor,
        )

    def add(self, other: StateDot) -> StateDot:
        """Return a new StateDot that is self + other."""
        return StateDot(
            velocity_eci=self.velocity_eci + other.velocity_eci,
            acceleration_eci=self.acceleration_eci + other.acceleration_eci,
            quaternion_dot=self.quaternion_dot + other.quaternion_dot,
            angular_acceleration_body=self.angular_acceleration_body + other.angular_acceleration_body,
            mass_rate_kg_s=self.mass_rate_kg_s + other.mass_rate_kg_s,
        )


DerivativesFn = Callable[[float, VehicleState], StateDot]


def _apply_state_dot(state: VehicleState, dot: StateDot, dt: float) -> VehicleState:
    """Apply derivatives to state over timestep dt.

    Builds the new state directly rather than copying first; the copy()
    allocated four arrays that were immediately overwritten.
    """
    return VehicleState(
        position_eci=state.position_eci + dot.velocity_eci * dt,
        velocity_eci=state.velocity_eci + dot.acceleration_eci * dt,
        quaternion=state.quaternion + dot.quaternion_dot * dt,
        angular_velocity_body=state.angular_velocity_body + dot.angular_acceleration_body * dt,
        mass_kg=max(0.0, state.mass_kg + dot.mass_rate_kg_s * dt),
        time_s=state.time_s + dt,
    )


def rk4_step(
    state: VehicleState,
    derivatives_fn: DerivativesFn,
    dt: float,
) -> VehicleState:
    """Advance state by one RK4 step.

    Args:
        state: Current vehicle state.
        derivatives_fn: Callback (t, state) -> StateDot.
        dt: Timestep (s).

    Returns:
        New vehicle state at t + dt.
    """
    t = state.time_s
    half_dt = 0.5 * dt

    k1 = derivatives_fn(t, state)
    s2 = _apply_state_dot(state, k1, half_dt)
    k2 = derivatives_fn(t + half_dt, s2)
    s3 = _apply_state_dot(state, k2, half_dt)
    k3 = derivatives_fn(t + half_dt, s3)
    s4 = _apply_state_dot(state, k3, dt)
    k4 = derivatives_fn(t + dt, s4)

    # Weighted RK4 increment applied directly: state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4).
    # Done inline (rather than via StateDot.scale/add helpers) to avoid allocating
    # several transient StateDot objects every step.
    sixth = dt / 6.0

    new_pos = state.position_eci + sixth * (
        k1.velocity_eci + 2.0 * (k2.velocity_eci + k3.velocity_eci) + k4.velocity_eci
    )
    new_vel = state.velocity_eci + sixth * (
        k1.acceleration_eci + 2.0 * (k2.acceleration_eci + k3.acceleration_eci) + k4.acceleration_eci
    )
    new_quat = state.quaternion + sixth * (
        k1.quaternion_dot + 2.0 * (k2.quaternion_dot + k3.quaternion_dot) + k4.quaternion_dot
    )
    new_omega = state.angular_velocity_body + sixth * (
        k1.angular_acceleration_body
        + 2.0 * (k2.angular_acceleration_body + k3.angular_acceleration_body)
        + k4.angular_acceleration_body
    )
    new_mass = state.mass_kg + sixth * (
        k1.mass_rate_kg_s + 2.0 * (k2.mass_rate_kg_s + k3.mass_rate_kg_s) + k4.mass_rate_kg_s
    )

    new_state = VehicleState(
        position_eci=new_pos,
        velocity_eci=new_vel,
        quaternion=new_quat,
        angular_velocity_body=new_omega,
        mass_kg=max(0.0, new_mass),
        time_s=state.time_s + dt,
    )
    new_state.normalize_quaternion()

    # NaN/Inf check
    if not (
        np.all(np.isfinite(new_state.position_eci))
        and np.all(np.isfinite(new_state.velocity_eci))
        and np.all(np.isfinite(new_state.quaternion))
        and np.all(np.isfinite(new_state.angular_velocity_body))
    ):
        raise RuntimeError(f"NaN/Inf detected in integrator at t={new_state.time_s:.3f}s")
    if not np.isfinite(new_state.mass_kg):
        raise RuntimeError(f"NaN/Inf in mass at t={new_state.time_s:.3f}s")

    return new_state
