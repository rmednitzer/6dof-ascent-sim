"""RK4 integrator — knows nothing about physics, only integrates."""

from __future__ import annotations

from typing import Callable

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
    """Apply derivatives to state over timestep dt."""
    new = state.copy()
    new.position_eci = state.position_eci + dot.velocity_eci * dt
    new.velocity_eci = state.velocity_eci + dot.acceleration_eci * dt
    new.quaternion = state.quaternion + dot.quaternion_dot * dt
    new.angular_velocity_body = state.angular_velocity_body + dot.angular_acceleration_body * dt
    new.mass_kg = max(0.0, state.mass_kg + dot.mass_rate_kg_s * dt)
    new.time_s = state.time_s + dt
    return new


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

    k1 = derivatives_fn(t, state)
    s2 = _apply_state_dot(state, k1, 0.5 * dt)
    k2 = derivatives_fn(t + 0.5 * dt, s2)
    s3 = _apply_state_dot(state, k2, 0.5 * dt)
    k3 = derivatives_fn(t + 0.5 * dt, s3)
    s4 = _apply_state_dot(state, k3, dt)
    k4 = derivatives_fn(t + dt, s4)

    # Weighted average: (k1 + 2*k2 + 2*k3 + k4) / 6
    combined = k1.add(k2.scale(2.0)).add(k3.scale(2.0)).add(k4)
    combined = combined.scale(1.0 / 6.0)

    new_state = _apply_state_dot(state, combined, dt)
    new_state.normalize_quaternion()

    # NaN/Inf check
    for arr in [new_state.position_eci, new_state.velocity_eci,
                new_state.quaternion, new_state.angular_velocity_body]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"NaN/Inf detected in integrator at t={new_state.time_s:.3f}s")
    if not np.isfinite(new_state.mass_kg):
        raise RuntimeError(f"NaN/Inf in mass at t={new_state.time_s:.3f}s")

    return new_state
