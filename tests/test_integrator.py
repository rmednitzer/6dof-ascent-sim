"""Tests for the RK4 integrator (sim.core.integrator).

Verifies correctness of the RK4 scheme by integrating a simple ODE with a
known analytical solution, and checks StateDot arithmetic helpers.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from sim.core.integrator import StateDot, rk4_step
from sim.core.state import VehicleState

# ---------------------------------------------------------------------------
# StateDot helpers
# ---------------------------------------------------------------------------


class TestStateDot:
    """Tests for StateDot construction, scale, and add operations."""

    def test_default_construction(self):
        """Default StateDot should have zero arrays and zero mass rate."""
        sd = StateDot()
        npt.assert_array_equal(sd.velocity_eci, np.zeros(3))
        npt.assert_array_equal(sd.acceleration_eci, np.zeros(3))
        npt.assert_array_equal(sd.quaternion_dot, np.zeros(4))
        npt.assert_array_equal(sd.angular_acceleration_body, np.zeros(3))
        assert sd.mass_rate_kg_s == 0.0

    def test_scale(self):
        """Scaling a StateDot multiplies every field by the factor."""
        sd = StateDot(
            velocity_eci=np.array([1.0, 2.0, 3.0]),
            acceleration_eci=np.array([4.0, 5.0, 6.0]),
            quaternion_dot=np.array([0.1, 0.2, 0.3, 0.4]),
            angular_acceleration_body=np.array([7.0, 8.0, 9.0]),
            mass_rate_kg_s=-10.0,
        )
        scaled = sd.scale(2.0)
        npt.assert_allclose(scaled.velocity_eci, [2.0, 4.0, 6.0])
        npt.assert_allclose(scaled.acceleration_eci, [8.0, 10.0, 12.0])
        npt.assert_allclose(scaled.quaternion_dot, [0.2, 0.4, 0.6, 0.8])
        npt.assert_allclose(scaled.angular_acceleration_body, [14.0, 16.0, 18.0])
        assert scaled.mass_rate_kg_s == -20.0

    def test_add(self):
        """Adding two StateDots produces element-wise sums."""
        a = StateDot(velocity_eci=np.array([1.0, 0.0, 0.0]), mass_rate_kg_s=-1.0)
        b = StateDot(velocity_eci=np.array([0.0, 2.0, 0.0]), mass_rate_kg_s=-3.0)
        c = a.add(b)
        npt.assert_allclose(c.velocity_eci, [1.0, 2.0, 0.0])
        assert c.mass_rate_kg_s == -4.0


# ---------------------------------------------------------------------------
# RK4 integrator correctness
# ---------------------------------------------------------------------------


class TestRK4Step:
    """Verify RK4 integration against known analytical solutions."""

    def test_exponential_growth_dx_dt_equals_x(self):
        """Integrate dx/dt = x with x(0)=1.  Exact solution: x(t) = e^t.

        We encode x in the position_eci[0] component and use the velocity
        as the derivative.  After many small steps the numerical solution
        should match e^t to high accuracy.
        """
        # Initial state: position_eci[0] = 1.0
        state = VehicleState(
            position_eci=np.array([1.0, 0.0, 0.0]),
            velocity_eci=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=100.0,
            time_s=0.0,
        )

        def derivatives_fn(_t, s):
            """dx/dt = x  =>  velocity = position (so position integrates)."""
            return StateDot(
                velocity_eci=np.array([s.position_eci[0], 0.0, 0.0]),
                acceleration_eci=np.array([s.velocity_eci[0], 0.0, 0.0]),
            )

        dt = 0.01
        n_steps = 100  # integrate to t=1.0
        for _ in range(n_steps):
            state = rk4_step(state, derivatives_fn, dt)

        # At t=1.0, position[0] should be e^1 = 2.71828...
        expected = math.exp(1.0)
        npt.assert_allclose(state.position_eci[0], expected, rtol=1e-8)
        npt.assert_allclose(state.time_s, 1.0, atol=1e-12)

    def test_constant_acceleration(self):
        """Integrate constant acceleration: x(t) = x0 + v0*t + 0.5*a*t^2.

        Set acceleration_eci = [0, 0, 10] (constant) and velocity starts at
        [100, 0, 0].  After 1 second position should match kinematics exactly.
        """
        state = VehicleState(
            position_eci=np.zeros(3),
            velocity_eci=np.array([100.0, 0.0, 0.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            mass_kg=50.0,
            time_s=0.0,
        )

        accel = np.array([0.0, 0.0, 10.0])

        def const_accel_fn(_t, s):
            return StateDot(
                velocity_eci=s.velocity_eci.copy(),
                acceleration_eci=accel.copy(),
            )

        dt = 0.01
        for _ in range(100):
            state = rk4_step(state, const_accel_fn, dt)

        t = 1.0
        npt.assert_allclose(state.position_eci[0], 100.0 * t, rtol=1e-8)
        npt.assert_allclose(state.position_eci[2], 0.5 * 10.0 * t**2, rtol=1e-8)
        npt.assert_allclose(state.velocity_eci[2], 10.0 * t, rtol=1e-8)

    def test_mass_decreases(self):
        """Mass should decrease when mass_rate_kg_s is negative."""
        state = VehicleState(mass_kg=1000.0, time_s=0.0)

        def burning(_t, _s):
            return StateDot(mass_rate_kg_s=-100.0)

        dt = 0.1
        for _ in range(5):
            state = rk4_step(state, burning, dt)

        # 1000 - 100*0.5 = 950
        npt.assert_allclose(state.mass_kg, 950.0, atol=1e-10)

    def test_mass_does_not_go_negative(self):
        """Mass should be clamped to zero, not become negative."""
        state = VehicleState(mass_kg=10.0, time_s=0.0)

        def burning(_t, _s):
            return StateDot(mass_rate_kg_s=-1000.0)

        state = rk4_step(state, burning, dt=1.0)
        assert state.mass_kg >= 0.0

    def test_quaternion_normalised_after_step(self):
        """Quaternion should be re-normalised after each RK4 step."""
        state = VehicleState(
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            time_s=0.0,
        )

        def spinning(_t, _s):
            return StateDot(
                quaternion_dot=np.array([0.01, 0.02, 0.03, 0.0]),
            )

        state = rk4_step(state, spinning, dt=0.1)
        npt.assert_allclose(np.linalg.norm(state.quaternion), 1.0, atol=1e-12)

    def test_nan_raises_runtime_error(self):
        """If derivatives return NaN the integrator should raise RuntimeError."""
        state = VehicleState(time_s=0.0)

        def nan_fn(_t, _s):
            return StateDot(velocity_eci=np.array([float("nan"), 0.0, 0.0]))

        with pytest.raises(RuntimeError, match="NaN"):
            rk4_step(state, nan_fn, dt=0.01)

    def test_time_advances(self):
        """The state time_s should advance by dt each step."""
        state = VehicleState(time_s=5.0)

        def zero_fn(_t, _s):
            return StateDot()

        state = rk4_step(state, zero_fn, dt=0.25)
        npt.assert_allclose(state.time_s, 5.25, atol=1e-15)
