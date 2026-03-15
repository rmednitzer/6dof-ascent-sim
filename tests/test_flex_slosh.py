"""Tests for FlexBody and SloshModel (sim.dynamics.flex_body, sim.dynamics.slosh).

Validates initialization, update output shapes, zero-input behaviour, and
basic physical response.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sim import config
from sim.dynamics.flex_body import FlexBody
from sim.dynamics.slosh import SloshModel

# ---------------------------------------------------------------------------
# FlexBody tests
# ---------------------------------------------------------------------------


class TestFlexBodyInit:
    """FlexBody initialization and configuration."""

    def test_default_mode_count(self):
        """Default FlexBody should have modes matching config."""
        fb = FlexBody()
        assert fb.n_modes == len(config.FLEX_MODE_FREQS_HZ)

    def test_custom_mode_count(self):
        """FlexBody with fewer modes than config."""
        fb = FlexBody(n_modes=2)
        assert fb.n_modes == 2

    def test_mode_count_clamped_to_config(self):
        """Requesting more modes than config should clamp."""
        fb = FlexBody(n_modes=100)
        assert fb.n_modes == len(config.FLEX_MODE_FREQS_HZ)

    def test_initial_displacements_are_zero(self):
        """All modal displacements and velocities should start at zero."""
        fb = FlexBody()
        npt.assert_array_equal(fb.modal_displacements(), np.zeros(fb.n_modes))
        npt.assert_array_equal(fb.modal_velocities(), np.zeros(fb.n_modes))


class TestFlexBodyUpdate:
    """FlexBody.update() behaviour."""

    def test_update_returns_correct_shape(self):
        """update() should return an array of shape (n_modes,)."""
        fb = FlexBody()
        result = fb.update(dt=0.01, tvc_force_n=0.0, propellant_fraction=1.0)
        assert result.shape == (fb.n_modes,)

    def test_zero_input_gives_near_zero_output(self):
        """With zero TVC force and zero initial state, output should be ~0."""
        fb = FlexBody()
        result = fb.update(dt=0.01, tvc_force_n=0.0, propellant_fraction=1.0)
        npt.assert_allclose(result, np.zeros(fb.n_modes), atol=1e-15)

    def test_zero_input_total_bending_rate_zero(self):
        """total_bending_rate_at_imu should be zero with no forcing."""
        fb = FlexBody()
        fb.update(dt=0.01, tvc_force_n=0.0, propellant_fraction=1.0)
        npt.assert_allclose(fb.total_bending_rate_at_imu(), 0.0, atol=1e-15)

    def test_nonzero_force_produces_nonzero_output(self):
        """Applying a TVC force should excite bending modes."""
        fb = FlexBody()
        # Apply force for several steps
        for _ in range(10):
            fb.update(dt=0.01, tvc_force_n=1000.0, propellant_fraction=0.8)

        # Modal velocities should be non-zero
        vel = fb.modal_velocities()
        assert np.any(np.abs(vel) > 1e-10)

    def test_reset_zeros_all_modes(self):
        """reset() should zero all modal states."""
        fb = FlexBody()
        for _ in range(10):
            fb.update(dt=0.01, tvc_force_n=500.0, propellant_fraction=0.5)

        fb.reset()
        npt.assert_array_equal(fb.modal_displacements(), np.zeros(fb.n_modes))
        npt.assert_array_equal(fb.modal_velocities(), np.zeros(fb.n_modes))

    def test_damping_dissipates_energy(self):
        """After removing forcing, energy should decay due to damping."""
        fb = FlexBody()

        # Excite modes
        for _ in range(100):
            fb.update(dt=0.01, tvc_force_n=5000.0, propellant_fraction=0.8)

        energy_after_forcing = fb.kinetic_energy() + fb.potential_energy(0.8)
        assert energy_after_forcing > 0

        # Let it ring down with no forcing
        for _ in range(1000):
            fb.update(dt=0.01, tvc_force_n=0.0, propellant_fraction=0.8)

        energy_after_decay = fb.kinetic_energy() + fb.potential_energy(0.8)
        assert energy_after_decay < energy_after_forcing

    def test_propellant_fraction_affects_frequency(self):
        """Different propellant fractions should change the response."""
        fb_full = FlexBody()
        fb_empty = FlexBody()

        # Apply same forcing with different propellant fractions
        for _ in range(50):
            fb_full.update(dt=0.01, tvc_force_n=1000.0, propellant_fraction=1.0)
            fb_empty.update(dt=0.01, tvc_force_n=1000.0, propellant_fraction=0.0)

        # Responses should differ (different natural frequencies)
        disp_full = fb_full.modal_displacements()
        disp_empty = fb_empty.modal_displacements()

        assert not np.allclose(disp_full, disp_empty), "Full and empty propellant should give different flex responses"


# ---------------------------------------------------------------------------
# SloshModel tests
# ---------------------------------------------------------------------------


class TestSloshModelInit:
    """SloshModel initialization."""

    def test_default_single_tank(self):
        """Default SloshModel should have 1 tank."""
        sm = SloshModel()
        assert sm.n_tanks == 1

    def test_custom_tank_count(self):
        """SloshModel with multiple tanks."""
        sm = SloshModel(n_tanks=3)
        assert sm.n_tanks == 3

    def test_initial_angles_zero(self):
        """Initial pendulum angles should be zero."""
        sm = SloshModel()
        npt.assert_array_equal(sm.pendulum_angles(), np.zeros(1))

    def test_initial_rates_zero(self):
        """Initial pendulum rates should be zero."""
        sm = SloshModel()
        npt.assert_array_equal(sm.pendulum_rates(), np.zeros(1))

    def test_mismatched_offsets_raises(self):
        """Providing wrong-length offsets should raise ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            SloshModel(n_tanks=2, tank_cg_offsets_m=np.array([1.0, 2.0, 3.0]))


class TestSloshModelUpdate:
    """SloshModel.update() behaviour."""

    def test_update_returns_correct_shapes(self):
        """update() should return (forces, torques) each of shape (n_tanks,)."""
        sm = SloshModel(n_tanks=2)
        forces, torques = sm.update(
            dt=0.01,
            lateral_accel_mps2=0.0,
            propellant_mass_kg=100000.0,
            propellant_fraction=1.0,
        )
        assert forces.shape == (2,)
        assert torques.shape == (2,)

    def test_zero_input_gives_near_zero_output(self):
        """With zero lateral acceleration and zero initial state, output ~0."""
        sm = SloshModel()
        forces, torques = sm.update(
            dt=0.01,
            lateral_accel_mps2=0.0,
            propellant_mass_kg=100000.0,
            propellant_fraction=1.0,
        )
        npt.assert_allclose(forces, np.zeros(1), atol=1e-15)
        npt.assert_allclose(torques, np.zeros(1), atol=1e-15)

    def test_nonzero_accel_produces_nonzero_force(self):
        """Lateral acceleration should excite slosh and produce forces."""
        sm = SloshModel()
        for _ in range(50):
            forces, _ = sm.update(
                dt=0.01,
                lateral_accel_mps2=5.0,
                propellant_mass_kg=100000.0,
                propellant_fraction=0.8,
            )

        # After repeated forcing, angles should be non-zero
        assert np.any(np.abs(sm.pendulum_angles()) > 1e-6)
        assert np.any(np.abs(forces) > 1e-6)

    def test_torque_with_offset(self):
        """Torque should be non-zero when tank CG offset is non-zero."""
        sm = SloshModel(n_tanks=1, tank_cg_offsets_m=np.array([5.0]))
        for _ in range(20):
            forces, torques = sm.update(
                dt=0.01,
                lateral_accel_mps2=2.0,
                propellant_mass_kg=50000.0,
                propellant_fraction=0.7,
            )

        assert abs(torques[0]) > 1e-6, "Non-zero offset should produce torque"

    def test_torque_zero_without_offset(self):
        """Torque should be zero when tank CG offset is zero."""
        sm = SloshModel(n_tanks=1, tank_cg_offsets_m=np.array([0.0]))
        for _ in range(20):
            _, torques = sm.update(
                dt=0.01,
                lateral_accel_mps2=2.0,
                propellant_mass_kg=50000.0,
                propellant_fraction=0.7,
            )

        npt.assert_allclose(torques[0], 0.0, atol=1e-15)

    def test_reset_zeros_state(self):
        """reset() should zero all pendulum states."""
        sm = SloshModel()
        for _ in range(20):
            sm.update(dt=0.01, lateral_accel_mps2=5.0, propellant_mass_kg=100000.0, propellant_fraction=0.5)

        sm.reset()
        npt.assert_array_equal(sm.pendulum_angles(), np.zeros(1))
        npt.assert_array_equal(sm.pendulum_rates(), np.zeros(1))

    def test_damping_dissipates_slosh_energy(self):
        """After removing excitation, slosh energy should decay."""
        sm = SloshModel()
        prop_mass = 100000.0
        prop_frac = 0.8

        # Excite
        for _ in range(200):
            sm.update(dt=0.01, lateral_accel_mps2=5.0, propellant_mass_kg=prop_mass, propellant_fraction=prop_frac)

        e_excited = sm.kinetic_energy(prop_mass) + sm.potential_energy(prop_mass, prop_frac)
        assert e_excited > 0

        # Ring down
        for _ in range(2000):
            sm.update(dt=0.01, lateral_accel_mps2=0.0, propellant_mass_kg=prop_mass, propellant_fraction=prop_frac)

        e_decayed = sm.kinetic_energy(prop_mass) + sm.potential_energy(prop_mass, prop_frac)
        assert e_decayed < e_excited
