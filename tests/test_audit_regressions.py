"""Regression tests for the code-audit fixes.

Each test here pins a specific defect that was found and fixed so it
cannot silently return. They intentionally target the highest-risk
correctness bugs (false orbital-insertion SUCCESS, IMU specific force,
orbit validity, AoA sign, slosh sign).
"""

from __future__ import annotations

import math

import numpy as np

from sim import config
from sim.gnc.sensors import IMU
from sim.main import _is_orbital_insertion
from sim.orbital.maneuvers import total_correction_budget
from sim.orbital.propagator import OrbitalElements, OrbitPropagator
from sim.vehicle.aerodynamics import AerodynamicsModel


def _circular_leo_state():
    """A clean, near-circular 400 km equatorial orbit state."""
    from sim.core.state import VehicleState

    r = config.EARTH_RADIUS_M + config.TARGET_ALTITUDE_M
    v_circ = math.sqrt(config.EARTH_MU / r)
    return VehicleState(
        position_eci=np.array([r, 0.0, 0.0]),
        velocity_eci=np.array([0.0, v_circ, 0.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity_body=np.zeros(3),
        mass_kg=10_000.0,
        time_s=500.0,
    )


class TestOrbitalInsertionCriterion:
    """The SUCCESS gate must validate a real orbit, not just alt/vel."""

    def test_clean_circular_orbit_is_insertion(self):
        ok, elements = _is_orbital_insertion(_circular_leo_state(), stage_index=1)
        assert ok is True
        assert elements is not None
        assert elements.eccentricity < config.INSERTION_MAX_ECCENTRICITY
        assert elements.periapsis_alt_km * 1000.0 > config.INSERTION_MIN_PERIAPSIS_ALT_M

    def test_stage_gate_blocks_insertion_on_first_stage(self):
        """Even a perfect orbit is not SUCCESS while on stage 1, and the
        gate lives in the single predicate so the in-loop and end-of-sim
        paths cannot disagree."""
        ok, elements = _is_orbital_insertion(_circular_leo_state(), stage_index=0)
        assert ok is False
        assert elements is None

    def test_suborbital_arc_is_not_insertion(self):
        """The original bug: high altitude but sub-orbital -> NOT SUCCESS.

        ~400 km altitude, 92% of target velocity, ~12 deg flight-path
        angle. The pre-fix gate accepted this; the resulting orbit has a
        deeply negative periapsis (it intersects the Earth).
        """
        from sim.core.state import VehicleState

        r = config.EARTH_RADIUS_M + config.TARGET_ALTITUDE_M
        speed = 0.92 * config.TARGET_VELOCITY_MS
        fpa = math.radians(12.0)
        # Velocity with a 12 deg climb angle (radial + tangential).
        vel = np.array([speed * math.sin(fpa), speed * math.cos(fpa), 0.0])
        state = VehicleState(
            position_eci=np.array([r, 0.0, 0.0]),
            velocity_eci=vel,
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=10_000.0,
            time_s=500.0,
        )
        ok, _ = _is_orbital_insertion(state, stage_index=1)
        assert ok is False

        # And the orbit it implies is genuinely not sustainable.
        elements = OrbitPropagator(state).state_to_elements()
        assert not elements.is_sustainable_leo(
            config.INSERTION_MIN_PERIAPSIS_ALT_M,
            config.INSERTION_MAX_ECCENTRICITY,
        )


class TestOrbitValidation:
    """OrbitalElements must flag non-bound / sub-surface orbits."""

    def test_subsurface_orbit_not_sustainable(self):
        # a < R_earth -> periapsis is below the surface.
        el = OrbitalElements(
            semi_major_axis_m=5_862_000.0,
            eccentricity=0.289,
            inclination_deg=34.0,
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            true_anomaly_deg=0.0,
            period_s=4464.0,
            apoapsis_alt_km=1177.0,
            periapsis_alt_km=-2209.0,
        )
        assert el.is_bound is True  # elliptical, but...
        assert el.is_sustainable_leo(config.INSERTION_MIN_PERIAPSIS_ALT_M, 0.05) is False

    def test_hyperbolic_orbit_not_bound(self):
        el = OrbitalElements(-7_000_000.0, 1.4, 28.0, 0.0, 0.0, 0.0, math.inf, 5000.0, 300.0)
        assert el.is_bound is False
        assert el.is_sustainable_leo(config.INSERTION_MIN_PERIAPSIS_ALT_M, 0.05) is False

    def test_correction_budget_no_crash_on_invalid_orbit(self):
        """maneuvers must return inf, not raise sqrt-domain, for a
        hyperbolic/sub-surface achieved orbit."""
        el = OrbitalElements(-7_000_000.0, 1.4, 28.0, 0.0, 0.0, 0.0, math.inf, 5000.0, 300.0)
        dv = total_correction_budget(el, config.TARGET_ALTITUDE_M, config.TARGET_INCLINATION_DEG)
        assert dv == math.inf


class TestIMUSpecificForce:
    """The IMU must sense specific force, not gravity."""

    def test_free_fall_reads_near_zero(self):
        """Zero specific force (free fall) -> accelerometer ~0."""
        from sim.core.state import VehicleState

        imu = IMU(rng=np.random.default_rng(0))
        state = VehicleState(
            position_eci=np.array([config.EARTH_RADIUS_M + 1e5, 0.0, 0.0]),
            velocity_eci=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=1000.0,
            time_s=1.0,
        )
        m = imu.measure(state, np.zeros(3), config.DT)
        # Only noise/bias remain; must be tiny (not ~9.8 m/s^2 of gravity).
        assert np.linalg.norm(m.accel_body_mps2) < 0.5

    def test_specific_force_rotated_into_body(self):
        """A known ECI specific force appears in the body frame."""
        from sim.core.state import VehicleState

        imu = IMU(rng=np.random.default_rng(0))
        state = VehicleState(
            position_eci=np.array([config.EARTH_RADIUS_M, 0.0, 0.0]),
            velocity_eci=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),  # identity: body == ECI
            angular_velocity_body=np.zeros(3),
            mass_kg=1000.0,
            time_s=1.0,
        )
        f_eci = np.array([30.0, 0.0, 0.0])
        m = imu.measure(state, f_eci, config.DT)
        np.testing.assert_allclose(m.accel_body_mps2, f_eci, atol=0.2)


class TestAeroAngleOfAttackSign:
    """AoA must span [0, pi]; backward flight is > 90 deg."""

    def test_backward_flight_alpha_exceeds_90deg(self):
        aero = AerodynamicsModel()
        # Velocity along -X ECI, vehicle pointing +X (identity quat):
        # the relative wind comes from behind -> AoA ~ 180 deg.
        res = aero.compute_aero_forces(
            vel_rel_eci=np.array([-250.0, 0.0, 0.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            omega_body=np.zeros(3),
            rho=0.4,
            speed_of_sound=300.0,
            com_offset_from_nose=20.0,
        )
        assert math.degrees(res.alpha_rad) > 90.0


class TestSloshReactionSign:
    """Slosh reaction force must oppose the driving lateral accel."""

    def test_reaction_opposes_drive(self):
        from sim.dynamics.slosh import SloshModel

        slosh = SloshModel()
        a_lat = 5.0  # positive lateral acceleration
        forces, _ = slosh.update(
            dt=config.DT,
            lateral_accel_mps2=a_lat,
            propellant_mass_kg=50_000.0,
            propellant_fraction=0.8,
        )
        # After one step the spring+damper reaction must point opposite
        # to the positive drive (negative), not reinforce it.
        assert float(np.sum(forces)) <= 0.0
