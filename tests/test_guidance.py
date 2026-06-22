"""Tests for the ascent guidance law (sim.gnc.guidance).

Focus: inclination targeting (AD-17 / ADR 0024) — the Earth-rotation azimuth
correction, the target inertial-plane normal, and the terminal yaw out-of-plane
steering that together reach ``TARGET_INCLINATION_DEG`` (a pure gravity turn
misses it by ~6°).
"""

from __future__ import annotations

import math

import numpy as np

from sim import config
from sim.core.state import VehicleState
from sim.gnc.guidance import GuidanceLaw


def _enu_basis():
    """East and North unit vectors at the launch site (ECEF ≈ ECI at t=0)."""
    lat, lon = math.radians(config.LAUNCH_LAT_DEG), math.radians(config.LAUNCH_LON_DEG)
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    return east, north


class TestInclinationTargeting:
    """AD-17 / ADR 0024: reach the target inclination, not the ~45° gravity-turn miss."""

    def test_rotation_correction_steers_more_northerly(self):
        # The flight azimuth subtracts the launch-site eastward rotation velocity,
        # so it is more northerly (smaller azimuth from north) than the inertial one.
        east, north = _enu_basis()
        dr_in = GuidanceLaw._compute_launch_downrange(rotation_correction=False)
        dr_fl = GuidanceLaw._compute_launch_downrange(rotation_correction=True)
        az_in = math.atan2(float(dr_in @ east), float(dr_in @ north))
        az_fl = math.atan2(float(dr_fl @ east), float(dr_fl @ north))
        assert az_fl < az_in
        assert not np.allclose(dr_in, dr_fl)

    def test_target_plane_normal_matches_target_inclination(self):
        g = GuidanceLaw()
        n = g._target_plane_normal
        npt_inc = math.degrees(math.acos(abs(float(n[2]))))
        assert abs(npt_inc - config.TARGET_INCLINATION_DEG) < 0.5

    def test_plane_steering_opposes_out_of_plane_velocity(self):
        g = GuidanceLaw()
        n = g._target_plane_normal
        # A commanded direction lying in the target plane, and a velocity purely
        # along +n (out of plane): the steered command must acquire a -n component.
        in_plane = np.array([1.0, 0.0, 0.0]) - float(np.array([1.0, 0.0, 0.0]) @ n) * n
        in_plane /= np.linalg.norm(in_plane)
        state = VehicleState(
            position_eci=np.array([config.EARTH_RADIUS_M + 400_000.0, 0.0, 0.0]),
            velocity_eci=n * 100.0,
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=10_000.0,
            time_s=300.0,
        )
        steered = g._apply_plane_steering(in_plane, state)
        assert float(steered @ n) < -1e-3  # thrust tilts against the out-of-plane velocity

    def test_plane_steering_noop_when_in_plane(self):
        g = GuidanceLaw()
        n = g._target_plane_normal
        in_plane = np.array([0.0, 1.0, 0.0]) - float(np.array([0.0, 1.0, 0.0]) @ n) * n
        in_plane /= np.linalg.norm(in_plane)
        state = VehicleState(
            position_eci=np.array([config.EARTH_RADIUS_M + 400_000.0, 0.0, 0.0]),
            velocity_eci=in_plane * 7000.0,  # velocity already in-plane -> v_op ≈ 0
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=10_000.0,
            time_s=300.0,
        )
        steered = g._apply_plane_steering(in_plane, state)
        assert abs(float(steered @ n)) < 1e-3  # no steering when there is no out-of-plane velocity
