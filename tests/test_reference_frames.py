"""Tests for reference frame transformations (sim.core.reference_frames).

Covers ECI/ECEF/LLA conversions, quaternion algebra, and body-frame rotations.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from sim.core.reference_frames import (
    eci_to_ecef,
    ecef_to_eci,
    ecef_to_lla,
    lla_to_ecef,
    quat_to_dcm,
    body_to_eci,
    eci_to_body,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_error,
    quaternion_from_axis_angle,
    quaternion_derivative,
)
from sim import config


# ---------------------------------------------------------------------------
# ECI <-> ECEF round-trip
# ---------------------------------------------------------------------------

class TestEciEcef:
    """ECI <-> ECEF rotation tests."""

    def test_round_trip_identity_at_t0(self):
        """At t=0 ECI and ECEF should coincide (no rotation)."""
        pos = np.array([7e6, 1e6, 2e6])
        ecef = eci_to_ecef(pos, 0.0)
        npt.assert_allclose(ecef, pos, atol=1e-6)

    def test_round_trip_arbitrary_time(self):
        """ECI -> ECEF -> ECI should return the original position."""
        pos_eci = np.array([6.5e6, 1.5e6, 3.0e6])
        t = 1234.5
        pos_ecef = eci_to_ecef(pos_eci, t)
        pos_eci_back = ecef_to_eci(pos_ecef, t)
        npt.assert_allclose(pos_eci_back, pos_eci, atol=1e-4)

    def test_ecef_to_eci_round_trip(self):
        """ECEF -> ECI -> ECEF should return the original position."""
        pos_ecef = np.array([6.4e6, -1e6, 2.5e6])
        t = 567.8
        pos_eci = ecef_to_eci(pos_ecef, t)
        pos_ecef_back = eci_to_ecef(pos_eci, t)
        npt.assert_allclose(pos_ecef_back, pos_ecef, atol=1e-4)

    def test_rotation_preserves_magnitude(self):
        """ECI -> ECEF should not change the position magnitude."""
        pos = np.array([1e7, 2e6, 3e6])
        ecef = eci_to_ecef(pos, 300.0)
        npt.assert_allclose(np.linalg.norm(ecef), np.linalg.norm(pos), rtol=1e-12)


# ---------------------------------------------------------------------------
# ECEF <-> LLA round-trip
# ---------------------------------------------------------------------------

class TestEcefLla:
    """ECEF <-> LLA geodetic conversion tests."""

    def test_equator_prime_meridian(self):
        """Point on equator at prime meridian, sea level."""
        lat, lon, alt = 0.0, 0.0, 0.0
        ecef = lla_to_ecef(lat, lon, alt)
        # Should be on the equator: x ~ R_earth, y=0, z=0
        npt.assert_allclose(ecef[0], config.EARTH_RADIUS_M, rtol=1e-6)
        npt.assert_allclose(ecef[1], 0.0, atol=1.0)
        npt.assert_allclose(ecef[2], 0.0, atol=1.0)

    def test_lla_round_trip(self):
        """LLA -> ECEF -> LLA should recover original coordinates."""
        lat_orig = math.radians(28.5)  # KSC latitude
        lon_orig = math.radians(-80.6)
        alt_orig = 100.0  # 100 m altitude

        ecef = lla_to_ecef(lat_orig, lon_orig, alt_orig)
        lat, lon, alt = ecef_to_lla(ecef)

        npt.assert_allclose(lat, lat_orig, atol=1e-8)
        npt.assert_allclose(lon, lon_orig, atol=1e-8)
        npt.assert_allclose(alt, alt_orig, atol=0.1)

    def test_north_pole(self):
        """North pole should have lat=90 deg, z ~ b (semi-minor axis)."""
        lat = math.pi / 2.0
        lon = 0.0
        alt = 0.0
        ecef = lla_to_ecef(lat, lon, alt)

        # At north pole, x and y should be ~0
        npt.assert_allclose(ecef[0], 0.0, atol=1.0)
        npt.assert_allclose(ecef[1], 0.0, atol=1.0)

        # z should be close to the semi-minor axis b = a*(1-f)
        b = config.EARTH_RADIUS_M * (1.0 - config.EARTH_FLATTENING)
        npt.assert_allclose(ecef[2], b, rtol=1e-6)

    def test_eci_ecef_lla_full_round_trip(self):
        """ECI -> ECEF -> LLA -> ECEF -> ECI should recover original."""
        pos_eci = np.array([6.5e6, 2.0e6, 3.5e6])
        t = 100.0

        pos_ecef = eci_to_ecef(pos_eci, t)
        lat, lon, alt = ecef_to_lla(pos_ecef)
        pos_ecef_2 = lla_to_ecef(lat, lon, alt)
        pos_eci_2 = ecef_to_eci(pos_ecef_2, t)

        npt.assert_allclose(pos_eci_2, pos_eci, atol=1.0)


# ---------------------------------------------------------------------------
# Quaternion operations
# ---------------------------------------------------------------------------

class TestQuaternionOps:
    """Quaternion algebra: multiply, conjugate, identity, axis-angle."""

    def test_identity_quaternion(self):
        """Identity quaternion [0,0,0,1] should produce identity DCM."""
        q_id = np.array([0.0, 0.0, 0.0, 1.0])
        dcm = quat_to_dcm(q_id)
        npt.assert_allclose(dcm, np.eye(3), atol=1e-15)

    def test_quaternion_conjugate(self):
        """Conjugate should negate the vector part and keep the scalar."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        qc = quaternion_conjugate(q)
        npt.assert_allclose(qc, [-0.1, -0.2, -0.3, 0.9])

    def test_multiply_with_identity(self):
        """q * identity = q."""
        q = np.array([0.1, 0.2, 0.3, math.sqrt(1 - 0.01 - 0.04 - 0.09)])
        q_id = np.array([0.0, 0.0, 0.0, 1.0])
        result = quaternion_multiply(q, q_id)
        npt.assert_allclose(result, q, atol=1e-14)

    def test_multiply_with_conjugate_gives_identity(self):
        """q * conj(q) should give the identity quaternion for unit q."""
        q = np.array([0.1, 0.2, 0.3, math.sqrt(1 - 0.01 - 0.04 - 0.09)])
        q /= np.linalg.norm(q)  # ensure unit
        result = quaternion_multiply(q, quaternion_conjugate(q))
        npt.assert_allclose(result, [0, 0, 0, 1], atol=1e-14)

    def test_quaternion_from_axis_angle_zero_rotation(self):
        """Zero rotation angle should give identity quaternion."""
        axis = np.array([1.0, 0.0, 0.0])
        q = quaternion_from_axis_angle(axis, 0.0)
        npt.assert_allclose(q, [0, 0, 0, 1], atol=1e-15)

    def test_quaternion_from_axis_angle_90_deg_z(self):
        """90 deg rotation about z-axis."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = math.pi / 2.0
        q = quaternion_from_axis_angle(axis, angle)

        # Expected: [0, 0, sin(pi/4), cos(pi/4)]
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        npt.assert_allclose(q, [0, 0, s, c], atol=1e-14)

    def test_quaternion_from_axis_angle_180_deg(self):
        """180 deg rotation about x-axis."""
        axis = np.array([1.0, 0.0, 0.0])
        angle = math.pi
        q = quaternion_from_axis_angle(axis, angle)

        # w = cos(pi/2) = 0, x = sin(pi/2) = 1
        npt.assert_allclose(q, [1, 0, 0, 0], atol=1e-14)

    def test_quaternion_error_identical_gives_identity(self):
        """Error between identical quaternions should be identity."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q /= np.linalg.norm(q)
        err = quaternion_error(q, q)
        npt.assert_allclose(err, [0, 0, 0, 1], atol=1e-14)

    def test_quaternion_multiply_associativity(self):
        """Quaternion multiplication should be associative: (q1*q2)*q3 = q1*(q2*q3)."""
        q1 = quaternion_from_axis_angle(np.array([1, 0, 0]), 0.3)
        q2 = quaternion_from_axis_angle(np.array([0, 1, 0]), 0.5)
        q3 = quaternion_from_axis_angle(np.array([0, 0, 1]), 0.7)

        lhs = quaternion_multiply(quaternion_multiply(q1, q2), q3)
        rhs = quaternion_multiply(q1, quaternion_multiply(q2, q3))
        npt.assert_allclose(lhs, rhs, atol=1e-14)


# ---------------------------------------------------------------------------
# body_to_eci / eci_to_body inverse relationship
# ---------------------------------------------------------------------------

class TestBodyEciConversion:
    """Verify that body_to_eci and eci_to_body are inverses."""

    def test_round_trip_identity_quaternion(self):
        """With identity quaternion, body=ECI, so round-trip is trivial."""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        vec = np.array([1.0, 2.0, 3.0])

        vec_eci = body_to_eci(vec, q)
        npt.assert_allclose(vec_eci, vec, atol=1e-14)

        vec_body = eci_to_body(vec, q)
        npt.assert_allclose(vec_body, vec, atol=1e-14)

    def test_round_trip_arbitrary_quaternion(self):
        """body_to_eci(eci_to_body(v, q), q) == v for arbitrary q."""
        q = quaternion_from_axis_angle(
            np.array([1, 1, 1]) / math.sqrt(3), math.radians(45)
        )
        vec_eci = np.array([10.0, -5.0, 3.0])

        vec_body = eci_to_body(vec_eci, q)
        vec_eci_back = body_to_eci(vec_body, q)
        npt.assert_allclose(vec_eci_back, vec_eci, atol=1e-10)

    def test_eci_to_body_inverse(self):
        """eci_to_body(body_to_eci(v, q), q) == v."""
        q = quaternion_from_axis_angle(np.array([0, 0, 1]), math.radians(90))
        vec_body = np.array([1.0, 0.0, 0.0])

        vec_eci = body_to_eci(vec_body, q)
        vec_body_back = eci_to_body(vec_eci, q)
        npt.assert_allclose(vec_body_back, vec_body, atol=1e-14)

    def test_90_deg_rotation_about_z(self):
        """90 deg rotation about z: body x-axis becomes ECI y-axis."""
        q = quaternion_from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        vec_body = np.array([1.0, 0.0, 0.0])
        vec_eci = body_to_eci(vec_body, q)
        npt.assert_allclose(vec_eci, [0, 1, 0], atol=1e-14)


class TestQcmProperties:
    """DCM from quaternion should be a proper rotation matrix."""

    def test_dcm_is_orthogonal(self):
        """DCM * DCM^T = I."""
        q = quaternion_from_axis_angle(np.array([1, 2, 3]) / math.sqrt(14), 1.2)
        dcm = quat_to_dcm(q)
        npt.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-14)

    def test_dcm_determinant_is_one(self):
        """det(DCM) = 1 (proper rotation, no reflection)."""
        q = quaternion_from_axis_angle(np.array([0, 1, 0]), 0.8)
        dcm = quat_to_dcm(q)
        npt.assert_allclose(np.linalg.det(dcm), 1.0, atol=1e-14)


class TestQuaternionDerivative:
    """Quaternion derivative from angular velocity."""

    def test_zero_omega_gives_zero_derivative(self):
        """Zero angular velocity should produce zero quaternion derivative."""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        omega = np.zeros(3)
        qdot = quaternion_derivative(q, omega)
        npt.assert_allclose(qdot, np.zeros(4), atol=1e-15)

    def test_derivative_magnitude_scales_with_omega(self):
        """Doubling omega should double the quaternion derivative magnitude."""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        omega1 = np.array([0.1, 0.0, 0.0])
        omega2 = np.array([0.2, 0.0, 0.0])

        qdot1 = quaternion_derivative(q, omega1)
        qdot2 = quaternion_derivative(q, omega2)

        npt.assert_allclose(
            np.linalg.norm(qdot2),
            2.0 * np.linalg.norm(qdot1),
            rtol=1e-14,
        )
