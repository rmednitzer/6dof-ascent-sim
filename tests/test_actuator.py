import math
from unittest.mock import patch

import pytest

from sim import config
from sim.vehicle.actuator import TVCActuator, TVCActuatorPair


def test_tvc_actuator_update():
    actuator = TVCActuator()

    # Initial state
    assert actuator.position_rad == 0.0

    # Update with 1 degree command
    dt = 0.01
    cmd_deg = 1.0
    actual_pos_deg = actuator.update(cmd_deg, dt)

    # Since it's second order, it won't reach 1.0 immediately
    assert 0.0 < actual_pos_deg < 1.0
    assert actuator.position_rad == math.radians(actual_pos_deg)


def test_tvc_actuator_limits():
    actuator = TVCActuator()
    dt = 1.0  # Large dt to reach limits quickly
    cmd_deg = config.TVC_MAX_DEFLECTION_DEG + 5.0  # Beyond limit

    actual_pos_deg = actuator.update(cmd_deg, dt)
    assert actual_pos_deg == pytest.approx(config.TVC_MAX_DEFLECTION_DEG)
    assert actuator.position_rad == pytest.approx(math.radians(config.TVC_MAX_DEFLECTION_DEG))


def test_tvc_actuator_bypass():
    """Verify behavior when TVC_ACTUATOR_DYNAMICS_ENABLED is False."""
    with patch("sim.vehicle.actuator.config.TVC_ACTUATOR_DYNAMICS_ENABLED", False):
        actuator = TVCActuator()
        dt = 0.01
        cmd_deg = 2.0
        # Should reach command immediately (clamped)
        actual_pos_deg = actuator.update(cmd_deg, dt)
        assert actual_pos_deg == 2.0
        assert actuator.position_rad == math.radians(2.0)

        # Should be limited even in bypass
        cmd_limit = config.TVC_MAX_DEFLECTION_DEG + 1.0
        actual_pos_deg = actuator.update(cmd_limit, dt)
        assert actual_pos_deg == config.TVC_MAX_DEFLECTION_DEG


def test_tvc_actuator_rate_limit():
    """Verify that the slew rate is correctly capped."""
    actuator = TVCActuator()
    dt = 0.01
    # Large command to force maximum rate
    cmd_deg = config.TVC_MAX_DEFLECTION_DEG

    # First step: calculate max possible change in one dt
    max_rate_deg_s = config.TVC_MAX_SLEW_RATE_DEG_S
    max_delta_pos_deg = max_rate_deg_s * dt

    actual_pos_deg = actuator.update(cmd_deg, dt)

    assert abs(actual_pos_deg) <= max_delta_pos_deg + 1e-10


def test_tvc_actuator_hard_stop():
    """Verify that rate is zeroed/clamped when hitting position limits."""
    actuator = TVCActuator()
    dt = 1.0  # Large dt
    cmd_deg = config.TVC_MAX_DEFLECTION_DEG + 10.0

    actuator.update(cmd_deg, dt)
    assert actuator.position_rad == pytest.approx(math.radians(config.TVC_MAX_DEFLECTION_DEG))

    # Internal rate should be <= 0 because we hit the positive limit
    assert actuator._rate <= 0.0


def test_tvc_actuator_pair():
    """Verify TVCActuatorPair updates both axes."""
    pair = TVCActuatorPair()
    dt = 0.01
    p_cmd, y_cmd = 1.0, -1.0

    p_act, y_act = pair.update(p_cmd, y_cmd, dt)

    assert 0.0 < p_act < 1.0
    assert -1.0 < y_act < 0.0
    assert pair.pitch.position_rad == math.radians(p_act)
    assert pair.yaw.position_rad == math.radians(y_act)


def test_tvc_actuator_step_response():
    """Verify second-order dynamics characteristics (rise time/overshoot)."""
    # Use a small dt for better integration accuracy
    dt = 0.001
    actuator = TVCActuator()
    cmd_deg = 1.0

    positions = []
    for _ in range(1000):  # 1 second
        positions.append(actuator.update(cmd_deg, dt))

    # Check for overshoot (damping ratio is 0.7, so there should be slight overshoot)
    peak = max(positions)
    assert peak > 1.0
    assert peak < 1.1  # Overshoot for zeta=0.7 is ~4.6%

    # Check that it eventually converges to the command
    assert positions[-1] == pytest.approx(1.0, abs=1e-2)
