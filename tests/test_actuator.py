import math

from sim import config
from sim.vehicle.actuator import TVCActuator


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
    assert actual_pos_deg <= config.TVC_MAX_DEFLECTION_DEG
    assert actuator.position_rad <= math.radians(config.TVC_MAX_DEFLECTION_DEG)
