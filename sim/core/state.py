"""6-DOF vehicle state vector definition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim import config


@dataclass
class VehicleState:
    """Complete 6-DOF state of the launch vehicle.

    Attributes:
        position_eci: ECI position vector (m).
        velocity_eci: ECI velocity vector (m/s).
        quaternion: Attitude quaternion [x, y, z, w] (scalar-last).
        angular_velocity_body: Body angular rates (rad/s).
        mass_kg: Current total mass (kg).
        time_s: Mission elapsed time (s).
    """

    position_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    angular_velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass_kg: float = 0.0
    time_s: float = 0.0

    def altitude_m(self) -> float:
        """Geodetic altitude above WGS84 ellipsoid (m)."""
        from sim.core.reference_frames import ecef_to_lla, eci_to_ecef

        r = np.linalg.norm(self.position_eci)
        if r < 1.0:
            return 0.0
        pos_ecef = eci_to_ecef(self.position_eci, self.time_s)
        _, _, alt = ecef_to_lla(pos_ecef)
        return alt

    def velocity_mag_ms(self) -> float:
        """Inertial speed magnitude (m/s)."""
        return float(np.linalg.norm(self.velocity_eci))

    def specific_orbital_energy(self) -> float:
        """Specific orbital energy (J/kg) via vis-viva."""
        r = np.linalg.norm(self.position_eci)
        v = np.linalg.norm(self.velocity_eci)
        if r < 1.0:
            return 0.0
        return 0.5 * v**2 - config.EARTH_MU / r

    def copy(self) -> VehicleState:
        """Return a deep copy of this state."""
        return VehicleState(
            position_eci=self.position_eci.copy(),
            velocity_eci=self.velocity_eci.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity_body=self.angular_velocity_body.copy(),
            mass_kg=self.mass_kg,
            time_s=self.time_s,
        )

    def normalize_quaternion(self) -> None:
        """Normalize the attitude quaternion in-place."""
        norm = np.linalg.norm(self.quaternion)
        if norm > 1e-10:
            self.quaternion /= norm

    def to_dict(self) -> dict:
        """JSON-serializable snapshot of the state."""
        return {
            "time_s": self.time_s,
            "position_eci_m": self.position_eci.tolist(),
            "velocity_eci_ms": self.velocity_eci.tolist(),
            "quaternion": self.quaternion.tolist(),
            "angular_velocity_body_rads": self.angular_velocity_body.tolist(),
            "mass_kg": self.mass_kg,
            "altitude_m": self.altitude_m(),
            "velocity_mag_ms": self.velocity_mag_ms(),
            "specific_orbital_energy_jkg": self.specific_orbital_energy(),
        }
