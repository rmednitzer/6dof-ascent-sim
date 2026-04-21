"""6-DOF vehicle state vector definition."""

from __future__ import annotations

import math
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
        """Geodetic altitude above WGS84 ellipsoid (m).

        Result is memoised on the instance to amortise across the multiple
        per-step consumers (sensors, recorder, insertion check).  The cache
        key is ``(id(position_eci), time_s)``; because ``rk4_step`` returns a
        fresh state (and fresh position array) every integration step, any
        genuine change automatically invalidates the cache.
        """
        from sim.core.reference_frames import ecef_to_lla, eci_to_ecef

        cache_key = (id(self.position_eci), self.time_s)
        try:
            if self._alt_cache_key == cache_key:  # type: ignore[has-type]
                return self._alt_cache_value  # type: ignore[has-type]
        except AttributeError:
            pass

        p0, p1, p2 = self.position_eci[0], self.position_eci[1], self.position_eci[2]
        r_sq = p0 * p0 + p1 * p1 + p2 * p2
        if r_sq < 1.0:
            alt = 0.0
        else:
            pos_ecef = eci_to_ecef(self.position_eci, self.time_s)
            _, _, alt = ecef_to_lla(pos_ecef)

        self._alt_cache_key = cache_key
        self._alt_cache_value = alt
        return alt

    def velocity_mag_ms(self) -> float:
        """Inertial speed magnitude (m/s)."""
        v = self.velocity_eci
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    def specific_orbital_energy(self) -> float:
        """Specific orbital energy (J/kg) via vis-viva."""
        p = self.position_eci
        v = self.velocity_eci
        r_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
        if r_sq < 1.0:
            return 0.0
        r = math.sqrt(r_sq)
        v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        return 0.5 * v_sq - config.EARTH_MU / r

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
        q = self.quaternion
        norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
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
