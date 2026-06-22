"""Golden-trajectory regression and independent-integrator cross-checks.

Two complementary guards, both motivated by the cross-repo audit
(``audit/05-cross-repo-lessons.md`` §B6/§C3, addressing backlog Q-04):

1. **Golden invariants** — ``tests/test_e2e_simulation.py`` already asserts the
   nominal run reaches orbit within *broad ranges*. This pins the *actual*
   summary numbers with tight tolerances, so a regression that stays inside
   those ranges (e.g. insertion drifting 407 -> 425 km) is still caught. The
   tolerances absorb cross-platform floating-point noise while flagging any
   real physics/GNC change.

2. **Independent-integrator cross-oracle** — the project's fixed-step RK4
   (``sim/core/integrator.rk4_step``) is validated on two-body Keplerian
   dynamics against SciPy's adaptive ``solve_ivp`` (the genesis-world "validate
   against an independent reference" pattern), plus a symplectic-free
   energy-conservation sanity bound. This exercises the exact integrator the
   simulation uses, on dynamics with a trusted reference.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
from scipy.integrate import solve_ivp

from sim import config
from sim.core.integrator import StateDot, rk4_step
from sim.core.state import VehicleState
from sim.main import run_simulation


class TestNominalGolden:
    """Pin the nominal-run summary to a committed golden baseline.

    Baseline captured 2026-06-21 after the chi-square innovation gate (ADR
    0013) and FTS attitude hysteresis (ADR 0014) landed; rebaselined 2026-06-22
    for the error-state attitude EKF + star-tracker attitude aiding (ADR 0020),
    then again for ADR 0020 Stage 2 — ``USE_ESTIMATED_ATTITUDE`` now defaults to
    True, so guidance and the controller fly on the EKF's *estimated* attitude.
    The trajectory is materially the same orbit; the moves of note are
    ``boundary_clamp_count`` (≈251 → ≈326: the noisier estimated rate/attitude
    exercises the TVC clamps more often) and ``peak_ekf_uncertainty_m`` (≈1700 m,
    still far below the 10 km FTS limit). If a deliberate physics/GNC change moves
    these, update the golden values here together with the committed example
    artifacts (``examples/output/``), per BACKLOG D-06. Re-baselined again
    2026-06-22 for ADR 0021 (S2 Isp 348->356 s, N-01 performance margin): the
    higher-Isp upper stage barely moves the nominal (insertion, peak-q, axial-g,
    and EKF uncertainty essentially unchanged; ``total_time_s`` 487.9->491.8 as
    the more-efficient burn shifts cutoff slightly). Re-baselined again 2026-06-22
    for ADR 0023 (ground-station range tracking, N-01): continuous position aiding
    through the coast drops ``peak_ekf_uncertainty_m`` 1707->521 m and, via the
    sharper position estimate fed to guidance, flies a slightly more circular
    insertion (fpa 0.80->0.32 deg, peak-q -2%). Re-baselined again 2026-06-22 for
    ADR 0024 (AD-17 inclination targeting): azimuth correction + terminal yaw
    out-of-plane steering reach ``insertion_inclination_deg`` ~45->51.04 deg
    (target 51.6), which shifts the terminal trajectory (fpa 0.32->0.70).
    """

    # Golden summary of the deterministic nominal run (seed 0).
    _GOLD = {
        "outcome": "SUCCESS",
        "insertion_altitude_m": 407_178.5,
        "insertion_velocity_ms": 7_601.4,
        "insertion_fpa_deg": 0.700,
        "insertion_inclination_deg": 51.04,
        "peak_q_pa": 31_725.4,
        "peak_axial_g": 5.40,
        "peak_ekf_uncertainty_m": 561.4,
        "boundary_clamp_count": 297,
        "total_time_s": 493.8,
    }

    def test_nominal_summary_matches_golden(self):
        result = run_simulation(config_override={}, quiet=True)
        g = self._GOLD

        assert result.outcome == g["outcome"]
        assert result.fts_trigger_time_s is None

        # Accumulated trajectory quantities: tight relative tolerance (a real
        # regression moves these by >> 0.3 %; FP/platform noise stays well under).
        npt.assert_allclose(result.insertion_altitude_m, g["insertion_altitude_m"], rtol=3e-3)
        npt.assert_allclose(result.insertion_velocity_ms, g["insertion_velocity_ms"], rtol=3e-3)
        npt.assert_allclose(result.peak_q_pa, g["peak_q_pa"], rtol=5e-3)
        npt.assert_allclose(result.peak_ekf_uncertainty_m, g["peak_ekf_uncertainty_m"], rtol=2e-2)
        # peak_axial_g is throttle-limited to a hard ceiling — essentially exact.
        npt.assert_allclose(result.peak_axial_g, g["peak_axial_g"], rtol=1e-3)
        # Flight-path angle is small; compare absolutely.
        npt.assert_allclose(result.insertion_fpa_deg, g["insertion_fpa_deg"], atol=0.25)
        # Inclination is actively targeted (ADR 0024); pin it near the target.
        npt.assert_allclose(result.insertion_inclination_deg, g["insertion_inclination_deg"], atol=0.30)
        # Insertion time: within a couple of integration steps' worth of drift.
        npt.assert_allclose(result.total_time_s, g["total_time_s"], atol=1.0)
        # Clamp count is an integer event tally — allow a small band for
        # borderline clamps that FP noise could flip, but catch gross changes.
        assert abs(result.boundary_clamp_count - g["boundary_clamp_count"]) <= 25, (
            f"boundary_clamp_count {result.boundary_clamp_count} far from golden {g['boundary_clamp_count']}"
        )


def _two_body_orbit_state(alt_m: float = 400_000.0) -> tuple[np.ndarray, np.ndarray]:
    """Circular-orbit position/velocity in ECI for a two-body cross-check."""
    mu = config.EARTH_MU
    r0 = np.array([config.EARTH_RADIUS_M + alt_m, 0.0, 0.0])
    v_circ = math.sqrt(mu / np.linalg.norm(r0))
    v0 = np.array([0.0, v_circ, 0.0])
    return r0, v0


class TestIntegratorCrossOracle:
    """Validate ``rk4_step`` against SciPy ``solve_ivp`` on two-body dynamics."""

    def test_rk4_matches_solve_ivp_two_body(self):
        mu = config.EARTH_MU
        r0, v0 = _two_body_orbit_state()
        t_end = 300.0
        dt = 0.1

        # --- Project integrator (gravity-only derivatives) ---
        def derivatives_fn(t: float, s: VehicleState) -> StateDot:
            r = s.position_eci
            rn = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            accel = -mu * r / rn**3
            return StateDot(
                velocity_eci=s.velocity_eci,
                acceleration_eci=accel,
                quaternion_dot=np.zeros(4),
                angular_acceleration_body=np.zeros(3),
                mass_rate_kg_s=0.0,
            )

        state = VehicleState(
            position_eci=r0.copy(),
            velocity_eci=v0.copy(),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=1000.0,
            time_s=0.0,
        )
        for _ in range(int(round(t_end / dt))):
            state = rk4_step(state, derivatives_fn, dt)

        # --- Independent reference: SciPy adaptive RK45, near machine precision ---
        def ode(t, y):
            r = y[:3]
            rn = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            return [y[3], y[4], y[5], *(-mu * r / rn**3)]

        sol = solve_ivp(ode, (0.0, t_end), [*r0, *v0], rtol=1e-11, atol=1e-9, dense_output=False)
        ref_r = sol.y[:3, -1]
        ref_v = sol.y[3:, -1]

        # RK4 at dt=0.1 s tracks the reference to well under a metre / mm/s.
        npt.assert_allclose(state.position_eci, ref_r, rtol=0.0, atol=1.0)
        npt.assert_allclose(state.velocity_eci, ref_v, rtol=0.0, atol=1e-3)

    def test_rk4_conserves_orbital_energy(self):
        """Fixed-step RK4 holds specific orbital energy over many steps."""
        mu = config.EARTH_MU
        r0, v0 = _two_body_orbit_state()

        def derivatives_fn(t: float, s: VehicleState) -> StateDot:
            r = s.position_eci
            rn = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            return StateDot(
                velocity_eci=s.velocity_eci,
                acceleration_eci=-mu * r / rn**3,
                quaternion_dot=np.zeros(4),
                angular_acceleration_body=np.zeros(3),
                mass_rate_kg_s=0.0,
            )

        def energy(pos: np.ndarray, vel: np.ndarray) -> float:
            return 0.5 * float(vel @ vel) - mu / float(np.linalg.norm(pos))

        state = VehicleState(
            position_eci=r0.copy(),
            velocity_eci=v0.copy(),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
            mass_kg=1000.0,
            time_s=0.0,
        )
        e0 = energy(state.position_eci, state.velocity_eci)
        for _ in range(6000):  # 600 s at dt = 0.1 s
            state = rk4_step(state, derivatives_fn, 0.1)
        e1 = energy(state.position_eci, state.velocity_eci)

        npt.assert_allclose(e1, e0, rtol=1e-6)
