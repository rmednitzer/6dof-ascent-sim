"""Simulation parameters.

Fixed physical / model constants are plain module globals below. The *overridable*
parameters — those a Monte-Carlo run may disperse — are NOT globals here: they are
declared (with types, bounds, and defaults) in :mod:`sim.config_schema` and
resolved per-run from a context-local config (see the bottom of this module and
ADR 0018). Read them the usual way, ``config.<NAME>``; set them for a run with
``with config.override_context({...}):`` — never assign ``config.<NAME> = ...``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from sim.config_schema import OVERRIDABLE_PARAM_NAMES, OverridableParams

# ---------- Orbital target ----------
TARGET_ALTITUDE_M = 400_000  # 400 km circular LEO
TARGET_INCLINATION_DEG = 51.6  # ISS inclination
TARGET_VELOCITY_MS = 7_670  # Approximate circular velocity at 400 km

# ---------- Orbital-insertion success criteria ----------
# A run is only SUCCESS if the osculating orbit is a real, sustainable
# LEO: bound (e < 1), above the sensible atmosphere, and near-circular.
INSERTION_MIN_PERIAPSIS_ALT_M = 140_000  # Periapsis must clear the atmosphere
INSERTION_MAX_ECCENTRICITY = 0.05  # Near-circular insertion
INSERTION_MAX_FPA_DEG = 5.0  # Flight-path angle at insertion
INSERTION_MIN_VELOCITY_FRAC = 0.97  # Fraction of TARGET_VELOCITY_MS
INSERTION_MIN_ALTITUDE_FRAC = 0.95  # Fraction of TARGET_ALTITUDE_M

# ---------- Earth model ----------
EARTH_RADIUS_M = 6_378_137.0  # WGS84 semi-major axis
EARTH_MU = 3.986004418e14  # GM (m³/s²)
EARTH_J2 = 1.08262668e-3  # J2 oblateness coefficient (EGM96)
EARTH_J3 = -2.53265649e-6  # J3 zonal harmonic (pear-shaped asymmetry)
EARTH_J4 = -1.61962159e-6  # J4 zonal harmonic
EARTH_J5 = -2.27296083e-7  # J5 zonal harmonic
EARTH_J6 = 5.40681239e-7  # J6 zonal harmonic
EARTH_OMEGA = 7.2921150e-5  # Rotation rate (rad/s)
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS84 flattening

# ---------- Simulation ----------
DT = 0.01  # Fixed timestep (s) — 100 Hz physics
T_MAX = 600.0  # Maximum sim time (s)
TELEMETRY_HZ = 10  # Downlink telemetry rate
INTERNAL_HZ = 100  # Internal loop rate
G0 = 9.80665  # Standard gravity (m/s²)

# ---------- Launch site (Kennedy Space Center) ----------
LAUNCH_LAT_DEG = 28.5729
LAUNCH_LON_DEG = -80.6490
LAUNCH_ALT_M = 0.0

# ---------- Stage 1 ----------
# Overridable (see config_schema): S1_DRY_MASS_KG, S1_PROPELLANT_KG,
# S1_THRUST_VAC_N, S1_ISP_VAC_S.
S1_THRUST_SL_N = 6_806_000  # Sea-level thrust
S1_ISP_SL_S = 282  # Sea-level Isp
S1_BURN_TIME_S = 162  # Nominal burn time
S1_THROTTLE_MIN = 0.4  # Minimum throttle

# ---------- Stage 2 ----------
# Overridable (see config_schema): S2_THRUST_VAC_N, S2_ISP_VAC_S.
S2_DRY_MASS_KG = 4_000
S2_PROPELLANT_KG = 92_670
S2_BURN_TIME_S = 397

# ---------- Aerodynamics ----------
# Overridable (see config_schema): CD_SCALE_FACTOR.
CD_TABLE_MACH = [0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0, 10.0]
CD_TABLE_VALUE = [0.30, 0.30, 0.35, 0.50, 0.45, 0.35, 0.30, 0.25, 0.20]
REFERENCE_AREA_M2 = 10.52  # Cross-section area (3.66m diameter)
COP_OFFSET_FROM_NOSE_M = 12.0  # Approximate center of pressure
VEHICLE_LENGTH_M = 70.0  # Total vehicle length
# Normal force coefficient slope (per radian) — slender body theory
# Barrowman (1967): CN_alpha ~ 2.0 for body-of-revolution, adjusted for fineness ratio
CN_ALPHA_TABLE_MACH = [0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0, 10.0]
CN_ALPHA_TABLE_VALUE = [2.0, 2.0, 2.2, 2.8, 2.6, 2.2, 2.0, 1.8, 1.5]  # per radian
# Pitch damping coefficient Cmq (per rad/s normalized by 2V/L)
# Typical: -0.5 to -2.0 for slender rockets (Nelson, "Flight Stability", 1998)
CMQ_PITCH_DAMPING = -1.2

# ---------- Structural limits ----------
MAX_Q_PA = 35_000  # Max dynamic pressure (Pa)
MAX_AXIAL_G = 6.0  # Max axial acceleration (g)
MAX_LATERAL_G = 0.5  # Max lateral acceleration (g)

# ---------- TVC actuator ----------
TVC_MAX_DEFLECTION_DEG = 5.0
TVC_MAX_SLEW_RATE_DEG_S = 20.0  # Increased for S2 exoatmospheric maneuvers
# Second-order actuator dynamics (Wie, "Space Vehicle Dynamics and Control",
# 2nd ed., Ch. 7). Typical hydraulic TVC servoactuator parameters:
TVC_ACTUATOR_NATURAL_FREQ_HZ = 25.0  # Actuator bandwidth (Hz)
TVC_ACTUATOR_DAMPING_RATIO = 0.7  # Damping ratio (critically damped ~0.7)
TVC_ACTUATOR_DYNAMICS_ENABLED = True  # Toggle actuator dynamics

# ---------- EKF parameters ----------
# Overridable (see config_schema): IMU_ACCEL_BIAS_MPS2, IMU_GYRO_BIAS_RADS,
# GPS_POS_NOISE_M.
IMU_ACCEL_NOISE_MPS2 = 0.01  # Accelerometer noise (m/s²)
IMU_GYRO_NOISE_RADS = 0.001  # Gyro noise (rad/s)
GPS_VEL_NOISE_MS = 0.1  # GPS velocity noise (m/s)
GPS_UPDATE_HZ = 1  # GPS update rate
# GPS availability ceiling (ADR 0020). Models the COCOM export limit (a
# simplified altitude gate ~ the 18 km / 515 m/s COTS-receiver restriction):
# GPS is denied above this altitude, so the upper stage flies the long coast/burn
# to orbit GPS-denied. Attitude observability through that GPS-denied phase is
# provided instead by the star tracker (below) — the realistic upper-stage
# avionics suite. Set to +inf to model an ITAR-cleared (SAASM / M-code) receiver
# with GPS through ascent.
GPS_AVAILABILITY_CEILING_M = 60_000.0  # COCOM altitude limit
BARO_ALT_NOISE_M = 10.0  # Barometer noise (m)
BARO_UPDATE_HZ = 10  # Barometer update rate
# ---------- Star tracker (inertial attitude reference) ----------
# A star tracker images star fields to measure inertial attitude directly
# (all three axes, including roll about the thrust axis, which GPS cannot
# observe). It is the attitude aid that keeps the error-state EKF (ADR 0020)
# observable during the GPS-denied upper-stage flight: without it, gyro-only
# attitude dead-reckoning diverges (>20°) and the attitude→velocity→position
# covariance coupling trips the FTS limit. Usable only above the sensible
# atmosphere (clear sky) and below a slew rate that would smear the star image —
# i.e. the upper-stage regime, complementary to GPS below the COCOM ceiling.
STAR_TRACKER_NOISE_ARCSEC = 10.0  # 1-sigma per-axis attitude noise (arcsec)
STAR_TRACKER_NOISE_RAD = STAR_TRACKER_NOISE_ARCSEC * math.pi / (180.0 * 3600.0)
STAR_TRACKER_UPDATE_HZ = 5  # update rate (Hz)
STAR_TRACKER_MIN_ALT_M = 100_000.0  # usable only above the sensible atmosphere
STAR_TRACKER_MAX_RATE_RADS = 0.05  # ~2.9 deg/s slew limit (image-smear threshold)
# ---------- Ground-station range tracking (ADR 0023) ----------
# A launch-range tracking network ranges the vehicle as a transponder/skin-track
# target. It is independent of GPS (the vehicle is *tracked*, not self-locating),
# so unlike the COTS GPS receiver it is NOT bound by the COCOM ceiling and keeps
# aiding the EKF position through the GPS-denied upper-stage coast — bounding the
# position covariance that otherwise grows unbounded and (for the most-lofted
# dispersions) trips the FTS (BACKLOG N-01). Each station contributes a slant-range
# measurement while the vehicle is above its elevation mask; the combined
# geometry multilaterates position. Range accuracy is coarser than GPS.
GROUND_RANGE_NOISE_M = 30.0  # slant-range 1-sigma noise (m)
GROUND_TRACK_UPDATE_HZ = 5  # ranging rate per station (Hz)
GROUND_TRACK_ELEV_MASK_DEG = 5.0  # min elevation above the local horizon for a usable track
# Eastern-Range stations along the north-east ascent track: (name, lat°, lon°, alt m).
GROUND_STATIONS = [
    ("KSC", LAUNCH_LAT_DEG, LAUNCH_LON_DEG, 0.0),
    ("Bermuda", 32.36, -64.68, 0.0),
]
# Innovation-consistency gate. A measurement is rejected when its normalised
# innovation squared (NIS = yᵀ S⁻¹ y, a chi-square statistic with one DOF per
# measurement component) exceeds the chi-square quantile at this tail
# probability. This Mahalanobis test accounts for the full innovation
# covariance S; it replaces the earlier per-component |yᵢ| > kσᵢ test, which
# used only the diagonal of S and ignored cross-covariance. At one DOF this
# reproduces the old 3-sigma intent exactly: 0.9973 = P(|N(0,1)| < 3), so
# chi2.ppf(0.9973, 1) = 3.0² = 9.0 (the previous per-component gate on baro).
EKF_INNOVATION_GATE_P = 0.9973  # chi-square tail probability for the NIS gate
# Navigation attitude authority (ADR 0020) is overridable (see config_schema):
# USE_ESTIMATED_ATTITUDE. The error-state EKF always estimates attitude (from the
# measured gyro, the GPS specific-force/velocity coupling, and the star-tracker
# update); the flag selects whether guidance/control/FTS consume that *estimated*
# attitude (default, full realism) or the true attitude (estimate still runs in
# parallel for validation/telemetry). Stage 2 enabled it by default.

# ---------- FTS abort criteria ----------
FTS_CROSSRANGE_LIMIT_M = 200_000  # Max cross-range deviation
# Max thrust-axis pointing error before FTS abort. This is now a
# genuine loss-of-control threshold (see FTS._compute_attitude_error,
# which measures thrust-axis divergence, not the uncontrolled roll).
# Nominal flight tracks to <5 deg, so 25 deg is protective without
# false-tripping. The old 90 deg never tripped before total loss.
FTS_ATTITUDE_LIMIT_DEG = 25.0  # Max thrust-axis pointing error (deg)
# Hysteresis (debounce) on the attitude criterion only: the thrust-axis error
# must exceed FTS_ATTITUDE_LIMIT_DEG *continuously* for this long before the FTS
# triggers on it. A single-sample marginal excursion — the dominant dispersed
# Monte-Carlo abort signature, where every observed abort is a ~25.0x° hit right
# at the limit during the PEG terminal phase (audit AD-19) — is filtered, while
# a genuine sustained loss of control still trips after a brief fixed delay.
# Set to 0.0 to recover the original instantaneous behaviour. The cross-range,
# covariance, and structural criteria remain instantaneous. This is an
# FTS-side mitigation and does NOT replace the AD-19 PEG terminal-guidance
# hardening (a separate control-design task).
FTS_ATTITUDE_HYSTERESIS_S = 0.2  # seconds (~20 frames at 100 Hz); 0.0 = instantaneous
FTS_COVARIANCE_LIMIT_M = 10_000  # Max EKF position uncertainty

# ---------- Flex body — first 3 lateral bending modes ----------
# Overridable (see config_schema): FLEX_ENABLED, FLEX_NOTCH_ENABLED,
# FLEX_NOTCH_Q, FLEX_MODAL_MASS_KG.
FLEX_MODE_FREQS_HZ = [1.2, 3.5, 7.0]  # Natural frequencies (full propellant)
FLEX_MODE_FREQS_EMPTY_HZ = [2.0, 5.5, 10.0]  # Natural frequencies (empty stage)
FLEX_DAMPING_RATIOS = [0.01, 0.01, 0.005]  # Modal damping ratios
FLEX_MODE_SLOPES_AT_IMU = [0.5, -0.3, 0.15]  # Mode shape slope at IMU location (rad/m)
FLEX_MODE_SLOPES_AT_ENGINE = [1.0, 0.8, 0.6]  # Mode shape slope at engine gimbal

# ---------- Propellant slosh — pendulum analogy ----------
# Overridable (see config_schema): SLOSH_ENABLED.
SLOSH_MASS_FRACTION = 0.30  # Fraction of propellant participating in slosh
SLOSH_FREQ_FULL_HZ = 0.3  # Slosh frequency at full tank
SLOSH_FREQ_EMPTY_HZ = 0.8  # Slosh frequency approaching empty
SLOSH_DAMPING_RATIO = 0.03  # Baffled tank damping
SLOSH_ARM_LENGTH_M = 2.0  # Effective pendulum length (full tank)

# ---------- Monte Carlo ----------
# Overridable (see config_schema): CD_SCALE_FACTOR, ATMO_DENSITY_SCALE.
MC_NUM_RUNS = 1000  # Default number of Monte Carlo runs
MC_SEED = 42  # Base random seed
MC_WORKERS = None  # None = os.cpu_count()

# ---------- Monte Carlo on SLURM HPC (experimental) ----------
# Campaign-level defaults for sim/montecarlo/hpc.py. Cluster-specific
# placement (partition, account, walltime, memory) lives in SlurmConfig,
# not here, because it is site-specific infrastructure rather than physics.
MC_RUNS_PER_TASK = 50  # Runs executed per SLURM array task
MC_HPC_OUTPUT_DIR = "output/montecarlo"  # Shared dir for shards + aggregate

# ---------- Wind ----------
# Overridable (see config_schema): WIND_SPEED_MS, WIND_DIRECTION_DEG.
WIND_GUST_SIGMA_MS = 5.0  # Gust standard deviation

# ---------- Guidance ----------
PITCH_KICK_DEG = 3.0  # Initial pitch-over angle
PITCH_KICK_TIME_S = 7.0  # Time to initiate gravity turn
VERTICAL_RISE_TIME_S = 7.0  # Duration of vertical rise phase
# Slew-rate limit on the commanded thrust direction. PEG can otherwise
# emit a jittery, physically-unrealisable attitude command (faster than
# the +/-5 deg TVC can track), inflating apparent attitude error. This
# keeps the command trackable so the FTS attitude check is meaningful.
GUIDANCE_MAX_CMD_RATE_DEG_S = 8.0

# ---------- Control gains ----------
# Baseline gains at reference condition (q=10kPa, full mass)
# Gain scheduling scales these with dynamic pressure and mass ratio
# Reference: Greensite, "Analysis and Design of Space Vehicle Flight
# Control Systems", NASA CR-820, 1967.
CONTROL_KP = 2.0
CONTROL_KD = 1.5
CONTROL_KI = 0.1
CONTROL_INTEGRATOR_LIMIT_DEG = 2.0  # Anti-windup limit
CONTROL_Q_REF_PA = 10_000.0  # Reference dynamic pressure for gain scheduling
CONTROL_MASS_REF_KG = 300_000.0  # Reference mass for gain scheduling
CONTROL_GAIN_SCHEDULE_ENABLED = True  # Toggle gain scheduling

# ---------- Atmospheric pressure at sea level ----------
P_SL = 101325.0  # Sea-level pressure (Pa)


# ---------- Per-run overridable parameters (context-local) — ADR 0018 ----------
# The overridable parameters (sim.config_schema.OverridableParams) are resolved
# from a context-local config rather than module globals, so a Monte-Carlo run
# applies its dispersion via override_context() instead of mutating this module.
# Concurrent runs (threads / async tasks) therefore cannot interfere, and
# run_simulation needs no save/restore of globals. This completes ADR-0009 and
# supersedes the global-override half of ADR-0004. PEP 562 module __getattr__
# serves the overridable names (which are deliberately not module globals).
_OVERRIDABLE: frozenset[str] = frozenset(OVERRIDABLE_PARAM_NAMES)
# Nominal values, shared across contexts (OverridableParams is frozen, so the
# singleton cannot be mutated). The ContextVar defaults to None = "use nominal".
_DEFAULT_OVERRIDES = OverridableParams()
_active_overrides: ContextVar[OverridableParams | None] = ContextVar("active_overrides", default=None)


def active_overrides() -> OverridableParams:
    """Return the overridable-parameter config active in the current context."""
    current = _active_overrides.get()
    return current if current is not None else _DEFAULT_OVERRIDES


def __getattr__(name: str) -> Any:
    """Resolve an overridable parameter from the active context-local config.

    Invoked by PEP 562 only for names not defined as module globals — i.e. the
    overridable parameters. Everything else raises ``AttributeError`` as usual.
    """
    if name in _OVERRIDABLE:
        return getattr(active_overrides(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextmanager
def override_context(overrides: Mapping[str, Any] | None) -> Iterator[None]:
    """Activate per-run parameter overrides for the duration of the block.

    Builds a validated :class:`OverridableParams` from *overrides* (ignoring
    ``_``-prefixed bookkeeping keys such as ``_seed`` / ``_run_index``) and
    installs it as the context-local config, restoring the previous one on exit.
    Thread- and task-safe; performs no global mutation.

    Args:
        overrides: Parameter-name -> value mapping, or ``None`` for the nominal
            configuration.
    """
    params = {k: v for k, v in (overrides or {}).items() if not k.startswith("_")}
    token = _active_overrides.set(OverridableParams(**params))
    try:
        yield
    finally:
        _active_overrides.reset(token)
