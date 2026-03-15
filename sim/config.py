"""Simulation parameters — single source of truth for all constants and tunables."""

# ---------- Orbital target ----------
TARGET_ALTITUDE_M = 400_000  # 400 km circular LEO
TARGET_INCLINATION_DEG = 51.6  # ISS inclination
TARGET_VELOCITY_MS = 7_670  # Approximate circular velocity at 400 km

# ---------- Earth model ----------
EARTH_RADIUS_M = 6_378_137.0  # WGS84 semi-major axis
EARTH_MU = 3.986004418e14  # GM (m³/s²)
EARTH_J2 = 1.08263e-3  # J2 oblateness coefficient
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
S1_DRY_MASS_KG = 22_200  # Structural mass
S1_PROPELLANT_KG = 395_700  # Fuel + oxidizer
S1_THRUST_VAC_N = 7_607_000  # Vacuum thrust (9 engines, Merlin-class)
S1_THRUST_SL_N = 6_806_000  # Sea-level thrust
S1_ISP_VAC_S = 311  # Vacuum Isp
S1_ISP_SL_S = 282  # Sea-level Isp
S1_BURN_TIME_S = 162  # Nominal burn time
S1_THROTTLE_MIN = 0.4  # Minimum throttle

# ---------- Stage 2 ----------
S2_DRY_MASS_KG = 4_000
S2_PROPELLANT_KG = 92_670
S2_THRUST_VAC_N = 981_000
S2_ISP_VAC_S = 348
S2_BURN_TIME_S = 397

# ---------- Aerodynamics ----------
CD_TABLE_MACH = [0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0, 10.0]
CD_TABLE_VALUE = [0.30, 0.30, 0.35, 0.50, 0.45, 0.35, 0.30, 0.25, 0.20]
REFERENCE_AREA_M2 = 10.52  # Cross-section area (3.66m diameter)
COP_OFFSET_FROM_NOSE_M = 12.0  # Approximate center of pressure
VEHICLE_LENGTH_M = 70.0  # Total vehicle length

# ---------- Structural limits ----------
MAX_Q_PA = 35_000  # Max dynamic pressure (Pa)
MAX_AXIAL_G = 6.0  # Max axial acceleration (g)
MAX_LATERAL_G = 0.5  # Max lateral acceleration (g)

# ---------- TVC actuator limits ----------
TVC_MAX_DEFLECTION_DEG = 5.0
TVC_MAX_SLEW_RATE_DEG_S = 10.0

# ---------- EKF parameters ----------
IMU_ACCEL_NOISE_MPS2 = 0.01  # Accelerometer noise (m/s²)
IMU_GYRO_NOISE_RADS = 0.001  # Gyro noise (rad/s)
IMU_ACCEL_BIAS_MPS2 = 0.001  # Accelerometer bias instability
IMU_GYRO_BIAS_RADS = 0.0001  # Gyro bias instability
GPS_POS_NOISE_M = 5.0  # GPS position noise (m)
GPS_VEL_NOISE_MS = 0.1  # GPS velocity noise (m/s)
GPS_UPDATE_HZ = 1  # GPS update rate
BARO_ALT_NOISE_M = 10.0  # Barometer noise (m)
BARO_UPDATE_HZ = 10  # Barometer update rate
EKF_RESIDUAL_SIGMA_THRESHOLD = 3.0  # Innovation gate (sigma)

# ---------- FTS abort criteria ----------
FTS_CROSSRANGE_LIMIT_M = 200_000  # Max cross-range deviation
FTS_ATTITUDE_LIMIT_DEG = 90.0  # Max attitude error (deg)
FTS_COVARIANCE_LIMIT_M = 10_000  # Max EKF position uncertainty

# ---------- Flex body — first 3 lateral bending modes ----------
FLEX_MODE_FREQS_HZ = [1.2, 3.5, 7.0]  # Natural frequencies (full propellant)
FLEX_MODE_FREQS_EMPTY_HZ = [2.0, 5.5, 10.0]  # Natural frequencies (empty stage)
FLEX_DAMPING_RATIOS = [0.01, 0.01, 0.005]  # Modal damping ratios
FLEX_MODE_SLOPES_AT_IMU = [0.5, -0.3, 0.15]  # Mode shape slope at IMU location (rad/m)
FLEX_MODE_SLOPES_AT_ENGINE = [1.0, 0.8, 0.6]  # Mode shape slope at engine gimbal
FLEX_ENABLED = True  # Toggle for comparison runs

# ---------- Propellant slosh — pendulum analogy ----------
SLOSH_MASS_FRACTION = 0.30  # Fraction of propellant participating in slosh
SLOSH_FREQ_FULL_HZ = 0.3  # Slosh frequency at full tank
SLOSH_FREQ_EMPTY_HZ = 0.8  # Slosh frequency approaching empty
SLOSH_DAMPING_RATIO = 0.03  # Baffled tank damping
SLOSH_ARM_LENGTH_M = 2.0  # Effective pendulum length (full tank)
SLOSH_ENABLED = True  # Toggle for comparison runs

# ---------- Monte Carlo ----------
MC_NUM_RUNS = 1000  # Default number of Monte Carlo runs
MC_SEED = 42  # Base random seed
MC_WORKERS = None  # None = os.cpu_count()
CD_SCALE_FACTOR = 1.0  # Multiplier on Cd table (dispersed in MC)
ATMO_DENSITY_SCALE = 1.0  # Multiplier on atmospheric density

# ---------- Wind ----------
WIND_SPEED_MS = 10.0  # Mean wind speed
WIND_DIRECTION_DEG = 270.0  # Wind coming from west
WIND_GUST_SIGMA_MS = 5.0  # Gust standard deviation

# ---------- Guidance ----------
PITCH_KICK_DEG = 3.0  # Initial pitch-over angle
PITCH_KICK_TIME_S = 7.0  # Time to initiate gravity turn
VERTICAL_RISE_TIME_S = 7.0  # Duration of vertical rise phase

# ---------- Control gains ----------
CONTROL_KP = 2.0
CONTROL_KD = 1.5
CONTROL_KI = 0.1
CONTROL_INTEGRATOR_LIMIT_DEG = 2.0  # Anti-windup limit

# ---------- Atmospheric pressure at sea level ----------
P_SL = 101325.0  # Sea-level pressure (Pa)
