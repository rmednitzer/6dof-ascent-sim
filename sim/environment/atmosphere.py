"""US Standard Atmosphere 1976 model.

Provides density, pressure, temperature, and speed of sound as functions of
geodetic altitude.  Valid from sea level to 86 km using the seven standard
lapse-rate layers; above 86 km an exponential decay approximation is used
that brings density to effectively zero by 200 km.

The returned density is multiplied by the ``ATMO_DENSITY_SCALE`` config
parameter to support Monte-Carlo dispersion studies.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from sim import config

# ---------------------------------------------------------------------------
# Physical constants for the atmosphere model (US Std Atmo 1976, Table 3)
# ---------------------------------------------------------------------------

#: Molar mass of dry air (kg/mol)
_MOLAR_MASS_AIR: float = 0.0289644

#: Universal gas constant (J/(mol*K))
_R_UNIVERSAL: float = 8.31447

#: Specific gas constant for dry air (J/(kg*K))
_R_SPECIFIC: float = _R_UNIVERSAL / _MOLAR_MASS_AIR  # ~287.058

#: Ratio of specific heats for diatomic ideal gas
_GAMMA: float = 1.4

#: Standard sea-level temperature (K)
_T0: float = 288.15

#: Standard sea-level pressure (Pa)
_P0: float = 101325.0

#: Standard sea-level density (kg/m^3)
_RHO0: float = 1.225

#: Standard gravitational acceleration used in barometric formula (m/s^2)
_G0: float = config.G0

# ---------------------------------------------------------------------------
# US Standard Atmosphere 1976 layer definitions (0 -- 86 km)
# Each tuple: (base_altitude_m, lapse_rate_K_per_m, base_temp_K, base_pressure_Pa)
# Values are derived from the standard; base_temp and base_pressure for each
# layer are computed once at import time to avoid repeated work.
# ---------------------------------------------------------------------------

# (geopotential altitude in m, temperature lapse rate in K/m)
_LAYER_PARAMS: list[tuple[float, float]] = [
    (0.0, -0.0065),  # Troposphere
    (11_000.0, 0.0),  # Tropopause
    (20_000.0, 0.001),  # Stratosphere (lower)
    (32_000.0, 0.0028),  # Stratosphere (upper)
    (47_000.0, 0.0),  # Stratopause
    (51_000.0, -0.0028),  # Mesosphere (lower)
    (71_000.0, -0.002),  # Mesosphere (upper)
]

#: Altitude ceiling of the tabulated model (m).
_TABULATED_CEILING_M: float = 86_000.0

#: Altitude above which density is treated as zero (m).
_EXOSPHERE_CEILING_M: float = 1_000_000.0


class _LayerData:
    """Pre-computed base values for a single atmospheric layer."""

    __slots__ = ("h_base", "lapse", "T_base", "P_base", "rho_base")

    def __init__(
        self,
        h_base: float,
        lapse: float,
        T_base: float,
        P_base: float,
        rho_base: float,
    ) -> None:
        self.h_base = h_base
        self.lapse = lapse
        self.T_base = T_base
        self.P_base = P_base
        self.rho_base = rho_base


def _build_layers() -> list[_LayerData]:
    """Pre-compute base temperature, pressure, and density for each layer."""
    layers: list[_LayerData] = []
    T = _T0
    P = _P0
    rho = _RHO0

    for i, (h_base, lapse) in enumerate(_LAYER_PARAMS):
        layers.append(_LayerData(h_base, lapse, T, P, rho))

        # Compute values at the top of this layer (= base of next layer)
        if i + 1 < len(_LAYER_PARAMS):
            h_top = _LAYER_PARAMS[i + 1][0]
        else:
            h_top = _TABULATED_CEILING_M

        dh = h_top - h_base

        if abs(lapse) > 1e-12:
            T_top = T + lapse * dh
            P_top = P * (T_top / T) ** (-_G0 / (_R_SPECIFIC * lapse))
        else:
            T_top = T
            P_top = P * math.exp(-_G0 * dh / (_R_SPECIFIC * T))

        rho_top = P_top / (_R_SPECIFIC * T_top)
        T, P, rho = T_top, P_top, rho_top

    return layers


# Layers are built once at module import.
_LAYERS: list[_LayerData] = _build_layers()


class AtmosphereResult(NamedTuple):
    """Atmospheric conditions at a given altitude."""

    density_kg_m3: float
    """Air density (kg/m^3), already scaled by ``ATMO_DENSITY_SCALE``."""

    pressure_pa: float
    """Static pressure (Pa)."""

    temperature_k: float
    """Static temperature (K)."""

    speed_of_sound_ms: float
    """Speed of sound (m/s)."""


def atmosphere(altitude_m: float) -> AtmosphereResult:
    """Evaluate the US Standard Atmosphere 1976 at a geodetic altitude.

    Below 0 m the sea-level values are returned (clamped).  Between 0 and
    86 km the standard seven-layer lapse-rate model is used.  Above 86 km
    an exponential decay approximation is applied, reaching near-zero
    density by 200 km.

    Args:
        altitude_m: Geodetic altitude above the WGS84 ellipsoid (m).

    Returns:
        An :class:`AtmosphereResult` named tuple with density, pressure,
        temperature, and speed of sound.
    """
    density_scale: float = config.ATMO_DENSITY_SCALE

    # Clamp below sea level
    if altitude_m <= 0.0:
        return AtmosphereResult(
            density_kg_m3=_RHO0 * density_scale,
            pressure_pa=_P0,
            temperature_k=_T0,
            speed_of_sound_ms=math.sqrt(_GAMMA * _R_SPECIFIC * _T0),
        )

    # ------------------------------------------------------------------
    # 0 -- 86 km: standard lapse-rate layers
    # ------------------------------------------------------------------
    if altitude_m <= _TABULATED_CEILING_M:
        T, P = _evaluate_standard_layers(altitude_m)
        rho = P / (_R_SPECIFIC * T)
        a = math.sqrt(_GAMMA * _R_SPECIFIC * T)
        return AtmosphereResult(
            density_kg_m3=rho * density_scale,
            pressure_pa=P,
            temperature_k=T,
            speed_of_sound_ms=a,
        )

    # ------------------------------------------------------------------
    # 86 -- 200 km: exponential decay approximation
    # ------------------------------------------------------------------
    if altitude_m < _EXOSPHERE_CEILING_M:
        return _high_altitude(altitude_m, density_scale)

    # Above 200 km — effectively vacuum
    return AtmosphereResult(
        density_kg_m3=0.0,
        pressure_pa=0.0,
        temperature_k=0.0,
        speed_of_sound_ms=0.0,
    )


def _evaluate_standard_layers(altitude_m: float) -> tuple[float, float]:
    """Return (temperature_K, pressure_Pa) for 0 < altitude <= 86 km."""
    # Find the correct layer (walk from top to bottom)
    layer = _LAYERS[0]
    for L in reversed(_LAYERS):
        if altitude_m >= L.h_base:
            layer = L
            break

    dh = altitude_m - layer.h_base

    if abs(layer.lapse) > 1e-12:
        T = layer.T_base + layer.lapse * dh
        P = layer.P_base * (T / layer.T_base) ** (-_G0 / (_R_SPECIFIC * layer.lapse))
    else:
        T = layer.T_base
        P = layer.P_base * math.exp(-_G0 * dh / (_R_SPECIFIC * T))

    return T, P


def _high_altitude(altitude_m: float, density_scale: float) -> AtmosphereResult:
    """Multi-layer exponential atmosphere from 86 km to 1000 km.

    Replaces a single-scale-height approximation with a piecewise-exponential
    model derived from NRLMSISE-00 / CIRA-2012 reference profiles.  Each layer
    uses a fitted scale height and base density that reproduces the mean
    thermospheric density profile to within ~20% (solar-cycle dependent
    variations can exceed this, handled via ATMO_DENSITY_SCALE in Monte Carlo).

    The thermosphere temperature profile rises from ~186 K at 86 km to
    ~1000 K above 250 km (moderate solar activity, F10.7 ~ 150 SFU).

    References:
        Picone et al., "NRLMSISE-00 empirical model of the atmosphere",
        J. Geophys. Res., 107(A12), 2002.
        Jacchia, "New static models of the thermosphere and exosphere
        with empirical temperature profiles", SAO Special Report 313, 1970.
    """
    # Piecewise-exponential layers for the thermosphere.
    # Each tuple: (base_alt_m, base_density_kg_m3, scale_height_m, temperature_K)
    # Densities and scale heights fitted to NRLMSISE-00 moderate solar activity.
    _THERMO_LAYERS: list[tuple[float, float, float, float]] = [
        (86_000.0, 6.958e-6, 5_900.0, 186.9),  # Mesopause → lower thermosphere
        (100_000.0, 5.604e-7, 6_400.0, 195.1),  # Kármán line region
        (115_000.0, 4.289e-8, 7_800.0, 304.0),  # Lower thermosphere
        (150_000.0, 2.076e-9, 22_800.0, 634.0),  # Mid-thermosphere (rapid T rise)
        (200_000.0, 2.541e-10, 37_500.0, 855.0),  # Upper thermosphere
        (300_000.0, 1.916e-11, 53_600.0, 976.0),  # Near-exobase
        (500_000.0, 5.215e-13, 75_800.0, 999.0),  # Exosphere transition
        (750_000.0, 3.561e-15, 100_000.0, 1000.0),  # Upper exosphere
    ]

    # Find the correct layer
    layer = _THERMO_LAYERS[0]
    for L in _THERMO_LAYERS:
        if altitude_m >= L[0]:
            layer = L
        else:
            break

    h_base, rho_base, scale_h, T = layer
    dh = altitude_m - h_base
    rho = rho_base * math.exp(-dh / scale_h)

    # At very high altitude, clamp to effective vacuum
    if rho < 1e-18:
        rho = 0.0

    P = rho * _R_SPECIFIC * T
    a = math.sqrt(_GAMMA * _R_SPECIFIC * max(T, 1.0)) if T > 0.0 else 0.0

    return AtmosphereResult(
        density_kg_m3=rho * density_scale,
        pressure_pa=P,
        temperature_k=T,
        speed_of_sound_ms=a,
    )
