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
    (0.0, -0.0065),          # Troposphere
    (11_000.0, 0.0),         # Tropopause
    (20_000.0, 0.001),       # Stratosphere (lower)
    (32_000.0, 0.0028),      # Stratosphere (upper)
    (47_000.0, 0.0),         # Stratopause
    (51_000.0, -0.0028),     # Mesosphere (lower)
    (71_000.0, -0.002),      # Mesosphere (upper)
]

#: Altitude ceiling of the tabulated model (m).
_TABULATED_CEILING_M: float = 86_000.0

#: Altitude above which density is treated as zero (m).
_EXOSPHERE_CEILING_M: float = 200_000.0


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
        P = layer.P_base * (T / layer.T_base) ** (
            -_G0 / (_R_SPECIFIC * layer.lapse)
        )
    else:
        T = layer.T_base
        P = layer.P_base * math.exp(-_G0 * dh / (_R_SPECIFIC * T))

    return T, P


def _high_altitude(altitude_m: float, density_scale: float) -> AtmosphereResult:
    """Exponential-decay model from 86 km to 200 km.

    At 86 km the standard atmosphere gives a known density (~5.0e-6 kg/m^3).
    A scale height of ~6.5 km provides a reasonable fit for the thermosphere
    transition region.  Temperature is held constant at the 86-km value for
    speed-of-sound purposes (the real thermosphere heats up, but the density
    is so low that aerodynamic forces are negligible).
    """
    # Conditions at 86 km from the tabulated model
    T_86, P_86 = _evaluate_standard_layers(_TABULATED_CEILING_M)
    rho_86 = P_86 / (_R_SPECIFIC * T_86)

    # Scale height for exponential decay above 86 km (m)
    _SCALE_HEIGHT_M: float = 6500.0

    dh = altitude_m - _TABULATED_CEILING_M
    rho = rho_86 * math.exp(-dh / _SCALE_HEIGHT_M)
    P = rho * _R_SPECIFIC * T_86  # approximate: ideal-gas with constant T
    a = math.sqrt(_GAMMA * _R_SPECIFIC * T_86)

    return AtmosphereResult(
        density_kg_m3=rho * density_scale,
        pressure_pa=P,
        temperature_k=T_86,
        speed_of_sound_ms=a,
    )
