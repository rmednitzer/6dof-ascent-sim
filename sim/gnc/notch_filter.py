"""Frequency-scheduled structural notch filter for control-structure interaction.

A cascade of second-order band-stop (notch) biquads, one per flexible bending
mode, applied to the measured body rate before the attitude controller's rate
feedback. Each notch is centred on the *current* modal frequency — which drifts
upward as propellant drains — so the controller never chases the lightly-damped
bending modes. This is the standard fix for the "tail-wags-dog" flutter
instability that a naive rate-feedback coupling of the flex model produces
(AD-04): the rigid-body control bandwidth (~0.3 Hz) passes through unattenuated
while the structural modes (~1–10 Hz) are rejected.

Each section uses the RBJ audio-EQ-cookbook notch design, discretised at the
control rate ``fs`` by the bilinear transform:

    w0 = 2*pi*f0/fs ;  alpha = sin(w0) / (2 Q)
    H(z) = (1 - 2 cos(w0) z^-1 + z^-2)
           / ((1 + alpha) - 2 cos(w0) z^-1 + (1 - alpha) z^-2)

H(z) is exactly unity at DC and Nyquist and has a transmission zero on the unit
circle at ``w0`` (full rejection at the notch frequency). The width is set by the
quality factor ``Q`` (larger Q = narrower notch, less phase distortion of the
rigid-body signal). Coefficients are recomputed every step from the scheduled
``f0`` while the section state is held continuous, i.e. a gain-scheduled
(linear time-varying) biquad — valid because the modal frequency varies slowly
relative to the 100 Hz update rate.

References:
    Robert Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter
    coefficients."
    Wie, *Space Vehicle Dynamics and Control*, 2nd ed., Ch. 7 (structural
    filtering and control-structure interaction).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


class _NotchSection:
    """Single second-order notch biquad with persistent state.

    Implemented in Direct Form II transposed, which is numerically well-behaved
    and keeps only two state variables per section.
    """

    __slots__ = ("_z1", "_z2")

    def __init__(self) -> None:
        self._z1: float = 0.0
        self._z2: float = 0.0

    def reset(self) -> None:
        """Zero the filter memory (e.g. on staging discontinuities)."""
        self._z1 = 0.0
        self._z2 = 0.0

    def process(self, x: float, f0_hz: float, fs_hz: float, q: float) -> float:
        """Filter one sample, with the notch centred at ``f0_hz``.

        A notch at or below DC, at or above Nyquist, or with a non-positive Q is
        treated as a pass-through (the modal frequency is then outside the
        controllable band, so there is nothing to reject).
        """
        if f0_hz <= 0.0 or f0_hz >= 0.5 * fs_hz or q <= 0.0:
            return x

        w0 = 2.0 * math.pi * f0_hz / fs_hz
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * q)
        a0 = 1.0 + alpha

        b0 = 1.0 / a0
        b1 = -2.0 * cos_w0 / a0
        b2 = 1.0 / a0
        a1 = -2.0 * cos_w0 / a0
        a2 = (1.0 - alpha) / a0

        y = b0 * x + self._z1
        self._z1 = b1 * x - a1 * y + self._z2
        self._z2 = b2 * x - a2 * y
        return y


class StructuralNotchFilter:
    """Cascade of frequency-scheduled notch sections (one per bending mode)."""

    def __init__(self, n_modes: int, fs_hz: float, q: float) -> None:
        if n_modes < 0:
            raise ValueError("n_modes must be non-negative")
        if fs_hz <= 0.0:
            raise ValueError("fs_hz must be positive")
        self._sections: list[_NotchSection] = [_NotchSection() for _ in range(n_modes)]
        self._fs_hz: float = fs_hz
        self._q: float = q

    @property
    def n_modes(self) -> int:
        """Number of notch sections in the cascade."""
        return len(self._sections)

    def reset(self) -> None:
        """Reset every section's state."""
        for section in self._sections:
            section.reset()

    def process(self, x: float, mode_freqs_hz: Sequence[float] | Iterable[float]) -> float:
        """Filter one sample through the cascade.

        Parameters
        ----------
        x : float
            Input sample (e.g. a measured body rate, rad/s).
        mode_freqs_hz : sequence of float
            Current notch centre frequency for each section (Hz). Extra
            frequencies beyond ``n_modes`` are ignored; missing ones leave the
            corresponding section as a pass-through.
        """
        y = x
        for section, f0 in zip(self._sections, mode_freqs_hz, strict=False):
            y = section.process(y, float(f0), self._fs_hz, self._q)
        return y
