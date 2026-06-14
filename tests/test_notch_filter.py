"""Unit tests for the frequency-scheduled structural notch filter (AD-04)."""

from __future__ import annotations

import math

from sim.gnc.notch_filter import StructuralNotchFilter

FS = 100.0


def _steady_state_amplitude(nf: StructuralNotchFilter, freq_hz: float, notch_freqs, n: int = 4000) -> float:
    """Peak |output| over the second half of a unit-sine sweep (steady state)."""
    nf.reset()
    peak = 0.0
    for i in range(n):
        t = i / FS
        y = nf.process(math.sin(2.0 * math.pi * freq_hz * t), notch_freqs)
        if i > n // 2:
            peak = max(peak, abs(y))
    return peak


class TestNotchResponse:
    def test_rejects_notch_frequency(self):
        nf = StructuralNotchFilter(n_modes=1, fs_hz=FS, q=2.0)
        # A sinusoid at the notch centre is strongly attenuated.
        assert _steady_state_amplitude(nf, 1.2, [1.2]) < 0.05

    def test_passes_low_frequency(self):
        nf = StructuralNotchFilter(n_modes=1, fs_hz=FS, q=2.0)
        # The rigid-body control band (well below the mode) passes ~unchanged.
        assert _steady_state_amplitude(nf, 0.2, [1.2]) > 0.95

    def test_cascade_rejects_all_modes(self):
        modes = [1.2, 3.5, 7.0]
        nf = StructuralNotchFilter(n_modes=3, fs_hz=FS, q=2.0)
        for f in modes:
            assert _steady_state_amplitude(nf, f, modes) < 0.05
        # ...while still passing DC.
        assert _steady_state_amplitude(nf, 0.2, modes) > 0.95

    def test_tracks_scheduled_frequency(self):
        # The same section rejects whatever centre frequency it is given,
        # i.e. it tracks the propellant-varying mode (gain scheduling).
        nf = StructuralNotchFilter(n_modes=1, fs_hz=FS, q=2.0)
        assert _steady_state_amplitude(nf, 2.0, [2.0]) < 0.05


class TestNotchEdgeCases:
    def test_passthrough_above_nyquist(self):
        nf = StructuralNotchFilter(n_modes=1, fs_hz=FS, q=2.0)
        # A notch above Nyquist is a no-op: output equals input every sample.
        for x in (1.0, -0.5, 0.3, 2.0):
            assert nf.process(x, [60.0]) == x

    def test_passthrough_zero_freq(self):
        nf = StructuralNotchFilter(n_modes=1, fs_hz=FS, q=2.0)
        for x in (1.0, -0.5, 0.3):
            assert nf.process(x, [0.0]) == x

    def test_reset_zeros_state(self):
        nf = StructuralNotchFilter(n_modes=1, fs_hz=FS, q=2.0)
        for _ in range(50):
            nf.process(1.0, [1.2])
        nf.reset()
        # With zero input and zero state, the output is exactly zero.
        assert nf.process(0.0, [1.2]) == 0.0

    def test_zero_modes_is_identity(self):
        nf = StructuralNotchFilter(n_modes=0, fs_hz=FS, q=2.0)
        assert nf.n_modes == 0
        for x in (1.0, -0.5, 0.3):
            assert nf.process(x, []) == x

    def test_dc_gain_is_unity(self):
        # A constant input settles to itself (unity DC gain) — the controller's
        # steady rate feedback is not biased by the filter.
        nf = StructuralNotchFilter(n_modes=3, fs_hz=FS, q=2.0)
        y = 0.0
        for _ in range(2000):
            y = nf.process(5.0, [1.2, 3.5, 7.0])
        assert abs(y - 5.0) < 1e-6
