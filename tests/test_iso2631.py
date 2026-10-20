"""
Unit tests for ISO 2631-1 Frequency Weighting and Comfort Metrics.
"""

import numpy as np
import pytest
from scipy.signal import sosfreqz
from src.evaluation.iso2631 import ISO2631Filter, ComfortCategory


def test_iso2631_filter_stability():
    """Verify all poles of the discrete digital SOS filter lie strictly inside unit circle."""
    fs = 500.0
    iso = ISO2631Filter(fs=fs)
    sos = iso.sos

    # Extract poles from each biquad section
    for section in sos:
        a0, a1, a2 = section[3], section[4], section[5]
        roots = np.roots([a0, a1, a2])
        # Stability condition: |z| < 1
        assert np.all(np.abs(roots) < 1.0)


def test_iso2631_wk_frequency_response_peak():
    """
    Verify ISO 2631-1 W_k weighting response peaks in the critical human sensitivity range (4 - 8 Hz).
    Standard specifies maximum sensitivity to vertical whole-body vibration between 4 Hz and 8 Hz.
    """
    fs = 500.0
    iso = ISO2631Filter(fs=fs)

    freqs = np.linspace(0.5, 40.0, 500)
    w = 2.0 * np.pi * freqs / fs
    _, h = sosfreqz(iso.sos, worN=w)
    mag = np.abs(h)

    # Frequency corresponding to maximum filter gain
    peak_freq = freqs[np.argmax(mag)]
    # Peak must lie in the human whole-body resonance zone (4.0 - 9.0 Hz)
    assert 4.0 <= peak_freq <= 9.0

    # Low frequency attenuation (e.g. at 0.5 Hz, magnitude should be significantly lower than peak)
    mag_05 = mag[np.argmin(np.abs(freqs - 0.5))]
    mag_peak = np.max(mag)
    assert mag_05 < 0.5 * mag_peak

    # High frequency attenuation (e.g. at 35 Hz, magnitude should roll off)
    mag_35 = mag[np.argmin(np.abs(freqs - 35.0))]
    assert mag_35 < 0.4 * mag_peak


def test_comfort_evaluation_outputs():
    """Verify ISO 2631 comfort metrics and classification categories."""
    fs = 500.0
    iso = ISO2631Filter(fs=fs)
    n = 1500
    time = np.linspace(0.0, 3.0, n)

    # 1. Very mild vibration (< 0.2 m/s^2) -> Comfortable
    mild_accel = 0.15 * np.sin(2.0 * np.pi * 5.0 * time)
    res_mild = iso.evaluate_comfort(mild_accel, time=time)
    assert res_mild.comfort_category == ComfortCategory.COMFORTABLE
    assert res_mild.crest_factor > 1.0
    assert res_mild.vibration_dose_value > 0.0

    # 2. Harsh vibration (> 1.5 m/s^2) -> Uncomfortable / Very Uncomfortable
    harsh_accel = 1.8 * np.sin(2.0 * np.pi * 5.0 * time)
    res_harsh = iso.evaluate_comfort(harsh_accel, time=time)
    assert res_harsh.weighted_rms_accel > 1.0
    assert res_harsh.comfort_category in [
        ComfortCategory.UNCOMFORTABLE,
        ComfortCategory.VERY_UNCOMFORTABLE,
        ComfortCategory.EXTREMELY_UNCOMFORTABLE,
    ]
