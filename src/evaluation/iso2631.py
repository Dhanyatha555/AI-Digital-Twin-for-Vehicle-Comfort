"""
ISO 2631-1 Frequency Weighting Filter (W_k) and Human Ride Comfort Scoring.

Implementation follows ISO 2631-1:1997:
"Mechanical vibration and shock -- Evaluation of human exposure to whole-body vibration -- Part 1: General requirements"

Frequency weighting W_k is defined for vertical (z-axis) seated / standing human exposure.
Analog filter transfer function H(s) = H_b(s) * H_a(s) * H_k(s) is transformed to digital
second-order sections (SOS) via the Bilinear Transform for optimal numerical stability.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np
from scipy.signal import bilinear_zpk, sosfilt, sosfiltfilt, zpk2sos


class ComfortCategory(str, Enum):
    COMFORTABLE = "Comfortable (< 0.315 m/s^2)"
    A_LITTLE_UNCOMFORTABLE = "A Little Uncomfortable (0.315 - 0.63 m/s^2)"
    FAIRLY_UNCOMFORTABLE = "Fairly Uncomfortable (0.5 - 1.0 m/s^2)"
    UNCOMFORTABLE = "Uncomfortable (0.8 - 1.6 m/s^2)"
    VERY_UNCOMFORTABLE = "Very Uncomfortable (1.25 - 2.5 m/s^2)"
    EXTREMELY_UNCOMFORTABLE = "Extremely Uncomfortable (> 2.0 m/s^2)"


@dataclass
class RideComfortEvaluation:
    """Comprehensive ride comfort metrics according to ISO 2631-1."""
    raw_rms_accel: float              # Unweighted RMS acceleration a_RMS [m/s^2]
    weighted_rms_accel: float         # Frequency-weighted RMS acceleration a_w,RMS [m/s^2]
    peak_accel: float                 # Peak absolute acceleration [m/s^2]
    crest_factor: float               # Crest factor CF = peak / a_w,RMS
    vibration_dose_value: float       # VDV = (integral(a_w^4 dt))^(1/4) [m/s^1.75]
    comfort_category: ComfortCategory # ISO 2631 subjective comfort class
    weighted_accel_series: np.ndarray # Filtered time series a_w(t)


class ISO2631Filter:
    """
    Constructs the discrete ISO 2631-1 W_k weighting filter using Second-Order Sections (SOS).
    """

    def __init__(self, fs: float = 500.0):
        self.fs = fs
        self.sos = self._build_w_k_sos(fs)

    @staticmethod
    def _build_w_k_sos(fs: float) -> np.ndarray:
        """
        Builds analog continuous poles, zeros, and gain for W_k filter,
        discretizes via Bilinear transform, and converts to SOS.
        """
        # ISO 2631-1 Table A.1 Parameters for W_k
        f1, Q1 = 0.4, 1.0 / np.sqrt(2.0)    # Band-limiting highpass
        f2, Q2 = 100.0, 1.0 / np.sqrt(2.0)  # Band-limiting lowpass
        f3 = 12.5                           # Acceleration-velocity transition zero
        f4, Q4 = 12.5, 0.63                 # Acceleration-velocity transition poles
        f5, Q5 = 2.37, 0.91                 # Upward step zeros
        f6, Q6 = 3.12, 0.69                 # Upward step poles

        w1 = 2.0 * np.pi * f1
        w2 = 2.0 * np.pi * f2
        w3 = 2.0 * np.pi * f3
        w4 = 2.0 * np.pi * f4
        w5 = 2.0 * np.pi * f5
        w6 = 2.0 * np.pi * f6

        zeros = []
        poles = []
        k = 1.0

        # 1. Band-limiting High-Pass: s^2 / (s^2 + w1/Q1 s + w1^2)
        zeros.extend([0.0, 0.0])
        hp_disc = (w1 / Q1)**2 - 4.0 * (w1**2)
        poles.append(-0.5 * (w1 / Q1) + 0.5 * np.sqrt(hp_disc + 0j))
        poles.append(-0.5 * (w1 / Q1) - 0.5 * np.sqrt(hp_disc + 0j))

        # 2. Band-limiting Low-Pass: w2^2 / (s^2 + w2/Q2 s + w2^2)
        lp_disc = (w2 / Q2)**2 - 4.0 * (w2**2)
        poles.append(-0.5 * (w2 / Q2) + 0.5 * np.sqrt(lp_disc + 0j))
        poles.append(-0.5 * (w2 / Q2) - 0.5 * np.sqrt(lp_disc + 0j))
        k *= (w2**2)

        # 3. Acceleration-Velocity Transition: (s + w3)/w3 * w4^2 / (s^2 + w4/Q4 s + w4^2)
        zeros.append(-w3)
        trans_disc = (w4 / Q4)**2 - 4.0 * (w4**2)
        poles.append(-0.5 * (w4 / Q4) + 0.5 * np.sqrt(trans_disc + 0j))
        poles.append(-0.5 * (w4 / Q4) - 0.5 * np.sqrt(trans_disc + 0j))
        k *= (w4**2 / w3)

        # 4. Upward Step Filter: (s^2 + w5/Q5 s + w5^2) / (s^2 + w6/Q6 s + w6^2) * (w6/w5)^2
        step_num_disc = (w5 / Q5)**2 - 4.0 * (w5**2)
        zeros.append(-0.5 * (w5 / Q5) + 0.5 * np.sqrt(step_num_disc + 0j))
        zeros.append(-0.5 * (w5 / Q5) - 0.5 * np.sqrt(step_num_disc + 0j))

        step_den_disc = (w6 / Q6)**2 - 4.0 * (w6**2)
        poles.append(-0.5 * (w6 / Q6) + 0.5 * np.sqrt(step_den_disc + 0j))
        poles.append(-0.5 * (w6 / Q6) - 0.5 * np.sqrt(step_den_disc + 0j))
        k *= ((w6 / w5)**2)

        # Bilinear transform to digital domain
        z_d, p_d, k_d = bilinear_zpk(zeros, poles, k, fs=fs)

        # Convert to Second-Order Sections (SOS)
        sos = zpk2sos(z_d, p_d, k_d)
        return sos

    def filter(self, acceleration: np.ndarray, zero_phase: bool = False) -> np.ndarray:
        """
        Apply ISO 2631 W_k weighting filter to vertical acceleration series.
        Args:
            acceleration: Raw body vertical acceleration [m/s^2]
            zero_phase: If True, uses sosfiltfilt (non-causal, zero phase shift).
                        If False, uses sosfilt (causal, real-time vehicle ECU simulation).
        """
        if zero_phase:
            return sosfiltfilt(self.sos, acceleration)
        return sosfilt(self.sos, acceleration)

    def evaluate_comfort(
        self,
        acceleration: np.ndarray,
        time: Optional[np.ndarray] = None,
        zero_phase: bool = False,
    ) -> RideComfortEvaluation:
        """
        Compute ISO 2631-1 ride comfort metrics on acceleration time series.
        """
        raw_rms = float(np.sqrt(np.mean(acceleration ** 2)))
        a_w = self.filter(acceleration, zero_phase=zero_phase)
        w_rms = float(np.sqrt(np.mean(a_w ** 2)))

        peak = float(np.max(np.abs(a_w)))
        crest_factor = float(peak / w_rms) if w_rms > 1e-6 else 1.0

        # Vibration Dose Value (VDV)
        dt = float(time[1] - time[0]) if time is not None and len(time) > 1 else (1.0 / self.fs)
        vdv = float((np.sum(a_w ** 4) * dt) ** 0.25)

        # Classify comfort according to ISO 2631-1 Section 8.2.2
        if w_rms < 0.315:
            cat = ComfortCategory.COMFORTABLE
        elif w_rms < 0.63:
            cat = ComfortCategory.A_LITTLE_UNCOMFORTABLE
        elif w_rms < 1.0:
            cat = ComfortCategory.FAIRLY_UNCOMFORTABLE
        elif w_rms < 1.6:
            cat = ComfortCategory.UNCOMFORTABLE
        elif w_rms < 2.5:
            cat = ComfortCategory.VERY_UNCOMFORTABLE
        else:
            cat = ComfortCategory.EXTREMELY_UNCOMFORTABLE

        return RideComfortEvaluation(
            raw_rms_accel=raw_rms,
            weighted_rms_accel=w_rms,
            peak_accel=peak,
            crest_factor=crest_factor,
            vibration_dose_value=vdv,
            comfort_category=cat,
            weighted_accel_series=a_w,
        )
