"""
Sensor Simulation, Noise Injection, Dropout Emulation, and Signal Conditioning.

Simulates raw vehicle telemetry sensors:
1. Body Accelerometer (z-axis IMU) with bias, drift, and white Gaussian noise.
2. Wheel Hub Accelerometer with harsh high-frequency noise.
3. Suspension Travel Sensor (LVDT / linear potentiometer) with quantization and noise.
4. Vehicle speed and wheel speed with CAN jitter.
5. Realistic sensor dropouts (packet loss, sensor freeze, disconnected pulses).
6. Signal conditioning: dropout detection, forward-fill/interpolation, and Butterworth filtering.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt
from ..physics.quarter_car import SimulationResult


@dataclass
class SensorNoiseConfig:
    """Noise parameters for automotive sensors."""
    body_accel_std: float = 0.12        # Body accelerometer noise std [m/s^2]
    body_accel_bias: float = 0.05       # Accelerometer constant bias [m/s^2]
    wheel_accel_std: float = 0.55       # Hub accelerometer noise std [m/s^2]
    suspension_travel_std: float = 0.0008  # LVDT travel sensor noise std [m] (0.8 mm)
    speed_std_kmh: float = 0.35         # Speed sensor noise std [km/h]

    # Dropout / Packet loss emulation
    dropout_probability: float = 0.02   # Probability of dropout burst initiation
    dropout_burst_len_min: int = 5      # Minimum consecutive samples lost
    dropout_burst_len_max: int = 25     # Maximum consecutive samples lost


@dataclass
class RawSensorMeasurements:
    """Raw, noisy sensor measurements with potential dropouts."""
    time: np.ndarray
    body_accel_raw: np.ndarray          # Body acceleration with noise & bias [m/s^2]
    wheel_accel_raw: np.ndarray         # Wheel hub acceleration [m/s^2]
    suspension_travel_raw: np.ndarray   # Suspension travel [m]
    vehicle_speed_kmh_raw: np.ndarray   # Vehicle speed [km/h]
    wheel_speed_kmh_raw: np.ndarray     # Wheel speed [km/h]
    dropout_mask: np.ndarray            # Boolean mask: True where samples were dropped


@dataclass
class ConditionedSensorData:
    """Conditioned, synchronized signals after dropout repair and filtering."""
    time: np.ndarray
    body_accel_filtered: np.ndarray
    wheel_accel_filtered: np.ndarray
    suspension_travel_filtered: np.ndarray
    vehicle_speed_kmh: np.ndarray
    wheel_speed_kmh: np.ndarray
    repaired_mask: np.ndarray           # Locations where dropouts were repaired
    detected_dropouts_count: int


class SensorSimulator:
    """
    Injects realistic noise, biases, and dropouts into ground-truth vehicle simulation signals.
    """

    def __init__(self, config: Optional[SensorNoiseConfig] = None, seed: int = 42):
        self.config = config if config is not None else SensorNoiseConfig()
        self.rng = np.random.default_rng(seed)

    def generate_measurements(
        self,
        sim_result: SimulationResult,
        vehicle_speed_kmh: float,
        apply_dropouts: bool = True,
    ) -> RawSensorMeasurements:
        """Add noise, bias, and dropouts to simulated signals."""
        c = self.config
        n = len(sim_result.time)

        # Body accelerometer: True accel + bias + Gaussian noise
        body_noise = self.rng.normal(0.0, c.body_accel_std, size=n)
        body_accel = sim_result.ddz_s + c.body_accel_bias + body_noise

        # Wheel accelerometer: True wheel accel + noise
        wheel_noise = self.rng.normal(0.0, c.wheel_accel_std, size=n)
        wheel_accel = sim_result.ddz_u + wheel_noise

        # Suspension travel sensor: True travel + noise
        travel_noise = self.rng.normal(0.0, c.suspension_travel_std, size=n)
        susp_travel = sim_result.suspension_travel + travel_noise

        # Vehicle and wheel speeds
        speed_noise_v = self.rng.normal(0.0, c.speed_std_kmh, size=n)
        v_speed = np.full(n, vehicle_speed_kmh) + speed_noise_v

        # Wheel speed has small slip/vibration oscillation
        slip_oscillation = 0.5 * np.sin(2.0 * np.pi * 12.0 * sim_result.time) * (sim_result.tire_deflection / 0.02)
        w_speed = v_speed + slip_oscillation + self.rng.normal(0.0, c.speed_std_kmh * 0.8, size=n)

        dropout_mask = np.zeros(n, dtype=bool)

        if apply_dropouts:
            # Inject burst dropouts
            idx = 20  # Keep first 20 samples intact for initialization
            while idx < n - 30:
                if self.rng.random() < c.dropout_probability:
                    burst_len = self.rng.integers(c.dropout_burst_len_min, c.dropout_burst_len_max + 1)
                    end_idx = min(idx + burst_len, n)
                    dropout_mask[idx:end_idx] = True

                    # Corrupt body accel and suspension travel with NaNs (packet loss)
                    body_accel[idx:end_idx] = np.nan
                    susp_travel[idx:end_idx] = np.nan
                    wheel_accel[idx:end_idx] = np.nan
                    idx = end_idx + 15
                else:
                    idx += 1

        return RawSensorMeasurements(
            time=sim_result.time.copy(),
            body_accel_raw=body_accel,
            wheel_accel_raw=wheel_accel,
            suspension_travel_raw=susp_travel,
            vehicle_speed_kmh_raw=v_speed,
            wheel_speed_kmh_raw=w_speed,
            dropout_mask=dropout_mask,
        )


class SignalConditioner:
    """
    Cleans raw sensor streams: detects dropouts, performs causal/non-causal interpolation,
    and applies digital low-pass Butterworth filtering to eliminate high-frequency instrumentation noise.
    """

    def __init__(self, cutoff_freq_hz: float = 40.0, order: int = 3):
        self.cutoff_freq_hz = cutoff_freq_hz
        self.order = order

    def repair_dropouts(self, time: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect NaNs and impute using shape-preserving linear interpolation.
        Returns:
            repaired_signal: array with all NaNs filled
            nan_mask: boolean mask indicating repaired points
        """
        nan_mask = np.isnan(signal)
        if not np.any(nan_mask):
            return signal.copy(), nan_mask

        repaired = signal.copy()
        valid_indices = np.where(~nan_mask)[0]

        if len(valid_indices) == 0:
            repaired[:] = 0.0
            return repaired, nan_mask

        # Linear interpolation over invalid samples
        repaired[nan_mask] = np.interp(
            time[nan_mask],
            time[valid_indices],
            signal[valid_indices],
        )

        return repaired, nan_mask

    def lowpass_filter(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Apply zero-phase Butterworth low-pass filter."""
        nyquist = 0.5 * fs
        normal_cutoff = min(self.cutoff_freq_hz / nyquist, 0.95)
        b, a = butter(self.order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, signal)

    def process(
        self,
        raw: RawSensorMeasurements,
        filter_signals: bool = True,
    ) -> ConditionedSensorData:
        """Run complete conditioning pipeline on raw sensor measurements."""
        dt = float(raw.time[1] - raw.time[0])
        fs = 1.0 / dt

        body_accel_clean, mask_ba = self.repair_dropouts(raw.time, raw.body_accel_raw)
        wheel_accel_clean, mask_wa = self.repair_dropouts(raw.time, raw.wheel_accel_raw)
        susp_travel_clean, mask_st = self.repair_dropouts(raw.time, raw.suspension_travel_raw)

        combined_repair_mask = mask_ba | mask_wa | mask_st
        dropout_count = int(np.sum(combined_repair_mask))

        if filter_signals:
            body_accel_filt = self.lowpass_filter(body_accel_clean, fs)
            wheel_accel_filt = self.lowpass_filter(wheel_accel_clean, fs)
            susp_travel_filt = self.lowpass_filter(susp_travel_clean, fs)
        else:
            body_accel_filt = body_accel_clean
            wheel_accel_filt = wheel_accel_clean
            susp_travel_filt = susp_travel_clean

        return ConditionedSensorData(
            time=raw.time.copy(),
            body_accel_filtered=body_accel_filt,
            wheel_accel_filtered=wheel_accel_filt,
            suspension_travel_filtered=susp_travel_filt,
            vehicle_speed_kmh=raw.vehicle_speed_kmh_raw.copy(),
            wheel_speed_kmh=raw.wheel_speed_kmh_raw.copy(),
            repaired_mask=combined_repair_mask,
            detected_dropouts_count=dropout_count,
        )
