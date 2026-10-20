"""
Unit tests for Kalman Filter Sensor Fusion and State Estimation.
"""

import numpy as np
import pytest
from src.physics.quarter_car import QuarterCarModel, QuarterCarParameters
from src.sensors.noise_dropout import SensorSimulator, SignalConditioner
from src.sensors.kalman_filter import SuspensionKalmanFilter


def test_kalman_filter_stability():
    """Verify state transition matrix F has eigenvalues inside unit circle."""
    params = QuarterCarParameters()
    kf = SuspensionKalmanFilter(params)
    F, H, Q = kf.build_system_matrices(dt=0.002)

    eigenvalues = np.linalg.eigvals(F)
    # Spectral radius must be strictly <= 1.0 for stability
    assert np.all(np.abs(eigenvalues) <= 1.0 + 1e-6)
    assert F.shape == (4, 4)
    assert H.shape == (3, 4)


def test_kalman_filter_state_tracking():
    """Verify Kalman filter accurately tracks suspension travel and velocities from noisy sensors."""
    params = QuarterCarParameters(c_s=1400.0)
    model = QuarterCarModel(params)
    t_eval = np.linspace(0.0, 3.0, 1500)
    t_span = (0.0, 3.0)

    # Road bump excitation
    def road_bump(t: float):
        if 0.5 <= t <= 0.8:
            z = 0.04 * np.sin(np.pi * (t - 0.5) / 0.3)
            dz = 0.04 * (np.pi / 0.3) * np.cos(np.pi * (t - 0.5) / 0.3)
            return z, dz
        return 0.0, 0.0

    sim = model.simulate(t_span, t_eval, road_bump)

    # Add realistic sensor noise (no dropouts for this tracking test)
    sensor_sim = SensorSimulator(seed=42)
    raw = sensor_sim.generate_measurements(sim, vehicle_speed_kmh=50.0, apply_dropouts=False)

    kf = SuspensionKalmanFilter(params)
    fused = kf.estimate(
        time=raw.time,
        body_accel=raw.body_accel_raw,
        wheel_accel=raw.wheel_accel_raw,
        susp_travel=raw.suspension_travel_raw,
    )

    # Travel estimation RMSE should be very small (< 1.5 mm)
    travel_rmse = np.sqrt(np.mean((sim.suspension_travel - fused.suspension_travel) ** 2))
    assert travel_rmse < 0.0015

    # Velocity estimation correlation with true body velocity should be > 0.90
    corr = np.corrcoef(sim.dz_s, fused.body_velocity)[0, 1]
    assert corr > 0.90


def test_kalman_filter_dropout_handling():
    """Verify filter maintains numerical stability and output continuity during dropouts."""
    params = QuarterCarParameters()
    kf = SuspensionKalmanFilter(params)
    n = 500
    time = np.linspace(0.0, 1.0, n)

    # Synthesize measurements with a 30-sample dropout burst of NaNs
    body_accel = 0.5 * np.sin(2.0 * np.pi * 1.5 * time)
    wheel_accel = 2.0 * np.sin(2.0 * np.pi * 11.0 * time)
    travel = 0.01 * np.sin(2.0 * np.pi * 1.5 * time)

    body_accel[100:130] = np.nan
    wheel_accel[100:130] = np.nan
    travel[100:130] = np.nan

    fused = kf.estimate(time, body_accel, wheel_accel, travel)

    # Filter outputs must not contain any NaNs or Infs
    assert not np.any(np.isnan(fused.suspension_travel))
    assert not np.any(np.isnan(fused.body_velocity))
    assert not np.any(np.isinf(fused.suspension_travel))
