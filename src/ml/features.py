"""
Physics-Informed Feature Engineering for Suspension Residual Learning.

Extracts physically meaningful state features from baseline physics simulations,
sensor measurements, and Kalman filter estimations.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ..physics.quarter_car import SimulationResult
from ..sensors.kalman_filter import EstimatedStates


@dataclass
class FeatureMatrix:
    """Extracted feature array and corresponding feature names."""
    X: np.ndarray             # 2D feature matrix (N x M)
    feature_names: List[str]  # Feature identifiers for explainability
    time: np.ndarray          # Timestamp array


class FeatureExtractor:
    """
    Transforms baseline physics simulation outputs and Kalman-fused states
    into tabular features for residual regression models.
    """

    def __init__(self, include_lags: bool = True, lag_steps: int = 2):
        self.include_lags = include_lags
        self.lag_steps = lag_steps

    def extract(
        self,
        base_sim: SimulationResult,
        fused_states: Optional[EstimatedStates] = None,
        vehicle_speed_kmh: float = 50.0,
    ) -> FeatureMatrix:
        """
        Extract features from baseline model and optional Kalman state estimates.
        """
        n = len(base_sim.time)
        features: List[np.ndarray] = []
        names: List[str] = []

        # 1. Baseline physical kinematics
        features.append(base_sim.ddz_s)
        names.append("base_body_accel")

        features.append(base_sim.suspension_travel)
        names.append("base_susp_travel")

        features.append(base_sim.suspension_velocity)
        names.append("base_susp_vel")

        features.append(base_sim.tire_deflection)
        names.append("base_tire_deflection")

        features.append(base_sim.z_s)
        names.append("base_body_disp")

        features.append(base_sim.dz_s)
        names.append("base_body_vel")

        # 2. Nonlinear physical damper indicators
        # Damper asymmetry indicator: sign of relative velocity (rebound vs compression)
        rebound_indicator = np.where(base_sim.suspension_velocity >= 0.0, 1.0, -1.0)
        features.append(rebound_indicator)
        names.append("rebound_direction_sign")

        # Damper velocity magnitude and progressive saturation term tanh(v / v_ref)
        vel_abs = np.abs(base_sim.suspension_velocity)
        features.append(vel_abs)
        names.append("susp_vel_magnitude")

        sat_indicator = np.tanh(base_sim.suspension_velocity / 0.45)
        features.append(sat_indicator)
        names.append("damper_saturation_tanh")

        # Kinetic and potential energy indicators
        p_energy = 0.5 * (base_sim.suspension_travel ** 2)
        features.append(p_energy)
        names.append("susp_potential_energy_proxy")

        k_energy = 0.5 * (base_sim.suspension_velocity ** 2)
        features.append(k_energy)
        names.append("susp_kinetic_energy_proxy")

        # 3. Vehicle operating condition
        v_speed_arr = np.full(n, vehicle_speed_kmh / 100.0)  # normalized
        features.append(v_speed_arr)
        names.append("vehicle_speed_norm")

        # 4. Kalman-fused states (if available)
        if fused_states is not None:
            features.append(fused_states.suspension_travel)
            names.append("fused_susp_travel")

            features.append(fused_states.suspension_velocity)
            names.append("fused_susp_velocity")

            features.append(fused_states.body_velocity)
            names.append("fused_body_velocity")

            features.append(fused_states.estimated_road_profile)
            names.append("fused_road_elevation")

        # 5. Temporal Lag features (capturing dynamic hysteresis and valve delay)
        if self.include_lags:
            for lag in range(1, self.lag_steps + 1):
                # Lagged suspension velocity
                lag_vel = np.pad(base_sim.suspension_velocity[:-lag], (lag, 0), mode="edge")
                features.append(lag_vel)
                names.append(f"base_susp_vel_lag_{lag}")

                # Lagged body acceleration
                lag_accel = np.pad(base_sim.ddz_s[:-lag], (lag, 0), mode="edge")
                features.append(lag_accel)
                names.append(f"base_body_accel_lag_{lag}")

        X = np.column_stack(features)

        # Handle any possible NaNs safely
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return FeatureMatrix(
            X=X,
            feature_names=names,
            time=base_sim.time.copy(),
        )
