"""
Physics-Guided Residual Learning Digital Twin.

Architecture:
    y_hybrid(t) = y_baseline_physics(t) + f_ML(features(t))

Where f_ML learns the discrepancy (residual error) caused by parameter mismatch,
unmodeled damping asymmetry, and damper valve saturation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureExtractor, FeatureMatrix
from ..physics.quarter_car import SimulationResult
from ..sensors.kalman_filter import EstimatedStates


class ModelType(str, Enum):
    HIST_GBM = "HistGradientBoosting"
    RANDOM_FOREST = "RandomForest"
    RIDGE = "Ridge (Linear Baseline)"


@dataclass
class ResidualEvaluation:
    """Metrics comparing baseline physics model and hybrid physics-ML twin against ground truth reference."""
    scenario_name: str
    target_name: str                     # e.g., 'body_accel' or 'suspension_travel'
    baseline_rmse: float
    hybrid_rmse: float
    baseline_mae: float
    hybrid_mae: float
    r2_residual: float                   # How well ML predicts the residual
    rmse_reduction_pct: float            # Positive indicates improvement, negative indicates degradation
    y_reference: np.ndarray
    y_baseline: np.ndarray
    y_hybrid: np.ndarray
    baseline_residuals: np.ndarray
    hybrid_residuals: np.ndarray
    time: np.ndarray


class SuspensionResidualTwin:
    """
    Lightweight ML residual learner trained to correct physical model discrepancies.
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.HIST_GBM,
        feature_extractor: Optional[FeatureExtractor] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.feature_extractor = feature_extractor if feature_extractor is not None else FeatureExtractor()
        self.random_state = random_state

        # Separate regression pipelines for body acceleration and suspension travel residuals
        self.model_accel: Optional[Pipeline] = None
        self.model_travel: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def _create_regressor(self):
        """Instantiate scikit-learn regressor pipeline with strictly fitted StandardScaler."""
        if self.model_type == ModelType.HIST_GBM:
            reg = HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=6,
                learning_rate=0.08,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=self.random_state,
            )
        elif self.model_type == ModelType.RANDOM_FOREST:
            reg = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                n_jobs=-1,
                random_state=self.random_state,
            )
        elif self.model_type == ModelType.RIDGE:
            reg = Ridge(alpha=10.0, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", reg),
        ])

    def fit(
        self,
        ref_sim: SimulationResult,
        base_sim: SimulationResult,
        fused_states: Optional[EstimatedStates] = None,
        vehicle_speed_kmh: float = 45.0,
    ) -> Dict[str, float]:
        """
        Train residual regressors on training simulation data.
        Residuals:
            e_accel = ref.ddz_s - base.ddz_s
            e_travel = ref.suspension_travel - base.suspension_travel
        """
        feats = self.feature_extractor.extract(base_sim, fused_states, vehicle_speed_kmh)
        self.feature_names = feats.feature_names
        X = feats.X

        # Target residuals to predict
        e_accel = ref_sim.ddz_s - base_sim.ddz_s
        e_travel = ref_sim.suspension_travel - base_sim.suspension_travel

        self.model_accel = self._create_regressor()
        self.model_travel = self._create_regressor()

        self.model_accel.fit(X, e_accel)
        self.model_travel.fit(X, e_travel)
        self.is_fitted = True

        r2_accel = float(self.model_accel.score(X, e_accel))
        r2_travel = float(self.model_travel.score(X, e_travel))

        return {
            "train_r2_accel_residual": r2_accel,
            "train_r2_travel_residual": r2_travel,
        }

    def predict(
        self,
        base_sim: SimulationResult,
        fused_states: Optional[EstimatedStates] = None,
        vehicle_speed_kmh: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict hybrid response for a given baseline simulation.
        Returns:
            ddz_s_hybrid: Corrected body acceleration [m/s^2]
            travel_hybrid: Corrected suspension travel [m]
            pred_e_accel: Predicted acceleration residual [m/s^2]
            pred_e_travel: Predicted travel residual [m]
        """
        if not self.is_fitted or self.model_accel is None or self.model_travel is None:
            raise RuntimeError("SuspensionResidualTwin must be fitted before predict() can be called.")

        feats = self.feature_extractor.extract(base_sim, fused_states, vehicle_speed_kmh)
        pred_e_accel = self.model_accel.predict(feats.X)
        pred_e_travel = self.model_travel.predict(feats.X)

        ddz_s_hybrid = base_sim.ddz_s + pred_e_accel
        travel_hybrid = base_sim.suspension_travel + pred_e_travel

        return ddz_s_hybrid, travel_hybrid, pred_e_accel, pred_e_travel

    def evaluate(
        self,
        ref_sim: SimulationResult,
        base_sim: SimulationResult,
        scenario_name: str,
        fused_states: Optional[EstimatedStates] = None,
        vehicle_speed_kmh: float = 50.0,
    ) -> Tuple[ResidualEvaluation, ResidualEvaluation]:
        """
        Evaluate physics-only vs hybrid predictions against reference simulation.
        Returns evaluation dataclasses for body acceleration and suspension travel.
        """
        ddz_s_hybrid, travel_hybrid, pred_e_accel, pred_e_travel = self.predict(
            base_sim, fused_states, vehicle_speed_kmh
        )

        time = ref_sim.time

        # --- Body Acceleration Evaluation ---
        y_ref_a = ref_sim.ddz_s
        y_base_a = base_sim.ddz_s
        y_hyb_a = ddz_s_hybrid

        base_res_a = y_ref_a - y_base_a
        hyb_res_a = y_ref_a - y_hyb_a

        base_rmse_a = float(np.sqrt(np.mean(base_res_a ** 2)))
        hyb_rmse_a = float(np.sqrt(np.mean(hyb_res_a ** 2)))
        base_mae_a = float(np.mean(np.abs(base_res_a)))
        hyb_mae_a = float(np.mean(np.abs(hyb_res_a)))

        # R^2 of residual prediction
        ss_res_a = np.sum(hyb_res_a ** 2)
        ss_tot_a = np.sum((base_res_a - np.mean(base_res_a)) ** 2)
        r2_res_a = float(1.0 - (ss_res_a / ss_tot_a)) if ss_tot_a > 1e-12 else 0.0

        improvement_pct_a = (
            float(100.0 * (base_rmse_a - hyb_rmse_a) / base_rmse_a)
            if base_rmse_a > 1e-6
            else 0.0
        )

        eval_accel = ResidualEvaluation(
            scenario_name=scenario_name,
            target_name="body_acceleration",
            baseline_rmse=base_rmse_a,
            hybrid_rmse=hyb_rmse_a,
            baseline_mae=base_mae_a,
            hybrid_mae=hyb_mae_a,
            r2_residual=r2_res_a,
            rmse_reduction_pct=improvement_pct_a,
            y_reference=y_ref_a,
            y_baseline=y_base_a,
            y_hybrid=y_hyb_a,
            baseline_residuals=base_res_a,
            hybrid_residuals=hyb_res_a,
            time=time,
        )

        # --- Suspension Travel Evaluation ---
        y_ref_t = ref_sim.suspension_travel
        y_base_t = base_sim.suspension_travel
        y_hyb_t = travel_hybrid

        base_res_t = y_ref_t - y_base_t
        hyb_res_t = y_ref_t - y_hyb_t

        base_rmse_t = float(np.sqrt(np.mean(base_res_t ** 2)))
        hyb_rmse_t = float(np.sqrt(np.mean(hyb_res_t ** 2)))
        base_mae_t = float(np.mean(np.abs(base_res_t)))
        hyb_mae_t = float(np.mean(np.abs(hyb_res_t)))

        ss_res_t = np.sum(hyb_res_t ** 2)
        ss_tot_t = np.sum((base_res_t - np.mean(base_res_t)) ** 2)
        r2_res_t = float(1.0 - (ss_res_t / ss_tot_t)) if ss_tot_t > 1e-12 else 0.0

        improvement_pct_t = (
            float(100.0 * (base_rmse_t - hyb_rmse_t) / base_rmse_t)
            if base_rmse_t > 1e-6
            else 0.0
        )

        eval_travel = ResidualEvaluation(
            scenario_name=scenario_name,
            target_name="suspension_travel",
            baseline_rmse=base_rmse_t,
            hybrid_rmse=hyb_rmse_t,
            baseline_mae=base_mae_t,
            hybrid_mae=hyb_mae_t,
            r2_residual=r2_res_t,
            rmse_reduction_pct=improvement_pct_t,
            y_reference=y_ref_t,
            y_baseline=y_base_t,
            y_hybrid=y_hyb_t,
            baseline_residuals=base_res_t,
            hybrid_residuals=hyb_res_t,
            time=time,
        )

        return eval_accel, eval_travel
