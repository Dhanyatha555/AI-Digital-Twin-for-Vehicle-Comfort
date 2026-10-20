"""
Unit tests for Machine Learning Residual Twin Model.
"""

import numpy as np
import pytest
from src.physics.quarter_car import QuarterCarModel, QuarterCarParameters
from src.road.generator import RoadClass, ObstacleType, RoadProfileGenerator
from src.ml.features import FeatureExtractor
from src.ml.residual_model import SuspensionResidualTwin, ModelType


def test_feature_extractor_dimensions():
    """Verify feature extractor extracts expected shape and columns."""
    params = QuarterCarParameters()
    model = QuarterCarModel(params)
    t_eval = np.linspace(0.0, 1.0, 500)
    t_span = (0.0, 1.0)
    sim = model.simulate(t_span, t_eval, lambda t: (0.0, 0.0))

    extractor = FeatureExtractor(include_lags=True, lag_steps=2)
    feats = extractor.extract(sim, vehicle_speed_kmh=60.0)

    assert feats.X.shape[0] == 500
    assert feats.X.shape[1] == len(feats.feature_names)
    assert "base_body_accel" in feats.feature_names
    assert "rebound_direction_sign" in feats.feature_names
    assert not np.any(np.isnan(feats.X))


def test_residual_twin_fit_and_error_reduction():
    """Verify that hybrid twin fits and achieves meaningful residual reduction on training scenario."""
    # Reference model with nonlinear damping
    ref_params = QuarterCarParameters(
        c_s=1450.0,
        is_nonlinear=True,
        rebound_ratio=1.8,
        compression_ratio=0.8,
    )
    # Baseline model with parameter mismatch (950 Ns/m linear)
    base_params = QuarterCarParameters(c_s=950.0, is_nonlinear=False)

    ref_model = QuarterCarModel(ref_params)
    base_model = QuarterCarModel(base_params)

    # Road bump scenario
    gen = RoadProfileGenerator(seed=42)
    road_data, road_func = gen.build_time_series(
        road_class=RoadClass.CLASS_B,
        obstacle_type=ObstacleType.SPEED_BUMP,
        speed_kmh=45.0,
        duration_s=3.0,
        fs=500.0,
    )

    t_eval = road_data.time
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    ref_sim = ref_model.simulate(t_span, t_eval, road_func)
    base_sim = base_model.simulate(t_span, t_eval, road_func)

    # Baseline model must show notable discrepancy vs reference
    base_rmse = np.sqrt(np.mean((ref_sim.ddz_s - base_sim.ddz_s) ** 2))
    assert base_rmse > 0.05

    # Fit residual model
    twin = SuspensionResidualTwin(model_type=ModelType.HIST_GBM, random_state=42)
    train_scores = twin.fit(ref_sim, base_sim, vehicle_speed_kmh=45.0)

    assert train_scores["train_r2_accel_residual"] > 0.60

    # Evaluate on this scenario
    eval_accel, eval_travel = twin.evaluate(
        ref_sim, base_sim, scenario_name="Train Bump", vehicle_speed_kmh=45.0
    )

    # Hybrid RMSE must be strictly lower than baseline RMSE
    assert eval_accel.hybrid_rmse < eval_accel.baseline_rmse
    # Residual error reduction should be at least 40% on training scenario
    assert eval_accel.rmse_reduction_pct > 40.0
