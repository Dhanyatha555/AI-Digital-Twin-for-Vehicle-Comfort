"""
Comprehensive performance metrics for suspension digital twins and ride comfort.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .iso2631 import ISO2631Filter, RideComfortEvaluation


@dataclass
class SuspensionMetrics:
    """Comprehensive performance scorecard for a model evaluation."""
    model_name: str
    scenario_name: str
    body_accel_rmse: float              # RMSE vs reference [m/s^2]
    body_accel_mae: float               # MAE vs reference [m/s^2]
    suspension_travel_rmse_mm: float    # Travel RMSE vs reference [mm]
    suspension_travel_mae_mm: float     # Travel MAE vs reference [mm]
    raw_rms_accel: float                # Raw RMS acceleration [m/s^2]
    iso_weighted_rms_accel: float       # ISO 2631 weighted RMS [m/s^2]
    vibration_dose_value: float         # VDV [m/s^1.75]
    crest_factor: float                 # Peak / RMS
    dynamic_tire_load_ratio: float      # RMS dynamic tire force / static weight
    max_suspension_travel_mm: float     # Maximum stroke utilization [mm]
    r2_accel: float                     # R^2 score against reference


def compute_metrics(
    model_name: str,
    scenario_name: str,
    pred_accel: np.ndarray,
    ref_accel: np.ndarray,
    pred_travel: np.ndarray,
    ref_travel: np.ndarray,
    dynamic_tire_force: np.ndarray,
    total_mass_kg: float,
    time: np.ndarray,
    iso_filter: Optional[ISO2631Filter] = None,
) -> SuspensionMetrics:
    """Compute complete telemetry metrics for a prediction against reference."""
    # Acceleration errors
    accel_err = ref_accel - pred_accel
    accel_rmse = float(np.sqrt(np.mean(accel_err ** 2)))
    accel_mae = float(np.mean(np.abs(accel_err)))

    ss_tot_a = np.sum((ref_accel - np.mean(ref_accel)) ** 2)
    ss_res_a = np.sum(accel_err ** 2)
    r2_a = float(1.0 - (ss_res_a / ss_tot_a)) if ss_tot_a > 1e-12 else 0.0

    # Travel errors
    travel_err = ref_travel - pred_travel
    travel_rmse_mm = float(np.sqrt(np.mean(travel_err ** 2)) * 1000.0)
    travel_mae_mm = float(np.mean(np.abs(travel_err)) * 1000.0)
    max_travel_mm = float(np.max(np.abs(pred_travel)) * 1000.0)

    # Raw RMS
    raw_rms = float(np.sqrt(np.mean(pred_accel ** 2)))

    # ISO 2631-1 weighted comfort
    dt = float(time[1] - time[0])
    fs = 1.0 / dt
    if iso_filter is None:
        iso_filter = ISO2631Filter(fs=fs)

    comfort = iso_filter.evaluate_comfort(pred_accel, time=time)

    # Road holding
    static_load = total_mass_kg * 9.81
    rms_tire_force = float(np.sqrt(np.mean(dynamic_tire_force ** 2)))
    tire_load_ratio = float(rms_tire_force / static_load)

    return SuspensionMetrics(
        model_name=model_name,
        scenario_name=scenario_name,
        body_accel_rmse=accel_rmse,
        body_accel_mae=accel_mae,
        suspension_travel_rmse_mm=travel_rmse_mm,
        suspension_travel_mae_mm=travel_mae_mm,
        raw_rms_accel=raw_rms,
        iso_weighted_rms_accel=comfort.weighted_rms_accel,
        vibration_dose_value=comfort.vibration_dose_value,
        crest_factor=comfort.crest_factor,
        dynamic_tire_load_ratio=tire_load_ratio,
        max_suspension_travel_mm=max_travel_mm,
        r2_accel=r2_a,
    )


def build_comparison_dataframe(
    baseline_metrics: SuspensionMetrics,
    hybrid_metrics: SuspensionMetrics,
) -> pd.DataFrame:
    """Build a side-by-side comparison table showing error reductions and improvements."""
    accel_improvement = (
        100.0 * (baseline_metrics.body_accel_rmse - hybrid_metrics.body_accel_rmse)
        / baseline_metrics.body_accel_rmse
        if baseline_metrics.body_accel_rmse > 1e-6 else 0.0
    )

    travel_improvement = (
        100.0 * (baseline_metrics.suspension_travel_rmse_mm - hybrid_metrics.suspension_travel_rmse_mm)
        / baseline_metrics.suspension_travel_rmse_mm
        if baseline_metrics.suspension_travel_rmse_mm > 1e-6 else 0.0
    )

    data = [
        {
            "Metric": "Body Accel RMSE (vs Ref)",
            "Unit": "m/s^2",
            "Baseline Physics": f"{baseline_metrics.body_accel_rmse:.4f}",
            "Hybrid Twin (Physics+ML)": f"{hybrid_metrics.body_accel_rmse:.4f}",
            "Improvement (%)": f"{accel_improvement:+.1f}%",
        },
        {
            "Metric": "Body Accel MAE (vs Ref)",
            "Unit": "m/s^2",
            "Baseline Physics": f"{baseline_metrics.body_accel_mae:.4f}",
            "Hybrid Twin (Physics+ML)": f"{hybrid_metrics.body_accel_mae:.4f}",
            "Improvement (%)": f"{100.0*(baseline_metrics.body_accel_mae - hybrid_metrics.body_accel_mae)/baseline_metrics.body_accel_mae:+.1f}%",
        },
        {
            "Metric": "Suspension Travel RMSE",
            "Unit": "mm",
            "Baseline Physics": f"{baseline_metrics.suspension_travel_rmse_mm:.2f}",
            "Hybrid Twin (Physics+ML)": f"{hybrid_metrics.suspension_travel_rmse_mm:.2f}",
            "Improvement (%)": f"{travel_improvement:+.1f}%",
        },
        {
            "Metric": "ISO 2631 Weighted RMS Accel (a_w)",
            "Unit": "m/s^2",
            "Baseline Physics": f"{baseline_metrics.iso_weighted_rms_accel:.4f}",
            "Hybrid Twin (Physics+ML)": f"{hybrid_metrics.iso_weighted_rms_accel:.4f}",
            "Improvement (%)": f"{100.0*(baseline_metrics.iso_weighted_rms_accel - hybrid_metrics.iso_weighted_rms_accel)/baseline_metrics.iso_weighted_rms_accel:+.1f}%",
        },
        {
            "Metric": "Vibration Dose Value (VDV)",
            "Unit": "m/s^1.75",
            "Baseline Physics": f"{baseline_metrics.vibration_dose_value:.3f}",
            "Hybrid Twin (Physics+ML)": f"{hybrid_metrics.vibration_dose_value:.3f}",
            "Improvement (%)": f"{100.0*(baseline_metrics.vibration_dose_value - hybrid_metrics.vibration_dose_value)/baseline_metrics.vibration_dose_value:+.1f}%",
        },
        {
            "Metric": "Body Accel R^2",
            "Unit": "-",
            "Baseline Physics": f"{baseline_metrics.r2_accel:.3f}",
            "Hybrid Twin (Physics+ML)": f"{hybrid_metrics.r2_accel:.3f}",
            "Improvement (%)": "N/A",
        },
    ]

    return pd.DataFrame(data)
