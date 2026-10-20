"""
Benchmark Execution and Plot Generator.

Executes the end-to-end digital twin pipeline across all 5 standard benchmark scenarios:
1. Training: Class B Road + Haversine Speed Bump at 45 km/h
2. Held-Out 1: Class C Local Road at 60 km/h
3. Held-Out 2: Severe Class D Track at 35 km/h
4. Held-Out 3: Washboard Corrugation (Resonant) at 50 km/h
5. Held-Out 4: Composite Bump + Pothole Sequence at 70 km/h

Saves comprehensive figures and prints exact benchmark telemetry.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import DigitalTwinPipeline
from src.road.generator import BENCHMARK_SCENARIOS


def main():
    print("=" * 80)
    print("PHYSICS-GUIDED SUSPENSION DIGITAL TWIN: BENCHMARK SUITE")
    print("=" * 80)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "plots")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = DigitalTwinPipeline(seed=42)

    # 1. Train residual model on training scenario
    print("\n[Step 1/3] Training physics-guided residual model on Scenario 0 (Class B + Speed Bump)...")
    train_scores = pipeline.train_on_scenario(BENCHMARK_SCENARIOS[0])
    print(f"  Training Accel Residual R^2: {train_scores['train_r2_accel_residual']:.4f}")
    print(f"  Training Travel Residual R^2: {train_scores['train_r2_travel_residual']:.4f}")

    # 2. Evaluate across all benchmark scenarios
    print("\n[Step 2/3] Evaluating generalization on held-out test scenarios...")
    results_df = pipeline.evaluate_benchmark_matrix()
    print("\n" + results_df.to_string(index=False))

    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics to {csv_path}")

    # 3. Generate high-resolution comparison plots
    print("\n[Step 3/3] Generating telemetry plots...")

    # Scenario 0 detailed time-series plot
    scen0 = BENCHMARK_SCENARIOS[0]
    res0 = pipeline.run_scenario(
        road_class=scen0.road_class,
        obstacle_type=scen0.obstacle,
        speed_kmh=scen0.vehicle_speed_kmh,
        duration_s=scen0.duration_s,
        scenario_name=scen0.name,
        seed=scen0.seed,
        include_semi_active=True,
    )

    t = res0.road_data.time
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Subplot 1: Road elevation profile and speed bump
    axs[0].plot(t, res0.road_data.z_r * 1000.0, "k-", lw=1.5, label="True Road Elevation $z_r$ (mm)")
    axs[0].plot(t, res0.fused_states.estimated_road_profile * 1000.0, "c--", lw=1.2, alpha=0.8, label="Kalman Reconstructed $\hat{z}_r$ (mm)")
    axs[0].set_ylabel("Elevation [mm]")
    axs[0].set_title(f"Telemetry: {scen0.name} (v = {scen0.vehicle_speed_kmh} km/h)", fontsize=13, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")

    # Subplot 2: Body Vertical Acceleration (Reference vs Baseline vs Hybrid)
    axs[1].plot(t, res0.ref_sim.ddz_s, color="#2ca02c", lw=1.8, label="Reference Ground Truth (Nonlinear Damping)")
    axs[1].plot(t, res0.base_sim.ddz_s, color="#d62728", ls="--", lw=1.2, alpha=0.85, label="Baseline Physics Model (-35% Damping Error)")
    axs[1].plot(t, res0.ddz_s_hybrid, color="#1f77b4", ls="-.", lw=1.4, label="Hybrid Digital Twin (Physics + ML Residual)")
    axs[1].set_ylabel("Body Accel $\ddot{z}_s$ [m/s²]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    # Subplot 3: Residual Prediction & Error Discrepancy
    axs[2].plot(t, res0.eval_accel.baseline_residuals, color="#d62728", lw=1.2, alpha=0.7, label=f"Baseline Error (RMSE: {res0.eval_accel.baseline_rmse:.3f} m/s²)")
    axs[2].plot(t, res0.eval_accel.hybrid_residuals, color="#1f77b4", lw=1.2, alpha=0.9, label=f"Hybrid Residual Error (RMSE: {res0.eval_accel.hybrid_rmse:.3f} m/s²)")
    axs[2].axhline(0.0, color="gray", lw=0.8, ls=":")
    axs[2].set_ylabel("Error Residual [m/s²]")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right")

    # Subplot 4: Relative Suspension Travel
    axs[3].plot(t, res0.ref_sim.suspension_travel * 1000.0, color="#2ca02c", lw=1.8, label="Reference Travel $z_s - z_u$")
    axs[3].plot(t, res0.base_sim.suspension_travel * 1000.0, color="#d62728", ls="--", lw=1.2, alpha=0.8, label="Baseline Model")
    axs[3].plot(t, res0.travel_hybrid * 1000.0, color="#1f77b4", ls="-.", lw=1.4, label="Hybrid Prediction")
    axs[3].set_ylabel("Travel [mm]")
    axs[3].set_xlabel("Time [s]")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "digital_twin_time_series.png")
    fig.savefig(plot1_path, dpi=200)
    plt.close(fig)
    print(f"  Saved time-series response plot: {plot1_path}")

    # Generalization Comparison Bar Chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    scenarios = [s.name.replace("Scenario: ", "").replace("Held-out ", "HO ") for s in BENCHMARK_SCENARIOS]
    x_pos = np.arange(len(scenarios))
    width = 0.35

    base_rmses = results_df["Baseline Accel RMSE"].values
    hyb_rmses = results_df["Hybrid Accel RMSE"].values

    rects1 = ax.bar(x_pos - width/2, base_rmses, width, label="Baseline Physics", color="#e74c3c", alpha=0.85)
    rects2 = ax.bar(x_pos + width/2, hyb_rmses, width, label="Hybrid Physics-ML Twin", color="#2980b9", alpha=0.85)

    ax.set_ylabel("Body Acceleration RMSE [m/s²]", fontsize=11, fontweight="bold")
    ax.set_title("Generalization Benchmark: Baseline Physics vs Hybrid Digital Twin", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")

    # Annotate percentage reductions above bars
    for i, r in enumerate(results_df["Accel RMSE Reduction (%)"]):
        ax.annotate(
            f"{r:+.1f}%",
            xy=(x_pos[i] + width/2, hyb_rmses[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1b4f72" if r > 0 else "#900c3f",
        )

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "generalization_benchmark.png")
    fig2.savefig(plot2_path, dpi=200)
    plt.close(fig2)
    print(f"  Saved generalization benchmark chart: {plot2_path}")

    print("\nBenchmark generation finished successfully!")


if __name__ == "__main__":
    main()
