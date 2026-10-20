"""
Semi-Active Suspension and Damping Mode Comparison.

Supports:
1. Predefined passive damping modes: Comfort (Soft), Normal (Balanced), Sport (Firm).
2. Continuous / 2-State Skyhook Semi-Active Controller.
3. Multi-objective trade-off evaluation and recommendation engine.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from .quarter_car import QuarterCarModel, QuarterCarParameters, SimulationResult


class DampingMode(str, Enum):
    COMFORT = "Comfort (Soft)"
    NORMAL = "Normal (Balanced)"
    SPORT = "Sport (Firm)"
    SKYHOOK = "Semi-Active (Skyhook)"


@dataclass
class DampingModeEvaluation:
    """Evaluation summary for a specific damping setting."""
    mode_name: str
    damping_value: float          # Nominal c_s [N*s/m]
    rms_body_accel: float         # Raw RMS acceleration [m/s^2]
    weighted_rms_accel: float     # ISO 2631 weighted RMS acceleration [m/s^2]
    max_suspension_travel: float  # Peak suspension travel [m]
    rms_suspension_travel: float  # RMS suspension travel [m]
    dynamic_tire_load_ratio: float  # RMS dynamic tire load / static load
    comfort_score: float          # Normalized 0-100 comfort score (100 is best)
    handling_score: float         # Normalized 0-100 road holding score (100 is best)
    composite_score: float        # Weighted overall score based on preference
    is_recommended: bool = False
    rationale: str = ""


class SemiActiveSuspension:
    """
    Evaluator and comparator for semi-active and selectable damping modes.
    """

    PREDEFINED_SETTINGS = {
        DampingMode.COMFORT: 850.0,
        DampingMode.NORMAL: 1450.0,
        DampingMode.SPORT: 2400.0,
    }

    def __init__(self, base_params: Optional[QuarterCarParameters] = None):
        self.base_params = base_params if base_params is not None else QuarterCarParameters()

    def simulate_mode(
        self,
        mode: DampingMode,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        road_func: Callable[[float], Tuple[float, float]],
    ) -> SimulationResult:
        """Simulate a quarter-car under a specific damping mode."""
        if mode in self.PREDEFINED_SETTINGS:
            params = QuarterCarParameters(
                m_s=self.base_params.m_s,
                m_u=self.base_params.m_u,
                k_s=self.base_params.k_s,
                c_s=self.PREDEFINED_SETTINGS[mode],
                k_t=self.base_params.k_t,
                c_t=self.base_params.c_t,
                is_nonlinear=False,
            )
            model = QuarterCarModel(params)
            return model.simulate(t_span, t_eval, road_func)

        if mode == DampingMode.SKYHOOK:
            return self._simulate_skyhook(t_span, t_eval, road_func)

        raise ValueError(f"Unknown damping mode: {mode}")

    def _simulate_skyhook(
        self,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        road_func: Callable[[float], Tuple[float, float]],
        c_sky: float = 2200.0,
        c_min: float = 650.0,
    ) -> SimulationResult:
        """
        Simulate 2-state Skyhook semi-active damper:
            F_damper = c_sky * dz_s        if dz_s * (dz_s - dz_u) >= 0
            F_damper = c_min * (dz_s - dz_u) if dz_s * (dz_s - dz_u) < 0
        """
        p = self.base_params

        def skyhook_derivs(t: float, state: np.ndarray) -> np.ndarray:
            z_s, dz_s, z_u, dz_u = state
            z_r, dz_r = road_func(t)
            v_rel = dz_s - dz_u

            # Skyhook logic
            if dz_s * v_rel >= 0.0:
                f_damper = c_sky * dz_s
            else:
                f_damper = c_min * v_rel

            f_spring = p.k_s * (z_s - z_u)
            f_susp = f_spring + f_damper

            f_tire = p.k_t * (z_u - z_r) + p.c_t * (dz_u - dz_r)

            ddz_s = -f_susp / p.m_s
            ddz_u = (f_susp - f_tire) / p.m_u
            return np.array([dz_s, ddz_s, dz_u, ddz_u], dtype=np.float64)

        dt = float(t_eval[1] - t_eval[0]) if len(t_eval) > 1 else 0.002
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            fun=skyhook_derivs,
            t_span=t_span,
            y0=np.zeros(4, dtype=np.float64),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
            max_step=min(0.005, dt),
        )

        z_s = sol.y[0]
        dz_s = sol.y[1]
        z_u = sol.y[2]
        dz_u = sol.y[3]

        n_pts = len(t_eval)
        z_r = np.zeros(n_pts, dtype=np.float64)
        dz_r = np.zeros(n_pts, dtype=np.float64)
        for i, t in enumerate(t_eval):
            zr, dzr = road_func(float(t))
            z_r[i] = zr
            dz_r[i] = dzr

        v_rel = dz_s - dz_u
        damping_force = np.where(dz_s * v_rel >= 0.0, c_sky * dz_s, c_min * v_rel)
        spring_force = p.k_s * (z_s - z_u)
        susp_force = spring_force + damping_force
        tire_force = p.k_t * (z_u - z_r) + p.c_t * (dz_u - dz_r)

        ddz_s = -susp_force / p.m_s
        ddz_u = (susp_force - tire_force) / p.m_u

        return SimulationResult(
            time=t_eval,
            z_r=z_r,
            dz_r=dz_r,
            z_s=z_s,
            dz_s=dz_s,
            ddz_s=ddz_s,
            z_u=z_u,
            dz_u=dz_u,
            ddz_u=ddz_u,
            suspension_travel=z_s - z_u,
            suspension_velocity=v_rel,
            tire_deflection=z_u - z_r,
            tire_force=tire_force,
            damping_force=damping_force,
            spring_force=spring_force,
        )

    def evaluate_all_modes(
        self,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        road_func: Callable[[float], Tuple[float, float]],
        weight_comfort: float = 0.6,
        weight_handling: float = 0.4,
        weighted_accel_calculator: Optional[Callable[[np.ndarray, float], float]] = None,
    ) -> Tuple[Dict[str, SimulationResult], List[DampingModeEvaluation]]:
        """
        Simulate and evaluate all damping modes, selecting the optimal recommendation.
        
        Args:
            weight_comfort: Importance weight for ride comfort [0 to 1]
            weight_handling: Importance weight for road holding [0 to 1]
            weighted_accel_calculator: Function(accel_array, fs) -> weighted_rms
        """
        results: Dict[str, SimulationResult] = {}
        evaluations: List[DampingModeEvaluation] = []
        static_wheel_load = (self.base_params.m_s + self.base_params.m_u) * 9.81
        dt = float(t_eval[1] - t_eval[0])
        fs = 1.0 / dt

        modes = [
            DampingMode.COMFORT,
            DampingMode.NORMAL,
            DampingMode.SPORT,
            DampingMode.SKYHOOK,
        ]

        for mode in modes:
            sim = self.simulate_mode(mode, t_span, t_eval, road_func)
            results[mode.value] = sim

            raw_rms = float(np.sqrt(np.mean(sim.ddz_s ** 2)))
            if weighted_accel_calculator is not None:
                w_rms = weighted_accel_calculator(sim.ddz_s, fs)
            else:
                w_rms = raw_rms

            max_travel = float(np.max(np.abs(sim.suspension_travel)))
            rms_travel = float(np.sqrt(np.mean(sim.suspension_travel ** 2)))
            rms_tire_force = float(np.sqrt(np.mean(sim.tire_force ** 2)))
            tire_load_ratio = float(rms_tire_force / static_wheel_load)

            d_val = (
                self.PREDEFINED_SETTINGS.get(mode, 1800.0)
                if mode in self.PREDEFINED_SETTINGS
                else 1800.0
            )

            evaluations.append(
                DampingModeEvaluation(
                    mode_name=mode.value,
                    damping_value=d_val,
                    rms_body_accel=raw_rms,
                    weighted_rms_accel=w_rms,
                    max_suspension_travel=max_travel,
                    rms_suspension_travel=rms_travel,
                    dynamic_tire_load_ratio=tire_load_ratio,
                    comfort_score=0.0,
                    handling_score=0.0,
                    composite_score=0.0,
                )
            )

        # Normalize metrics to 0-100 scores across modes
        accels = np.array([e.weighted_rms_accel for e in evaluations])
        tire_ratios = np.array([e.dynamic_tire_load_ratio for e in evaluations])

        min_a, max_a = np.min(accels), np.max(accels)
        min_t, max_t = np.min(tire_ratios), np.max(tire_ratios)

        # Invert so lower acceleration and lower tire force yield higher scores
        denom_a = (max_a - min_a) if (max_a - min_a) > 1e-6 else 1.0
        denom_t = (max_t - min_t) if (max_t - min_t) > 1e-6 else 1.0

        for e in evaluations:
            e.comfort_score = float(100.0 * (1.0 - (e.weighted_rms_accel - min_a) / denom_a))
            e.handling_score = float(100.0 * (1.0 - (e.dynamic_tire_load_ratio - min_t) / denom_t))
            e.composite_score = float(weight_comfort * e.comfort_score + weight_handling * e.handling_score)

        # Select best evaluated option
        best_idx = int(np.argmax([e.composite_score for e in evaluations]))
        for i, e in enumerate(evaluations):
            if i == best_idx:
                e.is_recommended = True
                e.rationale = (
                    f"Selected as optimal compromise: achieved lowest discomfort "
                    f"({e.weighted_rms_accel:.3f} m/s^2) while maintaining safe dynamic tire load "
                    f"({e.dynamic_tire_load_ratio:.3f}) and suspension travel ({e.max_suspension_travel*1000:.1f} mm)."
                )
            else:
                e.is_recommended = False
                e.rationale = (
                    f"Secondary option (composite score {e.composite_score:.1f}/100)."
                )

        return results, evaluations
