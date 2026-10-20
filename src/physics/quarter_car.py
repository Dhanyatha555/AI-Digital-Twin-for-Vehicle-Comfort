"""
Two-Degree-of-Freedom (2-DOF) Quarter-Car Suspension Model.

Governing Equations of Motion:
    m_s * d2z_s/dt2 + F_damper(dz_s/dt - dz_u/dt) + k_s * (z_s - z_u) = 0
    m_u * d2z_u/dt2 - F_damper(dz_s/dt - dz_u/dt) - k_s * (z_s - z_u) + k_t * (z_u - z_r) + c_t * (dz_u/dt - dz_r/dt) = 0

Where:
    m_s : Sprung mass (chassis / quarter body) [kg]
    m_u : Unsprung mass (wheel, tire, hub, half-axle) [kg]
    k_s : Suspension spring stiffness [N/m]
    c_s : Nominal suspension damping coefficient [N*s/m]
    k_t : Tire vertical stiffness [N/m]
    c_t : Tire vertical damping coefficient [N*s/m]
    z_s : Vertical displacement of sprung mass [m]
    z_u : Vertical displacement of unsprung mass [m]
    z_r : Road surface vertical displacement [m]
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class QuarterCarParameters:
    """Quarter-car physical parameters."""
    m_s: float = 320.0       # Sprung mass [kg] (e.g. 1/4 of ~1280 kg sedan body)
    m_u: float = 45.0        # Unsprung mass [kg] (wheel, hub, brake, tire)
    k_s: float = 28000.0     # Suspension stiffness [N/m]
    c_s: float = 1450.0      # Suspension damping coefficient [N*s/m]
    k_t: float = 210000.0    # Tire stiffness [N/m]
    c_t: float = 50.0        # Tire damping coefficient [N*s/m]

    # Non-linear characteristics (used for reference / high-fidelity simulation)
    is_nonlinear: bool = False
    rebound_ratio: float = 1.8       # Rebound damping is typically stiffer than compression
    compression_ratio: float = 0.8   # Compression damping is softer to absorb bump impact
    v_blowoff: float = 0.45          # Blow-off velocity for damper valve saturation [m/s]
    saturation_factor: float = 0.35  # Damping slope reduction past blow-off velocity

    @property
    def f_body_bounce(self) -> float:
        """Undamped body bounce natural frequency in Hz."""
        return float(np.sqrt(self.k_s / self.m_s) / (2.0 * np.pi))

    @property
    def f_wheel_hop(self) -> float:
        """Undamped wheel hop natural frequency in Hz."""
        return float(np.sqrt((self.k_s + self.k_t) / self.m_u) / (2.0 * np.pi))

    @property
    def body_damping_ratio(self) -> float:
        """Nominal body damping ratio zeta."""
        return float(self.c_s / (2.0 * np.sqrt(self.k_s * self.m_s)))


@dataclass
class SimulationResult:
    """Container for quarter-car dynamic simulation results."""
    time: np.ndarray             # Time array [s]
    z_r: np.ndarray              # Road displacement [m]
    dz_r: np.ndarray             # Road velocity [m/s]
    z_s: np.ndarray              # Body displacement [m]
    dz_s: np.ndarray             # Body velocity [m/s]
    ddz_s: np.ndarray            # Body acceleration [m/s^2]
    z_u: np.ndarray              # Wheel displacement [m]
    dz_u: np.ndarray             # Wheel velocity [m/s]
    ddz_u: np.ndarray            # Wheel acceleration [m/s^2]
    suspension_travel: np.ndarray  # Relative displacement s = z_s - z_u [m]
    suspension_velocity: np.ndarray  # Relative velocity ds = dz_s - dz_u [m/s]
    tire_deflection: np.ndarray  # Dynamic tire deflection d = z_u - z_r [m]
    tire_force: np.ndarray       # Dynamic tire force [N]
    damping_force: np.ndarray    # Suspension damper force [N]
    spring_force: np.ndarray     # Suspension spring force [N]


class QuarterCarModel:
    """
    Simulates a 2-DOF quarter-car suspension system with linear or nonlinear damper.
    """

    def __init__(self, params: Optional[QuarterCarParameters] = None):
        self.params = params if params is not None else QuarterCarParameters()

    def damper_force(self, v_rel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate suspension damper force as a function of relative velocity v_rel = dz_s/dt - dz_u/dt.
        Positive v_rel corresponds to rebound (extension), negative to compression (bump).
        """
        p = self.params
        if not p.is_nonlinear:
            return p.c_s * v_rel

        # Nonlinear damper with asymmetric rebound/compression and blow-off valve saturation
        v = np.asarray(v_rel)
        v_abs = np.abs(v)
        c_eff = np.where(v >= 0, p.c_s * p.rebound_ratio, p.c_s * p.compression_ratio)

        # Velocity saturation using softplus-like hyperbolic tangent transition
        v_b = p.v_blowoff
        sat_term = np.where(
            v_abs <= v_b,
            v,
            np.sign(v) * (v_b + p.saturation_factor * (v_abs - v_b))
        )
        force = c_eff * sat_term
        return force.item() if np.ndim(v_rel) == 0 else force

    def state_derivatives(
        self,
        t: float,
        state: np.ndarray,
        road_func: Callable[[float], Tuple[float, float]],
    ) -> np.ndarray:
        """
        Calculate state derivatives [dz_s, d2z_s, dz_u, d2z_u] at time t.
        State vector:
            state[0] = z_s (body displacement)
            state[1] = dz_s (body velocity)
            state[2] = z_u (wheel displacement)
            state[3] = dz_u (wheel velocity)
        """
        z_s, dz_s, z_u, dz_u = state
        z_r, dz_r = road_func(t)

        p = self.params
        v_rel = dz_s - dz_u
        f_spring = p.k_s * (z_s - z_u)
        f_damper = self.damper_force(v_rel)
        f_susp = f_spring + f_damper

        f_tire_spring = p.k_t * (z_u - z_r)
        f_tire_damper = p.c_t * (dz_u - dz_r)
        f_tire = f_tire_spring + f_tire_damper

        # Newton's 2nd Law for sprung mass and unsprung mass
        ddz_s = -f_susp / p.m_s
        ddz_u = (f_susp - f_tire) / p.m_u

        return np.array([dz_s, ddz_s, dz_u, ddz_u], dtype=np.float64)

    def simulate(
        self,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        road_func: Callable[[float], Tuple[float, float]],
        initial_state: Optional[np.ndarray] = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> SimulationResult:
        """
        Simulate vehicle response over time evaluation grid.
        
        Args:
            t_span: (t_start, t_end) tuple
            t_eval: evaluation timestamps (equally spaced recommended)
            road_func: function returning (z_r(t), dz_r(t))
            initial_state: [z_s0, dz_s0, z_u0, dz_u0], defaults to zeros
            rtol, atol: solver tolerances
        """
        if initial_state is None:
            initial_state = np.zeros(4, dtype=np.float64)

        dt = float(t_eval[1] - t_eval[0]) if len(t_eval) > 1 else 0.002
        sol = solve_ivp(
            fun=lambda t, y: self.state_derivatives(t, y, road_func),
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=min(0.005, dt),
        )

        if not sol.success:
            raise RuntimeError(f"Simulation integration failed: {sol.message}")

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
        p = self.params
        spring_force = p.k_s * (z_s - z_u)
        damping_force = self.damper_force(v_rel)
        susp_force = spring_force + damping_force

        tire_spring = p.k_t * (z_u - z_r)
        tire_damper = p.c_t * (dz_u - dz_r)
        tire_force = tire_spring + tire_damper

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
