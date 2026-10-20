"""
Kalman Filter Sensor Fusion for Vehicle Suspension State and Road Estimation.

State Vector:
    x = [s, dz_s, dz_u, d_t]^T
where:
    s    = z_s - z_u  (Suspension relative travel [m])
    dz_s = d(z_s)/dt  (Body vertical velocity [m/s])
    dz_u = d(z_u)/dt  (Wheel vertical velocity [m/s])
    d_t  = z_u - z_r  (Dynamic tire deflection [m])

Measurements:
    y = [ddz_s, ddz_u, s]^T
where:
    ddz_s = Measured body vertical acceleration [m/s^2]
    ddz_u = Measured wheel hub vertical acceleration [m/s^2]
    s     = Measured suspension travel [m]
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from ..physics.quarter_car import QuarterCarParameters


@dataclass
class KalmanFilterConfig:
    """Covariance parameters for suspension state estimation."""
    # Process noise covariance diagonals
    q_travel: float = 1e-6       # m^2
    q_vel_body: float = 0.05     # (m/s)^2
    q_vel_wheel: float = 0.40    # (m/s)^2
    q_tire_defl: float = 0.08    # m^2 (road roughness excitation)

    # Measurement noise covariance diagonals
    r_accel_body: float = 0.02   # (m/s^2)^2
    r_accel_wheel: float = 0.35  # (m/s^2)^2
    r_travel: float = 1e-6       # m^2 (LVDT sensor)


@dataclass
class EstimatedStates:
    """Estimated internal suspension and road states from Kalman Filter."""
    time: np.ndarray
    suspension_travel: np.ndarray        # Filtered travel s = z_s - z_u [m]
    suspension_velocity: np.ndarray      # Relative velocity ds = dz_s - dz_u [m/s]
    body_velocity: np.ndarray            # Estimated body velocity dz_s [m/s]
    wheel_velocity: np.ndarray           # Estimated wheel velocity dz_u [m/s]
    tire_deflection: np.ndarray          # Estimated tire deflection z_u - z_r [m]
    estimated_road_profile: np.ndarray   # Reconstructed road profile z_r [m]
    innovations: np.ndarray              # Measurement residual errors (innovation)
    estimation_variance: np.ndarray      # Diagonal of state covariance P over time


class SuspensionKalmanFilter:
    """
    Continuous-Discrete Extended/Linear Kalman Filter fusing body & wheel accelerations
    with suspension travel to estimate unmeasured velocities and road profile.
    """

    def __init__(
        self,
        params: Optional[QuarterCarParameters] = None,
        config: Optional[KalmanFilterConfig] = None,
    ):
        self.params = params if params is not None else QuarterCarParameters()
        self.config = config if config is not None else KalmanFilterConfig()

    def build_system_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build discretized continuous-time state-space matrices F and H.
        Continuous system: dx/dt = A x + G w
        Discretization via matrix exponential or first-order Euler approximation: F approx I + A*dt
        """
        p = self.params
        m_s, m_u = p.m_s, p.m_u
        k_s, c_s = p.k_s, p.c_s
        k_t = p.k_t

        # Continuous A matrix (4x4)
        # x = [s, dz_s, dz_u, d_t]
        A = np.array([
            [0.0,            1.0,           -1.0,           0.0],
            [-k_s / m_s,     -c_s / m_s,    c_s / m_s,      0.0],
            [k_s / m_u,      c_s / m_u,     -c_s / m_u,     -k_t / m_u],
            [0.0,            0.0,           1.0,            0.0],
        ], dtype=np.float64)

        # Discretize state transition: F = expm(A*dt)
        from scipy.linalg import expm
        F = expm(A * dt)

        # Measurement matrix H (3x4): y = [ddz_s, ddz_u, s]
        H = np.array([
            [-k_s / m_s, -c_s / m_s, c_s / m_s,  0.0],
            [k_s / m_u,  c_s / m_u,  -c_s / m_u, -k_t / m_u],
            [1.0,        0.0,        0.0,        0.0],
        ], dtype=np.float64)

        # Process noise covariance Q (4x4)
        c = self.config
        Q = np.diag([c.q_travel, c.q_vel_body, c.q_vel_wheel, c.q_tire_defl]) * dt

        return F, H, Q

    def estimate(
        self,
        time: np.ndarray,
        body_accel: np.ndarray,
        wheel_accel: np.ndarray,
        susp_travel: np.ndarray,
    ) -> EstimatedStates:
        """
        Run Kalman filter over time series with support for measurement dropouts (NaNs).
        """
        n = len(time)
        dt = float(time[1] - time[0])
        F, H, Q = self.build_system_matrices(dt)

        c = self.config
        R_full = np.diag([c.r_accel_body, c.r_accel_wheel, c.r_travel])

        # Initial state and covariance
        x_hat = np.zeros(4, dtype=np.float64)
        if not np.isnan(susp_travel[0]):
            x_hat[0] = susp_travel[0]
        P = np.diag([1e-4, 0.01, 0.05, 1e-4])

        # Logging arrays
        x_history = np.zeros((n, 4), dtype=np.float64)
        p_history = np.zeros((n, 4), dtype=np.float64)
        innovations = np.zeros((n, 3), dtype=np.float64)

        I = np.eye(4, dtype=np.float64)

        for k in range(n):
            # --- 1. Time Update (Predict) ---
            x_pred = F @ x_hat
            P_pred = F @ P @ F.T + Q

            # --- 2. Measurement Update (Correct) ---
            # Current raw measurements
            y_meas = np.array([body_accel[k], wheel_accel[k], susp_travel[k]], dtype=np.float64)
            valid_sensors = ~np.isnan(y_meas)

            if not np.any(valid_sensors):
                # Total dropout on this time step: rely purely on model prediction
                x_hat = x_pred
                P = P_pred
            else:
                # Subselect active sensors for this timestamp
                H_k = H[valid_sensors, :]
                R_k = R_full[np.ix_(valid_sensors, valid_sensors)]
                y_k = y_meas[valid_sensors]

                y_hat = H_k @ x_pred
                innov = y_k - y_hat
                innovations[k, valid_sensors] = innov

                S_k = H_k @ P_pred @ H_k.T + R_k
                K_k = P_pred @ H_k.T @ np.linalg.inv(S_k)

                x_hat = x_pred + K_k @ innov
                # Joseph form for numerical covariance symmetry and stability
                # P = (I - K H) P (I - K H)^T + K R K^T
                IKH = I - K_k @ H_k
                P = IKH @ P_pred @ IKH.T + K_k @ R_k @ K_k.T

            x_history[k, :] = x_hat
            p_history[k, :] = np.diag(P)

        s_hat = x_history[:, 0]
        dz_s_hat = x_history[:, 1]
        dz_u_hat = x_history[:, 2]
        d_t_hat = x_history[:, 3]
        v_rel_hat = dz_s_hat - dz_u_hat

        # Reconstruct road profile:
        # Numerically integrate wheel velocity to get z_u, then z_r = z_u - d_t
        from scipy.integrate import cumulative_trapezoid
        z_u_reconstructed = cumulative_trapezoid(dz_u_hat, time, initial=0.0)
        # Detrend any slight low-frequency drift in reconstructed wheel position
        from scipy.signal import detrend
        z_u_clean = detrend(z_u_reconstructed, type="linear")
        z_r_hat = z_u_clean - d_t_hat

        return EstimatedStates(
            time=time,
            suspension_travel=s_hat,
            suspension_velocity=v_rel_hat,
            body_velocity=dz_s_hat,
            wheel_velocity=dz_u_hat,
            tire_deflection=d_t_hat,
            estimated_road_profile=z_r_hat,
            innovations=innovations,
            estimation_variance=p_history,
        )
