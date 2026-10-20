"""
Road Profile Generator.

Generates realistic road profiles conforming to:
1. ISO 8608 Road Roughness Classification (Classes A through D).
2. Discrete obstacles: Haversine speed bumps, potholes, cleats, washboard corrugations.
3. Spatially defined profiles converted to time-domain excitation via vehicle speed v.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple
import numpy as np


class RoadClass(str, Enum):
    CLASS_A = "ISO Class A (Smooth Highway)"
    CLASS_B = "ISO Class B (Good / Rural Road)"
    CLASS_C = "ISO Class C (Average / Urban Road)"
    CLASS_D = "ISO Class D (Rough / Deteriorated Road)"


class ObstacleType(str, Enum):
    NONE = "None"
    SPEED_BUMP = "Speed Bump (Haversine)"
    POTHOLE = "Pothole (Dip)"
    DOUBLE_BUMP = "Consecutive Bumps"
    WASHBOARD = "Washboard (Corrugated)"
    COMPOSITE = "Bump + Pothole Sequence"


@dataclass
class RoadScenario:
    """Pre-defined road scenario configuration for reproducible experiments."""
    name: str
    road_class: RoadClass
    obstacle: ObstacleType
    vehicle_speed_kmh: float
    duration_s: float
    seed: int
    description: str


@dataclass
class RoadProfileData:
    """Evaluated road profile in spatial and time domains."""
    time: np.ndarray       # Time vector [s]
    distance: np.ndarray   # Spatial distance x = v * t [m]
    z_r: np.ndarray        # Road surface elevation [m]
    dz_r: np.ndarray       # Road surface vertical velocity [m/s]
    speed_mps: float       # Vehicle speed [m/s]
    speed_kmh: float       # Vehicle speed [km/h]
    road_class: RoadClass
    obstacle_type: ObstacleType


class RoadProfileGenerator:
    """
    Generates synthetic, physically consistent road elevation profiles.
    
    ISO 8608 Power Spectral Density:
        G_d(n) = G_d(n_0) * (n / n_0)^(-w)
    where:
        n_0 = 0.1 cycles/m
        w = 2.0 (standard waviness)
    """

    # Geometric mean values of G_d(n_0) in 10^-6 m^3 according to ISO 8608
    ISO_GD_VALUES = {
        RoadClass.CLASS_A: 16e-6,    # Smooth highway
        RoadClass.CLASS_B: 64e-6,    # Good / Minor highway
        RoadClass.CLASS_C: 256e-6,   # Average local road
        RoadClass.CLASS_D: 1024e-6,  # Rough unpaved road
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_iso_profile(
        self,
        road_class: RoadClass,
        length_m: float,
        dx: float = 0.01,
        n_harmonics: int = 500,
        n_min: float = 0.01,
        n_max: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate ISO 8608 spatial road profile using harmonic summation.
        Returns:
            x: distance array [m]
            z_iso: road elevation [m]
            dz_dx_iso: spatial slope dz/dx [m/m]
        """
        x = np.arange(0.0, length_m + dx, dx)
        gd_0 = self.ISO_GD_VALUES[road_class]
        n_0 = 0.1  # cycles/m

        # Logarithmic spatial frequency spacing for efficient spectrum coverage
        n_i = np.logspace(np.log10(n_min), np.log10(n_max), n_harmonics)
        delta_n = np.diff(np.pad(n_i, (1, 0), mode="edge"))
        delta_n[0] = n_i[1] - n_i[0]

        # ISO 8608 PSD: G_d(n) = G_d(n_0) * (n/n_0)^(-2)
        psd = gd_0 * (n_i / n_0) ** (-2.0)
        amplitudes = np.sqrt(2.0 * psd * delta_n)

        # Random phases in [0, 2pi)
        phases = self.rng.uniform(0.0, 2.0 * np.pi, size=n_harmonics)

        # Compute elevation and spatial derivative
        # z(x) = sum( A_i * cos(2*pi*n_i * x + phi_i) )
        # dz/dx = sum( -A_i * 2*pi*n_i * sin(2*pi*n_i * x + phi_i) )
        z_iso = np.zeros_like(x)
        dz_dx_iso = np.zeros_like(x)

        for a, n, phi in zip(amplitudes, n_i, phases):
            arg = 2.0 * np.pi * n * x + phi
            z_iso += a * np.cos(arg)
            dz_dx_iso += -a * (2.0 * np.pi * n) * np.sin(arg)

        # Remove DC offset so road starts at zero elevation
        z_iso -= z_iso[0]

        return x, z_iso, dz_dx_iso

    def add_obstacle(
        self,
        x: np.ndarray,
        z: np.ndarray,
        dz_dx: np.ndarray,
        obstacle_type: ObstacleType,
        x_start: float = 15.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Superimpose discrete geometric obstacles onto spatial road profile."""
        z_obs = np.zeros_like(z)
        dz_obs = np.zeros_like(dz_dx)

        if obstacle_type == ObstacleType.NONE:
            return z, dz_dx

        elif obstacle_type == ObstacleType.SPEED_BUMP:
            # Haversine bump: Height 0.065 m (65 mm), length 2.5 m
            h_bump = 0.065
            l_bump = 2.5
            mask = (x >= x_start) & (x <= x_start + l_bump)
            x_local = x[mask] - x_start
            z_obs[mask] = 0.5 * h_bump * (1.0 - np.cos(2.0 * np.pi * x_local / l_bump))
            dz_obs[mask] = (
                0.5 * h_bump * (2.0 * np.pi / l_bump) * np.sin(2.0 * np.pi * x_local / l_bump)
            )

        elif obstacle_type == ObstacleType.POTHOLE:
            # Pothole dip: Depth -0.045 m (-45 mm), length 0.8 m
            depth = -0.045
            l_pot = 0.8
            mask = (x >= x_start) & (x <= x_start + l_pot)
            x_local = x[mask] - x_start
            z_obs[mask] = 0.5 * depth * (1.0 - np.cos(2.0 * np.pi * x_local / l_pot))
            dz_obs[mask] = (
                0.5 * depth * (2.0 * np.pi / l_pot) * np.sin(2.0 * np.pi * x_local / l_pot)
            )

        elif obstacle_type == ObstacleType.DOUBLE_BUMP:
            # Two bumps in close succession
            for offset in [0.0, 4.0]:
                xs = x_start + offset
                l_b = 2.0
                h_b = 0.055
                mask = (x >= xs) & (x <= xs + l_b)
                x_local = x[mask] - xs
                z_obs[mask] += 0.5 * h_b * (1.0 - np.cos(2.0 * np.pi * x_local / l_b))
                dz_obs[mask] += (
                    0.5 * h_b * (2.0 * np.pi / l_b) * np.sin(2.0 * np.pi * x_local / l_b)
                )

        elif obstacle_type == ObstacleType.WASHBOARD:
            # Periodic washboard corrugations (e.g. wavelength 0.8 m, amplitude 15 mm)
            l_wash = 25.0
            lambda_w = 0.8
            a_w = 0.015
            mask = (x >= x_start) & (x <= x_start + l_wash)
            x_local = x[mask] - x_start
            # Envelope to taper in and out
            envelope = np.sin(np.pi * x_local / l_wash) ** 2
            z_obs[mask] = a_w * np.sin(2.0 * np.pi * x_local / lambda_w) * envelope
            dz_obs[mask] = (
                a_w * (2.0 * np.pi / lambda_w) * np.cos(2.0 * np.pi * x_local / lambda_w) * envelope
            )

        elif obstacle_type == ObstacleType.COMPOSITE:
            # Speed bump at 15m, pothole at 30m
            # Speed bump
            h_b, l_b = 0.06, 2.0
            m1 = (x >= x_start) & (x <= x_start + l_b)
            xl1 = x[m1] - x_start
            z_obs[m1] += 0.5 * h_b * (1.0 - np.cos(2.0 * np.pi * xl1 / l_b))
            dz_obs[m1] += 0.5 * h_b * (2.0 * np.pi / l_b) * np.sin(2.0 * np.pi * xl1 / l_b)

            # Pothole
            xs2 = x_start + 18.0
            d_p, l_p = -0.045, 0.75
            m2 = (x >= xs2) & (x <= xs2 + l_p)
            xl2 = x[m2] - xs2
            z_obs[m2] += 0.5 * d_p * (1.0 - np.cos(2.0 * np.pi * xl2 / l_p))
            dz_obs[m2] += 0.5 * d_p * (2.0 * np.pi / l_p) * np.sin(2.0 * np.pi * xl2 / l_p)

        return z + z_obs, dz_dx + dz_obs

    def build_time_series(
        self,
        road_class: RoadClass = RoadClass.CLASS_B,
        obstacle_type: ObstacleType = ObstacleType.NONE,
        speed_kmh: float = 50.0,
        duration_s: float = 6.0,
        fs: float = 500.0,
        x_obstacle_start: float = 15.0,
    ) -> Tuple[RoadProfileData, Callable[[float], Tuple[float, float]]]:
        """
        Generate time-domain road excitation at constant forward speed v.
        Returns:
            data: RoadProfileData dataclass
            road_func: continuous callable road_func(t) -> (z_r, dz_r) for ODE solver
        """
        speed_mps = speed_kmh / 3.6
        total_distance = speed_mps * duration_s + 10.0  # Buffer for interpolation
        dx = speed_mps / fs  # Spatial resolution matching sampling rate

        # Generate spatial road
        x_grid, z_iso, dz_dx_iso = self.generate_iso_profile(
            road_class=road_class,
            length_m=total_distance,
            dx=dx,
        )

        # Add obstacle
        z_total, dz_dx_total = self.add_obstacle(
            x=x_grid,
            z=z_iso,
            dz_dx=dz_dx_iso,
            obstacle_type=obstacle_type,
            x_start=x_obstacle_start,
        )

        # Time grid
        dt = 1.0 / fs
        time_vec = np.arange(0.0, duration_s, dt)
        dist_vec = speed_mps * time_vec

        # Interpolate elevation and compute vertical velocity: dz_r/dt = (dz/dx) * (dx/dt) = (dz/dx) * v
        z_r_vec = np.interp(dist_vec, x_grid, z_total)
        dz_r_vec = np.interp(dist_vec, x_grid, dz_dx_total) * speed_mps

        # Build fast continuous callable function for solve_ivp
        from scipy.interpolate import interp1d
        z_interp = interp1d(time_vec, z_r_vec, kind="linear", fill_value="extrapolate")
        dz_interp = interp1d(time_vec, dz_r_vec, kind="linear", fill_value="extrapolate")

        def road_func(t: float) -> Tuple[float, float]:
            return float(z_interp(t)), float(dz_interp(t))

        profile_data = RoadProfileData(
            time=time_vec,
            distance=dist_vec,
            z_r=z_r_vec,
            dz_r=dz_r_vec,
            speed_mps=speed_mps,
            speed_kmh=speed_kmh,
            road_class=road_class,
            obstacle_type=obstacle_type,
        )

        return profile_data, road_func


# Standard benchmark scenarios for rigorous generalization evaluation
BENCHMARK_SCENARIOS = [
    RoadScenario(
        name="Training Scenario: Class B + Bump",
        road_class=RoadClass.CLASS_B,
        obstacle=ObstacleType.SPEED_BUMP,
        vehicle_speed_kmh=45.0,
        duration_s=5.0,
        seed=101,
        description="Training environment: standard suburban road with isolated speed bump at 45 km/h.",
    ),
    RoadScenario(
        name="Held-out 1: Class C Local Road",
        road_class=RoadClass.CLASS_C,
        obstacle=ObstacleType.NONE,
        vehicle_speed_kmh=60.0,
        duration_s=5.0,
        seed=202,
        description="Held-out evaluation: rougher urban road with no obstacle at 60 km/h.",
    ),
    RoadScenario(
        name="Held-out 2: Severe Class D Track",
        road_class=RoadClass.CLASS_D,
        obstacle=ObstacleType.NONE,
        vehicle_speed_kmh=35.0,
        duration_s=5.0,
        seed=303,
        description="Held-out evaluation: severe deteriorated road roughness at low speed 35 km/h.",
    ),
    RoadScenario(
        name="Held-out 3: Washboard Corrugation",
        road_class=RoadClass.CLASS_B,
        obstacle=ObstacleType.WASHBOARD,
        vehicle_speed_kmh=50.0,
        duration_s=5.0,
        seed=404,
        description="Held-out evaluation: periodic washboard corrugations exciting wheel hop resonance at 50 km/h.",
    ),
    RoadScenario(
        name="Held-out 4: Bump + Pothole Sequence",
        road_class=RoadClass.CLASS_C,
        obstacle=ObstacleType.COMPOSITE,
        vehicle_speed_kmh=70.0,
        duration_s=5.0,
        seed=505,
        description="Held-out evaluation: highway-speed bump followed by sharp pothole impact at 70 km/h.",
    ),
]
