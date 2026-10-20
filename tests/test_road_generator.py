"""
Unit tests for ISO 8608 Road Profile Generator and Obstacle synthesis.
"""

import numpy as np
import pytest
from src.road.generator import (
    RoadClass,
    ObstacleType,
    RoadProfileGenerator,
)


def test_iso_road_roughness_scaling():
    """Verify elevation variance increases monotonically from ISO Class A to D."""
    gen = RoadProfileGenerator(seed=123)
    variances = []

    for rc in [RoadClass.CLASS_A, RoadClass.CLASS_B, RoadClass.CLASS_C, RoadClass.CLASS_D]:
        x, z, _ = gen.generate_iso_profile(road_class=rc, length_m=200.0)
        variances.append(float(np.var(z)))

    # Each subsequent road class should have higher elevation variance
    assert variances[0] < variances[1] < variances[2] < variances[3]


def test_speed_bump_geometry():
    """Verify haversine speed bump creates correct height and length."""
    gen = RoadProfileGenerator(seed=42)
    x = np.linspace(0.0, 50.0, 5000)
    z = np.zeros_like(x)
    dz = np.zeros_like(x)

    z_obs, _ = gen.add_obstacle(x, z, dz, obstacle_type=ObstacleType.SPEED_BUMP, x_start=20.0)

    # Elevation should peak near 0.065 m (65 mm)
    peak = np.max(z_obs)
    assert 0.060 < peak < 0.070
    # Before and after bump should be zero
    assert z_obs[0] == 0.0
    assert z_obs[-1] == 0.0


def test_pothole_geometry():
    """Verify pothole obstacle creates negative elevation dip."""
    gen = RoadProfileGenerator(seed=42)
    x = np.linspace(0.0, 50.0, 5000)
    z = np.zeros_like(x)
    dz = np.zeros_like(x)

    z_obs, _ = gen.add_obstacle(x, z, dz, obstacle_type=ObstacleType.POTHOLE, x_start=20.0)
    dip = np.min(z_obs)
    assert -0.050 < dip < -0.040


def test_deterministic_reproducibility():
    """Verify that same seed generates identical road profiles."""
    gen1 = RoadProfileGenerator(seed=999)
    p1, _ = gen1.build_time_series(speed_kmh=50.0, duration_s=2.0)

    gen2 = RoadProfileGenerator(seed=999)
    p2, _ = gen2.build_time_series(speed_kmh=50.0, duration_s=2.0)

    np.testing.assert_array_equal(p1.z_r, p2.z_r)
    np.testing.assert_array_equal(p1.dz_r, p2.dz_r)
