"""
Unit tests for Quarter-Car 2-DOF Physics Model.
"""

import numpy as np
import pytest
from src.physics.quarter_car import QuarterCarModel, QuarterCarParameters, SimulationResult


def test_quarter_car_eigenfrequencies():
    """Verify analytical undamped eigenfrequencies match passenger car targets."""
    params = QuarterCarParameters(
        m_s=320.0,
        m_u=45.0,
        k_s=28000.0,
        c_s=1450.0,
        k_t=210000.0,
        c_t=50.0,
    )
    # Body bounce target: ~1.4 - 1.6 Hz
    assert 1.3 < params.f_body_bounce < 1.7
    # Wheel hop target: ~10.5 - 12.5 Hz
    assert 10.5 < params.f_wheel_hop < 12.5
    # Damping ratio should be around 0.2 - 0.3 for passenger comfort
    assert 0.20 < params.body_damping_ratio < 0.35


def test_quarter_car_nonlinear_damper_asymmetry():
    """Verify nonlinear damper produces higher force in rebound than compression."""
    params = QuarterCarParameters(
        c_s=1000.0,
        is_nonlinear=True,
        rebound_ratio=1.8,
        compression_ratio=0.8,
        v_blowoff=0.5,
    )
    model = QuarterCarModel(params)

    # Rebound: positive relative velocity
    f_rebound = model.damper_force(0.2)
    # Compression: negative relative velocity
    f_compression = model.damper_force(-0.2)

    assert f_rebound > 0
    assert f_compression < 0
    assert abs(f_rebound) > abs(f_compression)
    assert pytest.approx(abs(f_rebound) / abs(f_compression), rel=1e-2) == (1.8 / 0.8)


def test_quarter_car_simulation_newton_balance():
    """Verify Newton's 2nd law dynamic force balance on simulated outputs."""
    params = QuarterCarParameters(m_s=320.0, m_u=45.0, k_s=28000.0, c_s=1400.0, k_t=210000.0)
    model = QuarterCarModel(params)

    t_eval = np.linspace(0.0, 2.0, 1000)
    t_span = (0.0, 2.0)

    # Road step bump excitation
    def road_step(t: float):
        if 0.5 <= t <= 0.8:
            return 0.05, 0.0
        return 0.0, 0.0

    res = model.simulate(t_span, t_eval, road_step)

    # Check output array lengths
    assert len(res.time) == 1000
    assert len(res.ddz_s) == 1000

    # Verify Newton's Second Law for sprung mass: m_s * ddz_s + F_susp == 0
    f_susp_calc = res.spring_force + res.damping_force
    net_body_force = params.m_s * res.ddz_s + f_susp_calc
    np.testing.assert_allclose(net_body_force, 0.0, atol=1e-3)

    # Verify Newton's Second Law for unsprung mass: m_u * ddz_u - F_susp + F_tire == 0
    net_wheel_force = params.m_u * res.ddz_u - f_susp_calc + res.tire_force
    np.testing.assert_allclose(net_wheel_force, 0.0, atol=1e-3)


def test_free_vibration_decay():
    """Verify free vibration converges to zero (asymptotic stability)."""
    params = QuarterCarParameters(c_s=1500.0)
    model = QuarterCarModel(params)

    t_eval = np.linspace(0.0, 4.0, 2000)
    t_span = (0.0, 4.0)

    # Zero road input
    def flat_road(t: float):
        return 0.0, 0.0

    # Initial displacement: 30 mm initial body displacement
    initial_state = np.array([0.03, 0.0, 0.0, 0.0])
    res = model.simulate(t_span, t_eval, flat_road, initial_state=initial_state)

    # Body displacement should damp out towards zero (< 0.5 mm at t=4s)
    assert abs(res.z_s[-1]) < 0.0005
    assert abs(res.dz_s[-1]) < 0.001
