"""Tests for Phi-coupled VPDE solver."""

import numpy as np
import pytest

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_IRREVERSIBLE
from volition.vpde import (
    ELEVATED_PLUNDER_CONTROLS,
    GOALDEN_CONTROLS,
    RampControls,
    VPDEConfig,
    calibrate_tau,
    crossing_time,
    default_calibrated_config,
    solve_vpde,
    solve_vpde_vector,
)
from volition.vpde.calibration import USA_TARGET


def test_vpde_returns_trajectory():
    t, dim4 = solve_vpde(0.85)
    assert len(t) == 200
    assert len(dim4) == 200
    assert np.all(np.isfinite(dim4))


def test_plunder_accelerates_dim4_growth():
    cfg = VPDEConfig(tau=2.0, t_span=(0.0, 20.0), n_eval=200)
    _, low = solve_vpde(0.85, cfg, controls=GOALDEN_CONTROLS)
    _, high = solve_vpde(0.85, cfg, controls=ELEVATED_PLUNDER_CONTROLS)
    assert high[-1] > low[-1]


def test_ramp_controls_time_varying():
    ramp = RampControls()
    early = ramp.at(0.0)
    late = ramp.at(6.0)
    assert late.rho_plunder > early.rho_plunder
    assert late.institutional_drift > early.institutional_drift


def test_solve_vpde_vector_shape():
    state = np.zeros(7)
    state[3] = 0.90
    cfg = VPDEConfig(tau=2.0, dim=7)
    t, traj = solve_vpde_vector(state, cfg)
    assert traj.shape == (7, len(t))
    assert np.all(np.isfinite(traj))


def test_crossing_time_above_threshold():
    cfg = VPDEConfig(
        tau=0.5,
        t_span=(0.0, 30.0),
        n_eval=300,
    )
    t_cross = crossing_time(
        0.94,
        DIM4_IRREVERSIBLE,
        cfg,
        controls=RampControls(),
    )
    assert t_cross is not None
    assert t_cross > 0


def test_calibrate_tau_usa_target():
    result = calibrate_tau(USA_TARGET)
    assert 0.1 < result.tau < 30.0
    assert result.achieved_crossing_time is not None
    assert abs(result.residual) < 0.5
    assert result.achieved_crossing_time == pytest.approx(6.0, abs=0.5)


def test_default_calibrated_config():
    cfg = default_calibrated_config()
    assert cfg.tau > 0
    t, traj = solve_vpde(0.94, cfg, controls=RampControls())
    assert traj[-1] >= DIM4_COOP_COLLAPSE or traj[-1] > 0.94