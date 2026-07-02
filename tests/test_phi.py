"""Tests for numeric Φ transition map evaluation."""

import numpy as np

from volition.equations.phi import PhiParameters, evaluate_phi, phi_dim4_equation


def test_phi_dim4_equation_is_equality():
    eq = phi_dim4_equation()
    assert eq.lhs is not None
    assert eq.rhs is not None


def test_evaluate_phi_shape():
    state = np.array([0.1, 0.2, 0.3, 0.85, 0.1, 0.1, 0.1])
    nxt = evaluate_phi(state)
    assert nxt.shape == (7,)
    assert np.all(np.isfinite(nxt))


def test_plunder_increases_dim4():
    state = np.zeros(7)
    state[3] = 0.80
    baseline = evaluate_phi(state, rho_plunder=0.0)
    plunder = evaluate_phi(state, rho_plunder=0.15)
    assert plunder[3] > baseline[3]


def test_intervention_shifts_dim4():
    state = np.zeros(7)
    state[3] = 0.90
    neg = evaluate_phi(state, intervention=-0.05)
    pos = evaluate_phi(state, intervention=0.05)
    assert pos[3] > neg[3]


def test_custom_parameters():
    params = PhiParameters.default()
    state = np.ones(7) * 0.5
    result = evaluate_phi(state, params=params)
    assert result.shape == (7,)