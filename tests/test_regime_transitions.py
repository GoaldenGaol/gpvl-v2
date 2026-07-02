"""Tests for regime transition matrix Θ."""

import numpy as np

from volition.constants import DEFAULT_THETA
from volition.regime import (
    Regime,
    RegimeTransitionMatrix,
    classify_regime,
    regime_from_index,
    regime_index,
)


def test_regime_index_roundtrip():
    for regime in Regime:
        assert regime_from_index(regime_index(regime)) is regime


def test_default_theta_shape_and_stochastic():
    theta = RegimeTransitionMatrix.default()
    assert theta.theta.shape == (3, 3)
    assert np.allclose(theta.theta.sum(axis=1), 1.0)
    assert np.allclose(theta.theta, DEFAULT_THETA)


def test_irreversible_is_absorbing():
    theta = RegimeTransitionMatrix.default()
    assert theta.transition_probability(Regime.IRREVERSIBLE, Regime.IRREVERSIBLE) == 1.0
    assert theta.transition_probability(Regime.IRREVERSIBLE, Regime.COOPERATIVE) == 0.0


def test_simulate_chain_length():
    theta = RegimeTransitionMatrix.default()
    chain = theta.simulate(Regime.COOPERATIVE, n_steps=10, seed=42)
    assert len(chain) == 11
    assert chain[0] == Regime.COOPERATIVE


def test_simulate_absorbing_eventually():
    theta = RegimeTransitionMatrix.default()
    chain = theta.simulate(Regime.PRE_COLLAPSE, n_steps=200, seed=7)
    assert chain[-1] in (Regime.PRE_COLLAPSE, Regime.IRREVERSIBLE)


def test_stationary_distribution():
    theta = RegimeTransitionMatrix.default()
    pi = theta.stationary_distribution()
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi, pi @ theta.theta, atol=1e-8)


def test_latex_export():
    theta = RegimeTransitionMatrix.default()
    latex = theta.as_latex()
    assert r"\begin{bmatrix}" in latex
    assert "0.85" in latex


def test_classify_regime_boundaries():
    assert classify_regime(0.91) == Regime.COOPERATIVE
    assert classify_regime(0.93) == Regime.PRE_COLLAPSE
    assert classify_regime(1.01) == Regime.IRREVERSIBLE