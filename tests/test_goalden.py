"""Tests for Goalden universal threshold."""

from volition.goalden import DEFAULT_GOALDEN, EquilibriumPrediction, GoaldenThreshold


def test_default_threshold_value():
    assert abs(DEFAULT_GOALDEN.threshold - 0.11648) < 1e-4


def test_cooperative_prediction():
    gt = GoaldenThreshold(rho_plunder=0.05, c_mean=1.28, r_mean=0.91)
    assert gt.predict() == EquilibriumPrediction.COOPERATIVE


def test_fragmentation_prediction():
    gt = GoaldenThreshold(rho_plunder=0.20, c_mean=1.28, r_mean=0.91)
    assert gt.predict() == EquilibriumPrediction.FRAGMENTATION


def test_default_backtest_cooperative():
    assert DEFAULT_GOALDEN.predict() == EquilibriumPrediction.COOPERATIVE