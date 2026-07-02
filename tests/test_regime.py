"""Tests for regime classification."""

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_IRREVERSIBLE
from volition.regime import Regime, classify_regime


def test_cooperative_regime():
    assert classify_regime(0.5) == Regime.COOPERATIVE
    assert classify_regime(DIM4_COOP_COLLAPSE) == Regime.COOPERATIVE


def test_pre_collapse_regime():
    assert classify_regime(0.95) == Regime.PRE_COLLAPSE
    assert classify_regime(DIM4_IRREVERSIBLE) == Regime.PRE_COLLAPSE


def test_irreversible_regime():
    assert classify_regime(1.05) == Regime.IRREVERSIBLE
    assert classify_regime(1.41) == Regime.IRREVERSIBLE