"""GPVL v2 — General Physics of Volition."""

from volition.constants import (
    DIM4_COOP_COLLAPSE,
    DIM4_IRREVERSIBLE,
    DIM4_NO_RETURN,
)
from volition.goalden import GoaldenThreshold, predict_equilibrium
from volition.hazard import hazard_of_collapse
from volition.regime import Regime, classify_regime
from volition.state import VolitionalState, VolitionalVector

__version__ = "2.0.0"

__all__ = [
    "DIM4_COOP_COLLAPSE",
    "DIM4_IRREVERSIBLE",
    "DIM4_NO_RETURN",
    "GoaldenThreshold",
    "Regime",
    "VolitionalState",
    "VolitionalVector",
    "classify_regime",
    "hazard_of_collapse",
    "predict_equilibrium",
]