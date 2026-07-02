"""GPVL v2 — General Physics of Volition."""

from volition.constants import (
    DIM4_COOP_COLLAPSE,
    DIM4_IRREVERSIBLE,
    DIM4_NO_RETURN,
)
from volition.goalden import GoaldenThreshold, predict_equilibrium
from volition.hazard import hazard_of_collapse
from volition.regime import (
    Regime,
    RegimeTransitionMatrix,
    classify_regime,
)
from volition.state import (
    FirmVolitionalState,
    FullState,
    StateSummary,
    VolitionalState,
    VolitionalVector,
    build_summary,
    tfr_band,
)

__version__ = "2.0.0"

__all__ = [
    "DIM4_COOP_COLLAPSE",
    "DIM4_IRREVERSIBLE",
    "DIM4_NO_RETURN",
    "FirmVolitionalState",
    "FullState",
    "GoaldenThreshold",
    "Regime",
    "RegimeTransitionMatrix",
    "StateSummary",
    "VolitionalState",
    "VolitionalVector",
    "build_summary",
    "classify_regime",
    "hazard_of_collapse",
    "predict_equilibrium",
    "tfr_band",
]