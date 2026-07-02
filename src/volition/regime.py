"""Regime classification and transition logic based on dim4 thresholds."""

from enum import Enum, auto

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_IRREVERSIBLE


class Regime(Enum):
    """Coarse regime classification based on dim4."""

    COOPERATIVE = auto()
    PRE_COLLAPSE = auto()
    IRREVERSIBLE = auto()


def classify_regime(dim4: float) -> Regime:
    """
    Classify a state into a regime based on dim4 thresholds.

    - dim4 > 1.00 -> IRREVERSIBLE
    - dim4 > 0.92 -> PRE_COLLAPSE
    - else        -> COOPERATIVE
    """
    if dim4 > DIM4_IRREVERSIBLE:
        return Regime.IRREVERSIBLE
    if dim4 > DIM4_COOP_COLLAPSE:
        return Regime.PRE_COLLAPSE
    return Regime.COOPERATIVE