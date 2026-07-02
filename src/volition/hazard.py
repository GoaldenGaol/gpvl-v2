"""Collapse hazard functions h(dim4)."""

import math

from volition.constants import DIM4_COOP_COLLAPSE, VPDE_LOGISTIC_STEEPNESS


def hazard_of_collapse(dim4: float, *, steepness: float = VPDE_LOGISTIC_STEEPNESS) -> float:
    """
    Smooth, monotone hazard function h(dim4) in (0, 1).

    Centered logistic around the cooperation-collapse threshold (+0.92).
    Replace with calibrated curve once MCMC posterior is available (Phase 5).
    """
    return 1.0 / (1.0 + math.exp(-steepness * (dim4 - DIM4_COOP_COLLAPSE)))