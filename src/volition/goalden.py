"""Goalden universal plunder threshold (from volition/Goalden.py)."""

from dataclasses import dataclass
from enum import Enum, auto

from volition.constants import GOALDEN_THRESHOLD_COEFF


class EquilibriumPrediction(Enum):
    COOPERATIVE = auto()
    FRAGMENTATION = auto()


@dataclass(frozen=True)
class GoaldenThreshold:
    """Measurable quantities for the universal cooperation threshold."""

    rho_plunder: float  # fraction of non-consensual actions (0–1)
    c_mean: float  # average agent competence
    r_mean: float  # average agent reputation

    @property
    def threshold(self) -> float:
        """Universal threshold: 0.1 * C_mean * R_mean."""
        return GOALDEN_THRESHOLD_COEFF * self.c_mean * self.r_mean

    def predict(self) -> EquilibriumPrediction:
        if self.rho_plunder < self.threshold:
            return EquilibriumPrediction.COOPERATIVE
        return EquilibriumPrediction.FRAGMENTATION


def predict_equilibrium(
    rho_plunder: float,
    c_mean: float,
    r_mean: float,
) -> EquilibriumPrediction:
    """Convenience wrapper around GoaldenThreshold.predict()."""
    return GoaldenThreshold(rho_plunder, c_mean, r_mean).predict()


# Default backtest values from 31-domain validation
DEFAULT_GOALDEN = GoaldenThreshold(
    rho_plunder=0.0312,
    c_mean=1.28,
    r_mean=0.91,
)