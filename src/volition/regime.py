"""Regime classification and Markov transition dynamics (matrix Θ)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_IRREVERSIBLE, DEFAULT_THETA


class Regime(Enum):
    """Coarse regime classification based on dim4."""

    COOPERATIVE = auto()
    PRE_COLLAPSE = auto()
    IRREVERSIBLE = auto()


REGIME_ORDER: tuple[Regime, ...] = (
    Regime.COOPERATIVE,
    Regime.PRE_COLLAPSE,
    Regime.IRREVERSIBLE,
)


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


def regime_index(regime: Regime) -> int:
    """Map regime to row/column index in Θ."""
    return REGIME_ORDER.index(regime)


def regime_from_index(index: int) -> Regime:
    """Map matrix index back to Regime."""
    return REGIME_ORDER[index]


@dataclass(frozen=True)
class RegimeTransitionMatrix:
    """
    Markov transition matrix Θ with Θ_ij = P(R_{t+1} = j | R_t = i).

    Rows sum to 1. IRREVERSIBLE is absorbing under default calibration.
    """

    theta: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=np.float64))
        if self.theta.shape != (3, 3):
            raise ValueError(f"Θ must be 3×3, got {self.theta.shape}")
        row_sums = self.theta.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError(f"Θ rows must sum to 1, got {row_sums}")

    @classmethod
    def default(cls) -> RegimeTransitionMatrix:
        """Framework-default Θ with monotonic drift toward fragmentation."""
        return cls(np.array(DEFAULT_THETA, dtype=np.float64))

    def transition_probability(self, from_regime: Regime, to_regime: Regime) -> float:
        i = regime_index(from_regime)
        j = regime_index(to_regime)
        return float(self.theta[i, j])

    def step(self, regime: Regime, rng: np.random.Generator | None = None) -> Regime:
        """Sample next regime given current regime."""
        gen = rng or np.random.default_rng()
        i = regime_index(regime)
        j = int(gen.choice(len(REGIME_ORDER), p=self.theta[i]))
        return regime_from_index(j)

    def simulate(
        self,
        initial: Regime,
        n_steps: int,
        *,
        seed: int | None = None,
    ) -> list[Regime]:
        """Simulate a regime chain of length n_steps + 1 (includes initial)."""
        gen = np.random.default_rng(seed)
        chain = [initial]
        current = initial
        for _ in range(n_steps):
            current = self.step(current, gen)
            chain.append(current)
        return chain

    def stationary_distribution(self, tol: float = 1e-12, max_iter: int = 10_000) -> NDArray[np.float64]:
        """
        Compute stationary distribution π via power iteration.

        π satisfies π = π Θ (row vector convention).
        """
        pi = np.ones(3, dtype=np.float64) / 3.0
        for _ in range(max_iter):
            pi_next = pi @ self.theta
            if np.linalg.norm(pi_next - pi, ord=1) < tol:
                return pi_next
            pi = pi_next
        return pi

    def as_latex(self) -> str:
        """Return Θ as a LaTeX bmatrix for export."""
        rows = []
        for i in range(3):
            row = " & ".join(f"{self.theta[i, j]:.2f}" for j in range(3))
            rows.append(row + r" \\")
        header = r"\text{COOP} & \text{PRE} & \text{IRREV} \\"
        return (
            r"\begin{bmatrix}" + "\n"
            + header + "\n"
            + r"\midrule" + "\n"
            + "\n".join(rows) + "\n"
            + r"\end{bmatrix}"
        )