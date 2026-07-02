"""Volitional state representations: V(t) and country-level snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class VolitionalVector:
    """
    7-dimensional volitional state vector V(t) = [dim1..dim7].

    Extracted from narrative-data embeddings (frozen Nov 2023).
    dim4 is the active latent scalar for fertility and cooperation prediction.
    """

    dim1: float = 0.0
    dim2: float = 0.0
    dim3: float = 0.0
    dim4: float = 0.0
    dim5: float = 0.0
    dim6: float = 0.0
    dim7: float = 0.0

    @property
    def as_array(self) -> NDArray[np.float64]:
        return np.array(
            [self.dim1, self.dim2, self.dim3, self.dim4,
             self.dim5, self.dim6, self.dim7],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> VolitionalVector:
        if arr.shape != (7,):
            raise ValueError(f"Expected 7-D vector, got shape {arr.shape}")
        return cls(*arr.tolist())

    def __repr__(self) -> str:
        vals = ", ".join(f"{v:.3f}" for v in self.as_array)
        return f"VolitionalVector([{vals}])"


@dataclass
class VolitionalState:
    """Country/year volitional state with optional TFR."""

    name: str
    year: int
    dim4: float
    tfr: float | None = None
    vector: VolitionalVector | None = field(default=None, repr=False)

    def summarize(self) -> str:
        from volition.hazard import hazard_of_collapse
        from volition.regime import classify_regime

        regime = classify_regime(self.dim4)
        hazard = hazard_of_collapse(self.dim4)
        tfr_part = f", TFR={self.tfr:.2f}" if self.tfr is not None else ""
        return (
            f"{self.name} ({self.year}) — "
            f"dim4={self.dim4:.2f}{tfr_part}, "
            f"regime={regime.name}, "
            f"hazard_of_collapse={hazard:.3f}"
        )