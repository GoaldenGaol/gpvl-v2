"""Octonion algebra stub for 8D volitional geometry (v2.0 interface)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Cayley-Dickson multiplication table indices (e0=1, e1..e7 imaginary units)
_BASIS_COUNT = 8


@dataclass(frozen=True)
class Octonion:
    """
    8-component octonion O = sum_{i=0}^{7} x_i * e_i.

    Non-associative; multiplication follows standard Cayley-Dickson construction.
    Full calibration to dim1–dim7 deferred to v2.1.
    """

    components: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.components.shape != (_BASIS_COUNT,):
            raise ValueError(f"Octonion requires 8 components, got {self.components.shape}")

    @classmethod
    def zero(cls) -> Octonion:
        return cls(np.zeros(_BASIS_COUNT, dtype=np.float64))

    @classmethod
    def unit(cls, index: int) -> Octonion:
        if not 0 <= index < _BASIS_COUNT:
            raise IndexError(f"Basis index must be in [0, 7], got {index}")
        c = np.zeros(_BASIS_COUNT, dtype=np.float64)
        c[index] = 1.0
        return cls(c)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.components))

    def conjugate(self) -> Octonion:
        c = self.components.copy()
        c[1:] *= -1
        return Octonion(c)

    def __add__(self, other: Octonion) -> Octonion:
        return Octonion(self.components + other.components)

    def scalar_part(self) -> float:
        return float(self.components[0])