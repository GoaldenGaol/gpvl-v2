"""Octonion algebra O with Cayley-Dickson / Fano-plane multiplication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_BASIS_COUNT = 8

# Fano plane lines (1-indexed imaginary units); cyclic order defines e_i e_j = e_k
# Standard Fano plane (Wikipedia / Baez convention)
_FANO_LINES: tuple[tuple[int, int, int], ...] = (
    (1, 2, 4),
    (1, 3, 7),
    (1, 5, 6),
    (2, 3, 6),
    (2, 5, 7),
    (3, 4, 5),
    (4, 6, 7),
)

# Build e_i * e_j -> sign * e_k table (i,j,k in 0..7, k=0 is scalar)
_MULT_SIGN: NDArray[np.int8] = np.zeros((8, 8), dtype=np.int8)
_MULT_INDEX: NDArray[np.int8] = np.zeros((8, 8), dtype=np.int8)


def _build_multiplication_table() -> None:
    for i in range(8):
        _MULT_INDEX[0, i] = i
        _MULT_SIGN[0, i] = 1
        _MULT_INDEX[i, 0] = i
        _MULT_SIGN[i, 0] = 1

    for i in range(1, 8):
        _MULT_INDEX[i, i] = 0
        _MULT_SIGN[i, i] = -1

    for a, b, c in _FANO_LINES:
        triples = [(a, b, c), (b, c, a), (c, a, b)]
        for x, y, z in triples:
            _MULT_INDEX[x, y] = z
            _MULT_SIGN[x, y] = 1
            _MULT_INDEX[y, x] = z
            _MULT_SIGN[y, x] = -1


_build_multiplication_table()


def _basis_product(i: int, j: int) -> NDArray[np.float64]:
    """Return e_i * e_j as 8-component coefficient vector."""
    out = np.zeros(8, dtype=np.float64)
    k = int(_MULT_INDEX[i, j])
    out[k] = float(_MULT_SIGN[i, j])
    return out


@dataclass(frozen=True)
class Octonion:
    """
    8-component octonion O = sum_{i=0}^{7} x_i * e_i.

    Multiplication is non-associative but alternative and normed.
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

    def __neg__(self) -> Octonion:
        return Octonion(-self.components)

    def __sub__(self, other: Octonion) -> Octonion:
        return self + (-other)

    def scalar_part(self) -> float:
        return float(self.components[0])

    def imaginary(self) -> NDArray[np.float64]:
        return self.components[1:].copy()

    def multiply(self, other: Octonion) -> Octonion:
        """Full octonion product (non-associative)."""
        result = np.zeros(8, dtype=np.float64)
        for i in range(8):
            if self.components[i] == 0:
                continue
            for j in range(8):
                if other.components[j] == 0:
                    continue
                result += self.components[i] * other.components[j] * _basis_product(i, j)
        return Octonion(result)

    def __mul__(self, other: Octonion) -> Octonion:
        return self.multiply(other)

    def inverse(self) -> Octonion:
        """Multiplicative inverse: O^{-1} = O* / |O|^2."""
        n2 = float(np.dot(self.components, self.components))
        if n2 < 1e-15:
            raise ZeroDivisionError("Cannot invert zero octonion")
        return Octonion(self.conjugate().components / n2)


def multiplication_table_latex() -> str:
    """Return LaTeX fragment for the Fano-plane multiplication rules."""
    lines = [r"\begin{align}"]
    for a, b, c in _FANO_LINES:
        lines.append(rf"e_{a} e_{b} &= e_{c} \\")
    lines.append(r"e_i^2 &= -1 \quad (i = 1,\ldots,7)")
    lines.append(r"\end{align}")
    return "\n".join(lines)