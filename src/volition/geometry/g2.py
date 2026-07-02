"""G2 structure group stub for octonion automorphisms (v2.0 interface)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volition.geometry.octonion import Octonion
from volition.state import VolitionalVector


@dataclass
class G2Projection:
    """
    Projection pi: O -> R^7 mapping octonion imaginary units to dim1–dim7.

    v2.0 provides identity-structured projection; G2 calibration in v2.1.
    """

    @staticmethod
    def to_volitional_vector(oct: Octonion) -> VolitionalVector:
        """Map imaginary octonion components (e1..e7) to V(t)."""
        c = oct.components
        return VolitionalVector(
            dim1=float(c[1]),
            dim2=float(c[2]),
            dim3=float(c[3]),
            dim4=float(c[4]),
            dim5=float(c[5]),
            dim6=float(c[6]),
            dim7=float(c[7]),
        )

    @staticmethod
    def from_volitional_vector(vec: VolitionalVector) -> Octonion:
        """Embed V(t) into octonion imaginary subspace (e0 = 0)."""
        arr = vec.as_array
        components = np.zeros(8, dtype=np.float64)
        components[1:] = arr
        return Octonion(components)