"""G2 structure: octonion automorphisms and calibrated 8D→7D projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volition.geometry.octonion import Octonion
from volition.state import VolitionalVector

# G2 Lie algebra dimension
G2_DIM = 14
IMAGINARY_DIM = 7


@dataclass(frozen=True)
class G2Calibration:
    """
    Calibrated projection parameters from frozen dim4 data.

    Maps dim4 stress scalar into e4 (primary volitional axis) with
    per-dimension scales derived from the empirical dim4 distribution.
    """

    dim4_mean: float
    dim4_std: float
    dim4_axis: int = 4  # octonion imaginary index for dim4 (e4)
    scales: NDArray[np.float64] = field(default_factory=lambda: np.ones(7))

    def normalize_dim4(self, dim4: float) -> float:
        if self.dim4_std < 1e-12:
            return dim4 - self.dim4_mean
        return (dim4 - self.dim4_mean) / self.dim4_std

    @classmethod
    def from_dim4_series(cls, dim4_values: NDArray[np.float64]) -> G2Calibration:
        """Calibrate from an array of dim4 observations."""
        mean = float(np.mean(dim4_values))
        std = float(np.std(dim4_values))
        if std < 1e-12:
            std = 1.0
        # Scale auxiliary dims relative to dim4 spread (for future 7D latent)
        scales = np.full(7, std, dtype=np.float64)
        scales[3] = std  # dim4 → e4
        return cls(dim4_mean=mean, dim4_std=std, scales=scales)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, column: str = "dim4") -> G2Calibration:
        return cls.from_dim4_series(df[column].to_numpy(dtype=np.float64))


@dataclass
class G2Projection:
    """
    Projection π: 𝕆 → ℝ⁷ with optional G2 calibration.

    Preserves dim4 as the primary stress axis (e4) under calibration.
    """

    calibration: G2Calibration | None = None

    @classmethod
    def default(cls) -> G2Projection:
        return cls()

    @classmethod
    def calibrated(cls, df: pd.DataFrame | None = None) -> G2Projection:
        from volition.data.dim4 import load_countries_full

        data = df if df is not None else load_countries_full()
        return cls(calibration=G2Calibration.from_dataframe(data))

    def to_volitional_vector(self, oct: Octonion) -> VolitionalVector:
        """Map imaginary octonion components (e1..e7) to V(t)."""
        c = oct.components
        cal = self.calibration
        if cal is None:
            return VolitionalVector(
                dim1=float(c[1]), dim2=float(c[2]), dim3=float(c[3]),
                dim4=float(c[4]), dim5=float(c[5]), dim6=float(c[6]), dim7=float(c[7]),
            )
        s = cal.scales
        return VolitionalVector(
            dim1=float(c[1]) * s[0],
            dim2=float(c[2]) * s[1],
            dim3=float(c[3]) * s[2],
            dim4=float(c[4]) * s[3] + cal.dim4_mean,
            dim5=float(c[5]) * s[4],
            dim6=float(c[6]) * s[5],
            dim7=float(c[7]) * s[6],
        )

    def from_volitional_vector(self, vec: VolitionalVector) -> Octonion:
        """Embed V(t) into octonion imaginary subspace (e0 = 0)."""
        arr = vec.as_array
        cal = self.calibration
        components = np.zeros(8, dtype=np.float64)
        if cal is None:
            components[1:] = arr
        else:
            s = cal.scales
            components[1] = arr[0] / s[0]
            components[2] = arr[1] / s[1]
            components[3] = arr[2] / s[2]
            components[4] = (arr[3] - cal.dim4_mean) / s[3]
            components[5] = arr[4] / s[4]
            components[6] = arr[5] / s[5]
            components[7] = arr[6] / s[6]
        return Octonion(components)

    def embed_dim4(self, dim4: float) -> Octonion:
        """Embed scalar dim4 stress into calibrated octonion (e4 axis)."""
        components = np.zeros(8, dtype=np.float64)
        if self.calibration is None:
            components[4] = dim4
        else:
            components[self.calibration.dim4_axis] = self.calibration.normalize_dim4(dim4)
        return Octonion(components)

    def extract_dim4(self, oct: Octonion) -> float:
        """Extract dim4 scalar from octonion (inverse of embed_dim4)."""
        if self.calibration is None:
            return float(oct.components[4])
        axis = self.calibration.dim4_axis
        normalized = float(oct.components[axis])
        return normalized * self.calibration.dim4_std + self.calibration.dim4_mean

    @staticmethod
    def to_volitional_vector_static(oct: Octonion) -> VolitionalVector:
        """Backward-compatible static wrapper."""
        return G2Projection().to_volitional_vector(oct)

    @staticmethod
    def from_volitional_vector_static(vec: VolitionalVector) -> Octonion:
        """Backward-compatible static wrapper."""
        return G2Projection().from_volitional_vector(vec)


def g2_generators_count() -> int:
    """Return the dimension of the G2 Lie algebra."""
    return G2_DIM


def _embed_r3(v: NDArray[np.float64]) -> Octonion:
    """Embed R³ into octonion imaginary units (e1, e2, e4)."""
    return Octonion(np.array([0.0, v[0], v[1], 0.0, v[2], 0.0, 0.0, 0.0]))


def g2_cross_product(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Cross product via octonion multiplication in the G2-invariant substructure.

    Embeds R³ into (e1, e2, e4); e1·e2 = e4 under the Fano plane.
    """
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Cross product requires 3-vectors")
    prod = _embed_r3(a) * _embed_r3(b)
    return np.array([prod.components[1], prod.components[2], prod.components[4]])