"""Calibrate G2 projection against frozen dim4 data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volition.geometry.g2 import G2Calibration, G2Projection


@dataclass(frozen=True)
class GeometryCalibrationResult:
    """Result of octonion/G2 calibration."""

    dim4_mean: float
    dim4_std: float
    n_samples: int
    roundtrip_max_error: float
    rank_correlation_preserved: bool
    spearman_r: float

    def summary(self) -> str:
        return (
            f"dim4 mean={self.dim4_mean:.4f}, std={self.dim4_std:.4f}, "
            f"n={self.n_samples}, roundtrip_err={self.roundtrip_max_error:.2e}, "
            f"Spearman r={self.spearman_r:.4f}"
        )


def _spearman_r(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Spearman rank correlation without scipy."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def calibrate_geometry(
    df: pd.DataFrame | None = None,
    *,
    column: str = "dim4",
) -> tuple[G2Projection, GeometryCalibrationResult]:
    """
    Calibrate G2 projection from frozen dim4 data.

    Validates that embed_dim4 → extract_dim4 roundtrip preserves rank order.
    """
    from volition.data.dim4 import load_countries_full

    data = df if df is not None else load_countries_full()
    dim4 = data[column].to_numpy(dtype=np.float64)

    cal = G2Calibration.from_dim4_series(dim4)
    proj = G2Projection(calibration=cal)

    errors = []
    extracted = []
    for d in dim4:
        oct = proj.embed_dim4(d)
        recovered = proj.extract_dim4(oct)
        errors.append(abs(recovered - d))
        extracted.append(recovered)

    extracted_arr = np.array(extracted)
    spearman = _spearman_r(dim4, extracted_arr)

    result = GeometryCalibrationResult(
        dim4_mean=cal.dim4_mean,
        dim4_std=cal.dim4_std,
        n_samples=len(dim4),
        roundtrip_max_error=float(max(errors)),
        rank_correlation_preserved=abs(spearman - 1.0) < 1e-9,
        spearman_r=spearman,
    )
    return proj, result