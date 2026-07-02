"""Load and validate frozen 2023 dim4 CSV datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volition.constants import DIM4_IRREVERSIBLE, DIM4_NO_RETURN
from volition.paths import data_file


def load_countries() -> pd.DataFrame:
    """Load country-level dim4 snapshot with matching TFR."""
    return pd.read_csv(data_file("dim4_frozen_2023.csv"))


def load_countries_full() -> pd.DataFrame:
    """Load complete 187-country frozen dim4 dataset."""
    return pd.read_csv(data_file("dim4_frozen_2023_full.csv"))


def load_firms() -> pd.DataFrame:
    """Load firm-level dim4 scalar (collapse vs survival)."""
    return pd.read_csv(data_file("dim4_firms_frozen_2023.csv"))


def validation_stats(df: pd.DataFrame | None = None) -> dict[str, float]:
    """
    Compute validation statistics for dim4 vs TFR mapping.

    Uses Pearson r and R² on the frozen 2023 cross-section.
    """
    if df is None:
        df = load_countries()

    x = df["dim4"].to_numpy(dtype=np.float64)
    y = df["tfr_future"].to_numpy(dtype=np.float64)

    r = float(np.corrcoef(x, y)[0, 1])
    r_squared = r**2

    above_irreversible = df[df["dim4"] > DIM4_IRREVERSIBLE]
    no_return_violations = int((above_irreversible["dim4"] < DIM4_NO_RETURN).sum())

    return {
        "pearson_r": r,
        "r_squared": r_squared,
        "n_countries": len(df),
        "n_above_irreversible": len(above_irreversible),
        "no_return_violations": no_return_violations,
    }