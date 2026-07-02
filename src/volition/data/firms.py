"""Firm-level dim4 data loading and fatal-pivot validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_FIRM_HIGH_RISK, DIM4_IRREVERSIBLE
from volition.paths import data_file
from volition.state import FirmVolitionalState


def load_firms_df() -> pd.DataFrame:
    """Load firm-level dim4 dataset as a DataFrame."""
    return pd.read_csv(data_file("dim4_firms_frozen_2023.csv"))


def load_firm_states() -> list[FirmVolitionalState]:
    """Load all firm volitional states."""
    df = load_firms_df()
    return [FirmVolitionalState.from_row(row) for _, row in df.iterrows()]


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC-AUC via the Mann-Whitney U statistic.

    Higher scores should predict positive class (collapse = 1).
    """
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # Count concordant pairs: P(score_pos > score_neg) + 0.5 * ties
    auc = 0.0
    for p in pos:
        auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(auc / (len(pos) * len(neg)))


def classify_firm_risk(dim4_firm: float) -> str:
    """Classify firm fatal-pivot risk band."""
    if dim4_firm > DIM4_IRREVERSIBLE:
        return "CRITICAL"
    if dim4_firm > DIM4_FIRM_HIGH_RISK:
        return "HIGH"
    if dim4_firm > DIM4_COOP_COLLAPSE:
        return "ELEVATED"
    return "LOW"


def firm_validation_stats(df: pd.DataFrame | None = None) -> dict[str, float]:
    """
    Validation statistics for dim4_firm → fatal pivot (5–7 yr).

    Reports ROC-AUC using dim4_firm as a single-variable rank score,
    plus threshold separation metrics.
    """
    if df is None:
        df = load_firms_df()

    scores = df["dim4_firm"].to_numpy(dtype=np.float64)
    labels = df["y_pivot_5yr"].to_numpy(dtype=np.int64)

    auc = _roc_auc(labels, scores)

    collapsed = df[df["y_pivot_5yr"] == 1]
    survived = df[df["y_pivot_5yr"] == 0]

    mean_collapsed = float(collapsed["dim4_firm"].mean())
    mean_survived = float(survived["dim4_firm"].mean())

    high_risk = df[df["dim4_firm"] > DIM4_FIRM_HIGH_RISK]
    precision_high = float(high_risk["y_pivot_5yr"].mean()) if len(high_risk) > 0 else float("nan")

    return {
        "roc_auc": auc,
        "n_firms": len(df),
        "n_collapsed": int(labels.sum()),
        "n_survived": int(len(labels) - labels.sum()),
        "mean_dim4_collapsed": mean_collapsed,
        "mean_dim4_survived": mean_survived,
        "precision_above_0.92": precision_high,
        "n_above_0.92": len(high_risk),
    }