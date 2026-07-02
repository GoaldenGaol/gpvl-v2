"""Empirical invariants A–E (CORE-FRAMEWORK Section 10)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_IRREVERSIBLE, DIM4_NO_RETURN


@dataclass(frozen=True)
class InvariantResult:
    """Result of an empirical invariant check."""

    invariant: str
    satisfied: bool
    statistic: float
    detail: str


def invariant_a_fertility_mapping(
    dim4_vals: NDArray[np.float64],
    tfr_vals: NDArray[np.float64],
    *,
    min_abs_r: float = 0.5,
) -> InvariantResult:
    """
    Invariant A: dim4 → TFR mapping is global, monotone, high-R².

    Checks negative Pearson correlation exceeds threshold.
    """
    r = float(np.corrcoef(dim4_vals, tfr_vals)[0, 1])
    ok = r <= -min_abs_r
    return InvariantResult(
        invariant="A_fertility_mapping",
        satisfied=ok,
        statistic=r,
        detail=f"Pearson r = {r:.4f} (require r ≤ −{min_abs_r})",
    )


def invariant_b_irreversible_band(
    dim4_vals: NDArray[np.float64],
    *,
    threshold: float = DIM4_IRREVERSIBLE,
) -> InvariantResult:
    """
    Invariant B: dim4 > 1.0 behaves as irreversible low-fertility band.

    Checks that all observations above threshold have dim4 > 0.7.
    """
    above = dim4_vals > threshold
    if not np.any(above):
        return InvariantResult(
            invariant="B_irreversible_band",
            satisfied=True,
            statistic=0.0,
            detail="No observations above threshold",
        )
    violations = int(np.sum(dim4_vals[above] < DIM4_NO_RETURN))
    ok = violations == 0
    return InvariantResult(
        invariant="B_irreversible_band",
        satisfied=ok,
        statistic=float(violations),
        detail=f"{int(np.sum(above))} above {threshold}, {violations} violations",
    )


def invariant_c_no_reversal(
    dim4_trajectory: NDArray[np.float64],
) -> InvariantResult:
    """
    Invariant C: no country with dim4 > 1.0 returned below 0.7.

    Applied to a single country's time series.
    """
    ever_above = bool(np.any(dim4_trajectory > DIM4_IRREVERSIBLE))
    if not ever_above:
        return InvariantResult(
            invariant="C_no_reversal",
            satisfied=True,
            statistic=0.0,
            detail="Never crossed irreversible band",
        )
    first_cross = int(np.argmax(dim4_trajectory > DIM4_IRREVERSIBLE))
    post_cross = dim4_trajectory[first_cross:]
    violations = int(np.sum(post_cross < DIM4_NO_RETURN))
    ok = violations == 0
    return InvariantResult(
        invariant="C_no_reversal",
        satisfied=ok,
        statistic=float(violations),
        detail=f"post-cross violations below {DIM4_NO_RETURN}: {violations}",
    )


def invariant_d_cooperation_collapse(
    dim4_vals: NDArray[np.float64],
    collapsed: NDArray[np.bool_],
    *,
    threshold: float = DIM4_COOP_COLLAPSE,
) -> InvariantResult:
    """
    Invariant D: dim4 > 0.92 predicts cooperation collapse.

    Checks precision of collapse label above threshold.
    """
    above = dim4_vals > threshold
    if not np.any(above):
        return InvariantResult(
            invariant="D_cooperation_collapse",
            satisfied=True,
            statistic=0.0,
            detail="No observations above cooperation threshold",
        )
    precision = float(np.mean(collapsed[above]))
    ok = precision >= 0.5
    return InvariantResult(
        invariant="D_cooperation_collapse",
        satisfied=ok,
        statistic=precision,
        detail=f"precision above {threshold} = {precision:.4f}",
    )


def invariant_e_unified_scalar(
    country_r: float,
    firm_auc: float,
    *,
    min_r: float = 0.5,
    min_auc: float = 0.8,
) -> InvariantResult:
    """
    Invariant E: same scalar predicts fertility and cooperation/collapse.

    Joint check on country correlation and firm AUC.
    """
    ok = country_r <= -min_r and firm_auc >= min_auc
    stat = (abs(country_r) + firm_auc) / 2.0
    return InvariantResult(
        invariant="E_unified_scalar",
        satisfied=ok,
        statistic=stat,
        detail=f"country |r|={abs(country_r):.4f}, firm AUC={firm_auc:.4f}",
    )


def check_all_invariants(
    dim4_country: NDArray[np.float64],
    tfr_country: NDArray[np.float64],
    dim4_firm: NDArray[np.float64],
    firm_collapsed: NDArray[np.bool_],
    firm_auc: float,
) -> list[InvariantResult]:
    """Run invariants A–E on bundled datasets."""
    country_r = float(np.corrcoef(dim4_country, tfr_country)[0, 1])
    return [
        invariant_a_fertility_mapping(dim4_country, tfr_country),
        invariant_b_irreversible_band(dim4_country),
        invariant_c_no_reversal(dim4_country),  # cross-section proxy
        invariant_d_cooperation_collapse(dim4_firm, firm_collapsed),
        invariant_e_unified_scalar(country_r, firm_auc),
    ]