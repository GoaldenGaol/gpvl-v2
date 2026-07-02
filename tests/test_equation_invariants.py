"""Tests for empirical invariant checks (equations.invariants)."""

import numpy as np

from volition.data.dim4 import load_countries
from volition.data.firms import firm_validation_stats, load_firms_df
from volition.equations.invariants import (
    check_all_invariants,
    invariant_a_fertility_mapping,
    invariant_b_irreversible_band,
    invariant_e_unified_scalar,
)


def test_invariant_a_on_frozen_data():
    df = load_countries()
    result = invariant_a_fertility_mapping(
        df["dim4"].to_numpy(),
        df["tfr_future"].to_numpy(),
        min_abs_r=0.5,
    )
    assert result.satisfied


def test_invariant_b_no_violations():
    df = load_countries()
    result = invariant_b_irreversible_band(df["dim4"].to_numpy())
    assert result.satisfied


def test_invariant_e_unified_scalar():
    df = load_countries()
    firm_df = load_firms_df()
    firm_stats = firm_validation_stats(firm_df)
    r = float(np.corrcoef(df["dim4"], df["tfr_future"])[0, 1])
    result = invariant_e_unified_scalar(r, firm_stats["roc_auc"])
    assert result.satisfied


def test_check_all_invariants():
    country_df = load_countries()
    firm_df = load_firms_df()
    firm_stats = firm_validation_stats(firm_df)
    results = check_all_invariants(
        country_df["dim4"].to_numpy(),
        country_df["tfr_future"].to_numpy(),
        firm_df["dim4_firm"].to_numpy(),
        firm_df["y_pivot_5yr"].to_numpy(dtype=bool),
        firm_stats["roc_auc"],
    )
    assert len(results) == 5
    assert sum(r.satisfied for r in results) >= 4