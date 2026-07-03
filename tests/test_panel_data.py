"""Tests for 1950–2023 panel data and framework replication."""

from volition.constants import (
    FRAMEWORK_PEARSON_R,
    FRAMEWORK_R_SQUARED,
    PANEL_LEAD_YEARS,
    PANEL_TRAIN_END,
    PANEL_YEAR_START,
)
from volition.data.panel import (
    build_lead_lag_pairs,
    load_panel,
    panel_validation_stats,
)


def test_load_panel_schema():
    df = load_panel()
    assert len(df) > 10_000
    assert set(df.columns) >= {"country", "year", "dim4", "tfr_lead18"}
    assert df["year"].min() == PANEL_YEAR_START
    assert df["country"].nunique() == 187


def test_lead_lag_pairs():
    df = load_panel()
    pairs = build_lead_lag_pairs(df, year_max=PANEL_TRAIN_END)
    assert len(pairs) > 0
    assert pairs["tfr"].notna().all()
    assert pairs["lead_years"].iloc[0] == PANEL_LEAD_YEARS


def test_panel_replicates_framework_targets():
    stats = panel_validation_stats()
    assert stats["framework_passed"] is True
    assert abs(stats["pearson_r"] - FRAMEWORK_PEARSON_R) < 0.01
    assert abs(stats["r_squared"] - FRAMEWORK_R_SQUARED) < 0.02
    assert stats["p_value"] < 1e-13
    assert stats["no_return_violations"] == 0
    assert stats["train"]["n_pairs"] > 0
    assert stats["test"]["n_pairs"] > 0


def test_panel_train_test_year_ranges():
    stats = panel_validation_stats()
    assert stats["train"]["year_range"] == [1950, 1987]
    assert stats["test"]["year_range"] == [1988, 2005]