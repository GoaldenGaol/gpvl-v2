"""Tests for firm-level dim4 data and validation."""

from volition.constants import DIM4_FIRM_HIGH_RISK
from volition.data.firms import (
    classify_firm_risk,
    firm_validation_stats,
    load_firm_states,
    load_firms_df,
)
from volition.state import FirmVolitionalState


def test_load_firms_df():
    df = load_firms_df()
    assert len(df) >= 80
    assert "dim4_firm" in df.columns
    assert "y_pivot_5yr" in df.columns


def test_load_firm_states():
    firms = load_firm_states()
    assert len(firms) >= 80
    assert all(isinstance(f, FirmVolitionalState) for f in firms)


def test_blockbuster_collapsed():
    firms = {f.name: f for f in load_firm_states()}
    bb = firms["Blockbuster"]
    assert bb.collapsed is True
    assert bb.dim4 > 1.0


def test_nvidia_survived():
    firms = {f.name: f for f in load_firm_states()}
    nv = firms["Nvidia"]
    assert nv.collapsed is False
    assert nv.dim4 < 0.92


def test_firm_validation_auc():
    stats = firm_validation_stats()
    assert stats["roc_auc"] > 0.85
    assert stats["n_firms"] >= 80
    assert stats["mean_dim4_collapsed"] > stats["mean_dim4_survived"]


def test_classify_firm_risk():
    assert classify_firm_risk(0.50) == "LOW"
    assert classify_firm_risk(0.95) == "HIGH"
    assert classify_firm_risk(1.10) == "CRITICAL"


def test_firm_from_row():
    df = load_firms_df()
    row = df.iloc[0]
    firm = FirmVolitionalState.from_row(row)
    assert firm.name == row["firm"]
    assert firm.dim4 == row["dim4_firm"]


def test_precision_above_threshold():
    stats = firm_validation_stats()
    assert stats["n_above_0.92"] > 0
    assert stats["precision_above_0.92"] > 0.5