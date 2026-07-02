"""Tests for dim4 data loading and validation."""

from volition.constants import DIM4_IRREVERSIBLE
from volition.data.dim4 import load_countries, validation_stats


def test_load_countries():
    df = load_countries()
    assert len(df) > 50
    assert "dim4" in df.columns
    assert "tfr_future" in df.columns


def test_south_korea_irreversible():
    df = load_countries()
    sk = df[df["country"] == "South Korea"].iloc[0]
    assert sk["dim4"] > DIM4_IRREVERSIBLE


def test_validation_stats_negative_correlation():
    stats = validation_stats()
    assert stats["pearson_r"] < 0
    assert stats["r_squared"] > 0.5
    assert stats["no_return_violations"] == 0