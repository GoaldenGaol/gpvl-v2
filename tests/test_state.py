"""Tests for volitional state representations."""

from volition.constants import DIM4_IRREVERSIBLE
from volition.regime import Regime
from volition.state import (
    FirmVolitionalState,
    FullState,
    VolitionalState,
    VolitionalVector,
    build_summary,
    tfr_band,
)


def test_tfr_band_classification():
    assert tfr_band(0.50) == "HIGH_TFR"
    assert tfr_band(0.85) == "MID_TFR"
    assert tfr_band(1.20) == "LOW_TFR"


def test_volitional_state_summarize():
    state = VolitionalState(name="Japan", year=2023, dim4=1.31, tfr=1.26)
    text = state.summarize()
    assert "Japan" in text
    assert "IRREVERSIBLE" in text
    assert "TFR=1.26" in text


def test_volitional_state_verbose():
    state = VolitionalState(name="Italy", year=2023, dim4=1.22, tfr=1.20)
    text = state.summarize(verbose=True)
    assert "kind=country" in text
    assert "irreversible=True" in text


def test_firm_volitional_state():
    firm = FirmVolitionalState(
        name="Blockbuster",
        year=2004,
        dim4=1.20,
        sector="media-rental",
        region="US",
        collapsed=True,
    )
    summary = firm.summary()
    assert summary.entity_kind == "firm"
    assert summary.collapsed is True
    assert summary.regime == Regime.IRREVERSIBLE


def test_firm_summarize_verbose():
    firm = FirmVolitionalState(
        name="Nvidia",
        year=2010,
        dim4=0.68,
        sector="semiconductors",
        region="US",
        collapsed=False,
    )
    text = firm.summarize(verbose=True)
    assert "sector=semiconductors" in text
    assert "collapsed=False" in text


def test_full_state_dim4():
    vec = VolitionalVector(dim4=1.05)
    xi = FullState(volitional_vector=vec, cooperation_density=0.42)
    assert xi.dim4 == 1.05


def test_build_summary_irreversible_flag():
    summary = build_summary("South Korea", 2023, 1.41, tfr=0.72)
    assert summary.is_irreversible
    assert not summary.violates_no_return
    assert summary.dim4 > DIM4_IRREVERSIBLE