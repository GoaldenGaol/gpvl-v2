"""Tests for full arXiv manuscript export."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from volition.export.manuscript import (
    export_introduction,
    export_main_manuscript,
    write_full_manuscript,
)


@pytest.fixture
def mock_stats(monkeypatch):
    """Fast mock stats to avoid MCMC/VPDE in manuscript tests."""
    mcmc = MagicMock()
    mcmc.alpha = -5.7
    mcmc.beta = 7.2
    mcmc.sigma = 0.65
    mcmc.summary.p16 = [-6.0, 7.0, 0.0]
    mcmc.summary.p84 = [-5.4, 7.4, 0.0]

    vpde = MagicMock()
    vpde.tau = 0.39
    vpde.achieved_crossing_time = 5.89
    vpde.residual = 0.11

    geom = MagicMock()
    geom.dim4_mean = 0.95
    geom.dim4_std = 0.18
    geom.n_samples = 195

    stats = {
        "country": {
            "pearson_r": -0.819,
            "r_squared": 0.671,
            "n_countries": 195,
            "n_above_irreversible": 24,
            "no_return_violations": 0,
        },
        "firm": {"roc_auc": 0.996, "n_firms": 83},
        "mcmc": mcmc,
        "vpde": vpde,
        "geom": geom,
    }
    monkeypatch.setattr("volition.export.manuscript._gather_stats", lambda: stats)
    return stats


def test_introduction_section():
    text = export_introduction()
    assert r"\section{Introduction}" in text
    assert "dim4" in text


def test_main_manuscript_structure():
    main = export_main_manuscript()
    assert r"\input{introduction.tex}" in main
    assert r"\input{methods.tex}" in main
    assert r"\input{appendix.tex}" in main
    assert r"\input{references.tex}" in main


def test_write_full_manuscript(tmp_path: Path, mock_stats):
    result = write_full_manuscript(tmp_path)
    assert result.main_tex.exists()
    assert (tmp_path / "00README.txt").exists()
    assert (tmp_path / "introduction.tex").exists()
    assert (tmp_path / "methods.tex").exists()
    assert (tmp_path / "results.tex").exists()
    assert (tmp_path / "discussion.tex").exists()
    assert (tmp_path / "appendix.tex").exists()
    assert (tmp_path / "references.tex").exists()

    results = (tmp_path / "results.tex").read_text(encoding="utf-8")
    assert "ROC-AUC" in results
    assert "-5.7" in results or "-5.70" in results

    appendix = (tmp_path / "appendix.tex").read_text(encoding="utf-8")
    assert "Octonion" in appendix
    assert "gpvl" in appendix