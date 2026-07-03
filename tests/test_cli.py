"""Tests for GPVL CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from volition.cli import main


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert "gpvl 2.0.0" in capsys.readouterr().out


def test_cli_validate(capsys):
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "Pearson r" in out
    assert "R²" in out


def test_cli_validate_json(capsys):
    assert main(["validate", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "pearson_r" in data
    assert data["pearson_r"] < 0


def test_cli_export_latex_bundle(tmp_path: Path):
    out = tmp_path / "arxiv"
    assert main(["export-latex", "-o", str(out)]) == 0
    assert (out / "main.tex").exists()
    assert (out / "equations.tex").exists()
    assert (out / "thresholds.tex").exists()
    assert (out / "validation.tex").exists()
    assert (out / "abstract.tex").exists()
    assert (out / "preamble.tex").exists()


def test_cli_export_equations_only(tmp_path: Path):
    out = tmp_path / "eq"
    assert main(["export-latex", "-o", str(out), "--equations-only"]) == 0
    assert (out / "equations.tex").exists()
    assert (out / "thresholds.tex").exists()
    assert not (out / "main.tex").exists()


@pytest.fixture
def mock_manuscript_stats(monkeypatch):
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


def test_cli_validate_panel(capsys):
    assert main(["validate", "--panel"]) == 0
    out = capsys.readouterr().out
    assert "Panel Validation" in out
    assert "Framework" in out
    assert "PASS" in out


def test_cli_validate_panel_json(capsys):
    assert main(["validate", "--panel", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["framework_passed"] is True
    assert "train" in data
    assert "test" in data


def test_cli_export_manuscript(tmp_path: Path, mock_manuscript_stats):
    out = tmp_path / "manuscript"
    assert main(["export-latex", "-o", str(out), "--manuscript"]) == 0
    assert (out / "main.tex").exists()
    assert (out / "introduction.tex").exists()
    assert (out / "methods.tex").exists()
    assert (out / "results.tex").exists()
    assert (out / "appendix.tex").exists()