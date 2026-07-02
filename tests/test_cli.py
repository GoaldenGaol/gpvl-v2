"""Tests for GPVL CLI."""

import json
from pathlib import Path

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