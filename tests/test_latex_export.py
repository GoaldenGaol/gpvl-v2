"""Tests for LaTeX export bundle."""

from pathlib import Path

from volition.export.latex import (
    export_validation_table,
    write_arxiv_bundle,
    write_equations_tex,
)


def test_validation_table_contains_stats():
    table = export_validation_table()
    assert "Pearson" in table
    assert "tabular" in table


def test_write_equations_tex(tmp_path: Path):
    path = write_equations_tex(tmp_path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert r"\begin{align}" in content
    assert (tmp_path / "thresholds.tex").exists()


def test_write_arxiv_bundle(tmp_path: Path):
    result = write_arxiv_bundle(tmp_path)
    assert result.main_tex.exists()
    main = result.main_tex.read_text(encoding="utf-8")
    assert r"\documentclass" in main
    assert r"\input{equations.tex}" in main
    assert result.validation_tex.read_text(encoding="utf-8").count("tabular") >= 1