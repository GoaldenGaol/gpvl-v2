"""Tests for symbolic equations."""

from volition.equations import export_equations_latex, fertility_mapping_linear


def test_fertility_mapping_has_symbols():
    eq = fertility_mapping_linear()
    assert eq.lhs is not None
    assert eq.rhs is not None


def test_latex_export_keys():
    latex = export_equations_latex()
    assert "fertility_linear" in latex
    assert "goalden_threshold" in latex
    assert len(latex["fertility_linear"]) > 0