"""Tests for symbolic equations package."""

import sympy as sp

from volition.equations import (
    all_axioms,
    export_equations_latex,
    fertility_mapping_linear,
    fertility_slope_sign_constraint,
    phi_transition_equations,
    system_transitions,
)


def test_fertility_mapping_has_symbols():
    eq = fertility_mapping_linear()
    assert eq.lhs is not None
    assert eq.rhs is not None


def test_fertility_monotone_constraint():
    rel = fertility_slope_sign_constraint()
    assert rel.rel_op == "<"


def test_phi_seven_equations():
    eqs = phi_transition_equations()
    assert len(eqs) == 7
    assert all(isinstance(e, sp.Eq) for e in eqs)


def test_system_transitions_count():
    eqs = system_transitions()
    assert len(eqs) == 5


def test_all_axioms_count():
    axioms = all_axioms()
    assert len(axioms) == 8


def test_latex_export_keys():
    latex = export_equations_latex()
    for key in (
        "fertility_linear",
        "phi_dim4",
        "phi_system",
        "dynamics",
        "axiom_8",
        "regime_theta",
    ):
        assert key in latex
        assert len(latex[key]) > 0