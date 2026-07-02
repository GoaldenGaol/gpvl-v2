"""Tests for axiom symbolic and numeric checks."""

from volition.equations.axioms import (
    axiom_8_competence_accumulation,
    check_axiom_5_consent_logic,
    check_axiom_8_numeric,
    check_goalden_cooperative,
)


def test_axiom_8_symbolic():
    eq = axiom_8_competence_accumulation()
    assert "C" in str(eq.lhs) or "C_{t+1}" in str(eq.lhs)


def test_axiom_8_numeric_pass():
    result = check_axiom_8_numeric(c_t=1.0, c_next=1.5, feedback=0.5)
    assert result.satisfied


def test_axiom_8_numeric_fail():
    result = check_axiom_8_numeric(c_t=1.0, c_next=2.0, feedback=0.5)
    assert not result.satisfied


def test_goalden_cooperative_pass():
    result = check_goalden_cooperative(0.05, 1.28, 0.91)
    assert result.satisfied


def test_goalden_cooperative_fail():
    result = check_goalden_cooperative(0.20, 1.28, 0.91)
    assert not result.satisfied


def test_axiom_5_consent_logic():
    assert check_axiom_5_consent_logic(True, False).satisfied
    assert check_axiom_5_consent_logic(False, True).satisfied
    assert not check_axiom_5_consent_logic(True, True).satisfied