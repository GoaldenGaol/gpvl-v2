"""Axioms of Volition (Section 8) as symbolic constraints."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from volition.equations.dynamics import competence_transition, property_transition
from volition.equations.symbols import (
    F_i,
    P,
    P_n,
    aggression,
    consent,
    rho_plunder,
)


def axiom_1_rationality() -> sp.Relational:
    """Agents select actions that increase expected utility: E[ΔU] ≥ 0."""
    dU = sp.Symbol("Delta_U", real=True)
    return sp.Ge(dU, 0)


def axiom_2_property_medium() -> sp.Eq:
    """All actions transform property: P(t+1) = T_P(P, X, ρ)."""
    return property_transition()


def axiom_3_interaction_transforms_property() -> sp.Eq:
    """Interactions are multi-agent property maps P(t) → P(t+1)."""
    return sp.Eq(P_n, sp.Function("T_P")(P, rho_plunder))


def axiom_4_consent() -> sp.Implies:
    """Non-aggression iff consent: ¬aggression ⟺ consent."""
    return sp.Implies(sp.Not(aggression), consent)


def axiom_5_aggression_definition() -> sp.Eq:
    """Aggression = non-consensual property transformation."""
    return sp.Eq(aggression, sp.Not(consent))


def axiom_6_aggression_penalty() -> sp.Relational:
    """E[F | aggression] ≤ 0 over long horizons."""
    E_F = sp.Symbol("E_F_aggression", real=True)
    return sp.Le(E_F, 0)


def axiom_7_profit() -> sp.Relational:
    """Voluntary interactions yield mutual utility gain: ΔU_a, ΔU_b > 0."""
    dU_a = sp.Symbol("Delta_U_a", real=True)
    dU_b = sp.Symbol("Delta_U_b", real=True)
    return sp.And(sp.Gt(dU_a, 0), sp.Gt(dU_b, 0))


def axiom_8_competence_accumulation() -> sp.Eq:
    """C(t+1) = C(t) + F(i_t)."""
    return competence_transition()


def all_axioms() -> dict[str, object]:
    """Return all eight axioms keyed by identifier."""
    return {
        "axiom_1_rationality": axiom_1_rationality(),
        "axiom_2_property_medium": axiom_2_property_medium(),
        "axiom_3_interaction_transforms": axiom_3_interaction_transforms_property(),
        "axiom_4_consent": axiom_4_consent(),
        "axiom_5_aggression": axiom_5_aggression_definition(),
        "axiom_6_penalty": axiom_6_aggression_penalty(),
        "axiom_7_profit": axiom_7_profit(),
        "axiom_8_competence": axiom_8_competence_accumulation(),
    }


@dataclass(frozen=True)
class AxiomCheckResult:
    """Result of a numeric axiom consistency check."""

    axiom: str
    satisfied: bool
    detail: str


def check_axiom_8_numeric(
    c_t: float,
    c_next: float,
    feedback: float,
    tol: float = 1e-9,
) -> AxiomCheckResult:
    """Verify C(t+1) = C(t) + F(i_t) numerically."""
    expected = c_t + feedback
    ok = abs(c_next - expected) <= tol
    return AxiomCheckResult(
        axiom="axiom_8_competence",
        satisfied=ok,
        detail=f"C_next={c_next:.6f}, expected={expected:.6f}",
    )


def check_goalden_cooperative(
    rho: float,
    c_mean_val: float,
    r_mean_val: float,
) -> AxiomCheckResult:
    """Verify cooperative equilibrium condition from Goalden threshold."""
    threshold = 0.1 * c_mean_val * r_mean_val
    ok = rho < threshold
    return AxiomCheckResult(
        axiom="goalden_cooperative",
        satisfied=ok,
        detail=f"rho={rho:.4f}, threshold={threshold:.4f}",
    )


def check_axiom_5_consent_logic(consented: bool, is_aggressive: bool) -> AxiomCheckResult:
    """Verify aggression ↔ ¬consent logical consistency."""
    expected_aggression = not consented
    ok = is_aggressive == expected_aggression
    return AxiomCheckResult(
        axiom="axiom_5_aggression",
        satisfied=ok,
        detail=f"consent={consented}, aggression={is_aggressive}",
    )