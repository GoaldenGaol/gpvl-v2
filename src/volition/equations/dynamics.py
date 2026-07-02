"""System dynamics S(t) = (P, C, B, X, Ω) transitions (Section 9)."""

from __future__ import annotations

import sympy as sp

from volition.equations.symbols import (
    B,
    B_n,
    C,
    C_n,
    F_i,
    Omega,
    Omega_n,
    P,
    P_n,
    X,
    X_n,
    delta_coop,
    lam,
    rho_plunder,
)


def property_transition() -> sp.Eq:
    """P(t+1) = P(t) + X(t)·(1 − ρ_plunder) − λ·ρ_plunder."""
    return sp.Eq(P_n, P + X * (1 - rho_plunder) - lam * rho_plunder)


def competence_transition() -> sp.Eq:
    """C(t+1) = C(t) + F(i_t)  [Axiom 8]."""
    return sp.Eq(C_n, C + F_i)


def belief_transition() -> sp.Eq:
    """B(t+1) = B(t) + α_B · F(i_t)  (Bayesian update surrogate)."""
    alpha_b = sp.Symbol("alpha_B", positive=True)
    return sp.Eq(B_n, B + alpha_b * F_i)


def action_set_transition() -> sp.Eq:
    """X(t+1) = X(t) + g_C · C(t)  (competence expands action set)."""
    g_c = sp.Symbol("g_C", positive=True)
    return sp.Eq(X_n, X + g_c * C)


def network_transition() -> sp.Eq:
    """Ω(t+1) = Ω(t) + δ_coop − ρ_plunder."""
    return sp.Eq(Omega_n, Omega + delta_coop - rho_plunder)


def system_transitions() -> list[sp.Eq]:
    """All S(t) component transitions."""
    return [
        property_transition(),
        competence_transition(),
        belief_transition(),
        action_set_transition(),
        network_transition(),
    ]


def stability_condition() -> sp.Eq:
    """lim_{t→∞} S(t) = S*  (equilibrium existence)."""
    S_star = sp.Symbol("S^*", real=True)
    return sp.Eq(sp.Limit(sp.Symbol("S(t)"), sp.Symbol("t"), sp.oo), S_star)