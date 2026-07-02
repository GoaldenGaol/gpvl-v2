"""Symbolic equations for the Science of Volition (sympy)."""

from __future__ import annotations

import sympy as sp

# --- Symbols (CORE-FRAMEWORK Section 7) ---
t = sp.Symbol("t", real=True)
dim4 = sp.Symbol("dim4", real=True)
tfr = sp.Symbol("TFR", positive=True)
rho_plunder = sp.Symbol("rho_plunder", nonnegative=True)
c_mean = sp.Symbol("C_mean", positive=True)
r_mean = sp.Symbol("R_mean", positive=True)

# Transition map parameters
alpha = sp.Symbol("alpha", real=True)
beta = sp.Symbol("beta", real=True)


def fertility_mapping_linear() -> sp.Eq:
    """TFR(t+18) = alpha * dim4(t) + beta  (linear form)."""
    return sp.Eq(tfr, alpha * dim4 + beta)


def fertility_mapping_quadratic() -> sp.Eq:
    """Quadratic fertility mapping (R² ≈ 0.85 on 2023 cross-section)."""
    gamma = sp.Symbol("gamma", real=True)
    return sp.Eq(tfr, alpha * dim4**2 + beta * dim4 + gamma)


def goalden_threshold_equation() -> sp.Eq:
    """rho_plunder < 0.1 * C_mean * R_mean  (cooperative equilibrium)."""
    return sp.Eq(sp.Lt(rho_plunder, sp.Rational(1, 10) * c_mean * r_mean), True)


def transition_map_rhs() -> sp.Expr:
    """
    Placeholder for V(t+1) = Phi(V(t), i_t, u(t)).

    Phase 3 will expand to full 7-D vector dynamics.
    """
    u = sp.Symbol("u", real=True)  # intervention
    return dim4 + alpha * rho_plunder + u


def export_equations_latex() -> dict[str, str]:
    """Return LaTeX strings for core equations."""
    return {
        "fertility_linear": sp.latex(fertility_mapping_linear()),
        "fertility_quadratic": sp.latex(fertility_mapping_quadratic()),
        "goalden_threshold": sp.latex(goalden_threshold_equation()),
        "transition_map": sp.latex(sp.Eq(sp.Symbol("dim4_{t+1}"), transition_map_rhs())),
    }