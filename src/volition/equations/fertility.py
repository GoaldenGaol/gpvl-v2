"""Fertility and Goalden threshold mappings."""

from __future__ import annotations

import sympy as sp

from volition.equations.symbols import alpha, beta, c_mean, dim4, gamma, r_mean, rho_plunder, tfr


def fertility_mapping_linear() -> sp.Eq:
    """TFR(t+18) = alpha * dim4(t) + beta."""
    return sp.Eq(tfr, alpha * dim4 + beta)


def fertility_mapping_quadratic() -> sp.Eq:
    """Quadratic fertility mapping (R² ≈ 0.85 on 2023 cross-section)."""
    return sp.Eq(tfr, alpha * dim4**2 + beta * dim4 + gamma)


def goalden_threshold_equation() -> sp.Relational:
    """rho_plunder < 0.1 * C_mean * R_mean (cooperative equilibrium)."""
    return sp.Lt(rho_plunder, sp.Rational(1, 10) * c_mean * r_mean)


def fertility_slope_sign_constraint() -> sp.Relational:
    """Monotone decreasing fertility mapping: dTFR/ddim4 < 0."""
    d_tfr = sp.diff(fertility_mapping_linear().rhs, dim4)
    return sp.Lt(d_tfr, 0)