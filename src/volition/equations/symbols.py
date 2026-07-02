"""SymPy symbol registry (CORE-FRAMEWORK Section 7)."""

from __future__ import annotations

import sympy as sp

# Time and agents
t = sp.Symbol("t", real=True)
t_next = sp.Symbol("t+1", real=True)

# 7-D volitional vector V(t) = [dim1..dim7]
dim1, dim2, dim3, dim4, dim5, dim6, dim7 = sp.symbols(
    "dim1 dim2 dim3 dim4 dim5 dim6 dim7", real=True
)
dim1_n, dim2_n, dim3_n, dim4_n, dim5_n, dim6_n, dim7_n = sp.symbols(
    "dim1_{t+1} dim2_{t+1} dim3_{t+1} dim4_{t+1} dim5_{t+1} dim6_{t+1} dim7_{t+1}",
    real=True,
)

# Full state S(t) = (P, C, B, X, Ω)
P = sp.Symbol("P", positive=True)
C = sp.Symbol("C", positive=True)
B = sp.Symbol("B", real=True)
X = sp.Symbol("X", positive=True)
Omega = sp.Symbol("Omega", nonnegative=True)

P_n = sp.Symbol("P_{t+1}", positive=True)
C_n = sp.Symbol("C_{t+1}", positive=True)
B_n = sp.Symbol("B_{t+1}", real=True)
X_n = sp.Symbol("X_{t+1}", positive=True)
Omega_n = sp.Symbol("Omega_{t+1}", nonnegative=True)

# Interaction and institutional quantities
rho_plunder = sp.Symbol("rho_plunder", nonnegative=True)
delta_coop = sp.Symbol("delta_coop", nonnegative=True)
DI = sp.Symbol("DI", nonnegative=True)  # institutional drift ||I(t) - I(t-1)||
F_i = sp.Symbol("F_i", real=True)  # feedback F(i_t)
u = sp.Symbol("u", real=True)  # intervention u(t)
lam = sp.Symbol("lambda", nonnegative=True)  # aggression penalty

# Demographics and Goalden
tfr = sp.Symbol("TFR", positive=True)
c_mean = sp.Symbol("C_mean", positive=True)
r_mean = sp.Symbol("R_mean", positive=True)

# Fertility fit parameters
alpha = sp.Symbol("alpha", real=True)
beta = sp.Symbol("beta", real=True)
gamma = sp.Symbol("gamma", real=True)

# Phi persistence and coupling parameters
phi_persist = sp.symbols("phi_p1:8", real=True, positive=True)
phi_couple = sp.symbols("phi_c1:8", real=True)

# Hazard / collapse
h = sp.Symbol("h", nonnegative=True)
Z = sp.Symbol("Z", integer=True)

# Utility / consent (axioms)
U = sp.Symbol("U", real=True)
consent = sp.Symbol("consent", bool=True)
aggression = sp.Symbol("aggression", bool=True)

VOLITIONAL_SYMBOLS: tuple[sp.Symbol, ...] = (
    dim1, dim2, dim3, dim4, dim5, dim6, dim7,
    dim1_n, dim2_n, dim3_n, dim4_n, dim5_n, dim6_n, dim7_n,
)

STATE_SYMBOLS: tuple[sp.Symbol, ...] = (P, C, B, X, Omega, P_n, C_n, B_n, X_n, Omega_n)

CONTROL_SYMBOLS: tuple[sp.Symbol, ...] = (rho_plunder, delta_coop, DI, F_i, u)


def volitional_vector() -> sp.Matrix:
    """V(t) as a 7×1 column vector."""
    return sp.Matrix([dim1, dim2, dim3, dim4, dim5, dim6, dim7])


def volitional_vector_next() -> sp.Matrix:
    """V(t+1) as a 7×1 column vector."""
    return sp.Matrix([dim1_n, dim2_n, dim3_n, dim4_n, dim5_n, dim6_n, dim7_n])