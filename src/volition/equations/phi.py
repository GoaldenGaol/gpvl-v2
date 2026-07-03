"""Transition map Φ: V(t+1) = Φ(V(t), i_t, u(t))."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from volition.equations.symbols import (
    DI,
    F_i,
    delta_coop,
    phi_couple,
    phi_persist,
    rho_plunder,
    u,
    volitional_vector,
    volitional_vector_next,
)

# dim4-specific coupling strengths
couple_rho4 = sp.Symbol("couple_rho4", real=True, positive=True)
couple_di4 = sp.Symbol("couple_di4", real=True, positive=True)
couple_delta4 = sp.Symbol("couple_delta4", real=True, positive=True)


def persistence_matrix() -> sp.Matrix:
    """Diagonal persistence matrix Λ = diag(φ_p1, …, φ_p7)."""
    return sp.diag(*phi_persist[:7])


def coupling_vector() -> sp.Matrix:
    """
    Control coupling vector b(i_t, u_t).

    dim4 row: dim4 + couple_rho4·ρ + couple_di4·DI − couple_delta4·δ_coop + u
    """
    d1, d2, d3, d4, d5, d6, d7 = volitional_vector()
    c1, c2, c3, c4, c5, c6, c7 = phi_couple[:7]
    # Λ V + b(i,u): persistence is in Λ; coupling is control-only (no duplicate d4).
    return sp.Matrix([
        c1 * rho_plunder,
        c2 * DI,
        c3 * rho_plunder,
        couple_rho4 * rho_plunder + couple_di4 * DI - couple_delta4 * delta_coop + u,
        c4 * delta_coop,
        c5 * DI,
        c6 * delta_coop + c7 * F_i,
    ])


def phi_rhs() -> sp.Matrix:
    """Φ(V, i, u) right-hand side: Λ V + b(i, u)."""
    return persistence_matrix() * volitional_vector() + coupling_vector()


def phi_transition_equations() -> list[sp.Eq]:
    """V(t+1) = Φ(V(t), i_t, u(t)) as 7 scalar equations."""
    v_next = volitional_vector_next()
    rhs = phi_rhs()
    return [sp.Eq(v_next[i], rhs[i]) for i in range(7)]


def phi_dim4_equation() -> sp.Eq:
    """Scalar dim4 component of Φ."""
    return phi_transition_equations()[3]


@dataclass(frozen=True)
class PhiParameters:
    """Numeric parameters for Φ evaluation."""

    persistence: NDArray[np.float64]  # shape (7,)
    coupling: NDArray[np.float64]  # shape (7,) — cross-dim coupling
    rho4: float = 0.08
    di4: float = 0.03
    delta4: float = 0.02

    @classmethod
    def default(cls) -> PhiParameters:
        # dim4 persistence = 1.0: stress accumulates; coupling drives upward drift.
        return cls(
            persistence=np.array([0.92, 0.90, 0.88, 1.00, 0.89, 0.87, 0.86]),
            coupling=np.array([0.02, 0.01, 0.05, 0.01, 0.01, 0.03, 0.02]),
            rho4=0.40,
            di4=0.12,
            delta4=0.04,
        )


_EVALUATOR_CACHE: dict[tuple, Callable[..., NDArray[np.float64]]] = {}


def build_phi_evaluator(
    params: PhiParameters | None = None,
) -> Callable[..., NDArray[np.float64]]:
    """Build numeric Φ evaluator via sympy.lambdify (cached per parameter set)."""
    p = params or PhiParameters.default()
    cache_key = (
        tuple(p.persistence.tolist()),
        tuple(p.coupling.tolist()),
        p.rho4,
        p.di4,
        p.delta4,
    )
    if cache_key in _EVALUATOR_CACHE:
        return _EVALUATOR_CACHE[cache_key]
    v = volitional_vector()
    rhs = phi_rhs()

    subs: dict[sp.Basic, float] = {}
    for i, sym in enumerate(phi_persist[:7]):
        subs[sym] = float(p.persistence[i])
    for i, sym in enumerate(phi_couple[:7]):
        subs[sym] = float(p.coupling[i])
    subs[couple_rho4] = p.rho4
    subs[couple_di4] = p.di4
    subs[couple_delta4] = p.delta4

    rhs_num = rhs.subs(subs)
    func = sp.lambdify(
        (v, rho_plunder, DI, delta_coop, F_i, u),
        rhs_num,
        modules="numpy",
    )

    def evaluate(
        state: NDArray[np.float64],
        rho: float,
        institutional_drift: float,
        coop_density: float,
        feedback: float,
        intervention: float,
    ) -> NDArray[np.float64]:
        result = func(
            sp.Matrix(state),
            rho,
            institutional_drift,
            coop_density,
            feedback,
            intervention,
        )
        return np.asarray(result, dtype=np.float64).flatten()

    _EVALUATOR_CACHE[cache_key] = evaluate
    return evaluate


def evaluate_phi(
    state: NDArray[np.float64],
    *,
    rho_plunder: float = 0.0,
    institutional_drift: float = 0.0,
    cooperation_density: float = 0.5,
    feedback: float = 0.0,
    intervention: float = 0.0,
    params: PhiParameters | None = None,
) -> NDArray[np.float64]:
    """Evaluate V(t+1) = Φ(V(t), i_t, u(t)) numerically."""
    return build_phi_evaluator(params)(
        state,
        rho_plunder,
        institutional_drift,
        cooperation_density,
        feedback,
        intervention,
    )