"""Symbolic equation core for the Science of Volition."""

from volition.equations.axioms import (
    all_axioms,
    axiom_8_competence_accumulation,
    check_axiom_5_consent_logic,
    check_axiom_8_numeric,
    check_goalden_cooperative,
)
from volition.equations.dynamics import system_transitions
from volition.equations.fertility import (
    fertility_mapping_linear,
    fertility_mapping_quadratic,
    fertility_slope_sign_constraint,
    goalden_threshold_equation,
)
from volition.equations.invariants import (
    check_all_invariants,
    invariant_a_fertility_mapping,
)
from volition.equations.phi import (
    PhiParameters,
    evaluate_phi,
    phi_dim4_equation,
    phi_rhs,
    phi_transition_equations,
)

__all__ = [
    "PhiParameters",
    "all_axioms",
    "axiom_8_competence_accumulation",
    "check_all_invariants",
    "check_axiom_5_consent_logic",
    "check_axiom_8_numeric",
    "check_goalden_cooperative",
    "evaluate_phi",
    "fertility_mapping_linear",
    "fertility_mapping_quadratic",
    "fertility_slope_sign_constraint",
    "goalden_threshold_equation",
    "invariant_a_fertility_mapping",
    "phi_dim4_equation",
    "phi_rhs",
    "phi_transition_equations",
    "system_transitions",
]


def export_equations_latex() -> dict[str, str]:
    """Return LaTeX strings for all symbolic equation groups."""
    import sympy as sp

    from volition.equations.axioms import all_axioms
    from volition.equations.dynamics import system_transitions
    from volition.regime import RegimeTransitionMatrix

    axioms = all_axioms()
    phi_eqs = phi_transition_equations()
    dyn_eqs = system_transitions()

    return {
        "fertility_linear": sp.latex(fertility_mapping_linear()),
        "fertility_quadratic": sp.latex(fertility_mapping_quadratic()),
        "fertility_monotone": sp.latex(fertility_slope_sign_constraint()),
        "goalden_threshold": sp.latex(goalden_threshold_equation()),
        "phi_dim4": sp.latex(phi_dim4_equation()),
        "phi_system": sp.latex(phi_eqs),
        "dynamics": sp.latex(dyn_eqs),
        "axiom_8": sp.latex(axiom_8_competence_accumulation()),
        "axiom_6": sp.latex(axioms["axiom_6_penalty"]),
        "regime_theta": RegimeTransitionMatrix.default().as_latex(),
        "stability": sp.latex(
            __import__("volition.equations.dynamics", fromlist=["stability_condition"])
            .stability_condition()
        ),
    }


# Backward-compatible alias
def transition_map_rhs():
    """Legacy alias: dim4 component of Φ."""
    return phi_dim4_equation().rhs