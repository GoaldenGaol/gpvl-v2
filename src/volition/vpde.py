"""Volitional PDE (VPDE) system — 1D numerical solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from volition.constants import DIM4_COOP_COLLAPSE, VPDE_DIFFUSION_DEFAULT


@dataclass
class VPDEConfig:
    """Configuration for 1D VPDE integration."""

    diffusion: float = VPDE_DIFFUSION_DEFAULT
    coop_threshold: float = DIM4_COOP_COLLAPSE
    t_span: tuple[float, float] = (0.0, 50.0)
    n_eval: int = 200


def _vpde_rhs(
    t: float,
    y: NDArray[np.float64],
    diffusion: float,
    coop_threshold: float,
) -> NDArray[np.float64]:
    """
    1D VPDE: dV/dt = -k*(V - V*) + D * d²V/dx² (spatial term collapsed to noise).

    For Phase 0, we solve the scalar dim4 evolution with logistic drift toward
    the cooperation-collapse threshold.
    """
    dim4_val = y[0]
    k = 0.05
    drift = -k * (dim4_val - coop_threshold)
    diffusion_term = diffusion * np.sin(t) * 0.01  # placeholder fluctuation
    return np.array([drift + diffusion_term])


def solve_vpde(
    dim4_initial: float,
    config: VPDEConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Integrate dim4(t) under the 1D VPDE.

    Returns (t_eval, dim4_trajectory).
    """
    cfg = config or VPDEConfig()
    sol = solve_ivp(
        _vpde_rhs,
        cfg.t_span,
        np.array([dim4_initial]),
        args=(cfg.diffusion, cfg.coop_threshold),
        t_eval=np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.n_eval),
        method="RK45",
    )
    return sol.t, sol.y[0]