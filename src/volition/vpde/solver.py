"""Phi-coupled VPDE solver: dV/dt = (Phi(V) - V) / tau + D*eps(t)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from volition.constants import DIM4_COOP_COLLAPSE, VPDE_DIFFUSION_DEFAULT
from volition.equations.phi import PhiParameters, evaluate_phi
from volition.vpde.controls import ConstantControls, ControlPath, VPDEControls


@dataclass
class VPDEConfig:
    """Configuration for VPDE integration."""

    tau: float = 1.0  # time-scale bridging discrete Phi to continuous ODE
    diffusion: float = VPDE_DIFFUSION_DEFAULT
    coop_threshold: float = DIM4_COOP_COLLAPSE
    t_span: tuple[float, float] = (0.0, 50.0)
    n_eval: int = 200
    phi_params: PhiParameters = field(default_factory=PhiParameters.default)
    dim: int = 1  # 1 = dim4 scalar, 7 = full volitional vector


def _initial_state(dim4_initial: float, *, dim: int = 1) -> NDArray[np.float64]:
    if dim == 1:
        return np.array([dim4_initial], dtype=np.float64)
    state = np.zeros(7, dtype=np.float64)
    state[3] = dim4_initial
    return state


def _vpde_rhs(
    t: float,
    y: NDArray[np.float64],
    tau: float,
    diffusion: float,
    control_path: ControlPath,
    phi_params: PhiParameters,
    dim: int,
) -> NDArray[np.float64]:
    """
    VPDE right-hand side.

    dV/dt = (Phi(V, i(t), u(t)) - V) / tau + D * eps(t)

    eps(t) is a bounded oscillatory shock on dim4 (1D reduction).
    """
    controls = control_path.at(t)

    if dim == 1:
        state = np.zeros(7, dtype=np.float64)
        state[3] = y[0]
    else:
        state = y

    phi_v = evaluate_phi(
        state,
        rho_plunder=controls.rho_plunder,
        institutional_drift=controls.institutional_drift,
        cooperation_density=controls.cooperation_density,
        feedback=controls.feedback,
        intervention=controls.intervention,
        params=phi_params,
    )

    if dim == 1:
        phi_component = phi_v[3]
        drift = (phi_component - y[0]) / tau
        shock = diffusion * np.sin(2.0 * np.pi * t / 10.0) * 0.01
        return np.array([drift + shock])
    drift = (phi_v - y) / tau
    if diffusion > 0:
        noise = diffusion * 0.005 * np.sin(2.0 * np.pi * t / 10.0 + np.arange(7))
        drift = drift + noise
    return drift


def solve_vpde(
    dim4_initial: float,
    config: VPDEConfig | None = None,
    *,
    controls: VPDEControls | ControlPath | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Integrate dim4(t) under the Phi-coupled 1D VPDE.

    Returns (t_eval, dim4_trajectory).
    """
    cfg = config or VPDEConfig(dim=1)
    path: ControlPath
    if controls is None:
        path = ConstantControls()
    elif isinstance(controls, VPDEControls):
        path = ConstantControls(controls)
    else:
        path = controls

    y0 = _initial_state(dim4_initial, dim=1)
    sol = solve_ivp(
        _vpde_rhs,
        cfg.t_span,
        y0,
        args=(cfg.tau, cfg.diffusion, path, cfg.phi_params, 1),
        t_eval=np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.n_eval),
        method="RK45",
        max_step=cfg.tau * 0.5,
    )
    return sol.t, sol.y[0]


def solve_vpde_vector(
    state_initial: NDArray[np.float64],
    config: VPDEConfig | None = None,
    *,
    controls: VPDEControls | ControlPath | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Integrate full 7-D V(t) under the Phi-coupled VPDE.

    Returns (t_eval, V_trajectory) where V_trajectory has shape (7, n_eval).
    """
    if state_initial.shape != (7,):
        raise ValueError(f"Expected initial state shape (7,), got {state_initial.shape}")

    cfg = config or VPDEConfig(dim=7)
    path: ControlPath
    if controls is None:
        path = ConstantControls()
    elif isinstance(controls, VPDEControls):
        path = ConstantControls(controls)
    else:
        path = controls

    sol = solve_ivp(
        _vpde_rhs,
        cfg.t_span,
        state_initial.astype(np.float64),
        args=(cfg.tau, cfg.diffusion, path, cfg.phi_params, 7),
        t_eval=np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.n_eval),
        method="RK45",
        max_step=cfg.tau * 0.5,
    )
    return sol.t, sol.y


def crossing_time(
    dim4_initial: float,
    threshold: float,
    config: VPDEConfig,
    *,
    controls: VPDEControls | ControlPath | None = None,
) -> float | None:
    """Return earliest t where dim4(t) >= threshold, or None if never crossed."""
    t, traj = solve_vpde(dim4_initial, config, controls=controls)
    crossed = np.where(traj >= threshold)[0]
    if len(crossed) == 0:
        return None
    return float(t[crossed[0]])