"""VPDE calibration against framework trajectory targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from volition.constants import DIM4_IRREVERSIBLE
from volition.vpde.controls import RampControls, VPDEControls
from volition.vpde.solver import VPDEConfig, crossing_time, solve_vpde


@dataclass(frozen=True)
class CalibrationTarget:
    """Empirical trajectory anchor from CORE-FRAMEWORK forecasts."""

    name: str
    dim4_initial: float
    dim4_target: float
    target_time: float
    controls: RampControls | VPDEControls


@dataclass(frozen=True)
class CalibrationResult:
    """Result of VPDE time-scale calibration."""

    tau: float
    target: CalibrationTarget
    achieved_crossing_time: float | None
    residual: float
    dim4_final: float


# USA crosses +1.0 in 2029±1 from dim4≈0.94 in 2023 → ~6 years
USA_TARGET = CalibrationTarget(
    name="USA_irreversibility",
    dim4_initial=0.94,
    dim4_target=DIM4_IRREVERSIBLE,
    target_time=6.0,
    controls=RampControls(),
)

# South Korea already above irreversibility — should remain elevated
KOREA_TARGET = CalibrationTarget(
    name="South_Korea_stable_irreversible",
    dim4_initial=1.41,
    dim4_target=DIM4_IRREVERSIBLE,
    target_time=0.0,
    controls=VPDEControls(rho_plunder=0.04, institutional_drift=0.012),
)


def _crossing_residual(tau: float, target: CalibrationTarget) -> float:
    cfg = VPDEConfig(
        tau=tau,
        t_span=(0.0, max(target.target_time * 4, 30.0)),
        n_eval=150,
    )
    t_cross = crossing_time(
        target.dim4_initial,
        target.dim4_target,
        cfg,
        controls=target.controls,
    )
    if t_cross is None:
        return 100.0 + tau
    return t_cross - target.target_time


def calibrate_tau(
    target: CalibrationTarget | None = None,
    *,
    tau_bounds: tuple[float, float] = (0.15, 5.0),
) -> CalibrationResult:
    """
    Calibrate VPDE time-scale τ so dim4 crosses target at target_time.

    Uses Brent's method on the crossing-time residual.
    """
    tgt = target or USA_TARGET

    if tgt.target_time <= 0:
        tau = 1.0
        cfg = VPDEConfig(tau=tau, t_span=(0.0, 10.0), n_eval=200)
        _, traj = solve_vpde(tgt.dim4_initial, cfg, controls=tgt.controls)
        return CalibrationResult(
            tau=tau,
            target=tgt,
            achieved_crossing_time=0.0,
            residual=0.0,
            dim4_final=float(traj[-1]),
        )

    lo, hi = tau_bounds
    f_lo = _crossing_residual(lo, tgt)
    f_hi = _crossing_residual(hi, tgt)
    if f_lo * f_hi > 0:
        # Expand upper bound until residual brackets zero
        for hi in (50.0, 80.0, 120.0, 200.0):
            f_hi = _crossing_residual(hi, tgt)
            if f_lo * f_hi <= 0:
                break
        else:
            hi = 200.0

    tau = brentq(
        lambda x: _crossing_residual(x, tgt),
        lo,
        hi,
        xtol=1e-2,
    )

    cfg = VPDEConfig(
        tau=tau,
        t_span=(0.0, max(tgt.target_time * 3, 20.0)),
        n_eval=500,
    )
    achieved = crossing_time(
        tgt.dim4_initial,
        tgt.dim4_target,
        cfg,
        controls=tgt.controls,
    )
    _, traj = solve_vpde(tgt.dim4_initial, cfg, controls=tgt.controls)

    return CalibrationResult(
        tau=float(tau),
        target=tgt,
        achieved_crossing_time=achieved,
        residual=abs((achieved or 0.0) - tgt.target_time),
        dim4_final=float(traj[-1]),
    )


def default_calibrated_config() -> VPDEConfig:
    """Return VPDEConfig with calibrated τ (cached on first call)."""
    return VPDEConfig(tau=get_calibrated_tau())


_CACHED_TAU: float | None = None


def get_calibrated_tau(*, refresh: bool = False) -> float:
    """Return calibrated τ, computing once unless refresh=True."""
    global _CACHED_TAU
    if _CACHED_TAU is None or refresh:
        _CACHED_TAU = calibrate_tau().tau
    return _CACHED_TAU


def validate_calibration(tolerance: float = 0.5) -> list[CalibrationResult]:
    """Run calibration checks for all framework anchors."""
    results = []
    for target in (USA_TARGET, KOREA_TARGET):
        if target.target_time > 0:
            results.append(calibrate_tau(target))
        else:
            cfg = VPDEConfig(tau=1.0, t_span=(0.0, 10.0))
            _, traj = solve_vpde(target.dim4_initial, cfg, controls=target.controls)
            results.append(CalibrationResult(
                tau=1.0,
                target=target,
                achieved_crossing_time=0.0,
                residual=0.0 if traj[-1] >= target.dim4_target else 1.0,
                dim4_final=float(traj[-1]),
            ))
    return results