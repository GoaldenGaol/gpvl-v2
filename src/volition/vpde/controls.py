"""Control inputs i(t) and u(t) for VPDE integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VPDEControls:
    """Frozen control vector at a point in time."""

    rho_plunder: float = 0.0312
    institutional_drift: float = 0.01
    cooperation_density: float = 0.5
    feedback: float = 0.0
    intervention: float = 0.0


class ControlPath(Protocol):
    """Protocol for time-varying controls."""

    def at(self, t: float) -> VPDEControls: ...


@dataclass(frozen=True)
class ConstantControls:
    """Constant control path."""

    controls: VPDEControls = VPDEControls()

    def at(self, t: float) -> VPDEControls:
        return self.controls


@dataclass(frozen=True)
class RampControls:
    """
    Linear ramp on plunder and institutional drift.

    Models slowly deteriorating institutional environment (USA 2023→2029).
    """

    rho_start: float = 0.0312
    rho_end: float = 0.050
    di_start: float = 0.008
    di_end: float = 0.022
    t_end: float = 6.0
    cooperation_density: float = 0.35
    feedback: float = 0.0
    intervention: float = 0.0

    def at(self, t: float) -> VPDEControls:
        if self.t_end <= 0:
            frac = 1.0
        else:
            frac = min(max(t / self.t_end, 0.0), 1.0)
        return VPDEControls(
            rho_plunder=self.rho_start + frac * (self.rho_end - self.rho_start),
            institutional_drift=self.di_start + frac * (self.di_end - self.di_start),
            cooperation_density=self.cooperation_density,
            feedback=self.feedback,
            intervention=self.intervention,
        )


# Framework defaults
GOALDEN_CONTROLS = VPDEControls(rho_plunder=0.0312)
ELEVATED_PLUNDER_CONTROLS = VPDEControls(rho_plunder=0.045, institutional_drift=0.015)
USA_FORECAST_CONTROLS = RampControls()