"""Volitional PDE (VPDE) system — Phi-coupled continuous dynamics."""

from volition.vpde.calibration import (
    CalibrationResult,
    CalibrationTarget,
    USA_TARGET,
    calibrate_tau,
    default_calibrated_config,
    get_calibrated_tau,
    validate_calibration,
)
from volition.vpde.controls import (
    ELEVATED_PLUNDER_CONTROLS,
    GOALDEN_CONTROLS,
    RampControls,
    VPDEControls,
)
from volition.vpde.solver import (
    VPDEConfig,
    crossing_time,
    solve_vpde,
    solve_vpde_vector,
)

__all__ = [
    "CalibrationResult",
    "CalibrationTarget",
    "ELEVATED_PLUNDER_CONTROLS",
    "GOALDEN_CONTROLS",
    "RampControls",
    "USA_TARGET",
    "VPDEConfig",
    "VPDEControls",
    "calibrate_tau",
    "crossing_time",
    "default_calibrated_config",
    "get_calibrated_tau",
    "solve_vpde",
    "solve_vpde_vector",
    "validate_calibration",
]