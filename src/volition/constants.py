"""Frozen 2023 dim4 thresholds from CORE-FRAMEWORK.md."""

DIM4_COOP_COLLAPSE = 0.92  # cooperation collapses in <40 rounds
DIM4_IRREVERSIBLE = 1.00  # irreversible demographic / institutional band
DIM4_NO_RETURN = 0.70  # no country has returned below this once > 1.0

# Goalden universal threshold coefficient (31-domain backtests)
GOALDEN_THRESHOLD_COEFF = 0.1

# VPDE default parameters (Phase 4 calibration)
VPDE_DIFFUSION_DEFAULT = 0.01
VPDE_LOGISTIC_STEEPNESS = 4.0