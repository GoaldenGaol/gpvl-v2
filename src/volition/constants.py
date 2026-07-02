"""Frozen 2023 dim4 thresholds from CORE-FRAMEWORK.md."""

DIM4_COOP_COLLAPSE = 0.92  # cooperation collapses in <40 rounds
DIM4_IRREVERSIBLE = 1.00  # irreversible demographic / institutional band
DIM4_NO_RETURN = 0.70  # no country has returned below this once > 1.0

# TFR regime bands (UNIFIED-THEORY.md L2)
DIM4_BAND_HIGH_TFR = 0.70  # <= 0.70: high TFR (~4.2)
DIM4_BAND_MID_TFR = 1.00  # 0.70–1.00: mid TFR (~2.0); >1.00: low TFR (~1.34)

# Firm-level fatal pivot threshold (UNIFIED-THEORY.md)
DIM4_FIRM_HIGH_RISK = 0.92

# Goalden universal threshold coefficient (31-domain backtests)
GOALDEN_THRESHOLD_COEFF = 0.1

# VPDE default parameters (Phase 4 calibration)
VPDE_DIFFUSION_DEFAULT = 0.01
VPDE_LOGISTIC_STEEPNESS = 4.0

# Default regime transition matrix Θ (rows = from, cols = to)
# COOPERATIVE → PRE_COLLAPSE drift; IRREVERSIBLE is absorbing.
DEFAULT_THETA = (
    (0.85, 0.14, 0.01),  # from COOPERATIVE
    (0.05, 0.70, 0.25),  # from PRE_COLLAPSE
    (0.00, 0.00, 1.00),  # from IRREVERSIBLE (absorbing)
)