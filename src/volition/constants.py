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

# Framework fertility mapping targets (CORE-FRAMEWORK Section 3, panel 1950–2023)
FRAMEWORK_PEARSON_R = -0.934
FRAMEWORK_R_SQUARED = 0.871
FRAMEWORK_P_VALUE_MAX = 1e-13

# Panel replication (Phase 10)
PANEL_YEAR_START = 1950
PANEL_YEAR_END = 2023
PANEL_LEAD_YEARS = 18
PANEL_TRAIN_END = 2005
PANEL_TEST_START = 2006
PANEL_R_TOLERANCE = 0.01
PANEL_R2_TOLERANCE = 0.02

# MCMC validation tolerances (cross-section replication)
MCMC_SLOPE_CI_LEVEL = 0.95
MCMC_DATA_R_TOLERANCE = 0.05

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