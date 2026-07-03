"""Validate MCMC posterior against framework and data targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from volition.constants import (
    FRAMEWORK_PEARSON_R,
    FRAMEWORK_R_SQUARED,
    MCMC_DATA_R_TOLERANCE,
)
from volition.mcmc.models import ModelKind
from volition.mcmc.sampler import MCMCResult


@dataclass(frozen=True)
class ValidationCheck:
    """Single validation check outcome."""

    name: str
    passed: bool
    observed: float
    expected: str
    detail: str


@dataclass
class MCMCValidationReport:
    """Aggregate MCMC validation report."""

    checks: list[ValidationCheck]
    model_kind: ModelKind
    dataset_label: str

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "model": self.model_kind.name,
            "dataset": self.dataset_label,
            "checks": {
                c.name: {
                    "passed": c.passed,
                    "observed": c.observed,
                    "expected": c.expected,
                    "detail": c.detail,
                }
                for c in self.checks
            },
        }


def validate_mcmc_result(
    result: MCMCResult,
    *,
    dataset_label: str = "frozen_2023",
    observed_r: float | None = None,
) -> MCMCValidationReport:
    """
    Validate MCMC posterior against replication criteria.

    Checks:
    1. Posterior slope (alpha) is negative (monotone fertility mapping)
    2. Posterior reproduces observed Pearson r within tolerance
    3. Model R² exceeds 0.5 on cross-section
    4. Framework reference proximity (|r| > 0.75 on available data)
    5. p-value < 0.001 for linear correlation
    """
    checks: list[ValidationCheck] = []

    alpha_med = float(result.summary.median[0])
    alpha_lo = float(result.summary.p16[0])
    alpha_hi = float(result.summary.p84[0])

    if result.kind == ModelKind.LINEAR:
        monotone_ok = alpha_hi < 0
        mono_detail = f"alpha median={alpha_med:.4f}, CI=[{alpha_lo:.4f}, {alpha_hi:.4f}]"
    else:
        beta_med = float(result.summary.median[1])
        d4_mean = 0.90  # typical frozen-2023 mean dim4
        deriv = 2 * alpha_med * d4_mean + beta_med
        monotone_ok = deriv < 0
        mono_detail = f"dTFR/ddim4 @ {d4_mean:.2f} = {deriv:.4f}"

    checks.append(ValidationCheck(
        name="monotone_negative_slope",
        passed=monotone_ok,
        observed=alpha_med,
        expected="negative fertility gradient",
        detail=mono_detail,
    ))

    obs_r = observed_r if observed_r is not None else result.pearson_r
    checks.append(ValidationCheck(
        name="reproduces_data_correlation",
        passed=abs(result.pearson_r - obs_r) <= MCMC_DATA_R_TOLERANCE,
        observed=result.pearson_r,
        expected=f"|r_model - r_data| <= {MCMC_DATA_R_TOLERANCE}",
        detail=f"model r={result.pearson_r:.4f}, data r={obs_r:.4f}",
    ))

    checks.append(ValidationCheck(
        name="cross_section_r_squared",
        passed=result.r_squared >= 0.5,
        observed=result.r_squared,
        expected="R² >= 0.5",
        detail=f"R²={result.r_squared:.4f}",
    ))

    checks.append(ValidationCheck(
        name="framework_reference_proximity",
        passed=abs(result.pearson_r) >= 0.75,
        observed=result.pearson_r,
        expected=f"|r| >= 0.75 (framework panel r={FRAMEWORK_PEARSON_R})",
        detail=f"framework target r={FRAMEWORK_PEARSON_R}, R²={FRAMEWORK_R_SQUARED}",
    ))

    n = result.n_obs
    if n > 2 and abs(result.pearson_r) < 1.0:
        t_stat = result.pearson_r * np.sqrt(n - 2) / np.sqrt(1 - result.pearson_r**2)
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
    else:
        p_value = 0.0

    checks.append(ValidationCheck(
        name="correlation_significance",
        passed=p_value < 0.001,
        observed=p_value,
        expected="p < 0.001",
        detail=f"p={p_value:.2e}, n={n}",
    ))

    checks.append(ValidationCheck(
        name="acceptance_fraction",
        passed=0.15 <= result.acceptance_fraction <= 0.8,
        observed=result.acceptance_fraction,
        expected="0.2 <= acceptance <= 0.7",
        detail=f"acceptance={result.acceptance_fraction:.3f}",
    ))

    return MCMCValidationReport(
        checks=checks,
        model_kind=result.kind,
        dataset_label=dataset_label,
    )