"""MCMC sampler for dim4 → TFR fertility mapping."""

from __future__ import annotations

from dataclasses import dataclass

import emcee
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volition.data.dim4 import load_countries, load_countries_full
from volition.mcmc.models import ModelKind, initial_guess, log_posterior, n_params, predict


@dataclass(frozen=True)
class PosteriorSummary:
    """Posterior percentiles for model parameters."""

    median: NDArray[np.float64]
    p16: NDArray[np.float64]
    p84: NDArray[np.float64]
    param_names: tuple[str, ...]


@dataclass
class MCMCResult:
    """Full MCMC inference result."""

    kind: ModelKind
    summary: PosteriorSummary
    samples: NDArray[np.float64]
    acceptance_fraction: float
    pearson_r: float
    r_squared: float
    n_obs: int
    n_walkers: int
    n_steps: int
    burn_in: int
    seed: int

    def predict(self, dim4: NDArray[np.float64]) -> NDArray[np.float64]:
        """Posterior median prediction."""
        return predict(dim4, self.summary.median, self.kind)

    @property
    def alpha(self) -> float:
        return float(self.summary.median[0])

    @property
    def beta(self) -> float:
        return float(self.summary.median[1])

    @property
    def sigma(self) -> float:
        return float(np.exp(self.summary.median[-1]))

    @property
    def gamma(self) -> float | None:
        if self.kind == ModelKind.QUADRATIC:
            return float(self.summary.median[2])
        return None


# Backward-compatible alias
@dataclass
class FertilityModelParams:
    """Posterior point estimate for linear fertility model."""

    alpha: float
    beta: float
    sigma: float
    pearson_r: float
    r_squared: float

    @classmethod
    def from_mcmc(cls, result: MCMCResult) -> FertilityModelParams:
        return cls(
            alpha=result.alpha,
            beta=result.beta,
            sigma=result.sigma,
            pearson_r=result.pearson_r,
            r_squared=result.r_squared,
        )


def _param_names(kind: ModelKind) -> tuple[str, ...]:
    if kind == ModelKind.LINEAR:
        return ("alpha", "beta", "log_sigma")
    return ("alpha", "beta", "gamma", "log_sigma")


def _compute_fit_stats(
    dim4: NDArray[np.float64],
    tfr: NDArray[np.float64],
    theta: NDArray[np.float64],
    kind: ModelKind,
) -> tuple[float, float]:
    mu = predict(dim4, theta, kind)
    r = float(np.corrcoef(dim4, tfr)[0, 1])
    ss_res = float(np.sum((tfr - mu) ** 2))
    ss_tot = float(np.sum((tfr - tfr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r, r_squared


def run_mcmc(
    df: pd.DataFrame | None = None,
    *,
    kind: ModelKind = ModelKind.LINEAR,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int = 200,
    seed: int = 42,
    use_full_dataset: bool = False,
) -> MCMCResult:
    """
    Fit TFR = f(dim4) via emcee MCMC.

    Default seed ensures reproducibility across runs.
    """
    if df is None:
        df = load_countries_full() if use_full_dataset else load_countries()

    dim4 = df["dim4"].to_numpy(dtype=np.float64)
    tfr = df["tfr_future"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_dim = n_params(kind)
    p0 = initial_guess(kind) + 1e-3 * rng.standard_normal((n_walkers, n_dim))

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        log_posterior,
        args=(dim4, tfr, kind),
    )
    sampler.run_mcmc(p0, n_steps, progress=False)

    chain = sampler.get_chain(discard=burn_in, flat=True)
    median = np.median(chain, axis=0)
    p16 = np.percentile(chain, 16, axis=0)
    p84 = np.percentile(chain, 84, axis=0)

    r, r_squared = _compute_fit_stats(dim4, tfr, median, kind)

    return MCMCResult(
        kind=kind,
        summary=PosteriorSummary(
            median=median,
            p16=p16,
            p84=p84,
            param_names=_param_names(kind),
        ),
        samples=chain,
        acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
        pearson_r=r,
        r_squared=r_squared,
        n_obs=len(df),
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in,
        seed=seed,
    )


def fit_fertility_model(
    df: pd.DataFrame | None = None,
    *,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int = 200,
    seed: int = 42,
) -> FertilityModelParams:
    """Backward-compatible wrapper returning linear model point estimates."""
    result = run_mcmc(
        df,
        kind=ModelKind.LINEAR,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in,
        seed=seed,
    )
    return FertilityModelParams.from_mcmc(result)