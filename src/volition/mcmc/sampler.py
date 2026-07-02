"""MCMC sampler for dim4 → TFR fertility mapping (Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass

import emcee
import numpy as np
import pandas as pd

from volition.data.dim4 import load_countries


@dataclass
class FertilityModelParams:
    """Posterior point estimate for linear fertility model."""

    alpha: float
    beta: float
    sigma: float
    pearson_r: float
    r_squared: float


def _log_likelihood(
    theta: np.ndarray,
    dim4: np.ndarray,
    tfr: np.ndarray,
) -> float:
    alpha, beta, log_sigma = theta
    sigma = np.exp(log_sigma)
    mu = alpha * dim4 + beta
    return float(-0.5 * np.sum(((tfr - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma**2)))


def _log_prior(theta: np.ndarray) -> float:
    alpha, beta, log_sigma = theta
    if log_sigma < -10 or log_sigma > 5:
        return -np.inf
    return 0.0


def _log_posterior(
    theta: np.ndarray,
    dim4: np.ndarray,
    tfr: np.ndarray,
) -> float:
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta, dim4, tfr)


def fit_fertility_model(
    df: pd.DataFrame | None = None,
    *,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int = 200,
    seed: int = 42,
) -> FertilityModelParams:
    """
    Fit TFR = alpha * dim4 + beta via emcee MCMC.

    Default seed ensures reproducibility across runs.
    """
    if df is None:
        df = load_countries()

    dim4 = df["dim4"].to_numpy(dtype=np.float64)
    tfr = df["tfr_future"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_dim = 3
    p0 = np.array([-2.0, 4.0, 0.0]) + 1e-3 * rng.standard_normal((n_walkers, n_dim))

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        _log_posterior,
        args=(dim4, tfr),
    )
    sampler.run_mcmc(p0, n_steps, progress=False)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    alpha, beta, log_sigma = np.median(samples, axis=0)
    sigma = float(np.exp(log_sigma))

    mu = alpha * dim4 + beta
    r = float(np.corrcoef(dim4, tfr)[0, 1])

    return FertilityModelParams(
        alpha=float(alpha),
        beta=float(beta),
        sigma=sigma,
        pearson_r=r,
        r_squared=r**2,
    )