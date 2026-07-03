"""Fertility model likelihoods for MCMC inference."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class ModelKind(Enum):
    LINEAR = auto()
    QUADRATIC = auto()


def n_params(kind: ModelKind) -> int:
    return 3 if kind == ModelKind.LINEAR else 4


def predict(dim4: NDArray[np.float64], theta: NDArray[np.float64], kind: ModelKind) -> NDArray[np.float64]:
    if kind == ModelKind.LINEAR:
        alpha, beta, _ = theta
        return alpha * dim4 + beta
    alpha, beta, gamma, _ = theta
    return alpha * dim4**2 + beta * dim4 + gamma


def log_likelihood(
    theta: NDArray[np.float64],
    dim4: NDArray[np.float64],
    tfr: NDArray[np.float64],
    kind: ModelKind,
) -> float:
    log_sigma = theta[-1]
    sigma = np.exp(log_sigma)
    mu = predict(dim4, theta, kind)
    return float(-0.5 * np.sum(((tfr - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma**2)))


def log_prior(theta: NDArray[np.float64], kind: ModelKind) -> float:
    log_sigma = theta[-1]
    if log_sigma < -10 or log_sigma > 5:
        return -np.inf
    if kind == ModelKind.LINEAR:
        return 0.0
    return 0.0


def log_posterior(
    theta: NDArray[np.float64],
    dim4: NDArray[np.float64],
    tfr: NDArray[np.float64],
    kind: ModelKind,
) -> float:
    lp = log_prior(theta, kind)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dim4, tfr, kind)


def initial_guess(kind: ModelKind) -> NDArray[np.float64]:
    if kind == ModelKind.LINEAR:
        return np.array([-2.0, 4.0, 0.0])
    return np.array([-0.5, -1.0, 5.0, 0.0])