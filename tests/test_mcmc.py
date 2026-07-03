"""Tests for MCMC fertility inference and validation."""

import numpy as np
import pytest

from volition.constants import FRAMEWORK_PEARSON_R, FRAMEWORK_R_SQUARED
from volition.mcmc import ModelKind, fit_fertility_model, run_mcmc, validate_mcmc_result


@pytest.fixture
def fast_mcmc_kwargs():
    return {"n_steps": 300, "burn_in": 80, "n_walkers": 16, "seed": 42}


def test_linear_mcmc_reproducible(fast_mcmc_kwargs):
    r1 = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    r2 = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    assert r1.alpha == pytest.approx(r2.alpha, rel=1e-6)
    assert r1.pearson_r == pytest.approx(r2.pearson_r, rel=1e-6)


def test_linear_negative_slope(fast_mcmc_kwargs):
    result = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    assert result.alpha < 0


def test_quadratic_improves_or_matches_r2(fast_mcmc_kwargs):
    linear = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    quad = run_mcmc(kind=ModelKind.QUADRATIC, **fast_mcmc_kwargs)
    assert quad.r_squared >= linear.r_squared - 0.01


def test_full_dataset_more_countries(fast_mcmc_kwargs):
    snap = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    full = run_mcmc(kind=ModelKind.LINEAR, use_full_dataset=True, **fast_mcmc_kwargs)
    assert full.n_obs == 195
    assert snap.n_obs == 59
    assert abs(full.pearson_r) > abs(snap.pearson_r)


def test_validation_passes_snapshot(fast_mcmc_kwargs):
    result = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    report = validate_mcmc_result(result, dataset_label="snapshot_59")
    assert report.all_passed


def test_validation_passes_full(fast_mcmc_kwargs):
    result = run_mcmc(
        kind=ModelKind.LINEAR,
        use_full_dataset=True,
        **fast_mcmc_kwargs,
    )
    report = validate_mcmc_result(result, dataset_label="full_195")
    assert report.all_passed


def test_predict_shape(fast_mcmc_kwargs):
    result = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    dim4 = np.linspace(0.5, 1.4, 10)
    pred = result.predict(dim4)
    assert pred.shape == (10,)
    assert np.all(pred > 0)


def test_fit_fertility_model_backward_compat(fast_mcmc_kwargs):
    params = fit_fertility_model(**fast_mcmc_kwargs)
    assert params.alpha < 0
    assert params.sigma > 0
    assert params.pearson_r < 0


def test_framework_targets_documented():
    assert FRAMEWORK_PEARSON_R < 0
    assert FRAMEWORK_R_SQUARED > 0.8


def test_mcmc_result_has_samples(fast_mcmc_kwargs):
    result = run_mcmc(kind=ModelKind.LINEAR, **fast_mcmc_kwargs)
    assert result.samples.shape[1] == 3
    assert len(result.samples) > 0