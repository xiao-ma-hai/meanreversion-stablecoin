from __future__ import annotations

import numpy as np
from scipy.stats import norm

from meanrev_stablecoin.models.mou_grid_filter import mou_grid_filter_loglik
from meanrev_stablecoin.models.mou_interval_likelihood import (
    fit_interval_mou_ifm,
    mixed_ou_interval_logprob_tau,
)
from meanrev_stablecoin.observation.rounding import (
    log_price_intervals,
    normal_interval_logprob,
    quantize_prices,
)


def test_normal_interval_probability_matches_direct_cdf_difference() -> None:
    lower = np.array([-2.0, -0.1, 3.0])
    upper = np.array([-1.5, 0.2, 3.1])
    actual = np.exp(normal_interval_logprob(lower, upper, 0.0, 1.0))
    expected = norm.cdf(upper) - norm.cdf(lower)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_tick_endpoint_convention_has_same_continuous_interval() -> None:
    price = np.array([0.9999, 1.0, 1.0001])
    left = log_price_intervals(price, 1e-4, endpoint="left_closed")
    right = log_price_intervals(price, 1e-4, endpoint="right_closed")
    np.testing.assert_allclose(left[0], right[0])
    np.testing.assert_allclose(left[1], right[1])


def test_interval_likelihood_is_finite_with_ties_and_has_numerical_optimum() -> None:
    previous = np.array([1.0] * 80 + [0.9999] * 20)
    following = np.array([1.0] * 70 + [1.0001] * 10 + [0.9999] * 10 + [1.0] * 10)
    delta = np.full(len(previous), 1 / 288)
    values = mixed_ou_interval_logprob_tau(
        following, previous, 1e-4, delta, theta=0.0, tau=0.001, kappa=1.0, lam=2.0
    )
    assert np.isfinite(values).all()
    fit = fit_interval_mou_ifm(
        following,
        previous,
        1e-4,
        delta,
        theta=0.0,
        tau=0.001,
        multistart=4,
    )
    assert fit.converged
    assert np.isfinite(fit.loglik)
    assert fit.params["kappa"] > 0
    assert fit.params["lambda"] >= 0


def test_quantization_creates_ties_and_grid_filter_is_finite() -> None:
    rng = np.random.default_rng(12)
    latent_log = np.cumsum(rng.normal(0, 2e-5, 70))
    observed = quantize_prices(np.exp(latent_log), 1e-4)
    assert np.mean(np.diff(observed) == 0) > 0
    likelihood = mou_grid_filter_loglik(
        observed,
        1e-4,
        1 / 288,
        theta=0.0,
        tau=0.001,
        kappa=2.0,
        lam=1.0,
        grid_size=101,
    )
    assert np.isfinite(likelihood)
