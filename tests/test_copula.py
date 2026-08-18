import numpy as np

from meanrev_stablecoin.models.copula import (
    EmpiricalMarginal,
    gaussian_copula_logpdf,
    rank_pseudo_observations,
    two_scale_mixed_gaussian_copula_logpdf,
    two_scale_rho,
)


def test_midrank_pseudo_observations_handle_ties_without_order_dependence():
    values = np.array([1.0, 2.0, 1.0, 3.0])
    u = rank_pseudo_observations(values)
    assert u[0] == u[2]
    assert np.isclose(u[0], 0.25)
    marginal = EmpiricalMarginal.fit(values)
    assert np.allclose(marginal.cdf(values), u)


def test_gaussian_copula_independence_density_is_one():
    u = np.linspace(0.01, 0.99, 101)
    v = u[::-1]
    assert np.allclose(np.exp(gaussian_copula_logpdf(u, v, 0.0)), 1.0)


def test_two_scale_dependence_and_lambda_zero_nesting():
    horizon = np.array([0.0, 1 / 288, 1.0])
    rho = two_scale_rho(horizon, kappa_slow=0.2, kappa_fast=5.0, weight_fast=0.6)
    assert rho[0] == 1.0
    assert np.all(np.diff(rho) < 0)
    u = np.linspace(0.01, 0.99, 101)
    v = u[::-1]
    h = 1 / 24
    expected = gaussian_copula_logpdf(u, v, float(two_scale_rho(h, 0.2, 5.0, 0.6)))
    actual = two_scale_mixed_gaussian_copula_logpdf(u, v, 0.2, 5.0, 0.6, 0.0, h)
    assert np.allclose(actual, expected)
