import numpy as np

from meanrev_stablecoin.models.copula import (
    gaussian_copula_logpdf,
    two_scale_mixed_gaussian_copula_logpdf,
    two_scale_rho,
)


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
