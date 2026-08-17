import numpy as np

from meanrev_stablecoin.models.mixed_ou import mixed_ou_logpdf, simulate_mixed_ou
from meanrev_stablecoin.models.ou import ou_logpdf


def test_mou_lambda_zero_equals_ou_pointwise():
    rng = np.random.default_rng(7)
    x_prev = rng.normal(size=100) * 0.01
    x_next = rng.normal(size=100) * 0.01
    delta = np.full(100, 1 / 288)
    args = (0.0, 30.0, 0.02)
    assert np.array_equal(mixed_ou_logpdf(x_next, x_prev, delta, *args, 0.0), ou_logpdf(x_next, x_prev, delta, *args))


def test_mou_simulated_conditional_mean():
    theta, kappa, sigma, lam = 0.001, 20.0, 0.03, 8.0
    delta, x0 = 1 / 288, 0.02
    draws = []
    for seed in range(20_000):
        rng = np.random.default_rng(seed)
        draws.append(simulate_mixed_ou(2, delta, theta, kappa, sigma, lam, rng, x0=x0)[1])
    expected = theta + np.exp(-(kappa + lam) * delta) * (x0 - theta)
    assert abs(np.mean(draws) - expected) < 4e-4

