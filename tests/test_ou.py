import numpy as np

from meanrev_stablecoin.models.ou import fit_ou, ou_logpdf, ou_moments, simulate_ou


def test_ou_exact_density_matches_ar1_formula():
    x_prev = np.array([-0.01, 0.0, 0.02])
    x_next = np.array([-0.008, 0.001, 0.018])
    delta = np.full(3, 1 / 288)
    theta, kappa, sigma = 0.001, 35.0, 0.025
    mean, variance = ou_moments(x_prev, delta, theta, kappa, sigma)
    expected = -0.5 * (np.log(2 * np.pi * variance) + (x_next - mean) ** 2 / variance)
    assert np.allclose(ou_logpdf(x_next, x_prev, delta, theta, kappa, sigma), expected)


def test_ou_simulation_parameter_recovery():
    rng = np.random.default_rng(20260806)
    theta, kappa, sigma = 0.001, 20.0, 0.03
    path = simulate_ou(12_000, 1 / 288, theta, kappa, sigma, rng)
    fit = fit_ou(path[1:], path[:-1], np.full(len(path) - 1, 1 / 288), multistart=4)
    assert abs(fit.params["theta"] - theta) < 0.004
    assert abs(fit.params["kappa"] - kappa) / kappa < 0.35
    assert abs(fit.params["sigma"] - sigma) / sigma < 0.12

