import numpy as np

from meanrev_stablecoin.models.mixed_ou import (
    fit_mou_ifm,
    fit_mou_peg,
    fit_mou_uqml,
    mixed_ou_logpdf,
    mixed_ou_logpdf_tau,
    simulate_mixed_ou,
)
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


def test_tau_parameterization_is_exact_reparameterization():
    xp = np.array([-0.001, 0.0, 0.002])
    xn = np.array([0.0003, -0.0002, 0.001])
    delta = np.full(3, 1 / 288)
    theta, tau, kappa, lam = 0.0001, 0.0012, 3.0, 0.8
    sigma = np.sqrt(2 * kappa) * tau
    direct = mixed_ou_logpdf(xn, xp, delta, theta, kappa, sigma, lam)
    reparameterized = mixed_ou_logpdf_tau(xn, xp, delta, theta, tau, kappa, lam)
    assert np.allclose(direct, reparameterized)


def test_margin_constrained_estimators_preserve_declared_margin():
    rng = np.random.default_rng(17)
    x = simulate_mixed_ou(900, 1 / 288, 0.0002, 4.0, 0.004, 0.5, rng)
    xp, xn = x[:-1], x[1:]
    delta = np.full(len(xp), 1 / 288)
    ifm = fit_mou_ifm(xn, xp, delta, x, multistart=3)
    peg = fit_mou_peg(xn, xp, delta, x, multistart=3)
    uqml = fit_mou_uqml(xn, xp, delta, multistart=3)
    assert np.isclose(ifm.params["theta"], np.mean(x))
    assert np.isclose(ifm.params["tau"], np.std(x, ddof=0))
    assert peg.params["theta"] == 0.0
    assert np.isclose(peg.params["tau"], np.sqrt(np.mean(x * x)))
    assert uqml.params["tau"] > 0
