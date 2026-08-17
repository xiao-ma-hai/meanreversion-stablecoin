import numpy as np

from meanrev_stablecoin.models.extended_copulas import (
    clayton_copula_logpdf,
    family_logpdf,
    frank_copula_logpdf,
    frank_tau,
    gumbel_copula_logpdf,
    mixed_family_logpdf,
    skew_t_cdf,
    skew_t_logpdf,
    student_t_copula_logpdf,
)


def _integral_on_midpoint_grid(logpdf, n=350):
    grid = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(grid, grid, indexing="ij")
    density = np.exp(logpdf(u.ravel(), v.ravel()))
    return float(np.mean(density))


def test_extended_copula_densities_integrate_to_one():
    assert abs(_integral_on_midpoint_grid(lambda u, v: clayton_copula_logpdf(u, v, 1.2)) - 1) < 0.02
    assert abs(_integral_on_midpoint_grid(lambda u, v: gumbel_copula_logpdf(u, v, 1.5)) - 1) < 0.02
    assert abs(_integral_on_midpoint_grid(lambda u, v: frank_copula_logpdf(u, v, 2.0)) - 1) < 0.02
    assert abs(_integral_on_midpoint_grid(lambda u, v: student_t_copula_logpdf(u, v, 0.4, 6.0)) - 1) < 0.03


def test_survival_copula_is_reflection():
    u = np.linspace(0.02, 0.98, 100)
    v = np.linspace(0.97, 0.01, 100)
    survival = family_logpdf("SurvivalClayton", u, v, 0.4)
    reflected = family_logpdf("Clayton", 1 - u, 1 - v, 0.4)
    assert np.allclose(survival, reflected)


def test_mixed_family_nests_base_at_zero_lambda():
    u = np.linspace(0.01, 0.99, 200)
    v = np.linspace(0.99, 0.01, 200)
    base = family_logpdf("Gumbel", u, v, np.exp(-0.8 / 24))
    mixed = mixed_family_logpdf("Gumbel", u, v, 0.8, 0.0, 1 / 24)
    assert np.allclose(base, mixed)


def test_skew_t_is_symmetric_at_unit_skew_and_cdf_monotone():
    x = np.linspace(-8, 8, 5000)
    logpdf = skew_t_logpdf(x, df=7.0, skew=1.0, loc=0.0, scale=1.0)
    assert np.allclose(logpdf, logpdf[::-1])
    cdf = skew_t_cdf(x, df=7.0, skew=1.0, loc=0.0, scale=1.0)
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[0] < 0.01 and cdf[-1] > 0.99


def test_frank_tau_is_increasing():
    values = [frank_tau(theta) for theta in (0.01, 0.1, 1.0, 10.0, 100.0)]
    assert all(0 < value < 1 for value in values)
    assert np.all(np.diff(values) > 0)

