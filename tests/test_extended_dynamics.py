import numpy as np

from meanrev_stablecoin.models.extended_dynamics import (
    jump_ou_approx_logpdf,
    nested_ou_pair_logpdf,
    nested_ou_rho,
    seasonal_integrated_intensity,
    seasonal_mou_logpdf,
    time_varying_gaussian_logpdf,
)
from meanrev_stablecoin.models.mixed_ou import mixed_ou_logpdf
from meanrev_stablecoin.models.ou import ou_logpdf


def test_jump_ou_nests_ou_at_zero_intensity():
    xp = np.linspace(-0.01, 0.01, 100)
    xn = xp + 0.0001
    h = np.full(len(xp), 1 / 288)
    base = ou_logpdf(xn, xp, h, 0.0, 1.2, 0.002)
    jump = jump_ou_approx_logpdf(xn, xp, h, 0.0, 1.2, 0.002, 0.0, 0.0, 0.01)
    assert np.allclose(base, jump)


def test_seasonal_intensity_is_nonnegative_and_constant_case_matches_mou():
    tp = np.linspace(0, 700, 100)
    tn = tp + 1 / 288
    integrated = seasonal_integrated_intensity(tp, tn, 0.7, 0.5, -0.2)
    assert np.all(integrated >= 0)
    xp = np.linspace(-0.01, 0.01, len(tp)); xn = xp + 0.0001; h = tn - tp
    seasonal = seasonal_mou_logpdf(xn, xp, tn, tp, h, 0.0, 1.2, 0.002, 0.7, 0.0, 0.0)
    constant = mixed_ou_logpdf(xn, xp, h, 0.0, 1.2, 0.002, 0.7)
    assert np.allclose(seasonal, constant)


def test_nested_ou_nests_single_ou_when_rates_equal_in_limit():
    h = np.array([1 / 288, 1 / 24, 1.0])
    rho = nested_ou_rho(h, 1.2, 1.2, 0.4)
    assert np.allclose(rho, np.exp(-1.2 * h))
    xp = np.linspace(-0.01, 0.01, 100); xn = xp + 0.0001
    ll = nested_ou_pair_logpdf(xn, xp, np.full(100, 1 / 288), 0.0, 1e-5, 0.4, 200, 0.01)
    assert np.isfinite(ll).all()


def test_time_varying_gaussian_reduces_to_stationary_gaussian_transition():
    xp = np.linspace(-0.01, 0.01, 100); xn = xp + 0.0001
    tp = np.linspace(0, 100, len(xp)); tn = tp + 1 / 288; h = tn - tp
    params = {"mean0": 0.0, "mean_trend": 0.0, "mean_sine": 0.0, "mean_cosine": 0.0,
              "log_scale0": np.log(0.01), "log_scale_trend": 0.0, "kappa": 1.2, "time_center_days": 50.0}
    tv = time_varying_gaussian_logpdf(xn, xp, tn, tp, h, params)
    sigma = np.sqrt(2 * 1.2) * 0.01
    stationary = ou_logpdf(xn, xp, h, 0.0, 1.2, sigma)
    assert np.allclose(tv, stationary)

