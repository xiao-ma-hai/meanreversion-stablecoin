import numpy as np

from meanrev_stablecoin.models.nonlinear_transforms import (
    cir_special_forward_and_derivative,
    exponential_cir_coefficients,
    exponential_cir_inverse,
    exponential_ou_coefficients,
    exponential_ou_inverse,
    ou_special_monotonicity_diagnostic,
    quadratic_cir_coefficients,
    quadratic_cir_inverse,
)


def test_exponential_ou_inverse_and_jacobian():
    t = np.linspace(0, 0.2, 101)
    params = dict(kappa=0.4, sigma=0.2, d=0.6, a0=0.7, b0=-0.1, c0=0.3)
    a, b, c, clipped = exponential_ou_coefficients(t, **params)
    latent = np.linspace(-0.5, 0.5, len(t))
    y = np.exp(a * latent + b) + c
    evaluated = exponential_ou_inverse(y, t, **params)
    assert not clipped.any()
    assert evaluated.valid.all()
    assert np.allclose(evaluated.latent, latent, atol=1e-11)
    eps = 1e-6
    plus = exponential_ou_inverse(y + eps, t, **params).latent
    minus = exponential_ou_inverse(y - eps, t, **params).latent
    numerical = (plus - minus) / (2 * eps)
    assert np.allclose(np.log(numerical), evaluated.log_abs_inverse_jacobian, atol=1e-7)


def test_ou_special_is_not_globally_monotone():
    diagnostic = ou_special_monotonicity_diagnostic(kappa=1.0, sigma=0.2, c=2.0, d=0.5)
    assert not diagnostic["global_monotonicity_pass"]
    assert diagnostic["positive_derivative_share"] > 0
    assert diagnostic["negative_derivative_share"] > 0


def test_quadratic_cir_inverse_and_jacobian_in_admissible_case():
    # At t=0 the article coefficients equal A0, B0, C0 exactly.
    t = np.zeros(101)
    params = dict(theta=1.0, kappa=0.8, sigma=0.2, d=2.0, a0=0.4, b0=0.3, c0=0.1)
    a, b, c, _ = quadratic_cir_coefficients(t, **params)
    latent = np.linspace(0.01, 2.0, len(t))
    y = a * latent**2 + b * latent + c
    evaluated = quadratic_cir_inverse(y, t, **params)
    assert evaluated.valid.all()
    assert np.allclose(evaluated.latent, latent, atol=1e-11)
    assert np.allclose(np.exp(evaluated.log_abs_inverse_jacobian), 1 / (2 * a * latent + b))


def test_exponential_cir_inverse_in_admissible_case():
    t = np.linspace(0, 0.1, 101)
    params = dict(theta=1.0, kappa=0.8, sigma=0.3, d=0.7, a0=0.5, b0=-0.2, c0=0.1)
    a, b, c, invalid = exponential_cir_coefficients(t, **params)
    latent = np.linspace(0.1, 1.5, len(t))
    y = np.exp(a * latent + b) + c
    evaluated = exponential_cir_inverse(y, t, **params)
    assert not invalid.any()
    assert evaluated.valid.all()
    assert np.allclose(evaluated.latent, latent, atol=1e-10)


def test_cir_special_derivative_matches_finite_difference():
    latent = np.linspace(0.1, 0.5, 30)
    t = np.zeros_like(latent)
    args = dict(theta=1.0, kappa=0.7, sigma=0.5, c=1.2, d=0.2, c1=0.4, c2=0.0)
    value, derivative = cir_special_forward_and_derivative(latent, t, **args)
    eps = 1e-6
    plus, _ = cir_special_forward_and_derivative(latent + eps, t, **args)
    minus, _ = cir_special_forward_and_derivative(latent - eps, t, **args)
    assert np.isfinite(value).all()
    assert np.allclose(derivative, (plus - minus) / (2 * eps), rtol=1e-5, atol=1e-6)

