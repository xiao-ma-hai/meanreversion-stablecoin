import numpy as np
from scipy.stats import gamma

from meanrev_stablecoin.models.gamma_copula import (
    article_gamma_case1_equation,
    audit_article_gamma_shape_condition,
    gaussian_gamma_logpdf,
)


def test_article_gamma_case1_has_no_finite_root_on_prespecified_range():
    audit = audit_article_gamma_shape_condition(points=500)
    assert audit["finite_share"] == 1.0
    assert audit["equation_minimum"] > 0
    assert not audit["finite_root_found"]
    assert article_gamma_case1_equation(1.0) > 0


def test_gaussian_gamma_conditional_density_integrates_to_one():
    shape, scale = 2.5, 0.3
    probability = (np.arange(20000) + 0.5) / 20000
    y = gamma.ppf(probability, a=shape, scale=scale)
    # Integrate in probability space: q(y|x)/f(y) averaged over v.
    log_joint = gaussian_gamma_logpdf(
        y, np.full_like(y, 0.7), np.full_like(y, 1 / 24), shape, scale, 1.2, 0.4
    )
    log_marginal = gamma.logpdf(y, a=shape, scale=scale)
    assert abs(np.mean(np.exp(log_joint - log_marginal)) - 1.0) < 0.01

