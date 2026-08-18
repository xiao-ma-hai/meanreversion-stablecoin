import numpy as np

from meanrev_stablecoin.models.latent_noise_ou import latent_noise_ou_loglik


def test_latent_noise_likelihood_is_finite_and_prefers_reasonable_center():
    rng = np.random.default_rng(91)
    y = 0.0002 + 0.001 * rng.standard_normal(80)
    delta = np.full(len(y) - 1, 1 / 288)
    centered = latent_noise_ou_loglik(y, delta, 0.0002, 0.001, 3.0, 0.0002)
    displaced = latent_noise_ou_loglik(y, delta, 0.1, 0.001, 3.0, 0.0002)
    assert np.isfinite(centered)
    assert centered > displaced
