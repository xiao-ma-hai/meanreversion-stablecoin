import numpy as np
from scipy.integrate import quad

from meanrev_stablecoin.models.cir import cir_logpdf


def test_cir_density_integrates_to_one():
    previous = np.array([1.0])
    delta = np.array([0.1])
    theta, kappa, sigma = 1.0, 2.0, 0.4
    density = lambda z: float(np.exp(cir_logpdf(np.array([z]), previous, delta, theta, kappa, sigma))[0])
    integral, _ = quad(density, 0, 5, epsabs=1e-8)
    assert abs(integral - 1) < 1e-6

