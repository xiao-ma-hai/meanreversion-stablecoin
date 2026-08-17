import numpy as np

from meanrev_stablecoin.transforms import affine_forward, affine_inverse, affine_inverse_jacobian


def test_affine_forward_inverse_accuracy():
    r = np.linspace(-2, 2, 101)
    a = np.exp(np.linspace(-0.2, 0.2, 101))
    b = np.linspace(-0.1, 0.1, 101)
    recovered = affine_inverse(affine_forward(r, a, b), a, b)
    assert np.max(np.abs(recovered - r)) < 1e-10


def test_affine_jacobian_matches_finite_difference():
    a = np.array([0.7, 1.2, 2.3])
    y = np.array([-0.1, 0.0, 0.2])
    b = np.array([0.1, 0.1, 0.1])
    eps = 1e-6
    numerical = (affine_inverse(y + eps, a, b) - affine_inverse(y - eps, a, b)) / (2 * eps)
    assert np.allclose(numerical, affine_inverse_jacobian(a), atol=1e-10)

