import numpy as np

from meanrev_stablecoin.models.copula import EmpiricalMarginal


def test_test_mapping_uses_training_distribution_only():
    train = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    test = np.array([100.0, 101.0])
    marginal = EmpiricalMarginal.fit(train)
    mapped = marginal.cdf(test)
    assert np.all(mapped >= 0.9)
    reranked_test = EmpiricalMarginal.fit(test).cdf(test)
    assert not np.allclose(mapped, reranked_test)
