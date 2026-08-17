import numpy as np

from meanrev_stablecoin.estimation.bootstrap import matched_joint_mou_lr_bootstrap
from meanrev_stablecoin.models.ou import simulate_ou


def test_matched_joint_bootstrap_uses_identical_design():
    rng = np.random.default_rng(19)
    path = simulate_ou(901, 1 / 288, 0.0, 3.0, 0.01, rng)
    summary, draws, observed = matched_joint_mou_lr_bootstrap(
        path[:-1], path[1:], replications=2, matched_pairs=600,
        multistart=2, parallel_jobs=1, seed=23,
    )
    assert summary["matched_pairs"] == 600
    assert summary["same_sample_size_observed_and_bootstrap"]
    assert summary["same_joint_alternative_fit_observed_and_bootstrap"]
    assert len(draws) == 2
    assert observed["lr"] >= 0
