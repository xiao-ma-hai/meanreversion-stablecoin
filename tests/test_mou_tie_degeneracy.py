from __future__ import annotations

import numpy as np

from meanrev_stablecoin.models.mixed_ou import mixed_ou_logpdf_tau
from meanrev_stablecoin.models.mou_interval_likelihood import mixed_ou_interval_logprob_tau


def test_continuous_mou_likelihood_degenerates_with_ties() -> None:
    previous_price = np.array([1.0] * 60 + [0.9999] * 20)
    following_price = np.array([1.0] * 50 + [1.0001] * 10 + [0.9999] * 10 + [1.0] * 10)
    previous = np.log(previous_price)
    following = np.log(following_price)
    delta = np.full(len(previous), 1 / 288)
    kappas = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    continuous = np.array([
        np.sum(mixed_ou_logpdf_tau(following, previous, delta, 0.0, 0.001, kappa, 2.0))
        for kappa in kappas
    ])
    assert np.all(np.diff(continuous) > 0)
    assert continuous[-1] - continuous[0] > 100


def test_interval_mou_likelihood_remains_bounded_as_kappa_decreases() -> None:
    previous = np.array([1.0] * 60 + [0.9999] * 20)
    following = np.array([1.0] * 50 + [1.0001] * 10 + [0.9999] * 10 + [1.0] * 10)
    delta = np.full(len(previous), 1 / 288)
    kappas = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    interval = np.array([
        np.sum(mixed_ou_interval_logprob_tau(following, previous, 1e-4, delta, 0.0, 0.001, kappa, 2.0))
        for kappa in kappas
    ])
    assert np.isfinite(interval).all()
    assert interval.max() - interval.min() < 500
