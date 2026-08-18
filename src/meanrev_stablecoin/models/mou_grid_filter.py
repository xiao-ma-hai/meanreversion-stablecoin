from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ..observation.rounding import log_price_intervals


def _normal_bin_probabilities(edges: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    upper = ndtr((edges[1:][None, :] - mean[:, None]) / sd[:, None])
    lower = ndtr((edges[:-1][None, :] - mean[:, None]) / sd[:, None])
    return np.maximum(upper - lower, 0.0)


def mou_grid_filter_loglik(
    observed_prices: np.ndarray,
    ticks: float | np.ndarray,
    delta_days: float | np.ndarray,
    *,
    theta: float,
    tau: float,
    kappa: float,
    lam: float,
    grid_size: int = 301,
) -> float:
    """Approximate the full rounded-observation likelihood by grid filtering.

    Cell masses are propagated with exact Gaussian interval probabilities.
    Observation-bin overlap fractions provide a deterministic discretization
    of the rounding indicator.  This implementation is intended for validation
    samples; full-data production fits require sparse/adaptive acceleration.
    """

    price = np.asarray(observed_prices, dtype=float)
    if len(price) < 2 or grid_size < 31:
        raise ValueError("the grid filter requires at least two prices and grid_size >= 31")
    tick_values = np.broadcast_to(np.asarray(ticks, dtype=float), price.shape)
    deltas = np.broadcast_to(np.asarray(delta_days, dtype=float), (len(price) - 1,))
    if tau <= 0 or kappa <= 0 or lam < 0 or np.any(deltas <= 0):
        return -np.inf
    lower_obs, upper_obs = log_price_intervals(price, tick_values)
    lower = min(float(np.min(lower_obs)), theta - 8 * tau)
    upper = max(float(np.max(upper_obs)), theta + 8 * tau)
    edges = np.linspace(lower, upper, grid_size + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    stationary = np.diff(ndtr((edges - theta) / tau))
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()

    def emission(index: int) -> np.ndarray:
        overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], upper_obs[index]) - np.maximum(edges[:-1], lower_obs[index]),
        )
        return overlap / widths

    weights = stationary * emission(0)
    first = float(weights.sum())
    if first <= 0:
        return -np.inf
    weights /= first
    loglik = np.log(first)

    for index, delta in enumerate(deltas, start=1):
        rho = np.exp(-kappa * delta)
        base_mean = theta + rho * (centers - theta)
        base_sd = np.full(grid_size, tau * np.sqrt(-np.expm1(-2 * kappa * delta)))
        base_transition = _normal_bin_probabilities(edges, base_mean, base_sd)
        a = np.exp(-lam * delta)
        transition = a * base_transition + (1 - a) * stationary[None, :]
        row_sum = transition.sum(axis=1)
        transition /= row_sum[:, None]
        predicted = weights @ transition
        filtered = predicted * emission(index)
        likelihood_increment = float(filtered.sum())
        if not np.isfinite(likelihood_increment) or likelihood_increment <= 0:
            return -np.inf
        loglik += np.log(likelihood_increment)
        weights = filtered / likelihood_increment
    return float(loglik)
