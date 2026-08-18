from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr


def _broadcast_tick(price: np.ndarray, tick: float | np.ndarray) -> np.ndarray:
    values = np.broadcast_to(np.asarray(tick, dtype=float), price.shape).copy()
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("tick sizes must be finite and strictly positive")
    if np.any(price <= values / 2):
        raise ValueError("observed prices must exceed half of their tick size")
    return values


def quantize_prices(latent_prices: np.ndarray, tick: float | np.ndarray) -> np.ndarray:
    """Round positive latent prices to the nearest tick, with half ticks rounded up."""

    price = np.asarray(latent_prices, dtype=float)
    if np.any(~np.isfinite(price)) or np.any(price <= 0):
        raise ValueError("latent prices must be finite and strictly positive")
    tick_values = np.broadcast_to(np.asarray(tick, dtype=float), price.shape)
    if np.any(~np.isfinite(tick_values)) or np.any(tick_values <= 0):
        raise ValueError("tick sizes must be finite and strictly positive")
    return np.floor(price / tick_values + 0.5) * tick_values


def log_price_intervals(
    observed_price: np.ndarray,
    tick: float | np.ndarray,
    *,
    endpoint: str = "left_closed",
) -> tuple[np.ndarray, np.ndarray]:
    """Return latent log-price bins implied by nearest-tick observations.

    The default convention is ``[p-delta/2, p+delta/2)``.  Continuous latent
    laws assign zero probability to the endpoints, so changing the endpoint
    convention does not change the likelihood; the argument is retained to
    make that convention explicit and testable.
    """

    if endpoint not in {"left_closed", "right_closed"}:
        raise ValueError("endpoint must be 'left_closed' or 'right_closed'")
    price = np.asarray(observed_price, dtype=float)
    if np.any(~np.isfinite(price)) or np.any(price <= 0):
        raise ValueError("observed prices must be finite and strictly positive")
    tick_values = _broadcast_tick(price, tick)
    return np.log(price - tick_values / 2), np.log(price + tick_values / 2)


def _log_difference(log_large: np.ndarray, log_small: np.ndarray) -> np.ndarray:
    gap = np.minimum(log_small - log_large, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = log_large + np.log1p(-np.exp(gap))
    return np.where(log_small == -np.inf, log_large, result)


def normal_interval_logprob(
    lower: np.ndarray,
    upper: np.ndarray,
    mean: float | np.ndarray,
    sd: float | np.ndarray,
) -> np.ndarray:
    """Stable log probability that a normal variable lies in ``[lower, upper)``."""

    lower_values, upper_values, mean_values, sd_values = np.broadcast_arrays(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(sd, dtype=float),
    )
    if np.any(upper_values <= lower_values):
        raise ValueError("every interval must have positive width")
    if np.any(~np.isfinite(sd_values)) or np.any(sd_values <= 0):
        return np.full(lower_values.shape, -np.inf, dtype=float)
    z_lower = (lower_values - mean_values) / sd_values
    z_upper = (upper_values - mean_values) / sd_values
    use_survival = z_lower > 0
    result = np.empty(z_lower.shape, dtype=float)
    cdf_mask = ~use_survival
    if np.any(cdf_mask):
        result[cdf_mask] = _log_difference(
            log_ndtr(z_upper[cdf_mask]), log_ndtr(z_lower[cdf_mask])
        )
    if np.any(use_survival):
        result[use_survival] = _log_difference(
            log_ndtr(-z_lower[use_survival]), log_ndtr(-z_upper[use_survival])
        )
    return result
