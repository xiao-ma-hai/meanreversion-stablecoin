from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.special import ndtr

from ..estimation.optimize import (
    FitResult,
    multistart_minimize,
    optimizer_diagnostics,
    repeated_solution_count,
)


def ou_moments(
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    kd = kappa * np.asarray(delta_days, dtype=float)
    rho = np.exp(-kd)
    mean = theta + rho * (np.asarray(x_prev, dtype=float) - theta)
    variance = sigma * sigma * (-np.expm1(-2.0 * kd)) / (2.0 * kappa)
    return mean, np.maximum(variance, np.finfo(float).tiny)


def ou_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
) -> np.ndarray:
    if kappa <= 0 or sigma <= 0:
        return np.full_like(np.asarray(x_next, dtype=float), -np.inf)
    mean, variance = ou_moments(x_prev, delta_days, theta, kappa, sigma)
    return -0.5 * (np.log(2.0 * np.pi * variance) + (np.asarray(x_next) - mean) ** 2 / variance)


def ou_cdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
) -> np.ndarray:
    mean, variance = ou_moments(x_prev, delta_days, theta, kappa, sigma)
    return ndtr((np.asarray(x_next) - mean) / np.sqrt(variance))


def ar1_initial_values(x_prev: np.ndarray, x_next: np.ndarray, delta_day: float) -> tuple[float, float, float]:
    x = np.asarray(x_prev, dtype=float)
    y = np.asarray(x_next, dtype=float)
    x_mean, y_mean = float(x.mean()), float(y.mean())
    denom = float(np.dot(x - x_mean, x - x_mean))
    phi_raw = float(np.dot(x - x_mean, y - y_mean) / denom) if denom > 0 else 0.9
    phi = float(np.clip(phi_raw, 1e-6, 1 - 1e-8))
    intercept = y_mean - phi * x_mean
    theta = intercept / (1 - phi)
    kappa = -math.log(phi) / delta_day
    resid = y - intercept - phi * x
    innovation_var = float(np.mean(resid * resid))
    sigma = math.sqrt(max(innovation_var * 2 * kappa / (-math.expm1(-2 * kappa * delta_day)), 1e-16))
    return theta, kappa, sigma


def fit_ou(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    fixed_theta: float | None = None,
    model_name: str | None = None,
    multistart: int = 12,
) -> FitResult:
    x_next = np.asarray(x_next, dtype=float)
    x_prev = np.asarray(x_prev, dtype=float)
    delta_days = np.asarray(delta_days, dtype=float)
    if not (len(x_next) == len(x_prev) == len(delta_days)) or len(x_next) < 10:
        raise ValueError("OU fitting arrays must have equal length and at least 10 observations")
    theta0, kappa0, sigma0 = ar1_initial_values(x_prev, x_next, float(np.median(delta_days)))
    if fixed_theta is not None:
        theta0 = float(fixed_theta)

        def unpack(beta: np.ndarray) -> tuple[float, float, float]:
            return float(fixed_theta), float(np.exp(beta[0])), float(np.exp(beta[1]))

        base = np.array([np.log(kappa0), np.log(sigma0)])
        bounds = [(-10, 14), (-16, 5)]
    else:

        def unpack(beta: np.ndarray) -> tuple[float, float, float]:
            return float(beta[0]), float(np.exp(beta[1])), float(np.exp(beta[2]))

        base = np.array([theta0, np.log(kappa0), np.log(sigma0)])
        bounds = [(-0.5, 0.5), (-10, 14), (-16, 5)]

    scales = np.linspace(-1.0, 1.0, max(multistart, 2))
    starts = []
    for idx, scale in enumerate(scales):
        start = base.copy()
        if fixed_theta is None:
            start[0] += scale * max(float(np.std(x_prev)), 1e-5) * 0.25
            start[1] += 0.6 * scale
            start[2] -= 0.3 * scale
        else:
            start[0] += 0.6 * scale
            start[1] -= 0.3 * scale
        starts.append(start)

    def objective(beta: np.ndarray) -> float:
        theta, kappa, sigma = unpack(beta)
        values = ou_logpdf(x_next, x_prev, delta_days, theta, kappa, sigma)
        return float(-np.mean(values)) if np.isfinite(values).all() else 1e100

    best, results = multistart_minimize(objective, starts, bounds=bounds)
    theta, kappa, sigma = unpack(best.x)
    loglik = float(np.sum(ou_logpdf(x_next, x_prev, delta_days, theta, kappa, sigma)))
    p = len(best.x)
    gradient, condition, covariance = optimizer_diagnostics(best, len(x_next))
    repeat_count = repeated_solution_count(results, tolerance=1e-8)
    weak = bool(condition is None or not np.isfinite(condition) or condition > 1e12 or repeat_count < 2)
    return FitResult(
        model_name=model_name or ("OU0" if fixed_theta is not None else "OUF"),
        params={"theta": theta, "kappa": kappa, "sigma": sigma},
        loglik=loglik,
        nobs=len(x_next),
        aic=-2 * loglik + 2 * p,
        bic=-2 * loglik + p * np.log(len(x_next)),
        converged=bool(best.success and np.isfinite(loglik)),
        optimizer_message=str(best.message),
        gradient_norm=gradient,
        hessian_condition=condition,
        covariance=covariance,
        robust_covariance=None,
        start_values={"theta": theta0, "kappa": kappa0, "sigma": sigma0},
        metadata={
            "half_life_minutes": 1440 * np.log(2) / kappa,
            "stationary_sd": sigma / np.sqrt(2 * kappa),
            "multistart": len(starts),
            "repeated_best_solutions": repeat_count,
            "boundary_hit": bool(np.any(np.isclose(best.x, np.array([b[0] for b in bounds]), atol=1e-6)) or np.any(np.isclose(best.x, np.array([b[1] for b in bounds]), atol=1e-6))),
            "weak_identification": weak,
        },
    )


def simulate_ou(
    n: int,
    delta_days: float | np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    rng: np.random.Generator,
    x0: float | None = None,
) -> np.ndarray:
    deltas = np.full(n - 1, float(delta_days)) if np.ndim(delta_days) == 0 else np.asarray(delta_days, dtype=float)
    values = np.empty(n)
    values[0] = theta if x0 is None else x0
    for idx, delta in enumerate(deltas, start=1):
        mean, variance = ou_moments(np.array([values[idx - 1]]), np.array([delta]), theta, kappa, sigma)
        values[idx] = mean[0] + np.sqrt(variance[0]) * rng.standard_normal()
    return values

