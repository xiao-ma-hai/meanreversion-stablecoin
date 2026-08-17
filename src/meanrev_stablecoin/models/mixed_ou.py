from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr

from ..estimation.optimize import FitResult, multistart_minimize, optimizer_diagnostics, repeated_solution_count
from .ou import fit_ou, ou_logpdf, ou_moments


def mixed_ou_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    lam: float,
) -> np.ndarray:
    if lam < 0:
        return np.full_like(np.asarray(x_next), -np.inf, dtype=float)
    delta = np.asarray(delta_days, dtype=float)
    base = ou_logpdf(x_next, x_prev, delta, theta, kappa, sigma)
    stationary_var = sigma * sigma / (2 * kappa)
    marginal = -0.5 * (np.log(2 * np.pi * stationary_var) + (np.asarray(x_next) - theta) ** 2 / stationary_var)
    if lam == 0:
        return base
    log_a = -lam * delta
    log_one_minus_a = np.log(-np.expm1(log_a))
    return logsumexp(np.vstack([log_a + base, log_one_minus_a + marginal]), axis=0)


def mixed_ou_cdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    lam: float,
) -> np.ndarray:
    mean, variance = ou_moments(x_prev, delta_days, theta, kappa, sigma)
    a = np.exp(-lam * np.asarray(delta_days))
    base = ndtr((np.asarray(x_next) - mean) / np.sqrt(variance))
    marginal = ndtr((np.asarray(x_next) - theta) / (sigma / np.sqrt(2 * kappa)))
    return a * base + (1 - a) * marginal


def fit_mixed_ou(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    fixed_theta: float | None = None,
    base_fit: FitResult | None = None,
    multistart: int = 12,
) -> FitResult:
    if base_fit is None:
        base_fit = fit_ou(x_next, x_prev, delta_days, fixed_theta=fixed_theta, multistart=min(multistart, 6))
    theta0, kappa0, sigma0 = (base_fit.params[k] for k in ("theta", "kappa", "sigma"))
    lambda_starts = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 20, 50, 100, 300, 1000, 3000])[:multistart]
    if fixed_theta is None:
        starts = [np.array([theta0, np.log(kappa0), np.log(sigma0), np.log(lam)]) for lam in lambda_starts]
        bounds = [(-0.5, 0.5), (-10, 14), (-16, 5), (-12, 12)]

        def unpack(beta: np.ndarray) -> tuple[float, float, float, float]:
            return float(beta[0]), float(np.exp(beta[1])), float(np.exp(beta[2])), float(np.exp(beta[3]))
    else:
        starts = [np.array([np.log(kappa0), np.log(sigma0), np.log(lam)]) for lam in lambda_starts]
        bounds = [(-10, 14), (-16, 5), (-12, 12)]

        def unpack(beta: np.ndarray) -> tuple[float, float, float, float]:
            return float(fixed_theta), float(np.exp(beta[0])), float(np.exp(beta[1])), float(np.exp(beta[2]))

    def objective(beta: np.ndarray) -> float:
        theta, kappa, sigma, lam = unpack(beta)
        ll = mixed_ou_logpdf(x_next, x_prev, delta_days, theta, kappa, sigma, lam)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    best, results = multistart_minimize(objective, starts, bounds=bounds, options={"maxiter": 350, "ftol": 1e-12, "gtol": 1e-7})
    theta, kappa, sigma, lam = unpack(best.x)
    loglik = float(np.sum(mixed_ou_logpdf(x_next, x_prev, delta_days, theta, kappa, sigma, lam)))
    p = len(best.x)
    gradient, condition, covariance = optimizer_diagnostics(best, len(x_next))
    repeats = repeated_solution_count(results, tolerance=1e-8)
    return FitResult(
        model_name="MOU0" if fixed_theta is not None else "MOUF",
        params={"theta": theta, "kappa": kappa, "sigma": sigma, "lambda": lam},
        loglik=loglik, nobs=len(x_next), aic=-2 * loglik + 2 * p,
        bic=-2 * loglik + p * np.log(len(x_next)), converged=bool(best.success),
        optimizer_message=str(best.message), gradient_norm=gradient,
        hessian_condition=condition, covariance=covariance, robust_covariance=None,
        start_values={"theta": theta0, "kappa": kappa0, "sigma": sigma0, "lambda": float(lambda_starts[0])},
        metadata={
            "base_half_life_minutes": 1440 * np.log(2) / kappa,
            "effective_half_life_minutes": 1440 * np.log(2) / (kappa + lam),
            "daily_reset_probability": 1 - np.exp(-lam),
            "repeated_best_solutions": repeats,
            "boundary_hit": bool(lam < 1e-5),
            "weak_identification": condition is None or condition > 1e12 or repeats < 2,
        },
    )


def simulate_mixed_ou(
    n: int,
    delta_days: float,
    theta: float,
    kappa: float,
    sigma: float,
    lam: float,
    rng: np.random.Generator,
    x0: float | None = None,
) -> np.ndarray:
    values = np.empty(n)
    stationary_sd = sigma / np.sqrt(2 * kappa)
    values[0] = theta if x0 is None else x0
    a = np.exp(-lam * delta_days)
    for idx in range(1, n):
        if rng.random() > a:
            values[idx] = theta + stationary_sd * rng.standard_normal()
        else:
            mean, variance = ou_moments(np.array([values[idx - 1]]), np.array([delta_days]), theta, kappa, sigma)
            values[idx] = mean[0] + np.sqrt(variance[0]) * rng.standard_normal()
    return values

