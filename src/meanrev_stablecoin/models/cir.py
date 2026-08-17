from __future__ import annotations

import numpy as np
from scipy.stats import ncx2

from ..estimation.optimize import FitResult, multistart_minimize, optimizer_diagnostics, repeated_solution_count
from .ou import ar1_initial_values


def cir_transition_parameters(
    z_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    kd = kappa * np.asarray(delta_days, dtype=float)
    one_minus = -np.expm1(-kd)
    c = sigma * sigma * one_minus / (4 * kappa)
    df = 4 * kappa * theta / (sigma * sigma)
    nc = 4 * kappa * np.exp(-kd) * np.asarray(z_prev, dtype=float) / (sigma * sigma * one_minus)
    return c, float(df), nc


def cir_logpdf(
    z_next: np.ndarray,
    z_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
) -> np.ndarray:
    z_next = np.asarray(z_next, dtype=float)
    if theta <= 0 or kappa <= 0 or sigma <= 0 or np.any(z_next < 0) or np.any(np.asarray(z_prev) < 0):
        return np.full_like(z_next, -np.inf)
    c, df, nc = cir_transition_parameters(z_prev, delta_days, theta, kappa, sigma)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = ncx2.logpdf(z_next / c, df, nc) - np.log(c)
    failed = ~np.isfinite(result)
    if np.any(failed):
        # For stablecoin prices the noncentrality and degrees of freedom can be
        # of order 1e8/1e6, where SciPy's exact ncx2 implementation underflows.
        # Use the high-noncentrality Gaussian limit with the exact CIR first two
        # conditional moments, and report the fallback share in fit metadata.
        delta = np.asarray(delta_days, dtype=float)
        rho = np.exp(-kappa * delta)
        mean = theta + rho * (np.asarray(z_prev, dtype=float) - theta)
        variance = (
            np.asarray(z_prev, dtype=float) * sigma**2 * rho * (1 - rho) / kappa
            + theta * sigma**2 * (1 - rho) ** 2 / (2 * kappa)
        )
        normal = -0.5 * (np.log(2 * np.pi * variance) + (z_next - mean) ** 2 / variance)
        result[failed] = normal[failed]
    return result


def cir_scipy_exact_failure_rate(z_next, z_prev, delta_days, theta, kappa, sigma) -> float:
    c, df, nc = cir_transition_parameters(z_prev, delta_days, theta, kappa, sigma)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = ncx2.logpdf(np.asarray(z_next) / c, df, nc) - np.log(c)
    return float(np.mean(~np.isfinite(raw)))


def cir_cdf(
    z_next: np.ndarray,
    z_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
) -> np.ndarray:
    c, df, nc = cir_transition_parameters(z_prev, delta_days, theta, kappa, sigma)
    return ncx2.cdf(np.asarray(z_next) / c, df, nc)


def fit_cir(
    z_next: np.ndarray,
    z_prev: np.ndarray,
    delta_days: np.ndarray,
    model_name: str = "CIR_Price",
    multistart: int = 6,
    max_nobs: int = 150_000,
) -> FitResult:
    z_next = np.asarray(z_next, dtype=float)
    z_prev = np.asarray(z_prev, dtype=float)
    delta_days = np.asarray(delta_days, dtype=float)
    if np.any(z_next <= 0) or np.any(z_prev <= 0):
        raise ValueError("CIR observations must be strictly positive")
    original_nobs = len(z_next)
    if original_nobs > max_nobs:
        idx = np.linspace(0, original_nobs - 1, max_nobs, dtype=int)
        z_next_fit, z_prev_fit, delta_fit = z_next[idx], z_prev[idx], delta_days[idx]
    else:
        z_next_fit, z_prev_fit, delta_fit = z_next, z_prev, delta_days
    theta0, kappa0, sigma0 = ar1_initial_values(z_prev_fit, z_next_fit, float(np.median(delta_fit)))
    theta0 = max(theta0, float(np.mean(z_next_fit)) * 0.5, 1e-8)
    sigma0 = max(sigma0, 1e-8)
    base = np.log([theta0, kappa0, sigma0])
    starts = [base + s * np.array([0.05, 0.7, -0.35]) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-12, 4), (-10, 14), (-16, 5)]

    def objective(beta: np.ndarray) -> float:
        theta, kappa, sigma = np.exp(beta)
        ll = cir_logpdf(z_next_fit, z_prev_fit, delta_fit, theta, kappa, sigma)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    best, results = multistart_minimize(objective, starts, bounds=bounds, options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-6})
    theta, kappa, sigma = map(float, np.exp(best.x))
    fit_loglik = float(np.sum(cir_logpdf(z_next_fit, z_prev_fit, delta_fit, theta, kappa, sigma)))
    scaled_loglik = fit_loglik * original_nobs / len(z_next_fit)
    gradient, condition, covariance = optimizer_diagnostics(best, len(z_next_fit))
    repeats = repeated_solution_count(results, tolerance=1e-7)
    p = 3
    feller = 2 * kappa * theta / (sigma * sigma)
    fallback_rate = cir_scipy_exact_failure_rate(z_next_fit, z_prev_fit, delta_fit, theta, kappa, sigma)
    return FitResult(
        model_name=model_name,
        params={"theta": theta, "kappa": kappa, "sigma": sigma},
        loglik=scaled_loglik,
        nobs=original_nobs,
        aic=-2 * scaled_loglik + 2 * p,
        bic=-2 * scaled_loglik + p * np.log(original_nobs),
        converged=bool(best.success and np.isfinite(fit_loglik)),
        optimizer_message=str(best.message),
        gradient_norm=gradient,
        hessian_condition=condition,
        covariance=covariance,
        robust_covariance=None,
        start_values={"theta": theta0, "kappa": kappa0, "sigma": sigma0},
        metadata={
            "half_life_minutes": 1440 * np.log(2) / kappa,
            "feller_index": feller,
            "feller_satisfied": bool(feller >= 1),
            "scipy_exact_logpdf_failure_rate": fallback_rate,
            "transition_density_backend": "scipy_ncx2_exact_with_high_noncentrality_moment_normal_fallback",
            "estimation_nobs": len(z_next_fit),
            "loglik_scaled_from_subsample": bool(original_nobs != len(z_next_fit)),
            "repeated_best_solutions": repeats,
            "weak_identification": condition is None or condition > 1e12 or repeats < 2 or fallback_rate > 0.5,
        },
    )
