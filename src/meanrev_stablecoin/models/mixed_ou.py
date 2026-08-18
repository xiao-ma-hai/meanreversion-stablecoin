from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr

from ..estimation.optimize import FitResult, multistart_minimize, optimizer_diagnostics, repeated_solution_count
from .ou import fit_ou, ou_logpdf, ou_moments


def mixed_ou_logpdf_tau(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    tau: float,
    kappa: float,
    lam: float,
) -> np.ndarray:
    """Mixed-OU transition log density in invariant-margin parameters.

    ``tau`` is the invariant standard deviation of the Gaussian base process,
    so that ``sigma = sqrt(2*kappa)*tau``.  This parameterization separates
    stationary-margin fit from dynamic persistence and reset intensity.
    """
    if tau <= 0 or kappa <= 0 or lam < 0:
        return np.full_like(np.asarray(x_next), -np.inf, dtype=float)
    sigma = np.sqrt(2.0 * kappa) * tau
    return mixed_ou_logpdf(x_next, x_prev, delta_days, theta, kappa, sigma, lam)


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


def _fit_mixed_ou_margin_parameterization(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    *,
    fixed_theta: float | None,
    fixed_tau: float | None,
    model_name: str,
    multistart: int = 12,
) -> FitResult:
    """Fit a mixed OU with optional externally fixed invariant margin."""
    x_next = np.asarray(x_next, dtype=float)
    x_prev = np.asarray(x_prev, dtype=float)
    delta_days = np.asarray(delta_days, dtype=float)
    if not (len(x_next) == len(x_prev) == len(delta_days)) or len(x_next) < 10:
        raise ValueError("mixed-OU fitting arrays must have equal length and at least 10 observations")
    pooled = np.concatenate([x_prev[:1], x_next])
    theta0 = float(np.mean(pooled)) if fixed_theta is None else float(fixed_theta)
    tau0 = max(float(np.std(pooled - theta0, ddof=0)), 1e-8) if fixed_tau is None else float(fixed_tau)
    if tau0 <= 0:
        raise ValueError("fixed invariant standard deviation must be positive")
    base = fit_ou(x_next, x_prev, delta_days, fixed_theta=fixed_theta, multistart=min(multistart, 4))
    kappa0 = float(base.params["kappa"])

    lambda_starts = np.array([0.0, 1e-3, 0.05, 0.5, 2.0, 10.0, 50.0, 250.0, 1000.0])
    starts: list[np.ndarray] = []
    for lam0 in lambda_starts[:max(multistart, 1)]:
        pieces: list[float] = []
        if fixed_theta is None:
            pieces.append(theta0)
        if fixed_tau is None:
            pieces.append(np.log(tau0))
        pieces.extend([np.log(max(kappa0, 1e-8)), np.log(max(lam0, 1e-8))])
        starts.append(np.asarray(pieces, dtype=float))

    bounds: list[tuple[float, float]] = []
    if fixed_theta is None:
        state_scale = max(float(np.std(pooled)), 1e-5)
        bounds.append((theta0 - 25 * state_scale, theta0 + 25 * state_scale))
    if fixed_tau is None:
        bounds.append((-16, 0))
    bounds.extend([(-10, 14), (-18, 12)])

    def unpack(beta: np.ndarray) -> tuple[float, float, float, float]:
        cursor = 0
        if fixed_theta is None:
            theta = float(beta[cursor]); cursor += 1
        else:
            theta = float(fixed_theta)
        if fixed_tau is None:
            tau = float(np.exp(beta[cursor])); cursor += 1
        else:
            tau = float(fixed_tau)
        kappa = float(np.exp(beta[cursor]))
        lam = float(np.exp(beta[cursor + 1]))
        return theta, tau, kappa, lam

    def objective(beta: np.ndarray) -> float:
        theta, tau, kappa, lam = unpack(beta)
        values = mixed_ou_logpdf_tau(x_next, x_prev, delta_days, theta, tau, kappa, lam)
        return float(-np.mean(values)) if np.isfinite(values).all() else 1e100

    best, results = multistart_minimize(
        objective,
        starts,
        bounds=bounds,
        options={"maxiter": 350, "ftol": 1e-12, "gtol": 1e-7},
    )
    theta, tau, kappa, lam = unpack(best.x)
    sigma = float(np.sqrt(2.0 * kappa) * tau)
    loglik = float(np.sum(mixed_ou_logpdf_tau(x_next, x_prev, delta_days, theta, tau, kappa, lam)))
    p = len(best.x)
    gradient, condition, covariance = optimizer_diagnostics(best, len(x_next))
    repeats = repeated_solution_count(results, tolerance=1e-8)
    return FitResult(
        model_name=model_name,
        params={"theta": theta, "tau": tau, "kappa": kappa, "sigma": sigma, "lambda": lam},
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
        start_values={"theta": theta0, "tau": tau0, "kappa": kappa0, "lambda": 0.0},
        metadata={
            "parameterization": "theta_tau_kappa_lambda",
            "margin_theta_fixed": fixed_theta is not None,
            "margin_tau_fixed": fixed_tau is not None,
            "base_half_life_minutes": 1440 * np.log(2) / kappa,
            "effective_half_life_minutes": 1440 * np.log(2) / (kappa + lam),
            "daily_reset_probability": 1 - np.exp(-lam),
            "repeated_best_solutions": repeats,
            "boundary_hit": bool(lam < 1e-5),
            "weak_identification": condition is None or condition > 1e12 or repeats < 2,
        },
    )


def fit_mou_uqml(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    multistart: int = 12,
) -> FitResult:
    """Unrestricted conditional QML; its invariant margin is model-implied."""
    return _fit_mixed_ou_margin_parameterization(
        x_next, x_prev, delta_days,
        fixed_theta=None, fixed_tau=None, model_name="MOU-UQML", multistart=multistart,
    )


def fit_mou_ifm(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    margin_values: np.ndarray,
    multistart: int = 12,
) -> FitResult:
    """Inference-functions-for-margins fit with a training Gaussian margin."""
    margin_values = np.asarray(margin_values, dtype=float)
    theta = float(np.mean(margin_values))
    tau = max(float(np.std(margin_values, ddof=0)), 1e-8)
    return _fit_mixed_ou_margin_parameterization(
        x_next, x_prev, delta_days,
        fixed_theta=theta, fixed_tau=tau, model_name="MOU-IFM", multistart=multistart,
    )


def fit_mou_peg(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    margin_values: np.ndarray,
    multistart: int = 12,
) -> FitResult:
    """Peg-constrained fit with theta=0 and training RMS as margin scale."""
    margin_values = np.asarray(margin_values, dtype=float)
    tau = max(float(np.sqrt(np.mean(margin_values * margin_values))), 1e-8)
    return _fit_mixed_ou_margin_parameterization(
        x_next, x_prev, delta_days,
        fixed_theta=0.0, fixed_tau=tau, model_name="MOU-Peg", multistart=multistart,
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
