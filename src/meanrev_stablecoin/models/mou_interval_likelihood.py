from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..estimation.optimize import FitResult, multistart_minimize, optimizer_diagnostics, repeated_solution_count
from ..observation.rounding import log_price_intervals, normal_interval_logprob


def ou_interval_logprob_tau(
    observed_next_price: np.ndarray,
    observed_previous_price: np.ndarray,
    next_tick: float | np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    tau: float,
    kappa: float,
) -> np.ndarray:
    """Conditional interval probability for an invariant-parameterized OU.

    The previous latent state is approximated by the log of the previous
    observed price.  This is therefore an interval-likelihood approximation,
    not the full filtered likelihood.
    """

    if tau <= 0 or kappa <= 0:
        return np.full_like(np.asarray(observed_next_price, dtype=float), -np.inf)
    previous = np.log(np.asarray(observed_previous_price, dtype=float))
    delta = np.asarray(delta_days, dtype=float)
    rho = np.exp(-kappa * delta)
    mean = theta + rho * (previous - theta)
    sd = tau * np.sqrt(np.maximum(-np.expm1(-2 * kappa * delta), np.finfo(float).tiny))
    lower, upper = log_price_intervals(observed_next_price, next_tick)
    return normal_interval_logprob(lower, upper, mean, sd)


def mixed_ou_interval_logprob_tau(
    observed_next_price: np.ndarray,
    observed_previous_price: np.ndarray,
    next_tick: float | np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    tau: float,
    kappa: float,
    lam: float,
) -> np.ndarray:
    """Bounded mixed-OU interval likelihood for tick-discrete observations."""

    if tau <= 0 or kappa <= 0 or lam < 0:
        return np.full_like(np.asarray(observed_next_price, dtype=float), -np.inf)
    delta = np.asarray(delta_days, dtype=float)
    base = ou_interval_logprob_tau(
        observed_next_price, observed_previous_price, next_tick, delta, theta, tau, kappa
    )
    lower, upper = log_price_intervals(observed_next_price, next_tick)
    reset = normal_interval_logprob(lower, upper, theta, tau)
    if lam == 0:
        return base
    log_a = -lam * delta
    log_one_minus_a = np.log(-np.expm1(log_a))
    return logsumexp(np.vstack([log_a + base, log_one_minus_a + reset]), axis=0)


def fit_rounded_normal_margin(
    observed_price: np.ndarray,
    tick: float | np.ndarray,
) -> tuple[float, float, float]:
    """Estimate a latent normal margin from rounded price bins."""

    price = np.asarray(observed_price, dtype=float)
    lower, upper = log_price_intervals(price, tick)
    midpoint = np.log(price)
    theta0 = float(np.mean(midpoint))
    tau0 = max(float(np.std(midpoint, ddof=0)), 1e-7)

    def objective(beta: np.ndarray) -> float:
        values = normal_interval_logprob(lower, upper, float(beta[0]), float(np.exp(beta[1])))
        return float(-np.mean(values)) if np.isfinite(values).all() else 1e100

    result = minimize(
        objective,
        np.array([theta0, np.log(tau0)]),
        method="L-BFGS-B",
        bounds=[(theta0 - 20 * tau0, theta0 + 20 * tau0), (-16, 0)],
        options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-8},
    )
    theta, tau = float(result.x[0]), float(np.exp(result.x[1]))
    return theta, tau, float(-result.fun * len(price))


def _fit_interval_dynamics(
    observed_next_price: np.ndarray,
    observed_previous_price: np.ndarray,
    next_tick: float | np.ndarray,
    delta_days: np.ndarray,
    *,
    theta: float,
    tau: float,
    mixed: bool,
    multistart: int,
    extra_starts: list[tuple[float, float]] | None = None,
) -> FitResult:
    following = np.asarray(observed_next_price, dtype=float)
    previous = np.asarray(observed_previous_price, dtype=float)
    delta = np.asarray(delta_days, dtype=float)
    ticks = np.broadcast_to(np.asarray(next_tick, dtype=float), following.shape)
    if not (len(following) == len(previous) == len(delta) == len(ticks)) or len(following) < 10:
        raise ValueError("interval-likelihood arrays must have equal length and at least 10 observations")

    previous_log = np.log(previous)
    following_lower, following_upper = log_price_intervals(following, ticks)
    reset_logprob = normal_interval_logprob(following_lower, following_upper, theta, tau)

    def base_values(kappa: float) -> np.ndarray:
        rho = np.exp(-kappa * delta)
        conditional_mean = theta + rho * (previous_log - theta)
        conditional_sd = tau * np.sqrt(
            np.maximum(-np.expm1(-2 * kappa * delta), np.finfo(float).tiny)
        )
        return normal_interval_logprob(
            following_lower, following_upper, conditional_mean, conditional_sd
        )

    centered_previous = np.log(previous) - theta
    centered_following = np.log(following) - theta
    denominator = float(np.dot(centered_previous, centered_previous))
    phi = float(np.dot(centered_previous, centered_following) / denominator) if denominator > 0 else 0.9
    phi = float(np.clip(phi, 1e-5, 1 - 1e-7))
    kappa0 = -np.log(phi) / max(float(np.median(delta)), 1e-12)
    lambda_starts = np.array([1e-8, 1e-3, 0.05, 0.5, 2.0, 10.0, 50.0, 250.0])
    if mixed:
        start_indices = np.linspace(0, len(lambda_starts) - 1, max(multistart, 1), dtype=int)
        starts = [np.array([np.log(kappa0), np.log(lambda_starts[index])]) for index in start_indices]
        for extra_kappa, extra_lambda in extra_starts or []:
            if extra_kappa > 0 and extra_lambda > 0:
                starts.append(np.array([np.log(extra_kappa), np.log(extra_lambda)]))
        bounds = [(-12, 14), (-18, 12)]

        def unpack(beta: np.ndarray) -> tuple[float, float]:
            return float(np.exp(beta[0])), float(np.exp(beta[1]))

        def values(beta: np.ndarray) -> np.ndarray:
            kappa, lam = unpack(beta)
            log_a = -lam * delta
            log_one_minus_a = np.log(-np.expm1(log_a))
            return logsumexp(
                np.vstack([log_a + base_values(kappa), log_one_minus_a + reset_logprob]),
                axis=0,
            )
    else:
        offsets = np.linspace(-1.0, 1.0, max(multistart, 2))
        starts = [np.array([np.log(kappa0) + offset]) for offset in offsets]
        bounds = [(-12, 14)]

        def unpack(beta: np.ndarray) -> tuple[float, float]:
            return float(np.exp(beta[0])), 0.0

        def values(beta: np.ndarray) -> np.ndarray:
            kappa, _ = unpack(beta)
            return base_values(kappa)

    def objective(beta: np.ndarray) -> float:
        result = values(beta)
        return float(-np.mean(result)) if np.isfinite(result).all() else 1e100

    best, results = multistart_minimize(
        objective,
        starts,
        bounds=bounds,
        options={"maxiter": 80, "ftol": 1e-9, "gtol": 5e-6},
    )
    kappa, lam = unpack(best.x)
    log_values = values(best.x)
    loglik = float(np.sum(log_values))
    gradient, condition, covariance = optimizer_diagnostics(best, len(following))
    repeats = repeated_solution_count(results, tolerance=1e-8)
    model_name = "MOU-Interval-IFM" if mixed else "OU-Interval-IFM"
    return FitResult(
        model_name=model_name,
        params={"theta": theta, "tau": tau, "kappa": kappa, "lambda": lam},
        loglik=loglik,
        nobs=len(following),
        aic=-2 * loglik + 2 * len(best.x),
        bic=-2 * loglik + len(best.x) * np.log(len(following)),
        converged=bool(best.success and np.isfinite(loglik)),
        optimizer_message=str(best.message),
        gradient_norm=gradient,
        hessian_condition=condition,
        covariance=covariance,
        robust_covariance=None,
        start_values={"theta": theta, "tau": tau, "kappa": kappa0, "lambda": 0.0},
        metadata={
            "likelihood_type": "conditional_interval_approximation",
            "previous_state_treatment": "observed_log_price_bin_midpoint",
            "margin_fixed": True,
            "multistart": len(starts),
            "repeated_best_solutions": repeats,
            "base_half_life_minutes": float(1440 * np.log(2) / kappa),
            "conditional_mean_half_life_minutes": float(1440 * np.log(2) / (kappa + lam)),
            "boundary_hit": bool(mixed and lam <= np.exp(bounds[1][0]) * 1.01),
            "weak_identification": bool(condition is None or not np.isfinite(condition) or condition > 1e12),
        },
    )


def fit_interval_ou_ifm(
    observed_next_price: np.ndarray,
    observed_previous_price: np.ndarray,
    next_tick: float | np.ndarray,
    delta_days: np.ndarray,
    *,
    theta: float,
    tau: float,
    multistart: int = 5,
) -> FitResult:
    return _fit_interval_dynamics(
        observed_next_price,
        observed_previous_price,
        next_tick,
        delta_days,
        theta=theta,
        tau=tau,
        mixed=False,
        multistart=multistart,
        extra_starts=None,
    )


def fit_interval_mou_ifm(
    observed_next_price: np.ndarray,
    observed_previous_price: np.ndarray,
    next_tick: float | np.ndarray,
    delta_days: np.ndarray,
    *,
    theta: float,
    tau: float,
    multistart: int = 7,
    extra_starts: list[tuple[float, float]] | None = None,
) -> FitResult:
    return _fit_interval_dynamics(
        observed_next_price,
        observed_previous_price,
        next_tick,
        delta_days,
        theta=theta,
        tau=tau,
        mixed=True,
        multistart=multistart,
        extra_starts=extra_starts,
    )
