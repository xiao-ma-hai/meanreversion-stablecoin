from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm

from ..estimation.optimize import FitResult, optimizer_diagnostics, repeated_solution_count
from .mixed_ou import mixed_ou_logpdf
from .ou import ar1_initial_values, ou_logpdf, ou_moments


def nested_ou_rho(horizon_days: np.ndarray | float, kappa_slow: float, kappa_fast: float, weight_fast: float) -> np.ndarray:
    h = np.asarray(horizon_days, dtype=float)
    return weight_fast * np.exp(-kappa_fast * h) + (1.0 - weight_fast) * np.exp(-kappa_slow * h)


def nested_ou_pair_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    stationary_variance: float,
    kappa_slow: float,
    kappa_fast: float,
    weight_fast: float,
) -> np.ndarray:
    rho = nested_ou_rho(delta_days, kappa_slow, kappa_fast, weight_fast)
    mean = theta + rho * (np.asarray(x_prev) - theta)
    variance = stationary_variance * np.maximum(1.0 - rho * rho, 1e-16)
    return norm.logpdf(x_next, mean, np.sqrt(variance))


def fit_nested_ou_pairwise(
    horizon_data: dict[int, tuple[np.ndarray, np.ndarray]],
    max_pairs_per_horizon: int = 60_000,
    multistart: int = 6,
) -> FitResult:
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_values = []
    for minutes, (x_prev, x_next) in horizon_data.items():
        idx = np.linspace(0, len(x_prev) - 1, min(len(x_prev), max_pairs_per_horizon), dtype=int)
        selected[minutes] = (np.asarray(x_prev)[idx], np.asarray(x_next)[idx])
        all_values.extend([np.asarray(x_prev)[idx], np.asarray(x_next)[idx]])
    values = np.concatenate(all_values); theta0 = float(np.mean(values)); variance0 = float(np.var(values))
    base = np.array([theta0, np.log(max(variance0, 1e-12)), np.log(0.3), np.log(200.0 - 0.3), 0.0])
    starts = [base + s * np.array([0.1 * np.sqrt(variance0), 0.3, 0.8, -0.8, 1.0]) for s in np.linspace(-1, 1, multistart)]
    bounds = [(theta0 - 10 * np.sqrt(variance0), theta0 + 10 * np.sqrt(variance0)), (np.log(variance0) - 6, np.log(variance0) + 5), (-9, 7), (-9, 9), (-8, 8)]

    def unpack(beta: np.ndarray):
        theta = float(beta[0]); variance = float(np.exp(beta[1])); slow = float(np.exp(beta[2])); fast = float(slow + np.exp(beta[3])); weight = float(expit(beta[4]))
        return theta, variance, slow, fast, weight

    def objective(beta: np.ndarray) -> float:
        theta, variance, slow, fast, weight = unpack(beta)
        scores = []
        for minutes, (xp, xn) in selected.items():
            scores.append(np.mean(nested_ou_pair_logpdf(xn, xp, np.full(len(xp), minutes / 1440.0), theta, variance, slow, fast, weight)))
        return -float(np.mean(scores))

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "ftol": 1e-11}) for start in starts]
    best = min(results, key=lambda result: result.fun); theta, variance, slow, fast, weight = unpack(best.x)
    nobs = sum(len(value[0]) for value in selected.values()); loglik = -float(best.fun) * nobs
    gradient, condition, covariance = optimizer_diagnostics(best, nobs); repeats = repeated_solution_count(results, 1e-7); p = 5
    boundary = bool(weight < 1e-3 or weight > 1 - 1e-3 or fast / slow < 1.01)
    weak = condition is None or condition > 1e12 or repeats < 2 or boundary
    return FitResult(
        "NestedOU", {"theta": theta, "stationary_variance": variance, "kappa_slow": slow, "kappa_fast": fast, "weight_fast": weight,
                     "sigma_slow": float(np.sqrt(2 * slow * variance * (1 - weight))), "sigma_fast": float(np.sqrt(2 * fast * variance * weight))},
        loglik, nobs, -2 * loglik + 2 * p, -2 * loglik + p * np.log(nobs), bool(best.success), str(best.message), gradient, condition, covariance, None,
        {"theta": theta0, "stationary_variance": variance0, "kappa_slow": 0.3, "kappa_fast": 200.0, "weight_fast": 0.01},
        {"composite_equal_horizon_objective": True, "scalar_markov": False, "latent_vector_markov": True,
         "repeated_best_solutions": repeats, "boundary_hit": boundary, "weak_identification": weak},
    )


def jump_ou_approx_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_sd: float,
) -> np.ndarray:
    """OU plus compound-normal jumps, with an at-least-one-jump moment component.

    The no-jump component is exact. Conditional on at least one jump, the
    implementation uses the exact one-uniform-jump first two moments. The
    approximation is explicitly reported and is most accurate at five minutes.
    """
    h = np.asarray(delta_days, dtype=float)
    base_mean, base_var = ou_moments(x_prev, h, theta, kappa, sigma)
    if jump_intensity <= 0:
        return norm.logpdf(x_next, base_mean, np.sqrt(base_var))
    kh = np.maximum(kappa * h, 1e-12)
    damping_mean = -np.expm1(-kh) / kh
    damping_second = -np.expm1(-2 * kh) / (2 * kh)
    jump_component_mean = base_mean + jump_mean * damping_mean
    jump_component_var = base_var + jump_sd**2 * damping_second + jump_mean**2 * np.maximum(
        damping_second - damping_mean**2, 0.0
    )
    log_no_jump = -jump_intensity * h
    no_jump = log_no_jump + norm.logpdf(x_next, base_mean, np.sqrt(base_var))
    with_jump = np.log(-np.expm1(log_no_jump)) + norm.logpdf(
        x_next, jump_component_mean, np.sqrt(np.maximum(jump_component_var, 1e-18))
    )
    return logsumexp(np.vstack([no_jump, with_jump]), axis=0)


def fit_jump_ou_approx(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    base_params: dict[str, float] | None = None,
    multistart: int = 6,
) -> FitResult:
    x_next = np.asarray(x_next, dtype=float); x_prev = np.asarray(x_prev, dtype=float)
    if base_params is None:
        theta, kappa, sigma = ar1_initial_values(x_prev, x_next, float(np.median(delta_days)))
    else:
        theta, kappa, sigma = (float(base_params[key]) for key in ("theta", "kappa", "sigma"))
    residual_scale = max(float(np.std(x_next - x_prev)), 1e-8)
    base = np.array([np.log(0.2), 0.0, np.log(5 * residual_scale)])
    starts = [base + s * np.array([1.5, residual_scale, 0.8]) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-10, 8), (-0.05, 0.05), (np.log(residual_scale) - 5, np.log(residual_scale) + 6)]

    def objective(beta: np.ndarray) -> float:
        ll = jump_ou_approx_logpdf(
            x_next, x_prev, delta_days, theta, kappa, sigma,
            float(np.exp(beta[0])), float(beta[1]), float(np.exp(beta[2])),
        )
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "ftol": 1e-11}) for start in starts]
    best = min(results, key=lambda result: result.fun)
    intensity, jump_mean, jump_sd = float(np.exp(best.x[0])), float(best.x[1]), float(np.exp(best.x[2]))
    ll = jump_ou_approx_logpdf(x_next, x_prev, delta_days, theta, kappa, sigma, intensity, jump_mean, jump_sd)
    loglik = float(np.sum(ll)); gradient, condition, covariance = optimizer_diagnostics(best, len(x_next))
    repeats = repeated_solution_count(results, tolerance=1e-7); p = 3
    boundary = bool(np.any(np.isclose(best.x, [bound[0] for bound in bounds], atol=1e-5)) or np.any(np.isclose(best.x, [bound[1] for bound in bounds], atol=1e-5)))
    weak = condition is None or condition > 1e12 or repeats < 2 or boundary
    return FitResult(
        "JumpOU", {"theta": theta, "kappa": kappa, "sigma": sigma, "jump_intensity": intensity, "jump_mean": jump_mean, "jump_sd": jump_sd},
        loglik, len(x_next), -2 * loglik + 2 * p, -2 * loglik + p * np.log(len(x_next)), bool(best.success), str(best.message),
        gradient, condition, covariance, None,
        {"jump_intensity": 0.2, "jump_mean": 0.0, "jump_sd": 5 * residual_scale},
        {"base_ou_parameters_fixed": True, "transition_approximation": "exact no-jump plus one-uniform-jump moment component",
         "repeated_best_solutions": repeats, "boundary_hit": boundary, "weak_identification": weak},
    )


def seasonal_integrated_intensity(
    t_prev_days: np.ndarray,
    t_next_days: np.ndarray,
    lambda0: float,
    sine: float,
    cosine: float,
    period_days: float = 365.2425,
) -> np.ndarray:
    start, end = np.asarray(t_prev_days, dtype=float), np.asarray(t_next_days, dtype=float)
    omega = 2 * np.pi / period_days
    integral = lambda0 * (
        end - start
        + sine * (np.cos(omega * start) - np.cos(omega * end)) / omega
        + cosine * (np.sin(omega * end) - np.sin(omega * start)) / omega
    )
    return np.maximum(integral, 0.0)


def seasonal_mou_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    t_next_days: np.ndarray,
    t_prev_days: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    lambda0: float,
    sine: float,
    cosine: float,
) -> np.ndarray:
    integrated = seasonal_integrated_intensity(t_prev_days, t_next_days, lambda0, sine, cosine)
    base_mean, base_var = ou_moments(x_prev, delta_days, theta, kappa, sigma)
    stationary_var = sigma * sigma / (2 * kappa)
    log_a = -integrated
    first = log_a + norm.logpdf(x_next, base_mean, np.sqrt(base_var))
    second = np.log(-np.expm1(log_a)) + norm.logpdf(x_next, theta, np.sqrt(stationary_var))
    return logsumexp(np.vstack([first, second]), axis=0)


def fit_seasonal_mou(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    t_next_days: np.ndarray,
    t_prev_days: np.ndarray,
    delta_days: np.ndarray,
    base_params: dict[str, float],
    multistart: int = 6,
) -> FitResult:
    theta, kappa, sigma = (float(base_params[key]) for key in ("theta", "kappa", "sigma"))

    def unpack(beta: np.ndarray):
        lambda0 = float(np.exp(beta[0]))
        radius = float(0.95 * expit(beta[1]))
        angle = float(beta[2])
        return lambda0, radius * np.cos(angle), radius * np.sin(angle)

    starts = [np.array([np.log(lam), radial, angle]) for lam in (0.1, 0.5, 2.0) for radial, angle in ((-2, 0), (0, 1), (2, -1))]
    starts = starts[: max(multistart, 3)]
    bounds = [(-12, 9), (-8, 8), (-4 * np.pi, 4 * np.pi)]

    def objective(beta: np.ndarray) -> float:
        lambda0, sine, cosine = unpack(beta)
        ll = seasonal_mou_logpdf(x_next, x_prev, t_next_days, t_prev_days, delta_days, theta, kappa, sigma, lambda0, sine, cosine)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "ftol": 1e-11}) for start in starts]
    best = min(results, key=lambda result: result.fun)
    lambda0, sine, cosine = unpack(best.x)
    ll = seasonal_mou_logpdf(x_next, x_prev, t_next_days, t_prev_days, delta_days, theta, kappa, sigma, lambda0, sine, cosine)
    loglik = float(np.sum(ll)); gradient, condition, covariance = optimizer_diagnostics(best, len(x_next)); repeats = repeated_solution_count(results, 1e-7)
    min_intensity = lambda0 * (1 - np.sqrt(sine * sine + cosine * cosine)); max_intensity = lambda0 * (1 + np.sqrt(sine * sine + cosine * cosine))
    weak = condition is None or condition > 1e12 or repeats < 2
    p = 3
    return FitResult(
        "SeasonalIntensityMOU", {"theta": theta, "kappa": kappa, "sigma": sigma, "lambda0": lambda0, "seasonal_sine": sine, "seasonal_cosine": cosine},
        loglik, len(x_next), -2 * loglik + 2 * p, -2 * loglik + p * np.log(len(x_next)), bool(best.success), str(best.message),
        gradient, condition, covariance, None, {"lambda0": 0.5, "seasonal_sine": 0.0, "seasonal_cosine": 0.0},
        {"base_ou_parameters_fixed": True, "minimum_intensity": min_intensity, "maximum_intensity": max_intensity,
         "repeated_best_solutions": repeats, "weak_identification": weak, "deterministic_intensity": True},
    )


def time_varying_normal_parameters(t_days: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_days, dtype=float); years = (t - params["time_center_days"]) / 365.2425
    omega = 2 * np.pi / 365.2425
    mean = params["mean0"] + params["mean_trend"] * years + params["mean_sine"] * np.sin(omega * t) + params["mean_cosine"] * np.cos(omega * t)
    log_scale = params["log_scale0"] + params["log_scale_trend"] * years
    return mean, np.exp(np.clip(log_scale, -20, 2))


def time_varying_gaussian_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    t_next_days: np.ndarray,
    t_prev_days: np.ndarray,
    delta_days: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    mean_prev, scale_prev = time_varying_normal_parameters(t_prev_days, params)
    mean_next, scale_next = time_varying_normal_parameters(t_next_days, params)
    rho = np.exp(-params["kappa"] * np.asarray(delta_days, dtype=float))
    z_prev = (np.asarray(x_prev) - mean_prev) / scale_prev
    conditional_mean = mean_next + scale_next * rho * z_prev
    conditional_sd = scale_next * np.sqrt(np.maximum(1 - rho * rho, 1e-16))
    return norm.logpdf(x_next, conditional_mean, conditional_sd)


def fit_time_varying_gaussian(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    t_next_days: np.ndarray,
    t_prev_days: np.ndarray,
    delta_days: np.ndarray,
    multistart: int = 6,
) -> FitResult:
    x_all = np.r_[x_prev, x_next]; mean0, scale0 = float(np.mean(x_all)), max(float(np.std(x_all)), 1e-8)
    center = float(np.mean(t_prev_days)); _, k0, _ = ar1_initial_values(x_prev, x_next, float(np.median(delta_days)))
    base = np.array([mean0, 0.0, 0.0, 0.0, np.log(scale0), 0.0, np.log(k0)])
    starts = [base + s * np.array([0.1 * scale0, 0.1 * scale0, 0.1 * scale0, 0.1 * scale0, 0.2, 0.1, 0.5]) for s in np.linspace(-1, 1, multistart)]
    bounds = [
        (mean0 - 10 * scale0, mean0 + 10 * scale0), (-10 * scale0, 10 * scale0),
        (-10 * scale0, 10 * scale0), (-10 * scale0, 10 * scale0),
        (np.log(scale0) - 6, np.log(scale0) + 4), (-3, 3), (-9, 9),
    ]

    def unpack(beta: np.ndarray) -> dict[str, float]:
        return {"mean0": float(beta[0]), "mean_trend": float(beta[1]), "mean_sine": float(beta[2]), "mean_cosine": float(beta[3]),
                "log_scale0": float(beta[4]), "log_scale_trend": float(beta[5]), "kappa": float(np.exp(beta[6])), "time_center_days": center}

    def objective(beta: np.ndarray) -> float:
        ll = time_varying_gaussian_logpdf(x_next, x_prev, t_next_days, t_prev_days, delta_days, unpack(beta))
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 350, "ftol": 1e-11}) for start in starts]
    best = min(results, key=lambda result: result.fun); params = unpack(best.x)
    ll = time_varying_gaussian_logpdf(x_next, x_prev, t_next_days, t_prev_days, delta_days, params)
    loglik = float(np.sum(ll)); gradient, condition, covariance = optimizer_diagnostics(best, len(x_next)); repeats = repeated_solution_count(results, 1e-7)
    boundary = bool(np.any(np.isclose(best.x, [b[0] for b in bounds], atol=1e-5)) or np.any(np.isclose(best.x, [b[1] for b in bounds], atol=1e-5)))
    weak = condition is None or condition > 1e12 or repeats < 2 or boundary; p = len(best.x)
    return FitResult(
        "FullyTimeVaryingMarginalCopula", params, loglik, len(x_next), -2 * loglik + 2 * p, -2 * loglik + p * np.log(len(x_next)),
        bool(best.success), str(best.message), gradient, condition, covariance, None,
        {"mean0": mean0, "scale0": scale0, "kappa": k0},
        {"copula": "Gaussian semigroup", "marginal": "deterministic trend/seasonal Normal", "repeated_best_solutions": repeats,
         "boundary_hit": boundary, "weak_identification": weak},
    )


@dataclass
class MarkovSwitchingFit:
    fit: FitResult
    transition_matrix: np.ndarray


def _stationary_probabilities(matrix: np.ndarray) -> np.ndarray:
    p01, p10 = matrix[0, 1], matrix[1, 0]
    total = p01 + p10
    return np.array([p10 / total, p01 / total]) if total > 0 else np.array([0.5, 0.5])


def markov_switching_filter_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappas: np.ndarray,
    sigmas: np.ndarray,
    transition: np.ndarray,
    initial_probabilities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Compute both emission-density vectors once.  The previous implementation
    # allocated six length-one arrays at every observation and likelihood call;
    # this algebraically identical scalar recursion is orders of magnitude
    # faster and retains per-observation predictive log scores.
    xn = np.asarray(x_next, dtype=float); xp = np.asarray(x_prev, dtype=float)
    h = np.asarray(delta_days, dtype=float)
    log_emissions = np.empty((len(xn), 2))
    for regime in range(2):
        rho = np.exp(-float(kappas[regime]) * h)
        mean = theta + (xp - theta) * rho
        variance = float(sigmas[regime]) ** 2 * (-np.expm1(-2 * float(kappas[regime]) * h)) / (2 * float(kappas[regime]))
        log_emissions[:, regime] = norm.logpdf(xn, mean, np.sqrt(np.maximum(variance, 1e-300)))

    initial = _stationary_probabilities(transition) if initial_probabilities is None else np.asarray(initial_probabilities, dtype=float)
    alpha0, alpha1 = float(initial[0]), float(initial[1])
    p00, p01 = float(transition[0, 0]), float(transition[0, 1])
    p10, p11 = float(transition[1, 0]), float(transition[1, 1])
    log_scores = np.empty(len(xn)); filtered = np.empty((len(xn), 2))
    tiny = np.finfo(float).tiny
    for idx in range(len(xn)):
        predicted0 = alpha0 * p00 + alpha1 * p10
        predicted1 = alpha0 * p01 + alpha1 * p11
        log0, log1 = float(log_emissions[idx, 0]), float(log_emissions[idx, 1])
        maximum = max(log0, log1)
        weighted0 = predicted0 * math.exp(log0 - maximum)
        weighted1 = predicted1 * math.exp(log1 - maximum)
        scaled_likelihood = max(weighted0 + weighted1, tiny)
        log_scores[idx] = maximum + math.log(scaled_likelihood)
        alpha0, alpha1 = weighted0 / scaled_likelihood, weighted1 / scaled_likelihood
        filtered[idx, 0], filtered[idx, 1] = alpha0, alpha1
    return log_scores, filtered


def fit_markov_switching_ou(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    multistart: int = 5,
) -> MarkovSwitchingFit:
    theta0, k0, s0 = ar1_initial_values(x_prev, x_next, float(np.median(delta_days)))
    # kappa_2 = kappa_1 + exp(beta_2) imposes smooth label ordering.  Sorting
    # kappas without sorting their sigmas made the former objective nonsmooth
    # and could silently mismatch regime-specific volatility parameters.
    base = np.array([theta0, np.log(k0 / 2), np.log(k0 * 1.5), np.log(s0 * 0.7), np.log(s0 * 1.5), 3.0, 3.0])
    starts = [base + s * np.array([0.1 * np.std(x_prev), 0.5, -0.5, 0.3, -0.3, 1.0, -1.0]) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-0.1, 0.1), (-9, 9), (-9, 9), (-16, 2), (-16, 2), (-8, 8), (-8, 8)]

    def unpack(beta: np.ndarray):
        theta = float(beta[0]); kappa_1 = float(np.exp(beta[1])); kappas = np.array([kappa_1, kappa_1 + float(np.exp(beta[2]))]); sigmas = np.exp(beta[3:5])
        p00, p11 = expit(beta[5]), expit(beta[6]); transition = np.array([[p00, 1 - p00], [1 - p11, p11]])
        return theta, kappas, sigmas, transition

    def objective(beta: np.ndarray) -> float:
        theta, kappas, sigmas, transition = unpack(beta)
        ll, _ = markov_switching_filter_logpdf(x_next, x_prev, delta_days, theta, kappas, sigmas, transition)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 180, "ftol": 1e-9}) for start in starts]
    best = min(results, key=lambda result: result.fun); theta, kappas, sigmas, transition = unpack(best.x)
    ll, filtered = markov_switching_filter_logpdf(x_next, x_prev, delta_days, theta, kappas, sigmas, transition)
    loglik = float(np.sum(ll)); gradient, condition, covariance = optimizer_diagnostics(best, len(x_next)); repeats = repeated_solution_count(results, 1e-6); p = 7
    observed_span_days = float(np.sum(delta_days))
    slow_regime_reversion_units = float(kappas[0] * observed_span_days)
    boundary = bool(
        np.any(best.x - np.array([item[0] for item in bounds]) < 0.5)
        or np.any(np.array([item[1] for item in bounds]) - best.x < 0.5)
    )
    weak = bool(
        condition is None or condition > 1e12 or repeats < 2
        or abs(kappas[1] - kappas[0]) < 1e-4
        or slow_regime_reversion_units < 1.0 or boundary
    )
    fit = FitResult(
        "MarkovSwitchingOU", {"theta": theta, "kappa_1": float(kappas[0]), "kappa_2": float(kappas[1]), "sigma_1": float(sigmas[0]), "sigma_2": float(sigmas[1]),
                              "p00": float(transition[0, 0]), "p11": float(transition[1, 1])},
        loglik, len(x_next), -2 * loglik + 2 * p, -2 * loglik + p * np.log(len(x_next)), bool(best.success), str(best.message), gradient, condition, covariance, None,
        {"theta": theta0, "kappa": k0, "sigma": s0},
        {"repeated_best_solutions": repeats, "weak_identification": weak, "last_filtered_probability_regime_1": float(filtered[-1, 0]),
         "ordered_kappa_parameterization": "kappa_2=kappa_1+exp(beta_2)", "vectorized_emissions": True,
         "boundary_hit": boundary, "observed_span_days": observed_span_days,
         "slow_regime_reversion_units_observed": slow_regime_reversion_units},
    )
    return MarkovSwitchingFit(fit, transition)
