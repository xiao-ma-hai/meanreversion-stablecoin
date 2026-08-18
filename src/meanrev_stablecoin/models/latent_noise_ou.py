from __future__ import annotations

import numpy as np

from ..estimation.optimize import FitResult, multistart_minimize, optimizer_diagnostics, repeated_solution_count
from .ou import ar1_initial_values


def latent_noise_ou_loglik(
    observations: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    tau: float,
    kappa: float,
    observation_sd: float,
) -> float:
    """Exact scalar Kalman likelihood for latent stationary OU plus noise."""
    y = np.asarray(observations, dtype=float)
    delta = np.asarray(delta_days, dtype=float)
    if len(y) < 2 or len(delta) != len(y) - 1:
        raise ValueError("delta_days must have len(observations)-1 entries")
    if tau <= 0 or kappa <= 0 or observation_sd < 0 or np.any(delta <= 0):
        return -np.inf
    obs_var = max(observation_sd * observation_sd, np.finfo(float).tiny)
    state_mean = float(theta)
    state_var = float(tau * tau)
    loglik = 0.0
    for idx, value in enumerate(y):
        innovation_var = state_var + obs_var
        innovation = value - state_mean
        loglik += -0.5 * (np.log(2 * np.pi * innovation_var) + innovation * innovation / innovation_var)
        gain = state_var / innovation_var
        filtered_mean = state_mean + gain * innovation
        filtered_var = max((1.0 - gain) * state_var, 0.0)
        if idx == len(y) - 1:
            break
        rho = float(np.exp(-kappa * delta[idx]))
        state_mean = theta + rho * (filtered_mean - theta)
        state_var = rho * rho * filtered_var + tau * tau * (1.0 - rho * rho)
    return float(loglik)


def fit_latent_noise_ou(
    observations: np.ndarray,
    delta_days: np.ndarray,
    multistart: int = 8,
) -> FitResult:
    y = np.asarray(observations, dtype=float)
    delta = np.asarray(delta_days, dtype=float)
    if len(y) < 20 or len(delta) != len(y) - 1:
        raise ValueError("latent-noise OU requires at least 20 ordered observations")
    theta0, kappa0, sigma0 = ar1_initial_values(y[:-1], y[1:], float(np.median(delta)))
    tau0 = max(float(sigma0 / np.sqrt(2 * kappa0)), float(np.std(y)) * 0.2, 1e-8)
    observed_increment_sd = max(float(np.std(np.diff(y))) / np.sqrt(2), 1e-8)
    noise_scales = np.array([0.02, 0.1, 0.3, 0.6, 0.9, 1.2, 2.0])
    starts = [
        np.array([theta0, np.log(tau0), np.log(kappa0), np.log(observed_increment_sd * scale)])
        for scale in noise_scales[:max(multistart, 1)]
    ]
    state_scale = max(float(np.std(y)), 1e-5)
    bounds = [
        (theta0 - 25 * state_scale, theta0 + 25 * state_scale),
        (-16, 0),
        (-10, 14),
        (-20, 0),
    ]

    def unpack(beta: np.ndarray) -> tuple[float, float, float, float]:
        return float(beta[0]), float(np.exp(beta[1])), float(np.exp(beta[2])), float(np.exp(beta[3]))

    def objective(beta: np.ndarray) -> float:
        params = unpack(beta)
        value = latent_noise_ou_loglik(y, delta, *params)
        return -value / len(y) if np.isfinite(value) else 1e100

    best, results = multistart_minimize(
        objective, starts, bounds=bounds,
        options={"maxiter": 260, "ftol": 1e-11, "gtol": 2e-6},
    )
    theta, tau, kappa, observation_sd = unpack(best.x)
    loglik = latent_noise_ou_loglik(y, delta, theta, tau, kappa, observation_sd)
    gradient, condition, covariance = optimizer_diagnostics(best, len(y))
    repeats = repeated_solution_count(results, tolerance=1e-7)
    p = 4
    return FitResult(
        model_name="LatentNoiseOU",
        params={
            "theta": theta,
            "tau": tau,
            "kappa": kappa,
            "sigma": float(np.sqrt(2 * kappa) * tau),
            "observation_sd": observation_sd,
        },
        loglik=loglik,
        nobs=len(y),
        aic=-2 * loglik + 2 * p,
        bic=-2 * loglik + p * np.log(len(y)),
        converged=bool(best.success and np.isfinite(loglik)),
        optimizer_message=str(best.message),
        gradient_norm=gradient,
        hessian_condition=condition,
        covariance=covariance,
        robust_covariance=None,
        start_values={
            "theta": theta0, "tau": tau0, "kappa": kappa0,
            "observation_sd": observed_increment_sd,
        },
        metadata={
            "likelihood_type": "exact_linear_gaussian_state_space",
            "latent_half_life_minutes": 1440 * np.log(2) / kappa,
            "observed_stationary_sd": float(np.sqrt(tau * tau + observation_sd * observation_sd)),
            "noise_variance_share": float(observation_sd**2 / (tau**2 + observation_sd**2)),
            "repeated_best_solutions": repeats,
            "weak_identification": condition is None or condition > 1e12 or repeats < 2,
        },
    )
