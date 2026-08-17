from __future__ import annotations

import numpy as np

from ..estimation.optimize import FitResult, multistart_minimize, optimizer_diagnostics, repeated_solution_count


def fit_threshold_ou(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float = 0.0,
    heterogeneous_sigma: bool = False,
    multistart: int = 8,
) -> FitResult:
    x_next = np.asarray(x_next, dtype=float)
    x_prev = np.asarray(x_prev, dtype=float)
    delta = np.asarray(delta_days, dtype=float)
    change = x_next - x_prev
    plus = x_prev >= theta
    scale0 = max(float(np.std(change) / np.sqrt(np.mean(delta))), 1e-8)
    base = np.log([50.0, 50.0, scale0, scale0]) if heterogeneous_sigma else np.log([50.0, 50.0, scale0])
    starts = [base + s * np.linspace(-0.6, 0.6, len(base)) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-10, 14), (-10, 14)] + [(-16, 5)] * (2 if heterogeneous_sigma else 1)

    def unpack(beta: np.ndarray) -> tuple[float, float, float, float]:
        values = np.exp(beta)
        return (
            float(values[0]), float(values[1]), float(values[2]),
            float(values[3] if heterogeneous_sigma else values[2]),
        )

    def objective(beta: np.ndarray) -> float:
        kp, km, sp, sm = unpack(beta)
        kappa = np.where(plus, kp, km)
        sigma = np.where(plus, sp, sm)
        mean_change = kappa * (theta - x_prev) * delta
        variance = sigma * sigma * delta
        ll = -0.5 * (np.log(2 * np.pi * variance) + (change - mean_change) ** 2 / variance)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    best, results = multistart_minimize(objective, starts, bounds=bounds)
    kp, km, sp, sm = unpack(best.x)
    loglik = -float(best.fun) * len(x_next)
    gradient, condition, covariance = optimizer_diagnostics(best, len(x_next))
    repeats = repeated_solution_count(results)
    p = len(best.x)
    return FitResult(
        model_name="ThresholdOU_Heteroskedastic" if heterogeneous_sigma else "ThresholdOU",
        params={"theta": theta, "kappa_plus": kp, "kappa_minus": km, "sigma_plus": sp, "sigma_minus": sm},
        loglik=loglik,
        nobs=len(x_next),
        aic=-2 * loglik + 2 * p,
        bic=-2 * loglik + p * np.log(len(x_next)),
        converged=bool(best.success),
        optimizer_message=str(best.message),
        gradient_norm=gradient,
        hessian_condition=condition,
        covariance=covariance,
        robust_covariance=None,
        start_values={"kappa_plus": 50.0, "kappa_minus": 50.0, "sigma": scale0},
        metadata={
            "half_life_plus_minutes": 1440 * np.log(2) / kp,
            "half_life_minus_minutes": 1440 * np.log(2) / km,
            "repeated_best_solutions": repeats,
            "weak_identification": condition is None or condition > 1e12 or repeats < 2,
            "likelihood_type": "Euler_Gaussian_QMLE",
        },
    )

