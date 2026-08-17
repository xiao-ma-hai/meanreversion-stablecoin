from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..estimation.optimize import FitResult, optimizer_diagnostics
from ..transforms import affine_inverse
from .ou import ar1_initial_values, ou_logpdf


def transformed_ou_inverse(
    x: np.ndarray,
    t_days: np.ndarray,
    kappa: float,
    d: float,
    b0: float = 0.0,
    a0: float = 1.0,
) -> np.ndarray:
    a = a0 * np.exp(np.clip((kappa - d) * np.asarray(t_days), -50, 50))
    b = b0 * np.exp(np.clip(-d * np.asarray(t_days), -50, 50))
    return affine_inverse(x, a, b)


def transformed_ou_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    kappa: float,
    d: float,
    sigma: float,
    b0: float = 0.0,
) -> np.ndarray:
    r_next = transformed_ou_inverse(x_next, t_next, kappa, d, b0)
    r_prev = transformed_ou_inverse(x_prev, t_prev, kappa, d, b0)
    log_jac = -np.clip((kappa - d) * np.asarray(t_next), -50, 50)
    return ou_logpdf(r_next, r_prev, delta_days, 0.0, kappa, sigma) + log_jac


def fit_transformed_ou(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
) -> FitResult:
    theta0, kappa0, sigma0 = ar1_initial_values(x_prev, x_next, float(np.median(delta_days)))
    base = np.array([np.log(kappa0), np.log(sigma0), 0.0])
    starts = [base + np.array([s * 0.5, -s * 0.25, s]) for s in np.linspace(-1, 1, 8)]

    def unpack(beta: np.ndarray) -> tuple[float, float, float]:
        kappa = float(np.exp(beta[0]))
        sigma = float(np.exp(beta[1]))
        gamma = float(0.002 * np.tanh(beta[2]))
        d = kappa - gamma
        return kappa, d, sigma

    def objective(beta: np.ndarray) -> float:
        kappa, d, sigma = unpack(beta)
        if d <= 0:
            return 1e100
        ll = transformed_ou_logpdf(x_next, x_prev, t_next, t_prev, delta_days, kappa, d, sigma)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    results = [minimize(objective, s, method="L-BFGS-B", bounds=[(-10, 14), (-16, 5), (-6, 6)], options={"maxiter": 300, "ftol": 1e-11}) for s in starts]
    best = min((r for r in results if np.isfinite(r.fun)), key=lambda r: r.fun)
    kappa, d, sigma = unpack(best.x)
    loglik = -float(best.fun) * len(x_next)
    gradient, condition, covariance = optimizer_diagnostics(best, len(x_next))
    repeated = sum(r.success and abs(r.fun - best.fun) < 1e-8 for r in results)
    return FitResult(
        model_name="AffineTransformedOU",
        params={"A0": 1.0, "B0": 0.0, "kappa": kappa, "d": d, "sigma": sigma, "gamma_kappa_minus_d": kappa - d},
        loglik=loglik,
        nobs=len(x_next),
        aic=-2 * loglik + 6,
        bic=-2 * loglik + 3 * np.log(len(x_next)),
        converged=bool(best.success), optimizer_message=str(best.message),
        gradient_norm=gradient, hessian_condition=condition, covariance=covariance,
        robust_covariance=None,
        start_values={"kappa": kappa0, "d": kappa0, "sigma": sigma0, "B0": 0.0},
        metadata={
            "variance_regime": "decaying" if d > kappa else ("growing" if d < kappa else "constant"),
            "A0_fixed_for_identification": True,
            "B0_fixed_simplified_version": True,
            "repeated_best_solutions": repeated,
            "weak_identification": condition is None or condition > 1e12 or repeated < 2,
        },
    )

