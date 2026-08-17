from __future__ import annotations

import numpy as np

from ..estimation.optimize import FitResult
from ..transforms import affine_inverse
from .cir import cir_logpdf


def transformed_cir_inverse(
    price: np.ndarray,
    t_days: np.ndarray,
    theta: float,
    kappa: float,
    d: float,
    b0: float = 0.0,
    a0: float = 1.0,
) -> np.ndarray:
    a = a0 * np.exp(np.clip((kappa - d) * np.asarray(t_days), -50, 50))
    b = theta * (1.0 - a) + b0 * np.exp(np.clip(-d * np.asarray(t_days), -50, 50))
    return affine_inverse(price, a, b)


def transformed_cir_logpdf(
    p_next: np.ndarray,
    p_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    d: float,
    sigma: float,
) -> np.ndarray:
    r_next = transformed_cir_inverse(p_next, t_next, theta, kappa, d)
    r_prev = transformed_cir_inverse(p_prev, t_prev, theta, kappa, d)
    if np.any(r_next <= 0) or np.any(r_prev <= 0):
        return np.full_like(r_next, -np.inf)
    return cir_logpdf(r_next, r_prev, delta_days, theta, kappa, sigma) - np.clip((kappa - d) * t_next, -50, 50)


def fit_transformed_cir_profile(
    p_next: np.ndarray,
    p_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    base_fit: FitResult,
    max_nobs: int = 75_000,
) -> tuple[FitResult, np.ndarray]:
    if len(p_next) > max_nobs:
        idx = np.linspace(0, len(p_next) - 1, max_nobs, dtype=int)
    else:
        idx = np.arange(len(p_next))
    theta = base_fit.params["theta"]
    kappa = base_fit.params["kappa"]
    sigma = base_fit.params["sigma"]
    gamma_grid = np.linspace(-0.001, 0.001, 81)
    profile = []
    for gamma in gamma_grid:
        d = kappa - gamma
        if d <= 0:
            profile.append(-np.inf)
            continue
        ll = transformed_cir_logpdf(
            p_next[idx], p_prev[idx], t_next[idx], t_prev[idx], delta_days[idx], theta, kappa, d, sigma
        )
        profile.append(float(np.sum(ll)) if np.isfinite(ll).all() else -np.inf)
    profile_arr = np.asarray(profile)
    best_idx = int(np.nanargmax(profile_arr))
    gamma = float(gamma_grid[best_idx])
    d = kappa - gamma
    scaled_ll = float(profile_arr[best_idx] * len(p_next) / len(idx))
    curvature = np.nan
    if 0 < best_idx < len(gamma_grid) - 1:
        step = gamma_grid[1] - gamma_grid[0]
        curvature = -(profile_arr[best_idx + 1] - 2 * profile_arr[best_idx] + profile_arr[best_idx - 1]) / step**2
    weak = (not np.isfinite(curvature) or curvature <= 0 or best_idx in (0, len(gamma_grid) - 1)
            or bool(base_fit.metadata.get("weak_identification", False)))
    result = FitResult(
        model_name="AffineTransformedCIR",
        params={"A0": 1.0, "B0": 0.0, "theta": theta, "kappa": kappa, "d": d, "sigma": sigma, "gamma_kappa_minus_d": gamma},
        loglik=scaled_ll, nobs=len(p_next), aic=-2 * scaled_ll + 8,
        bic=-2 * scaled_ll + 4 * np.log(len(p_next)), converged=np.isfinite(scaled_ll),
        optimizer_message="Profile likelihood with base CIR parameters fixed",
        gradient_norm=np.nan, hessian_condition=None, covariance=None, robust_covariance=None,
        start_values={"theta": theta, "kappa": kappa, "d": kappa, "sigma": sigma},
        metadata={
            "A0_fixed_for_identification": True, "B0_fixed_simplified_version": True,
            "base_cir_parameters_fixed": True, "profile_curvature": curvature,
            "estimation_nobs": len(idx), "weak_identification": weak,
            "variance_regime": "decaying" if gamma < 0 else ("growing" if gamma > 0 else "constant"),
        },
    )
    return result, np.column_stack([gamma_grid, profile_arr])
