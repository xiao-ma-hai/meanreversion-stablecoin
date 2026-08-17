from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import hyp1f1, hyperu

from ..estimation.optimize import FitResult, optimizer_diagnostics, repeated_solution_count
from .cir import cir_logpdf
from .ou import ar1_initial_values, ou_logpdf


_TINY = np.finfo(float).tiny


@dataclass
class TransformEvaluation:
    latent: np.ndarray
    log_abs_inverse_jacobian: np.ndarray
    valid: np.ndarray
    clipped: np.ndarray


def _exp_with_flag(value: np.ndarray, limit: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=float)
    clipped = np.abs(value) > limit
    return np.exp(np.clip(value, -limit, limit)), clipped


def exponential_ou_coefficients(
    t_days: np.ndarray,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients in article OU Example 2, with overflow flags."""
    t = np.asarray(t_days, dtype=float)
    ek, clipped_a = _exp_with_flag(kappa * t)
    e2k, clipped_b = _exp_with_flag(2.0 * kappa * t)
    ed, clipped_c = _exp_with_flag(-d * t)
    a = a0 * ek
    b = b0 - sigma * sigma * a0 * a0 * (e2k - 1.0) / (4.0 * kappa) - d * t
    c = c0 * ed
    return a, b, c, clipped_a | clipped_b | clipped_c


def exponential_ou_inverse(
    y: np.ndarray,
    t_days: np.ndarray,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> TransformEvaluation:
    y = np.asarray(y, dtype=float)
    a, b, c, clipped = exponential_ou_coefficients(t_days, kappa, sigma, d, a0, b0, c0)
    gap = y - c
    valid = (a > 0) & (gap > 0) & np.isfinite(a + b + c + gap)
    latent = np.full_like(y, np.nan)
    log_jac = np.full_like(y, -np.inf)
    latent[valid] = (np.log(gap[valid]) - b[valid]) / a[valid]
    log_jac[valid] = -np.log(a[valid]) - np.log(gap[valid])
    valid &= np.isfinite(latent) & np.isfinite(log_jac)
    return TransformEvaluation(latent, log_jac, valid, clipped)


def exponential_ou_logpdf(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> tuple[np.ndarray, dict[str, float]]:
    nxt = exponential_ou_inverse(y_next, t_next, kappa, sigma, d, a0, b0, c0)
    prv = exponential_ou_inverse(y_prev, t_prev, kappa, sigma, d, a0, b0, c0)
    valid = nxt.valid & prv.valid
    result = np.full_like(np.asarray(y_next, dtype=float), -np.inf)
    if np.any(valid):
        result[valid] = ou_logpdf(
            nxt.latent[valid], prv.latent[valid], np.asarray(delta_days)[valid], 0.0, kappa, sigma
        ) + nxt.log_abs_inverse_jacobian[valid]
    return result, {
        "support_violation_rate": float(1.0 - np.mean(valid)),
        "coefficient_clipping_rate": float(np.mean(nxt.clipped | prv.clipped)),
    }


def ou_special_forward_and_derivative(
    latent: np.ndarray,
    t_days: np.ndarray,
    kappa: float,
    sigma: float,
    c: float,
    d: float,
    c1: float = 1.0,
    c2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Article OU Example 3. The derivative exposes its global non-monotonicity."""
    x = np.asarray(latent, dtype=float)
    z = kappa * x * x / (sigma * sigma)
    p, q = (c - d) / (2.0 * kappa), 0.5
    j = c1 * hyp1f1(p, q, z) + c2 * hyperu(p, q, z)
    dzdx = 2.0 * kappa * x / (sigma * sigma)
    derivative_z = c1 * (p / q) * hyp1f1(p + 1.0, q + 1.0, z)
    if c2 != 0:
        derivative_z += c2 * (-p) * hyperu(p + 1.0, q + 1.0, z)
    decay = np.exp(np.clip(-c * np.asarray(t_days, dtype=float), -80, 80))
    return decay * j, decay * derivative_z * dzdx


def ou_special_monotonicity_diagnostic(
    kappa: float, sigma: float, c: float, d: float, stationary_sd_multiple: float = 6.0, grid_points: int = 401
) -> dict[str, float | bool | str]:
    sd = sigma / np.sqrt(2.0 * kappa)
    grid = np.linspace(-stationary_sd_multiple * sd, stationary_sd_multiple * sd, grid_points)
    _, derivative = ou_special_forward_and_derivative(grid, np.zeros_like(grid), kappa, sigma, c, d)
    positive = float(np.mean(derivative > 1e-12))
    negative = float(np.mean(derivative < -1e-12))
    return {
        "global_monotonicity_pass": bool(positive == 1.0 or negative == 1.0),
        "positive_derivative_share": positive,
        "negative_derivative_share": negative,
        "zero_or_nonfinite_derivative_share": float(np.mean(~np.isfinite(derivative) | (np.abs(derivative) <= 1e-12))),
        "failure_reason": "dependence on latent squared state makes the map even and non-injective on R",
    }


def quadratic_cir_coefficients(
    t_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t_days, dtype=float)
    e2, clip2 = _exp_with_flag((2.0 * kappa - d) * t)
    e1, clip1 = _exp_with_flag((kappa - d) * t)
    ed, clipd = _exp_with_flag(-d * t)
    s = sigma * sigma + 2.0 * kappa * theta
    a = a0 * e2
    b = -(a0 * s / kappa) * e2 + (b0 + a0 * s / kappa) * e1
    c = (
        theta
        + theta * a0 * s * e2 / (2.0 * kappa)
        - theta * (kappa * b0 + a0 * s) * e1 / kappa
        + (c0 - theta + theta * b0 + theta * a0 * s / (2.0 * kappa)) * ed
    )
    return a, b, c, clip2 | clip1 | clipd


def quadratic_cir_inverse(
    y: np.ndarray,
    t_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> TransformEvaluation:
    y = np.asarray(y, dtype=float)
    a, b, c, clipped = quadratic_cir_coefficients(t_days, theta, kappa, sigma, d, a0, b0, c0)
    discriminant = b * b + 4.0 * a * (y - c)
    valid = (a > 0) & (b >= 0) & (discriminant > 0) & np.isfinite(discriminant)
    latent = np.full_like(y, np.nan)
    log_jac = np.full_like(y, -np.inf)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    latent[valid] = 2.0 * (y[valid] - c[valid]) / (b[valid] + root[valid])
    derivative = 2.0 * a * latent + b
    valid &= (latent >= 0) & (derivative > 0) & np.isfinite(latent)
    log_jac[valid] = -np.log(derivative[valid])
    return TransformEvaluation(latent, log_jac, valid, clipped)


def quadratic_cir_logpdf(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> tuple[np.ndarray, dict[str, float]]:
    nxt = quadratic_cir_inverse(y_next, t_next, theta, kappa, sigma, d, a0, b0, c0)
    prv = quadratic_cir_inverse(y_prev, t_prev, theta, kappa, sigma, d, a0, b0, c0)
    valid = nxt.valid & prv.valid
    result = np.full_like(np.asarray(y_next, dtype=float), -np.inf)
    if np.any(valid):
        result[valid] = cir_logpdf(
            nxt.latent[valid], prv.latent[valid], np.asarray(delta_days)[valid], theta, kappa, sigma
        ) + nxt.log_abs_inverse_jacobian[valid]
    a_n, b_n, _, _ = quadratic_cir_coefficients(t_next, theta, kappa, sigma, d, a0, b0, c0)
    a_p, b_p, _, _ = quadratic_cir_coefficients(t_prev, theta, kappa, sigma, d, a0, b0, c0)
    return result, {
        "support_violation_rate": float(1.0 - np.mean(valid)),
        "monotonicity_violation_rate": float(np.mean((a_n <= 0) | (b_n < 0) | (a_p <= 0) | (b_p < 0))),
        "coefficient_clipping_rate": float(np.mean(nxt.clipped | prv.clipped)),
    }


def exponential_cir_coefficients(
    t_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t_days, dtype=float)
    sigma2 = sigma * sigma
    first = 2.0 * kappa / a0 - sigma2
    valid_first = first > 0
    safe_first = max(first, _TINY)
    exp_neg, clipped = _exp_with_flag(-kappa * t)
    a = 2.0 * kappa / (sigma2 + first * exp_neg)
    log_term = np.logaddexp(np.log(safe_first), np.log(sigma2) + kappa * t)
    q = 2.0 * kappa * theta / sigma2
    b = -d * t - q * log_term + b0 + q * np.log(2.0 * kappa / a0)
    c = theta + (c0 - theta) * np.exp(np.clip(-d * t, -80, 80))
    invalid = clipped | (~np.isfinite(a + b + c)) | (not valid_first)
    return a, b, c, np.asarray(invalid, dtype=bool)


def exponential_cir_inverse(
    y: np.ndarray,
    t_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> TransformEvaluation:
    y = np.asarray(y, dtype=float)
    a, b, c, clipped = exponential_cir_coefficients(t_days, theta, kappa, sigma, d, a0, b0, c0)
    gap = y - c
    valid = (a > 0) & (gap > 0) & np.isfinite(a + b + c + gap) & ~clipped
    latent = np.full_like(y, np.nan)
    log_jac = np.full_like(y, -np.inf)
    latent[valid] = (np.log(gap[valid]) - b[valid]) / a[valid]
    valid &= (latent >= 0) & np.isfinite(latent)
    log_jac[valid] = -np.log(a[valid]) - np.log(gap[valid])
    return TransformEvaluation(latent, log_jac, valid, clipped)


def exponential_cir_logpdf(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    d: float,
    a0: float,
    b0: float,
    c0: float,
) -> tuple[np.ndarray, dict[str, float]]:
    nxt = exponential_cir_inverse(y_next, t_next, theta, kappa, sigma, d, a0, b0, c0)
    prv = exponential_cir_inverse(y_prev, t_prev, theta, kappa, sigma, d, a0, b0, c0)
    valid = nxt.valid & prv.valid
    result = np.full_like(np.asarray(y_next, dtype=float), -np.inf)
    if np.any(valid):
        result[valid] = cir_logpdf(
            nxt.latent[valid], prv.latent[valid], np.asarray(delta_days)[valid], theta, kappa, sigma
        ) + nxt.log_abs_inverse_jacobian[valid]
    return result, {
        "support_violation_rate": float(1.0 - np.mean(valid)),
        "coefficient_clipping_rate": float(np.mean(nxt.clipped | prv.clipped)),
    }


def cir_special_forward_and_derivative(
    latent: np.ndarray,
    t_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    c: float,
    d: float,
    c1: float = 1.0,
    c2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(latent, dtype=float)
    p, q = (c - d) / kappa, 2.0 * kappa * theta / (sigma * sigma)
    z = 2.0 * kappa * r / (sigma * sigma)
    j = c1 * hyp1f1(p, q, z) + c2 * hyperu(p, q, z)
    derivative_z = c1 * (p / q) * hyp1f1(p + 1.0, q + 1.0, z)
    if c2 != 0:
        derivative_z += c2 * (-p) * hyperu(p + 1.0, q + 1.0, z)
    decay = np.exp(np.clip(-c * np.asarray(t_days, dtype=float), -80, 80))
    return theta + decay * j, decay * derivative_z * (2.0 * kappa / (sigma * sigma))


def _failed_fit(name: str, nobs: int, message: str, metadata: dict) -> FitResult:
    return FitResult(
        model_name=name, params={}, loglik=float("-inf"), nobs=nobs, aic=float("inf"), bic=float("inf"),
        converged=False, optimizer_message=message, gradient_norm=float("nan"), hessian_condition=None,
        covariance=None, robust_covariance=None, start_values={}, metadata={"weak_identification": True, **metadata},
    )


def fit_exponential_ou(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    multistart: int = 6,
) -> FitResult:
    y_next = np.asarray(y_next, dtype=float); y_prev = np.asarray(y_prev, dtype=float)
    # Base latent scale is initialized from log prices. This is only a starting value.
    _, k0, s0 = ar1_initial_values(np.log(y_prev), np.log(y_next), float(np.median(delta_days)))
    minimum = float(min(y_next.min(), y_prev.min()))

    def unpack(beta: np.ndarray) -> tuple[float, float, float, float, float, float]:
        kappa, sigma, d, a0 = np.exp(beta[:4])
        b0 = float(beta[4])
        c0 = float(0.995 * minimum / (1.0 + np.exp(-beta[5])))
        return map(float, (kappa, sigma, d, a0, b0, c0))

    base = np.array([np.log(max(k0, 1e-3)), np.log(max(s0, 1e-5)), np.log(max(k0, 1e-3)), 0.0, 0.0, -6.0])
    starts = [base + s * np.array([0.6, -0.3, 0.5, 0.4, 0.05, 1.0]) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-10, 2), (-16, 1), (-10, 3), (-8, 5), (-5, 5), (-12, 5)]

    def objective(beta: np.ndarray) -> float:
        params = tuple(unpack(beta))
        ll, diag = exponential_ou_logpdf(y_next, y_prev, t_next, t_prev, delta_days, *params)
        if diag["support_violation_rate"] > 0 or not np.isfinite(ll).all():
            return 1e6 + 1e4 * diag["support_violation_rate"] + 1e3 * diag["coefficient_clipping_rate"]
        return float(-np.mean(ll))

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 350, "ftol": 1e-10}) for start in starts]
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        return _failed_fit("ExponentialTransformedOU", len(y_next), "all optimization starts non-finite", {})
    best = min(finite, key=lambda result: result.fun)
    kappa, sigma, d, a0, b0, c0 = tuple(unpack(best.x))
    ll, diag = exponential_ou_logpdf(y_next, y_prev, t_next, t_prev, delta_days, kappa, sigma, d, a0, b0, c0)
    valid_fit = np.isfinite(ll).all() and diag["support_violation_rate"] == 0 and diag["coefficient_clipping_rate"] == 0
    if not valid_fit:
        return _failed_fit(
            "ExponentialTransformedOU", len(y_next), "no admissible all-observation solution",
            {**diag, "best_penalized_objective": float(best.fun), "formula_implemented": True},
        )
    loglik = float(np.sum(ll)); gradient, condition, covariance = optimizer_diagnostics(best, len(y_next))
    repeats = repeated_solution_count(results, tolerance=1e-7); p = 6
    weak = condition is None or condition > 1e12 or repeats < 2 or np.any(np.isclose(best.x, [b[0] for b in bounds], atol=1e-5)) or np.any(np.isclose(best.x, [b[1] for b in bounds], atol=1e-5))
    return FitResult(
        "ExponentialTransformedOU", {"kappa": kappa, "sigma": sigma, "d": d, "A0": a0, "B0": b0, "C0": c0},
        loglik, len(y_next), -2 * loglik + 2 * p, -2 * loglik + p * np.log(len(y_next)), bool(best.success),
        str(best.message), gradient, condition, covariance, None,
        {"kappa": k0, "sigma": s0, "d": k0, "A0": 1.0, "B0": 0.0, "C0": 0.0},
        {**diag, "repeated_best_solutions": repeats, "boundary_hit": weak, "weak_identification": weak, "formula_implemented": True},
    )


def fit_quadratic_cir_fixed_base(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    multistart: int = 6,
) -> FitResult:
    y_next = np.asarray(y_next, dtype=float); y_prev = np.asarray(y_prev, dtype=float)
    base = np.array([0.0, -6.0, 0.0, -1.0])  # log A0, log B0, C0, unconstrained gamma
    starts = [base + s * np.array([0.6, 1.0, 0.05, 0.8]) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-10, 5), (-14, 8), (-2, 1.1), (-5, 5)]

    def unpack(beta: np.ndarray):
        a0, b0 = np.exp(beta[0]), np.exp(beta[1])
        c0 = beta[2]
        gamma = 0.01 * np.tanh(beta[3])
        d = 2.0 * kappa - gamma
        return float(d), float(a0), float(b0), float(c0)

    def objective(beta: np.ndarray) -> float:
        d, a0, b0, c0 = unpack(beta)
        ll, diag = quadratic_cir_logpdf(y_next, y_prev, t_next, t_prev, delta_days, theta, kappa, sigma, d, a0, b0, c0)
        if not np.isfinite(ll).all():
            return 1e6 + 1e5 * diag["support_violation_rate"] + 1e5 * diag["monotonicity_violation_rate"]
        return float(-np.mean(ll))

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "ftol": 1e-10}) for start in starts]
    best = min(results, key=lambda result: result.fun)
    d, a0, b0, c0 = unpack(best.x)
    ll, diag = quadratic_cir_logpdf(y_next, y_prev, t_next, t_prev, delta_days, theta, kappa, sigma, d, a0, b0, c0)
    if not np.isfinite(ll).all():
        return _failed_fit("QuadraticTransformedCIR", len(y_next), "no globally monotone/support-valid solution", {**diag, "formula_implemented": True, "base_parameters_fixed": True})
    loglik = float(np.sum(ll)); gradient, condition, covariance = optimizer_diagnostics(best, len(y_next))
    repeats = repeated_solution_count(results, tolerance=1e-7); p = 4
    weak = condition is None or condition > 1e12 or repeats < 2
    return FitResult(
        "QuadraticTransformedCIR", {"theta": theta, "kappa": kappa, "sigma": sigma, "d": d, "A0": a0, "B0": b0, "C0": c0},
        loglik, len(y_next), -2 * loglik + 2 * p, -2 * loglik + p * np.log(len(y_next)), bool(best.success), str(best.message),
        gradient, condition, covariance, None, {"A0": 1.0, "B0": 0.0, "C0": 0.0, "d": 2 * kappa},
        {**diag, "base_parameters_fixed": True, "repeated_best_solutions": repeats, "weak_identification": weak, "formula_implemented": True},
    )


def fit_exponential_cir_fixed_base(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    t_next: np.ndarray,
    t_prev: np.ndarray,
    delta_days: np.ndarray,
    theta: float,
    kappa: float,
    sigma: float,
    multistart: int = 6,
) -> FitResult:
    maximum_a0 = 0.999 * 2.0 * kappa / (sigma * sigma)
    minimum = float(min(np.min(y_next), np.min(y_prev)))
    base = np.array([-5.0, 0.0, -6.0, 0.0])  # A0 logistic, B0, C0 logistic, log d
    starts = [base + s * np.array([1.0, 0.2, 1.0, 0.5]) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-15, 10), (-20, 20), (-15, 8), (-10, 4)]

    def unpack(beta: np.ndarray):
        a0 = maximum_a0 / (1.0 + np.exp(-beta[0]))
        b0 = beta[1]
        c0 = 0.995 * minimum / (1.0 + np.exp(-beta[2]))
        d = np.exp(beta[3])
        return float(d), float(a0), float(b0), float(c0)

    def objective(beta: np.ndarray) -> float:
        d, a0, b0, c0 = unpack(beta)
        ll, diag = exponential_cir_logpdf(y_next, y_prev, t_next, t_prev, delta_days, theta, kappa, sigma, d, a0, b0, c0)
        if not np.isfinite(ll).all():
            return 1e6 + 1e5 * diag["support_violation_rate"] + 1e4 * diag["coefficient_clipping_rate"]
        return float(-np.mean(ll))

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "ftol": 1e-10}) for start in starts]
    best = min(results, key=lambda result: result.fun)
    d, a0, b0, c0 = unpack(best.x)
    ll, diag = exponential_cir_logpdf(y_next, y_prev, t_next, t_prev, delta_days, theta, kappa, sigma, d, a0, b0, c0)
    if not np.isfinite(ll).all():
        return _failed_fit("ExponentialTransformedCIR", len(y_next), "no support-valid finite solution", {**diag, "formula_implemented": True, "base_parameters_fixed": True})
    loglik = float(np.sum(ll)); gradient, condition, covariance = optimizer_diagnostics(best, len(y_next))
    repeats = repeated_solution_count(results, tolerance=1e-7); p = 4
    weak = condition is None or condition > 1e12 or repeats < 2
    return FitResult(
        "ExponentialTransformedCIR", {"theta": theta, "kappa": kappa, "sigma": sigma, "d": d, "A0": a0, "B0": b0, "C0": c0},
        loglik, len(y_next), -2 * loglik + 2 * p, -2 * loglik + p * np.log(len(y_next)), bool(best.success), str(best.message),
        gradient, condition, covariance, None, {"A0": 1.0, "B0": 0.0, "C0": 0.0, "d": kappa},
        {**diag, "base_parameters_fixed": True, "repeated_best_solutions": repeats, "weak_identification": weak, "formula_implemented": True},
    )

