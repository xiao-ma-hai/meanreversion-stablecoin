from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma, norm

from .copula import gaussian_copula_logpdf


def article_gamma_case1_equation(alpha: float) -> float:
    """Left side of article Equation (gamma example 1)."""
    alpha = float(alpha)
    x_alpha = gamma.cdf(alpha, a=alpha, scale=1.0)
    z_alpha = norm.ppf(x_alpha)
    first = np.exp(gammaln(alpha) + alpha * (1.0 - np.log(alpha))) * norm.pdf(z_alpha)
    return float(first - 2.0 * z_alpha)


def audit_article_gamma_shape_condition(lower: float = 0.01, upper: float = 1000.0, points: int = 2000) -> dict[str, Any]:
    grid = np.logspace(np.log10(lower), np.log10(upper), points)
    values = np.array([article_gamma_case1_equation(value) for value in grid])
    finite = np.isfinite(values)
    sign_changes = np.flatnonzero(values[:-1] * values[1:] < 0)
    minimization = minimize_scalar(
        lambda log_alpha: abs(article_gamma_case1_equation(np.exp(log_alpha))),
        bounds=(np.log(lower), np.log(upper)), method="bounded",
    )
    return {
        "searched_alpha_lower": lower,
        "searched_alpha_upper": upper,
        "grid_points": points,
        "finite_share": float(np.mean(finite)),
        "equation_minimum": float(np.min(values[finite])),
        "equation_maximum": float(np.max(values[finite])),
        "sign_change_count": int(len(sign_changes)),
        "finite_root_found": bool(len(sign_changes) > 0),
        "closest_alpha": float(np.exp(minimization.x)),
        "closest_absolute_residual": float(minimization.fun),
        "diagnosis": "no finite admissible shape root on the prespecified wide range" if not len(sign_changes) else "root bracket found",
    }


def gaussian_gamma_logpdf(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    delta_days: np.ndarray,
    shape: float,
    scale: float,
    kappa: float,
    lam: float = 0.0,
) -> np.ndarray:
    y_next = np.asarray(y_next, dtype=float); y_prev = np.asarray(y_prev, dtype=float)
    if shape <= 0 or scale <= 0 or kappa <= 0 or lam < 0 or np.any(y_next <= 0) or np.any(y_prev <= 0):
        return np.full_like(y_next, -np.inf)
    u = np.clip(gamma.cdf(y_prev, a=shape, scale=scale), 1e-10, 1 - 1e-10)
    v = np.clip(gamma.cdf(y_next, a=shape, scale=scale), 1e-10, 1 - 1e-10)
    rho = np.exp(-kappa * np.asarray(delta_days, dtype=float))
    base = gaussian_copula_logpdf(u, v, rho)
    if lam > 0:
        log_a = -lam * np.asarray(delta_days, dtype=float)
        base = logsumexp(np.vstack([log_a + base, np.log(-np.expm1(log_a))]), axis=0)
    return base + gamma.logpdf(y_next, a=shape, scale=scale)


def fit_gaussian_gamma(
    y_next: np.ndarray,
    y_prev: np.ndarray,
    delta_days: np.ndarray,
    mixed: bool = False,
    multistart: int = 6,
) -> dict[str, Any]:
    y_next = np.asarray(y_next, dtype=float); y_prev = np.asarray(y_prev, dtype=float)
    shape0, _, scale0 = gamma.fit(np.r_[y_prev, y_next], floc=0)
    base = np.log([shape0, scale0, 1.0] + ([0.2] if mixed else []))
    starts = [base + s * np.array([0.4, -0.2, 0.8] + ([-0.6] if mixed else [])) for s in np.linspace(-1, 1, multistart)]
    bounds = [(-8, 14), (-20, 5), (-9, 9)] + ([(-12, 9)] if mixed else [])

    def objective(beta: np.ndarray) -> float:
        values = np.exp(beta); shape, scale, kappa = values[:3]; lam = values[3] if mixed else 0.0
        ll = gaussian_gamma_logpdf(y_next, y_prev, delta_days, shape, scale, kappa, lam)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "ftol": 1e-11}) for start in starts]
    best = min(results, key=lambda result: result.fun); values = np.exp(best.x)
    shape, scale, kappa = map(float, values[:3]); lam = float(values[3]) if mixed else 0.0
    ll = gaussian_gamma_logpdf(y_next, y_prev, delta_days, shape, scale, kappa, lam)
    condition_audit = audit_article_gamma_shape_condition()
    repeats = sum(bool(result.success) and abs(result.fun - best.fun) < 1e-7 for result in results)
    return {
        "model": "MixedGaussianGammaStationary" if mixed else "GaussianGammaStationary",
        "params": {"shape": shape, "scale": scale, "kappa": kappa, "lambda": lam},
        "loglik": float(np.sum(ll)), "mean_loglik": float(np.mean(ll)), "nobs": len(ll),
        "converged": bool(best.success), "optimizer_message": str(best.message), "repeated_best_solutions": repeats,
        "weak_identification": bool(not best.success or repeats < 2),
        "article_case1_shape_condition_pass": bool(condition_audit["finite_root_found"] and abs(article_gamma_case1_equation(shape)) < 1e-5),
        "article_case1_equation_at_estimate": article_gamma_case1_equation(shape),
        "article_shape_condition_audit": condition_audit,
        "interpretation": "observable density estimated, but article mean-reversion shape restriction must pass separately",
    }


def gamma_case2_endpoint_diagnostic() -> dict[str, Any]:
    audit = audit_article_gamma_shape_condition()
    return {
        "model": "GaussianGammaTimeVarying",
        "density_formula_implemented": True,
        "estimation_attempted": False,
        "convergent_stationary_endpoint_available": audit["finite_root_found"],
        "failure_reason": (
            "Case 2 requires a stationary endpoint satisfying Case 1; no shape root was found on the prespecified [0.01, 1000] range"
            if not audit["finite_root_found"] else "endpoint available; numerical ODE calibration may proceed"
        ),
        **audit,
    }
