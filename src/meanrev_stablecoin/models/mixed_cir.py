from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import gamma

from ..estimation.optimize import FitResult
from .cir import cir_logpdf


def mixed_cir_logpdf(z_next, z_prev, delta_days, theta, kappa, sigma, lam):
    base = cir_logpdf(z_next, z_prev, delta_days, theta, kappa, sigma)
    shape = 2 * kappa * theta / (sigma * sigma)
    scale = sigma * sigma / (2 * kappa)
    marginal = gamma.logpdf(z_next, a=shape, scale=scale)
    if lam == 0:
        return base
    log_a = -lam * np.asarray(delta_days)
    return logsumexp(np.vstack([log_a + base, np.log(-np.expm1(log_a)) + marginal]), axis=0)


def fit_mixed_cir_lambda(z_next, z_prev, delta_days, base_fit: FitResult, max_nobs: int = 100_000) -> FitResult:
    theta, kappa, sigma = (base_fit.params[k] for k in ("theta", "kappa", "sigma"))
    n = len(z_next)
    idx = np.linspace(0, n - 1, min(n, max_nobs), dtype=int)
    zn, zp, dd = np.asarray(z_next)[idx], np.asarray(z_prev)[idx], np.asarray(delta_days)[idx]

    def objective(log_lam):
        ll = mixed_cir_logpdf(zn, zp, dd, theta, kappa, sigma, np.exp(log_lam))
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    opt = minimize_scalar(objective, bounds=(-12, 12), method="bounded", options={"xatol": 1e-7})
    lam = float(np.exp(opt.x))
    loglik = -float(opt.fun) * n
    return FitResult(
        model_name="MCIR", params={"theta": theta, "kappa": kappa, "sigma": sigma, "lambda": lam},
        loglik=loglik, nobs=n, aic=-2 * loglik + 8, bic=-2 * loglik + 4 * np.log(n),
        converged=bool(opt.success), optimizer_message=str(opt.message), gradient_norm=np.nan,
        hessian_condition=None, covariance=None, robust_covariance=None,
        start_values={**base_fit.params, "lambda": 1.0},
        metadata={"base_parameters_fixed": True, "estimation_nobs": len(idx),
                  "effective_half_life_minutes": 1440 * np.log(2) / (kappa + lam),
                  "weak_identification": True},
    )

