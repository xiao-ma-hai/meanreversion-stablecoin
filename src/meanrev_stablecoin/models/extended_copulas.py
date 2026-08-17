from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, spence
from scipy.stats import gennorm, norm, t as student_t

from .copula import gaussian_copula_logpdf


_EPS = 1e-10


@dataclass
class FittedMarginal:
    name: str
    params: dict[str, Any]
    train_mean_logpdf: float
    converged: bool
    weak_identification: bool
    message: str
    _spline_q: np.ndarray | None = None
    _spline_p: np.ndarray | None = None

    def cdf(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if self.name == "Normal":
            result = norm.cdf(x, self.params["loc"], self.params["scale"])
        elif self.name == "StudentT":
            result = student_t.cdf((x - self.params["loc"]) / self.params["scale"], self.params["df"])
        elif self.name == "SkewStudentT":
            result = skew_t_cdf(x, **self.params)
        elif self.name == "GED":
            result = gennorm.cdf(x, self.params["beta"], loc=self.params["loc"], scale=self.params["scale"])
        elif self.name == "MonotoneSpline":
            result = np.interp(x, self._spline_q, self._spline_p, left=self._spline_p[0], right=self._spline_p[-1])
        else:
            raise ValueError(self.name)
        return np.clip(result, _EPS, 1 - _EPS)

    def logpdf(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if self.name == "Normal":
            return norm.logpdf(x, self.params["loc"], self.params["scale"])
        if self.name == "StudentT":
            z = (x - self.params["loc"]) / self.params["scale"]
            return student_t.logpdf(z, self.params["df"]) - np.log(self.params["scale"])
        if self.name == "SkewStudentT":
            return skew_t_logpdf(x, **self.params)
        if self.name == "GED":
            return gennorm.logpdf(x, self.params["beta"], loc=self.params["loc"], scale=self.params["scale"])
        if self.name == "MonotoneSpline":
            u = self.cdf(x)
            spline = PchipInterpolator(self._spline_p, self._spline_q, extrapolate=True)
            derivative = np.maximum(spline.derivative()(u), 1e-14)
            return -np.log(derivative)
        raise ValueError(self.name)

    def serializable(self) -> dict[str, Any]:
        result = {
            "name": self.name, "params": self.params, "train_mean_logpdf": self.train_mean_logpdf,
            "converged": self.converged, "weak_identification": self.weak_identification, "message": self.message,
        }
        if self.name == "MonotoneSpline":
            result["spline_probabilities"] = self._spline_p.tolist()
            result["spline_quantiles"] = self._spline_q.tolist()
        return result


def skew_t_logpdf(values: np.ndarray, df: float, skew: float, loc: float, scale: float) -> np.ndarray:
    """Fernandez--Steel skew-Student-t density; skew=1 is symmetric t."""
    x = np.asarray(values, dtype=float)
    z = (x - loc) / scale
    constant = np.log(2.0) - np.log(skew + 1.0 / skew) - np.log(scale)
    transformed = np.where(z >= 0, z / skew, skew * z)
    return constant + student_t.logpdf(transformed, df)


def skew_t_cdf(values: np.ndarray, df: float, skew: float, loc: float, scale: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    z = (x - loc) / scale
    left_mass = 1.0 / (skew * skew + 1.0)
    left = 2.0 / (skew * skew + 1.0) * student_t.cdf(skew * z, df)
    right = left_mass + 2.0 * skew * skew / (skew * skew + 1.0) * (student_t.cdf(z / skew, df) - 0.5)
    return np.where(z < 0, left, right)


def fit_marginals(values: np.ndarray, max_nobs: int = 150_000) -> list[FittedMarginal]:
    x_full = np.asarray(values, dtype=float)
    idx = np.linspace(0, len(x_full) - 1, min(len(x_full), max_nobs), dtype=int)
    x = x_full[idx]
    loc0, scale0 = float(np.mean(x)), max(float(np.std(x, ddof=1)), 1e-10)
    fits: list[FittedMarginal] = []

    def add(name: str, params: dict[str, float], converged: bool = True, weak: bool = False, message: str = ""):
        placeholder = FittedMarginal(name, params, np.nan, converged, weak, message)
        placeholder.train_mean_logpdf = float(np.mean(placeholder.logpdf(x)))
        fits.append(placeholder)

    add("Normal", {"loc": loc0, "scale": scale0})

    try:
        df, loc, scale = student_t.fit(x)
        add("StudentT", {"df": float(df), "loc": float(loc), "scale": float(scale)}, weak=bool(df > 500), message="scipy Student-t MLE")
    except Exception as exc:
        fits.append(FittedMarginal("StudentT", {"df": np.nan, "loc": loc0, "scale": scale0}, -np.inf, False, True, str(exc)))

    def skew_objective(beta: np.ndarray) -> float:
        df = 2.01 + np.exp(beta[0]); skew = np.exp(beta[1]); loc = beta[2]; scale = np.exp(beta[3])
        ll = skew_t_logpdf(x, df, skew, loc, scale)
        return float(-np.mean(ll)) if np.isfinite(ll).all() else 1e100

    skew_starts = [
        np.array([np.log(v - 2.01), s, loc0, np.log(scale0)])
        for v in (4.0, 8.0, 20.0) for s in (-0.2, 0.0, 0.2)
    ]
    skew_results = [minimize(skew_objective, start, method="L-BFGS-B", bounds=[(-4, 8), (-3, 3), (loc0 - 5 * scale0, loc0 + 5 * scale0), (np.log(scale0) - 6, np.log(scale0) + 4)]) for start in skew_starts]
    skew_best = min(skew_results, key=lambda result: result.fun)
    sp = skew_best.x
    add(
        "SkewStudentT",
        {"df": float(2.01 + np.exp(sp[0])), "skew": float(np.exp(sp[1])), "loc": float(sp[2]), "scale": float(np.exp(sp[3]))},
        converged=bool(skew_best.success), weak=bool(not skew_best.success), message=str(skew_best.message),
    )

    try:
        beta, loc, scale = gennorm.fit(x)
        add("GED", {"beta": float(beta), "loc": float(loc), "scale": float(scale)}, weak=bool(beta < 0.2 or beta > 20), message="scipy generalized-normal MLE")
    except Exception as exc:
        fits.append(FittedMarginal("GED", {"beta": np.nan, "loc": loc0, "scale": scale0}, -np.inf, False, True, str(exc)))

    probabilities = np.unique(np.r_[np.linspace(1e-4, 0.01, 50), np.linspace(0.011, 0.989, 979), np.linspace(0.99, 1 - 1e-4, 50)])
    quantiles = np.quantile(x, probabilities, method="linear")
    # Strictly increasing support is required by the quantile-density identity.
    keep = np.r_[True, np.diff(quantiles) > 1e-14]
    probabilities, quantiles = probabilities[keep], quantiles[keep]
    spline = FittedMarginal(
        "MonotoneSpline", {"grid_points": int(len(probabilities)), "tail_probability": float(probabilities[0])},
        np.nan, True, False, "PCHIP empirical quantile density", quantiles, probabilities,
    )
    spline.train_mean_logpdf = float(np.mean(spline.logpdf(x)))
    fits.append(spline)
    return fits


def student_t_copula_logpdf(u: np.ndarray, v: np.ndarray, rho: float, df: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), _EPS, 1 - _EPS)
    v = np.clip(np.asarray(v, dtype=float), _EPS, 1 - _EPS)
    rho = float(np.clip(rho, -0.999999, 0.999999)); df = float(df)
    x, y = student_t.ppf(u, df), student_t.ppf(v, df)
    return student_t_copula_logpdf_from_quantiles(x, y, rho, df)


def student_t_copula_logpdf_from_quantiles(
    x: np.ndarray, y: np.ndarray, rho: float, df: float
) -> np.ndarray:
    """Student-t copula log density given cached univariate t quantiles.

    The probability-integral transforms depend on ``df`` but not on ``rho``.
    Profiling the dependence parameters therefore reuses these arrays without
    changing either the likelihood or the resulting estimator.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    one_minus = 1.0 - rho * rho
    q = (x * x - 2 * rho * x * y + y * y) / one_minus
    log_joint = (
        gammaln((df + 2) / 2) - gammaln(df / 2) - np.log(df * np.pi)
        - 0.5 * np.log(one_minus) - (df + 2) / 2 * np.log1p(q / df)
    )
    log_uni = (
        2 * (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi))
        - (df + 1) / 2 * (np.log1p(x * x / df) + np.log1p(y * y / df))
    )
    return log_joint - log_uni


def clayton_copula_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), _EPS, 1 - _EPS); v = np.clip(np.asarray(v, dtype=float), _EPS, 1 - _EPS)
    theta = max(float(theta), 1e-8)
    a, b = -theta * np.log(u), -theta * np.log(v)
    m = np.maximum(np.maximum(a, b), 0.0)
    inside = np.exp(a - m) + np.exp(b - m) - np.exp(-m)
    log_s = m + np.log(np.maximum(inside, _EPS))
    return np.log1p(theta) + (-theta - 1) * (np.log(u) + np.log(v)) + (-2 - 1 / theta) * log_s


def gumbel_copula_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), _EPS, 1 - _EPS); v = np.clip(np.asarray(v, dtype=float), _EPS, 1 - _EPS)
    theta = max(float(theta), 1.0 + 1e-8)
    a, b = -np.log(u), -np.log(v)
    log_s = logsumexp(np.vstack([theta * np.log(a), theta * np.log(b)]), axis=0)
    s_inv_theta = np.exp(-log_s / theta)
    log_cdf = -np.exp(log_s / theta)
    return (
        log_cdf + (theta - 1) * (np.log(a) + np.log(b)) - np.log(u) - np.log(v)
        + (2 / theta - 2) * log_s + np.log1p((theta - 1) * s_inv_theta)
    )


def frank_copula_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), _EPS, 1 - _EPS); v = np.clip(np.asarray(v, dtype=float), _EPS, 1 - _EPS)
    theta = max(float(theta), 1e-7)
    terms = np.vstack([-theta * u, -theta * v, np.full_like(u, -theta), -theta * (u + v)])
    signs = np.array([1.0, 1.0, -1.0, -1.0])[:, None]
    log_den, sign = logsumexp(terms, b=signs, axis=0, return_sign=True)
    log_num = np.log(theta) + np.log(-np.expm1(-theta)) - theta * (u + v)
    result = log_num - 2.0 * log_den
    result[sign <= 0] = -np.inf
    return result


def frank_tau(theta: float | np.ndarray) -> float | np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    small = theta_array < 1e-3
    safe = np.maximum(theta_array, 1e-3)
    exp_neg = np.exp(-safe)
    integral = safe * np.log1p(-exp_neg) - spence(1.0 - exp_neg) + np.pi * np.pi / 6.0
    regular = 1.0 - 4.0 / safe + 4.0 * integral / (safe * safe)
    result = np.where(small, theta_array / 9.0, regular)
    return float(result) if result.ndim == 0 else result


# A dense monotone table avoids solving an implicit Debye-function equation at
# every likelihood call.  Interpolation error is far below the optimizer and
# reporting tolerances, while unlike the previous rounded cache this map is
# continuous and does not create zero numerical gradients.
_FRANK_LOG_THETA_GRID = np.linspace(-12.0, 11.0, 50_001)
_FRANK_THETA_GRID = np.exp(_FRANK_LOG_THETA_GRID)
_FRANK_TAU_GRID = np.maximum.accumulate(np.asarray(frank_tau(_FRANK_THETA_GRID)))


def frank_theta_from_tau(tau: float) -> float:
    target = float(np.clip(tau, _FRANK_TAU_GRID[0], min(_FRANK_TAU_GRID[-1], 0.9995)))
    log_theta = np.interp(target, _FRANK_TAU_GRID, _FRANK_LOG_THETA_GRID)
    return float(np.exp(log_theta))


def family_logpdf(family: str, u: np.ndarray, v: np.ndarray, dependence: float, df: float | None = None) -> np.ndarray:
    reflected = family.startswith("Survival")
    base_family = family.replace("Survival", "")
    if reflected:
        u, v = 1.0 - np.asarray(u), 1.0 - np.asarray(v)
    tau = float(np.clip(dependence, 1e-8, 0.9995))
    if base_family == "Gaussian":
        return gaussian_copula_logpdf(u, v, tau)
    if base_family == "StudentT":
        return student_t_copula_logpdf(u, v, tau, float(df))
    if base_family == "Clayton":
        return clayton_copula_logpdf(u, v, 2 * tau / (1 - tau))
    if base_family == "Gumbel":
        return gumbel_copula_logpdf(u, v, 1 / (1 - tau))
    if base_family == "Frank":
        theta = frank_theta_from_tau(tau)
        return frank_copula_logpdf(u, v, theta)
    raise ValueError(family)


def mixed_family_logpdf(
    family: str, u: np.ndarray, v: np.ndarray, kappa: float, lam: float, horizon_days: float, df: float | None = None
) -> np.ndarray:
    dependence = float(np.exp(-kappa * horizon_days))
    base = family_logpdf(family, u, v, dependence, df)
    if lam <= 0:
        return base
    log_a = -lam * horizon_days
    return logsumexp(np.vstack([log_a + base, np.full_like(base, np.log(-np.expm1(log_a)))]), axis=0)


def fit_extended_copula_families(
    horizon_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    families: list[str],
    fit_mixed: bool = True,
    max_pairs_per_horizon: int = 40_000,
    multistart: int = 6,
) -> list[dict[str, Any]]:
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for minutes, (u, v) in horizon_pairs.items():
        idx = np.linspace(0, len(u) - 1, min(len(u), max_pairs_per_horizon), dtype=int)
        selected[minutes] = (np.asarray(u)[idx], np.asarray(v)[idx])

    fits: list[dict[str, Any]] = []
    for family in families:
        variants = [False, True] if fit_mixed else [False]
        df_grid = [2.05, 2.10, 2.25, 2.50, 3.0, 4.0, 5.0, 8.0, 15.0, 30.0, 100.0] if family == "StudentT" else [None]
        for mixed in variants:
            candidates = []
            for df in df_grid:
                cached_t_quantiles = None
                if family == "StudentT":
                    cached_t_quantiles = {
                        minutes: (
                            student_t.ppf(np.clip(u, _EPS, 1 - _EPS), float(df)),
                            student_t.ppf(np.clip(v, _EPS, 1 - _EPS), float(df)),
                        )
                        for minutes, (u, v) in selected.items()
                    }

                def objective(beta: np.ndarray) -> float:
                    kappa = float(np.exp(beta[0])); lam = float(np.exp(beta[1])) if mixed else 0.0
                    scores = []
                    for minutes, (u, v) in selected.items():
                        horizon_days = minutes / 1440.0
                        if family == "StudentT":
                            dependence = float(np.exp(-kappa * horizon_days))
                            x, y = cached_t_quantiles[minutes]
                            base = student_t_copula_logpdf_from_quantiles(x, y, dependence, float(df))
                            if mixed:
                                log_a = -lam * horizon_days
                                ll = logsumexp(
                                    np.vstack([log_a + base, np.full_like(base, np.log(-np.expm1(log_a)))]),
                                    axis=0,
                                )
                            else:
                                ll = base
                        else:
                            ll = mixed_family_logpdf(family, u, v, kappa, lam, horizon_days, df)
                        if not np.isfinite(ll).all():
                            return 1e100
                        scores.append(np.mean(ll))
                    return -float(np.mean(scores))

                k_grid = np.geomspace(0.1, 10.0, max(2, multistart))
                if mixed:
                    lam_grid = np.geomspace(0.05, 2.0, max(2, multistart))
                    starts = [
                        np.log([k_grid[i], lam_grid[(2 * i + 1) % len(lam_grid)]])
                        for i in range(max(2, multistart))
                    ]
                else:
                    starts = [np.log([value]) for value in k_grid]
                bounds = [(-9, 9)] + ([(-12, 9)] if mixed else [])
                results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250, "ftol": 1e-10}) for start in starts]
                best = min(results, key=lambda result: result.fun)
                candidates.append((best.fun, df, best, results))
            _, df, best, results = min(candidates, key=lambda value: value[0])
            kappa = float(np.exp(best.x[0])); lam = float(np.exp(best.x[1])) if mixed else 0.0
            repeats = sum(bool(result.success) and abs(result.fun - best.fun) < 1e-7 for result in results)
            boundary = bool(np.any(np.isclose(best.x, [(-9, -12)[mixed and i == 1] for i in range(len(best.x))], atol=1e-5)))
            df_profile_boundary = bool(df is not None and (df == df_grid[0] or df == df_grid[-1]))
            fits.append({
                "model": family + ("Mixed" if mixed else ""), "family": family, "mixed_reset": mixed,
                "kappa": kappa, "lambda": lam, "df": df, "n_parameters": len(best.x) + (1 if df is not None else 0),
                "objective_mean_log_copula": -float(best.fun), "converged": bool(best.success),
                "optimizer_message": str(best.message), "repeated_best_solutions": repeats,
                "boundary_hit": boundary, "df_profile_boundary": df_profile_boundary,
                "df_profile_grid": df_grid if df is not None else None,
                "weak_identification": bool(not best.success or repeats < 2 or boundary or df_profile_boundary),
                "max_pairs_per_horizon": max_pairs_per_horizon,
                "multistart": max(2, multistart),
                "markov_semigroup_status": "exact" if family == "Gaussian" and not mixed else (
                    "exact_poisson_mixture" if family == "Gaussian" else "discrete-horizon diagnostic; semigroup not established"
                ),
            })
    return fits


def evaluate_extended_copulas(
    fits: list[dict[str, Any]],
    horizon_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    period: str,
) -> list[dict[str, Any]]:
    rows = []
    for fit in fits:
        horizon_scores = []
        total = 0
        for minutes, (u, v) in sorted(horizon_pairs.items()):
            ll = mixed_family_logpdf(
                fit["family"], u, v, fit["kappa"], fit["lambda"], minutes / 1440.0, fit.get("df")
            )
            score = float(np.mean(ll)); horizon_scores.append(score); total += len(u)
            rows.append({"period": period, "model": fit["model"], "horizon_minutes": str(minutes), "n_pairs": len(u), "mean_log_copula_score": score})
        rows.append({"period": period, "model": fit["model"], "horizon_minutes": "equal_horizon_mean", "n_pairs": total, "mean_log_copula_score": float(np.mean(horizon_scores))})
    return rows
