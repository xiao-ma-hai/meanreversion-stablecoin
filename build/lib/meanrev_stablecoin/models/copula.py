from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, ndtr, ndtri


@dataclass
class EmpiricalMarginal:
    sorted_values: np.ndarray
    clip: float = 1e-8

    @classmethod
    def fit(cls, values: np.ndarray, clip: float = 1e-8) -> "EmpiricalMarginal":
        return cls(np.sort(np.asarray(values, dtype=float)), clip)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        ranks = np.searchsorted(self.sorted_values, np.asarray(values), side="right")
        u = (ranks - 0.5) / len(self.sorted_values)
        return np.clip(u, self.clip, 1 - self.clip)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.quantile(self.sorted_values, np.clip(u, self.clip, 1 - self.clip), method="linear")


def rank_pseudo_observations(values: np.ndarray, clip: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    u = (ranks - 0.5) / len(values)
    return np.clip(u, clip, 1 - clip)


def gaussian_copula_logpdf(u: np.ndarray, v: np.ndarray, rho: float | np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12)
    v = np.clip(np.asarray(v, dtype=float), 1e-12, 1 - 1e-12)
    rho = np.clip(np.asarray(rho, dtype=float), -0.99999999, 0.99999999)
    zu, zv = ndtri(u), ndtri(v)
    one_minus = 1 - rho * rho
    return -0.5 * np.log(one_minus) + (2 * rho * zu * zv - rho * rho * (zu * zu + zv * zv)) / (2 * one_minus)


def mixed_gaussian_copula_logpdf(
    u: np.ndarray,
    v: np.ndarray,
    kappa: float,
    lam: float,
    horizon_days: float,
) -> np.ndarray:
    rho = np.exp(-kappa * horizon_days)
    base = gaussian_copula_logpdf(u, v, rho)
    if lam == 0:
        return base
    log_a = -lam * horizon_days
    return logsumexp(np.vstack([log_a + base, np.full_like(base, np.log(-np.expm1(log_a)))]), axis=0)


def two_scale_rho(
    horizon_days: float | np.ndarray,
    kappa_slow: float,
    kappa_fast: float,
    weight_fast: float,
) -> np.ndarray:
    if not (0 < kappa_slow < kappa_fast) or not (0 < weight_fast < 1):
        raise ValueError("two-scale dependence requires 0 < kappa_slow < kappa_fast and 0 < weight_fast < 1")
    horizon = np.asarray(horizon_days, dtype=float)
    return (
        weight_fast * np.exp(-kappa_fast * horizon)
        + (1 - weight_fast) * np.exp(-kappa_slow * horizon)
    )


def two_scale_mixed_gaussian_copula_logpdf(
    u: np.ndarray,
    v: np.ndarray,
    kappa_slow: float,
    kappa_fast: float,
    weight_fast: float,
    lam: float,
    horizon_days: float,
) -> np.ndarray:
    rho = float(two_scale_rho(horizon_days, kappa_slow, kappa_fast, weight_fast))
    base = gaussian_copula_logpdf(u, v, rho)
    if lam == 0:
        return base
    log_a = -lam * horizon_days
    return logsumexp(
        np.vstack([log_a + base, np.full_like(base, np.log(-np.expm1(log_a)))]), axis=0
    )


def estimate_gaussian_rho(u: np.ndarray, v: np.ndarray) -> float:
    zu = ndtri(np.clip(u, 1e-8, 1 - 1e-8))
    zv = ndtri(np.clip(v, 1e-8, 1 - 1e-8))
    return float(np.clip(np.corrcoef(zu, zv)[0, 1], -0.999, 0.999))


def fit_multihorizon_composite_likelihood(
    horizon_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    random_seed: int = 20260806,
    max_pairs_per_horizon: int = 150_000,
) -> dict:
    rng = np.random.default_rng(random_seed)
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    empirical_rows = []
    for minutes, (u, v) in sorted(horizon_pairs.items()):
        u, v = np.asarray(u), np.asarray(v)
        rho = estimate_gaussian_rho(u, v)
        empirical_rows.append({"horizon_minutes": minutes, "n_pairs": len(u), "empirical_gaussian_rho": rho})
        if len(u) > max_pairs_per_horizon:
            idx = np.sort(rng.choice(len(u), max_pairs_per_horizon, replace=False))
            selected[minutes] = (u[idx], v[idx])
        else:
            selected[minutes] = (u, v)

    positive_rows = [r for r in empirical_rows if r["empirical_gaussian_rho"] > 0]
    if len(positive_rows) >= 2:
        h = np.array([r["horizon_minutes"] / 1440 for r in positive_rows])
        log_rho = np.log([r["empirical_gaussian_rho"] for r in positive_rows])
        kappa0 = max(float(-np.dot(h, log_rho) / np.dot(h, h)), 1e-4)
    else:
        kappa0 = 10.0

    def objective(beta: np.ndarray) -> float:
        kappa, lam = np.exp(beta)
        horizon_scores = []
        for minutes, (u, v) in selected.items():
            ll = mixed_gaussian_copula_logpdf(u, v, kappa, lam, minutes / 1440)
            horizon_scores.append(float(np.mean(ll)))
        return -float(np.mean(horizon_scores))

    starts = [np.log([kappa0 * scale, lam]) for scale in (0.5, 1, 2) for lam in (1e-4, 0.1, 5, 50)]
    results = [minimize(objective, s, method="L-BFGS-B", bounds=[(-10, 14), (-12, 12)], options={"maxiter": 300, "ftol": 1e-12}) for s in starts]
    best = min((r for r in results if np.isfinite(r.fun)), key=lambda r: r.fun)
    kappa, lam = map(float, np.exp(best.x))
    repeated = sum(r.success and abs(r.fun - best.fun) < 1e-8 for r in results)
    for row in empirical_rows:
        hday = row["horizon_minutes"] / 1440
        row["fitted_base_rho"] = np.exp(-kappa * hday)
        row["fitted_mixed_linear_dependence"] = np.exp(-(kappa + lam) * hday)
        row["reset_weight"] = 1 - np.exp(-lam * hday)
    return {
        "kappa": kappa,
        "lambda": lam,
        "objective_mean_log_copula": -float(best.fun),
        "converged": bool(best.success),
        "optimizer_message": str(best.message),
        "gradient_norm": float(np.linalg.norm(best.jac)),
        "repeated_best_solutions": repeated,
        "weak_identification": repeated < 2,
        "horizon_rows": empirical_rows,
    }


def _selected_horizon_pairs(
    horizon_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    random_seed: int,
    max_pairs_per_horizon: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[dict]]:
    rng = np.random.default_rng(random_seed)
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    empirical_rows = []
    for minutes, (u, v) in sorted(horizon_pairs.items()):
        u, v = np.asarray(u), np.asarray(v)
        empirical_rows.append({
            "horizon_minutes": int(minutes),
            "n_pairs": len(u),
            "empirical_gaussian_rho": estimate_gaussian_rho(u, v),
        })
        if len(u) > max_pairs_per_horizon:
            idx = np.sort(rng.choice(len(u), max_pairs_per_horizon, replace=False))
            selected[int(minutes)] = (u[idx], v[idx])
        else:
            selected[int(minutes)] = (u, v)
    return selected, empirical_rows


def _copula_model_logpdf(model: dict, u: np.ndarray, v: np.ndarray, horizon_days: float) -> np.ndarray:
    params = model["params"]
    if model["family"] == "single_scale":
        return mixed_gaussian_copula_logpdf(
            u, v, params["kappa"], params.get("lambda", 0.0), horizon_days
        )
    return two_scale_mixed_gaussian_copula_logpdf(
        u, v,
        params["kappa_slow"], params["kappa_fast"], params["weight_fast"],
        params.get("lambda", 0.0), horizon_days,
    )


def fit_copula_model_comparison(
    horizon_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    random_seed: int = 20260806,
    max_pairs_per_horizon: int = 100_000,
) -> tuple[list[dict], list[dict]]:
    """Fit fair single/two-scale Gaussian and mixed-copula comparators."""
    selected, horizon_rows = _selected_horizon_pairs(
        horizon_pairs, random_seed, max_pairs_per_horizon
    )
    h = np.array([row["horizon_minutes"] / 1440 for row in horizon_rows])
    empirical = np.array([row["empirical_gaussian_rho"] for row in horizon_rows])
    positive = empirical > 0
    kappa0 = max(float(-np.dot(h[positive], np.log(empirical[positive])) / np.dot(h[positive], h[positive])), 1e-4)

    specifications = [
        ("SingleGaussian", "single_scale", False),
        ("SingleMixed", "single_scale", True),
        ("TwoScaleGaussian", "two_scale", False),
        ("TwoScaleMixed", "two_scale", True),
    ]
    fitted: list[dict] = []
    for name, family, mixed in specifications:
        if family == "single_scale":
            if mixed:
                starts = [np.log([kappa0 * scale, lam]) for scale in (0.5, 1, 2) for lam in (1e-4, 0.1, 1.0)]
                bounds = [(-10, 12), (-16, 10)]
            else:
                starts = [np.array([np.log(kappa0 * scale)]) for scale in (0.5, 1, 2)]
                bounds = [(-10, 12)]

            def unpack(beta, mixed=mixed):
                return {"kappa": float(np.exp(beta[0])), "lambda": float(np.exp(beta[1])) if mixed else 0.0}
        else:
            slow_candidates = (max(kappa0 * 0.03, 1e-3), max(kappa0 * 0.15, 5e-3), max(kappa0 * 0.5, 1e-2))
            starts = []
            for slow in slow_candidates:
                for fast_multiple, weight in ((2.0, 0.25), (5.0, 0.5), (15.0, 0.75)):
                    fast = max(kappa0 * fast_multiple, slow * 1.5)
                    base = [np.log(slow), np.log(fast - slow), logit(weight)]
                    if mixed:
                        for lam in (1e-4, 0.1):
                            starts.append(np.array(base + [np.log(lam)]))
                    else:
                        starts.append(np.array(base))
            bounds = [(-10, 10), (-10, 12), (-7, 7)] + ([(-16, 10)] if mixed else [])

            def unpack(beta, mixed=mixed):
                slow = float(np.exp(beta[0]))
                fast = slow + float(np.exp(beta[1]))
                return {
                    "kappa_slow": slow,
                    "kappa_fast": fast,
                    "weight_fast": float(expit(beta[2])),
                    "lambda": float(np.exp(beta[3])) if mixed else 0.0,
                }

        def objective(beta: np.ndarray) -> float:
            params = unpack(beta)
            model = {"family": family, "params": params}
            scores = [
                float(np.mean(_copula_model_logpdf(model, u, v, minutes / 1440)))
                for minutes, (u, v) in selected.items()
            ]
            return -float(np.mean(scores))

        results = [
            minimize(
                objective, start, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-7},
            )
            for start in starts
        ]
        best = min((result for result in results if np.isfinite(result.fun)), key=lambda result: result.fun)
        params = unpack(best.x)
        model = {
            "model": name,
            "family": family,
            "mixed_reset": mixed,
            "params": params,
            "n_parameters": len(best.x),
            "objective_mean_log_copula": -float(best.fun),
            "converged": bool(best.success),
            "gradient_norm": float(np.linalg.norm(best.jac)),
            "repeated_best_solutions": int(sum(result.success and abs(result.fun - best.fun) < 1e-8 for result in results)),
            "max_pairs_per_horizon": max_pairs_per_horizon,
            "latent_state_note": (
                "one-dimensional Markov" if family == "single_scale"
                else "sum of two latent OU factors; Markov only in the two-dimensional latent state"
            ),
        }
        fitted_dependence = []
        for minutes in [row["horizon_minutes"] for row in horizon_rows]:
            hd = minutes / 1440
            if family == "single_scale":
                base_rho = np.exp(-params["kappa"] * hd)
            else:
                base_rho = float(two_scale_rho(hd, params["kappa_slow"], params["kappa_fast"], params["weight_fast"]))
            fitted_dependence.append(np.exp(-params.get("lambda", 0.0) * hd) * base_rho)
        model["dependence_rmse"] = float(np.sqrt(np.mean((empirical - fitted_dependence) ** 2)))
        fitted.append(model)

    for row_idx, row in enumerate(horizon_rows):
        minutes = row["horizon_minutes"]
        for model in fitted:
            params = model["params"]
            hd = minutes / 1440
            if model["family"] == "single_scale":
                base_rho = np.exp(-params["kappa"] * hd)
            else:
                base_rho = float(two_scale_rho(hd, params["kappa_slow"], params["kappa_fast"], params["weight_fast"]))
            row[model["model"]] = float(np.exp(-params.get("lambda", 0.0) * hd) * base_rho)
    return fitted, horizon_rows


def evaluate_copula_models(
    models: list[dict],
    horizon_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    period: str,
) -> list[dict]:
    rows = []
    for model in models:
        horizon_scores = []
        for minutes, (u, v) in sorted(horizon_pairs.items()):
            values = _copula_model_logpdf(model, np.asarray(u), np.asarray(v), minutes / 1440)
            score = float(np.mean(values))
            horizon_scores.append(score)
            rows.append({
                "period": period,
                "model": model["model"],
                "horizon_minutes": int(minutes),
                "n_pairs": len(u),
                "mean_log_copula_score": score,
            })
        rows.append({
            "period": period,
            "model": model["model"],
            "horizon_minutes": "equal_horizon_mean",
            "n_pairs": int(sum(len(pair[0]) for pair in horizon_pairs.values())),
            "mean_log_copula_score": float(np.mean(horizon_scores)),
        })
    return rows


def copula_conditional_mean(
    x_grid: np.ndarray,
    marginal: EmpiricalMarginal,
    horizon_days: float,
    kappa: float,
    lam: float,
    quadrature_nodes: int = 128,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    v = (nodes + 1) / 2
    w = weights / 2
    y = marginal.quantile(v)
    u = marginal.cdf(np.asarray(x_grid))
    zu = ndtri(u)[:, None]
    zv = ndtri(v)[None, :]
    rho = np.exp(-kappa * horizon_days)
    conditional_u_density = np.exp(
        -0.5 * np.log(1 - rho * rho)
        + (2 * rho * zu * zv - rho * rho * (zu * zu + zv * zv)) / (2 * (1 - rho * rho))
    )
    a = np.exp(-lam * horizon_days)
    density = a * conditional_u_density + (1 - a)
    return (density * y[None, :] * w[None, :]).sum(axis=1)


def gaussian_copula_cdf_conditional(v: np.ndarray, u: np.ndarray, rho: float) -> np.ndarray:
    return ndtr((ndtri(v) - rho * ndtri(u)) / np.sqrt(1 - rho * rho))
