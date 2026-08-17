from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, ndtr
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from .models.mixed_ou import mixed_ou_logpdf
from .models.ou import ar1_initial_values, ou_logpdf, ou_moments
from .pairs import build_exact_horizon_pairs


@dataclass
class FastOUParameters:
    theta: float
    kappa: float
    sigma: float
    lam: float = 0.0
    slow_mean: float | None = None
    rw_step_sd: float | None = None


def fast_ou_parameters(x_prev: np.ndarray, x_next: np.ndarray, delta_day: float = 1 / 288) -> FastOUParameters:
    theta, kappa, sigma = ar1_initial_values(x_prev, x_next, delta_day)
    return FastOUParameters(theta, kappa, sigma)


def fit_conditional_lambda(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    delta_days: np.ndarray,
    params: FastOUParameters,
    max_nobs: int = 75_000,
) -> tuple[float, float]:
    n = len(x_prev)
    idx = np.linspace(0, n - 1, min(n, max_nobs), dtype=int)
    xp, xn, dd = x_prev[idx], x_next[idx], delta_days[idx]

    def objective(log_lam: float) -> float:
        ll = mixed_ou_logpdf(xn, xp, dd, params.theta, params.kappa, params.sigma, float(np.exp(log_lam)))
        return float(-np.mean(ll))

    result = minimize_scalar(objective, bounds=(-12, 12), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(result.x)), -float(result.fun) * n


def nested_ou_proxy_parameters(values: np.ndarray, delta_day: float = 1 / 288, slow_days: int = 30) -> FastOUParameters:
    slow = pd.Series(values).ewm(span=slow_days * 288, adjust=False).mean().to_numpy()
    deviation = values - slow
    theta, kappa, sigma = ar1_initial_values(deviation[:-1], deviation[1:], delta_day)
    return FastOUParameters(theta=theta, kappa=kappa, sigma=sigma, slow_mean=float(slow[-1]))


def random_walk_parameters(values: np.ndarray) -> FastOUParameters:
    return FastOUParameters(theta=0.0, kappa=0.0, sigma=0.0, rw_step_sd=float(np.std(np.diff(values), ddof=1)))


def _a_function(diff: np.ndarray, sd: np.ndarray | float) -> np.ndarray:
    sd = np.asarray(sd, dtype=float)
    z = np.asarray(diff) / sd
    return 2 * sd * norm.pdf(z) + np.asarray(diff) * (2 * norm.cdf(z) - 1)


def normal_crps(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return _a_function(y - mean, sd) - sd / np.sqrt(np.pi)


def mixture_normal_crps(y, w1, m1, s1, m2, s2):
    w2 = 1 - w1
    first = w1 * _a_function(y - m1, s1) + w2 * _a_function(y - m2, s2)
    pair = 0.5 * (
        w1 * w1 * _a_function(np.zeros_like(y), np.sqrt(2) * s1)
        + 2 * w1 * w2 * _a_function(m1 - m2, np.sqrt(s1 * s1 + s2 * s2))
        + w2 * w2 * _a_function(np.zeros_like(y), np.sqrt(2) * s2)
    )
    return first - pair


def _normal_depeg_probability(mean, sd, threshold):
    lower, upper = np.log(1 - threshold), np.log(1 + threshold)
    return ndtr((lower - mean) / sd) + 1 - ndtr((upper - mean) / sd)


def _forecast_distribution(model: str, x: np.ndarray, horizon_days: float, params: FastOUParameters):
    if model == "RandomWalk":
        steps = horizon_days * 288
        mean = x.copy()
        sd = np.full_like(x, max(params.rw_step_sd * np.sqrt(steps), 1e-10))
        return {"mean": mean, "sd": sd, "kind": "normal"}
    if model == "NestedOU_proxy":
        center = float(params.slow_mean)
        rho = np.exp(-params.kappa * horizon_days)
        mean = center + params.theta + rho * (x - center - params.theta)
        variance = params.sigma**2 * (-np.expm1(-2 * params.kappa * horizon_days)) / (2 * params.kappa)
        return {"mean": mean, "sd": np.full_like(x, np.sqrt(variance)), "kind": "normal"}
    base_mean, base_var = ou_moments(x, np.full_like(x, horizon_days), params.theta, params.kappa, params.sigma)
    if model == "OUF":
        return {"mean": base_mean, "sd": np.sqrt(base_var), "kind": "normal"}
    if model == "MOUF":
        weight = np.exp(-params.lam * horizon_days)
        stationary_sd = params.sigma / np.sqrt(2 * params.kappa)
        point_mean = params.theta + np.exp(-(params.kappa + params.lam) * horizon_days) * (x - params.theta)
        return {"mean": point_mean, "kind": "mixture", "weight": weight,
                "m1": base_mean, "s1": np.sqrt(base_var), "m2": np.full_like(x, params.theta),
                "s2": np.full_like(x, stationary_sd)}
    raise ValueError(model)


def _evaluate_distribution(y: np.ndarray, dist: dict, x: np.ndarray, thresholds: list[float]):
    if dist["kind"] == "normal":
        logpdf = norm.logpdf(y, loc=dist["mean"], scale=dist["sd"])
        crps = normal_crps(y, dist["mean"], dist["sd"])
        probabilities = {b: _normal_depeg_probability(dist["mean"], dist["sd"], b) for b in thresholds}
    else:
        w = dist["weight"]
        lp1 = np.log(w) + norm.logpdf(y, dist["m1"], dist["s1"])
        lp2 = np.log1p(-w) + norm.logpdf(y, dist["m2"], dist["s2"])
        logpdf = logsumexp(np.vstack([lp1, lp2]), axis=0)
        crps = mixture_normal_crps(y, w, dist["m1"], dist["s1"], dist["m2"], dist["s2"])
        probabilities = {}
        for b in thresholds:
            probabilities[b] = w * _normal_depeg_probability(dist["m1"], dist["s1"], b) + (1 - w) * _normal_depeg_probability(dist["m2"], dist["s2"], b)
    direction_actual = np.sign(y - x)
    direction_predicted = np.sign(dist["mean"] - x)
    return logpdf, crps, probabilities, direction_actual == direction_predicted


def rolling_forecast_evaluation(
    df: pd.DataFrame,
    horizons_minutes: list[int],
    thresholds: list[float],
    refit_days: int = 30,
    window_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2025-12-31 23:55:00", tz="UTC")
    refit_dates = pd.date_range(start, end, freq=f"{refit_days}D")
    metric_acc: dict[tuple[str, str, int], dict[str, float]] = {}
    probability_acc: dict[tuple[str, str, int, float], dict[str, list[np.ndarray]]] = {}
    fit_rows = []
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    values = df["log_price"].to_numpy(dtype=float)

    for refit_idx, refit in enumerate(refit_dates):
        next_refit = refit_dates[refit_idx + 1] if refit_idx + 1 < len(refit_dates) else end + pd.Timedelta(minutes=5)
        fit_start = refit - pd.Timedelta(days=window_days)
        train = df[df["datetime"].between(fit_start, refit, inclusive="left")]
        pi, pj = build_exact_horizon_pairs(train, 300)
        x_prev, x_next = train["log_price"].to_numpy()[pi], train["log_price"].to_numpy()[pj]
        if len(x_prev) < 1000:
            continue
        ou = fast_ou_parameters(x_prev, x_next)
        lam, _ = fit_conditional_lambda(x_prev, x_next, np.full(len(x_prev), 1 / 288), ou)
        mou = FastOUParameters(ou.theta, ou.kappa, ou.sigma, lam=lam)
        nested = nested_ou_proxy_parameters(train["log_price"].to_numpy())
        rw = random_walk_parameters(train["log_price"].to_numpy())
        models = {"RandomWalk": rw, "NestedOU_proxy": nested, "OUF": ou, "MOUF": mou}
        for name, pars in models.items():
            fit_rows.append({"refit_utc": refit, "fit_start_utc": fit_start, "fit_end_exclusive_utc": refit,
                             "forecast_end_exclusive_utc": next_refit, "model": name,
                             "theta": pars.theta, "kappa": pars.kappa, "sigma": pars.sigma,
                             "lambda": pars.lam, "slow_mean": pars.slow_mean, "rw_step_sd": pars.rw_step_sd,
                             "fit_nobs": len(train)})
        origin_mask = (df["datetime"] >= refit) & (df["datetime"] < next_refit)
        origins = np.flatnonzero(origin_mask.to_numpy())
        for horizon in horizons_minutes:
            target_ts = timestamps[origins] + horizon * 60
            target_idx = np.searchsorted(timestamps, target_ts)
            valid = target_idx < len(timestamps)
            valid_pos = np.flatnonzero(valid)
            valid[valid_pos] &= timestamps[target_idx[valid_pos]] == target_ts[valid_pos]
            oi, tj = origins[valid], target_idx[valid]
            if len(oi) == 0:
                continue
            x, y = values[oi], values[tj]
            period = np.where(pd.to_datetime(timestamps[oi], unit="s", utc=True).year == 2024, "validation_2024", "test_2025")
            for model, pars in models.items():
                dist = _forecast_distribution(model, x, horizon / 1440, pars)
                logpdf, crps, probabilities, direction_ok = _evaluate_distribution(y, dist, x, thresholds)
                error = y - dist["mean"]
                price_y = np.exp(y)
                for period_name in np.unique(period):
                    pmask = period == period_name
                    key = (str(period_name), model, horizon)
                    acc = metric_acc.setdefault(key, {"n": 0.0, "abs_error": 0.0, "sq_error": 0.0,
                                                     "negative_logpdf": 0.0, "crps": 0.0, "direction": 0.0})
                    acc["n"] += int(pmask.sum())
                    acc["abs_error"] += float(np.abs(error[pmask]).sum())
                    acc["sq_error"] += float(np.square(error[pmask]).sum())
                    acc["negative_logpdf"] += float((-logpdf[pmask]).sum())
                    acc["crps"] += float(crps[pmask].sum())
                    acc["direction"] += float(direction_ok[pmask].sum())
                    for threshold, prob in probabilities.items():
                        observed = np.abs(price_y[pmask] - 1) > threshold
                        pkey = (str(period_name), model, horizon, float(threshold))
                        pacc = probability_acc.setdefault(pkey, {"p": [], "observed": []})
                        pacc["p"].append(np.clip(prob[pmask], 1e-12, 1 - 1e-12))
                        pacc["observed"].append(observed.astype(np.int8))

    rows = []
    for (period, model, horizon), acc in metric_acc.items():
        n = acc["n"]
        base = {"period": period, "model": model, "horizon_minutes": horizon, "nobs": int(n),
                "MAE": acc["abs_error"] / n, "RMSE": np.sqrt(acc["sq_error"] / n),
                "NLS": acc["negative_logpdf"] / n, "CRPS": acc["crps"] / n,
                "direction_accuracy": acc["direction"] / n}
        for threshold in thresholds:
            pacc = probability_acc[(period, model, horizon, float(threshold))]
            p = np.concatenate(pacc["p"]); obs = np.concatenate(pacc["observed"]).astype(int)
            row = dict(base); row["threshold"] = threshold
            row["Brier"] = np.mean((p - obs) ** 2)
            row["event_log_score"] = -np.mean(obs * np.log(p) + (1 - obs) * np.log1p(-p))
            row["AUC"] = roc_auc_score(obs, p) if np.unique(obs).size == 2 else np.nan
            rows.append(row)
    score_table = pd.DataFrame(rows)

    reliability_rows = []
    for keys, acc in probability_acc.items():
        p = np.concatenate(acc["p"])
        observed = np.concatenate(acc["observed"])
        group = pd.DataFrame({"probability": p, "observed": observed})
        try:
            bins = pd.qcut(group["probability"], 10, duplicates="drop")
        except ValueError:
            bins = pd.cut(group["probability"], np.linspace(0, 1, 11), include_lowest=True)
        grouped = group.groupby(bins, observed=True)
        for bin_id, (_, bg) in enumerate(grouped):
            reliability_rows.append({"period": keys[0], "model": keys[1], "horizon_minutes": keys[2],
                                     "threshold": keys[3], "bin": bin_id,
                                     "mean_predicted_probability": bg["probability"].mean(),
                                     "observed_frequency": bg["observed"].mean(), "nobs": len(bg)})
    return score_table, pd.DataFrame(reliability_rows), pd.DataFrame(fit_rows)
