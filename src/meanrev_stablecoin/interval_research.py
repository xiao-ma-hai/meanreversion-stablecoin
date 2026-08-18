from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .models.mou_interval_likelihood import (
    fit_interval_mou_ifm,
    fit_interval_ou_ifm,
    fit_rounded_normal_margin,
    mixed_ou_interval_logprob_tau,
    ou_interval_logprob_tau,
)


def normalized_tick(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError("empirical tick must be positive")
    return float(10.0 ** np.round(np.log10(value)))


def tick_schedule_from_audit(audit: pd.DataFrame, asset: str, minutes: int = 5) -> dict[int, float]:
    selected = audit[
        (audit["asset"] == asset)
        & (audit["frequency_minutes"].astype(int) == minutes)
        & (audit["price_proxy"] == "close")
        & (audit["period"] != "full")
    ]
    return {
        int(row.period): normalized_tick(float(row.min_nonzero_price_change))
        for row in selected.itertuples()
        if str(row.period).isdigit()
    }


def ticks_for_frame(frame: pd.DataFrame, schedule: dict[int, float]) -> np.ndarray:
    years = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.year.to_numpy()
    missing = sorted(set(map(int, np.unique(years))) - set(schedule))
    if missing:
        raise ValueError(f"tick schedule is missing years: {missing}")
    return np.asarray([schedule[int(year)] for year in years], dtype=float)


def exact_pair_indices(frame: pd.DataFrame, seconds: int = 300) -> tuple[np.ndarray, np.ndarray]:
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    left = np.flatnonzero(np.diff(timestamps) == seconds)
    return left, left + 1


def fit_interval_pair(frame: pd.DataFrame, schedule: dict[int, float], multistart: int = 4) -> dict[str, Any]:
    ticks = ticks_for_frame(frame, schedule)
    price = frame["close"].to_numpy(dtype=float)
    left, right = exact_pair_indices(frame)
    theta, tau, margin_loglik = fit_rounded_normal_margin(price, ticks)
    delta = np.full(len(left), 1 / 288)
    ou = fit_interval_ou_ifm(
        price[right], price[left], ticks[right], delta, theta=theta, tau=tau, multistart=min(multistart, 5)
    )
    mou = fit_interval_mou_ifm(
        price[right], price[left], ticks[right], delta, theta=theta, tau=tau, multistart=multistart
    )
    return {
        "theta": theta,
        "tau": tau,
        "margin_loglik": margin_loglik,
        "price": price,
        "ticks": ticks,
        "left": left,
        "right": right,
        "fits": [ou, mou],
    }


def fit_rows(asset: str, fit_sample: str, frame: pd.DataFrame, result: dict[str, Any], data_hash: str, config_hash: str) -> list[dict[str, Any]]:
    start = pd.to_datetime(frame["timestamp"].iloc[0], unit="s", utc=True)
    end = pd.to_datetime(frame["timestamp"].iloc[-1], unit="s", utc=True)
    rows = []
    for fit in result["fits"]:
        rows.append({
            "asset": asset,
            "fit_sample": fit_sample,
            "sample_start_utc": start,
            "sample_end_utc": end,
            "state_observations": len(frame),
            "exact_pairs_available": len(result["left"]),
            "pairs_used": fit.nobs,
            "pair_sampling": "all exact 300-second pairs",
            "margin_estimation_sample": fit_sample,
            "dynamic_estimation_sample": fit_sample,
            "model": fit.model_name,
            **fit.params,
            "base_half_life_minutes": fit.metadata["base_half_life_minutes"],
            "conditional_mean_half_life_minutes": fit.metadata["conditional_mean_half_life_minutes"],
            "loglik": fit.loglik,
            "aic": fit.aic,
            "bic": fit.bic,
            "converged": fit.converged,
            "gradient_norm": fit.gradient_norm,
            "hessian_condition": fit.hessian_condition,
            "likelihood_type": fit.metadata["likelihood_type"],
            "previous_state_treatment": fit.metadata["previous_state_treatment"],
            "parameter_bounds": "log(kappa)[-12,14]; log(lambda)[-18,12] for MOU",
            "raw_data_sha256": data_hash,
            "config_sha256": config_hash,
        })
    return rows


def score_fit(fit: Any, frame: pd.DataFrame, schedule: dict[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ticks = ticks_for_frame(frame, schedule)
    price = frame["close"].to_numpy(dtype=float)
    left, right = exact_pair_indices(frame)
    delta = np.full(len(left), 1 / 288)
    p = fit.params
    if fit.model_name.startswith("OU-"):
        scores = ou_interval_logprob_tau(price[right], price[left], ticks[right], delta, p["theta"], p["tau"], p["kappa"])
    else:
        scores = mixed_ou_interval_logprob_tau(
            price[right], price[left], ticks[right], delta, p["theta"], p["tau"], p["kappa"], p["lambda"]
        )
    lower = np.log(price[right] - ticks[right] / 2)
    upper = np.log(price[right] + ticks[right] / 2)
    density_equivalent = scores - np.log(upper - lower)
    days = pd.to_datetime(frame["timestamp"].to_numpy(dtype=np.int64)[left], unit="s", utc=True).floor("D").to_numpy()
    return scores, density_equivalent, days


def cluster_mean_summary(values: np.ndarray, days: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    table = pd.DataFrame({"value": values, "day": days}).groupby("day")["value"].agg(["sum", "count"])
    mean = float(values.mean())
    clusters = len(table)
    centered = table["sum"].to_numpy() - table["count"].to_numpy() * mean
    variance = clusters / max(clusters - 1, 1) * float(np.sum(centered * centered)) / (len(values) ** 2)
    se = float(np.sqrt(max(variance, 0.0)))
    return {
        "mean": mean,
        "cluster_se": se,
        "ci_lower": mean - norm.ppf(0.975) * se,
        "ci_upper": mean + norm.ppf(0.975) * se,
        "days": clusters,
        "nobs": len(values),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(normalized).hexdigest()
