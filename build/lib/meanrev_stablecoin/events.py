from __future__ import annotations

import numpy as np
import pandas as pd

from .forecasting import fast_ou_parameters, fit_conditional_lambda
from .pairs import build_exact_horizon_pairs


def identify_depeg_events(
    df: pd.DataFrame,
    b_out: float = 0.005,
    b_in: float = 0.0025,
    prior_inside_hours: int = 12,
) -> pd.DataFrame:
    price = df["close"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    deviation = np.abs(price - 1)
    bars = prior_inside_hours * 12
    exact = np.r_[False, np.diff(timestamps) == 300]
    inside = deviation <= b_out
    eligible = pd.Series(inside & exact).rolling(bars, min_periods=bars).sum().shift(1).eq(bars).fillna(False).to_numpy()
    rows = []
    in_event = False
    start = None
    for idx in range(len(df)):
        if not in_event and eligible[idx] and deviation[idx] > b_out:
            in_event = True
            start = idx
        elif in_event and deviation[idx] <= b_in:
            segment = slice(start, idx + 1)
            local = deviation[segment]
            peak_rel = int(np.argmax(local)); peak_idx = start + peak_rel
            rows.append({"start_utc": df["datetime"].iloc[start], "end_utc": df["datetime"].iloc[idx],
                         "peak_utc": df["datetime"].iloc[peak_idx], "direction": "premium" if price[start] > 1 else "discount",
                         "start_price": price[start], "peak_price": price[peak_idx],
                         "max_abs_deviation": deviation[peak_idx],
                         "recovery_minutes": (timestamps[idx] - timestamps[start]) / 60,
                         "censored": False, "b_out": b_out, "b_in": b_in})
            in_event = False
            start = None
    if in_event and start is not None:
        local = deviation[start:]; peak_idx = start + int(np.argmax(local))
        rows.append({"start_utc": df["datetime"].iloc[start], "end_utc": pd.NaT,
                     "peak_utc": df["datetime"].iloc[peak_idx], "direction": "premium" if price[start] > 1 else "discount",
                     "start_price": price[start], "peak_price": price[peak_idx], "max_abs_deviation": deviation[peak_idx],
                     "recovery_minutes": np.nan, "censored": True, "b_out": b_out, "b_in": b_in})
    return pd.DataFrame(rows)


def rolling_parameter_estimates(
    df: pd.DataFrame,
    windows_days: list[int] = [30, 90, 180, 365],
    step_days: int = 30,
) -> pd.DataFrame:
    endpoints = pd.date_range(pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC"), freq=f"{step_days}D")
    rows = []
    for window in windows_days:
        for end in endpoints:
            start = end - pd.Timedelta(days=window)
            part = df[df["datetime"].between(start, end, inclusive="left")]
            i, j = build_exact_horizon_pairs(part, 300)
            if len(i) < 1000:
                continue
            x = part["log_price"].to_numpy(); xp, xn = x[i], x[j]
            ou = fast_ou_parameters(xp, xn)
            lam, mix_ll = fit_conditional_lambda(xp, xn, np.full(len(xp), 1 / 288), ou, max_nobs=50_000)
            rows.append({"window_days": window, "start_utc": start, "end_utc": end, "nobs": len(xp),
                         "theta": ou.theta, "kappa": ou.kappa, "sigma": ou.sigma, "lambda": lam,
                         "half_life_minutes": 1440 * np.log(2) / ou.kappa,
                         "effective_half_life_minutes": 1440 * np.log(2) / (ou.kappa + lam),
                         "conditional_mixed_loglik": mix_ll})
    return pd.DataFrame(rows)

