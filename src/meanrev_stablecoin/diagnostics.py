from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from .estimation.inference import ljung_box_summary


def descriptive_statistics(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    variables = ["close", "deviation", "log_price", "log_range", "volume", "trades", "log_volume", "log_trades"]
    rows = []
    for variable in variables:
        values = df[variable].to_numpy(dtype=float)
        rows.append(
            {
                "sample": sample_name,
                "variable": variable,
                "n": len(values),
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "min": np.min(values),
                "p01": np.quantile(values, 0.01),
                "p05": np.quantile(values, 0.05),
                "median": np.median(values),
                "p95": np.quantile(values, 0.95),
                "p99": np.quantile(values, 0.99),
                "max": np.max(values),
                "skew": pd.Series(values).skew(),
                "excess_kurtosis": pd.Series(values).kurt(),
            }
        )
    return pd.DataFrame(rows)


def pit_diagnostics(pit: np.ndarray, model_name: str, max_lag: int = 20) -> dict:
    pit = np.clip(np.asarray(pit, dtype=float), 1e-10, 1 - 1e-10)
    z = norm.ppf(pit)
    q_z, p_z = ljung_box_summary(z, max_lag)
    q_z2, p_z2 = ljung_box_summary(z * z, max_lag)
    ks = kstest(pit, "uniform")
    return {
        "model": model_name,
        "nobs": len(pit),
        "pit_mean": np.mean(pit),
        "pit_variance": np.var(pit),
        "pit_ks_stat": ks.statistic,
        "pit_ks_pvalue": ks.pvalue,
        "ljung_box_z_q20": q_z,
        "ljung_box_z_p20": p_z,
        "ljung_box_z2_q20": q_z2,
        "ljung_box_z2_p20": p_z2,
        "lower_1pct_coverage": np.mean(pit < 0.01),
        "upper_99pct_coverage": np.mean(pit > 0.99),
    }


def pit_liquidity_groups(pit: np.ndarray, source_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    result = []
    for variable in ("volume", "trades"):
        groups = pd.qcut(source_df[variable], 4, labels=False, duplicates="drop")
        for group in sorted(groups.dropna().unique()):
            values = np.asarray(pit)[groups.to_numpy() == group]
            result.append({"model": model_name, "group_variable": variable, "quartile": int(group) + 1,
                           "nobs": len(values), "pit_mean": np.mean(values), "pit_variance": np.var(values),
                           "lower_1pct": np.mean(values < 0.01), "upper_99pct": np.mean(values > 0.99)})
    return pd.DataFrame(result)

