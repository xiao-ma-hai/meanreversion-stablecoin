from __future__ import annotations

import numpy as np
import pandas as pd


def build_exact_horizon_pairs(
    df: pd.DataFrame,
    horizon_seconds: int,
    value_column: str = "log_price",
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    values = df[value_column].to_numpy(dtype=float)
    target = timestamps + int(horizon_seconds)
    j = np.searchsorted(timestamps, target)
    valid = (j < len(timestamps))
    valid_idx = np.flatnonzero(valid)
    valid[valid_idx] &= timestamps[j[valid_idx]] == target[valid_idx]
    i_idx = np.flatnonzero(valid)
    return i_idx, j[valid]


def build_pair_arrays(
    df: pd.DataFrame,
    horizon_seconds: int,
    value_column: str = "log_price",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i, j = build_exact_horizon_pairs(df, horizon_seconds, value_column)
    x = df[value_column].to_numpy(dtype=float)
    delta = (df["timestamp"].to_numpy(dtype=np.int64)[j] - df["timestamp"].to_numpy(dtype=np.int64)[i]) / 86400.0
    return x[i], x[j], delta, i


def build_irregular_adjacent_pairs(
    df: pd.DataFrame,
    value_column: str = "log_price",
    maximum_gap_class: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = df["gap_class"].between(0, maximum_gap_class).to_numpy()
    valid = np.flatnonzero(mask)
    x = df[value_column].to_numpy(dtype=float)
    return x[valid - 1], x[valid], df["delta_days"].to_numpy(dtype=float)[valid]

