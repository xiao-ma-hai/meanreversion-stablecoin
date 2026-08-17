from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    BAR_SECONDS,
    EXPECTED_FIRST_TIMESTAMP,
    EXPECTED_LAST_TIMESTAMP,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
    HARD_ANOMALY_TIMESTAMPS,
)
from .data_io import resolve_path, sha256_file


def audit_raw_data(
    df: pd.DataFrame,
    path: str | Path,
    expected_sha256: str = EXPECTED_SHA256,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = resolve_path(path)
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    deltas = np.diff(timestamps)
    first, last = int(timestamps[0]), int(timestamps[-1])
    theoretical = int((last - first) // BAR_SECONDS + 1)
    exact_pairs = int(np.sum(deltas == BAR_SECONDS))
    gap_pairs = int(np.sum(deltas > BAR_SECONDS))
    missing_grid = int(np.sum(np.maximum(deltas // BAR_SECONDS - 1, 0)))
    missing_fields = int(df.isna().sum().sum())
    duplicate_timestamps = int(df["timestamp"].duplicated().sum())
    timestamp_grid_violations = int(np.sum(timestamps % BAR_SECONDS != 0))
    strictly_increasing = bool(np.all(deltas > 0))
    ohlc_valid = (
        (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df[["open", "close"]].max(axis=1) <= df["high"])
    )
    nonnegative = (df[["open", "high", "low", "close", "volume", "trades"]] >= 0).all(axis=1)
    digest = sha256_file(path)
    attrs = getattr(path.stat(), "st_file_attributes", 0)
    is_readonly = bool(attrs & getattr(stat, "FILE_ATTRIBUTE_READONLY", 1))

    observed = {
        "sha256": digest,
        "rows": len(df),
        "first_timestamp": first,
        "last_timestamp": last,
        "first_utc": pd.to_datetime(first, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
        "last_utc": pd.to_datetime(last, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
        "duplicate_timestamps": duplicate_timestamps,
        "missing_fields": missing_fields,
        "exact_300s_pairs": exact_pairs,
        "greater_than_300s_pairs": gap_pairs,
        "theoretical_grid_rows": theoretical,
        "missing_grid_rows": missing_grid,
        "coverage_pct": 100.0 * len(df) / theoretical,
        "timestamp_grid_violations": timestamp_grid_violations,
        "strictly_increasing": strictly_increasing,
        "ohlc_violations": int((~ohlc_valid).sum()),
        "negative_value_rows": int((~nonnegative).sum()),
        "raw_readonly": is_readonly or not os.access(path, os.W_OK),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 2**20),
    }
    expected = {
        "sha256": expected_sha256,
        "rows": EXPECTED_ROWS,
        "first_timestamp": EXPECTED_FIRST_TIMESTAMP,
        "last_timestamp": EXPECTED_LAST_TIMESTAMP,
        "duplicate_timestamps": 0,
        "missing_fields": 0,
        "exact_300s_pairs": 750_220,
        "greater_than_300s_pairs": 46_957,
        "theoretical_grid_rows": 921_395,
        "missing_grid_rows": 124_217,
        "coverage_pct": 86.5186,
        "timestamp_grid_violations": 0,
        "strictly_increasing": True,
        "ohlc_violations": 0,
        "negative_value_rows": 0,
        "raw_readonly": True,
    }
    rows = []
    for metric, value in observed.items():
        target = expected.get(metric)
        if isinstance(target, float):
            match = bool(abs(float(value) - target) < 5e-5)
        else:
            match = True if target is None else bool(value == target)
        rows.append({"metric": metric, "observed": value, "expected": target, "match": match})
    audit_table = pd.DataFrame(rows)

    gap_bars = deltas[deltas > BAR_SECONDS] // BAR_SECONDS - 1
    gap_minutes = gap_bars * 5
    gap_class = pd.cut(
        deltas / 60,
        bins=[5, 30, 360, np.inf],
        labels=["G1_5_to_30m", "G2_30m_to_6h", "G3_over_6h"],
        right=True,
    )
    gap_summary = (
        pd.Series(gap_class[deltas > BAR_SECONDS])
        .value_counts(sort=False)
        .rename_axis("gap_class")
        .reset_index(name="pair_count")
    )
    if len(gap_minutes):
        extras = pd.DataFrame(
            {
                "gap_class": ["all_gaps"],
                "pair_count": [len(gap_minutes)],
                "missing_bars": [int(gap_bars.sum())],
                "median_gap_minutes": [float(np.median(gap_minutes + 5))],
                "max_gap_minutes": [float(np.max(deltas[deltas > BAR_SECONDS]) / 60)],
            }
        )
        gap_summary = pd.concat([gap_summary, extras], ignore_index=True)

    dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    years = dt.dt.year
    coverage_rows = []
    for year, group in df.groupby(years, sort=True):
        start = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), dt.iloc[0])
        end = min(pd.Timestamp(f"{year}-12-31 23:55:00", tz="UTC"), dt.iloc[-1])
        expected_year = int((end.timestamp() - start.timestamp()) // BAR_SECONDS + 1)
        coverage_rows.append(
            {
                "year": int(year),
                "observations": len(group),
                "theoretical_bars": expected_year,
                "coverage_pct": 100 * len(group) / expected_year,
            }
        )
    annual = pd.DataFrame(coverage_rows)

    anomaly_rows = df[df["timestamp"].isin(HARD_ANOMALY_TIMESTAMPS)].copy()
    anomaly_rows.insert(1, "utc", pd.to_datetime(anomaly_rows["timestamp"], unit="s", utc=True))
    observed["hard_anomalies_found"] = int(len(anomaly_rows))
    critical = [
        "sha256", "rows", "first_timestamp", "last_timestamp", "duplicate_timestamps",
        "missing_fields", "exact_300s_pairs", "greater_than_300s_pairs",
        "theoretical_grid_rows", "missing_grid_rows", "timestamp_grid_violations",
        "strictly_increasing", "ohlc_violations", "negative_value_rows",
    ]
    observed["audit_pass"] = bool(audit_table.set_index("metric").loc[critical, "match"].all())
    return audit_table, gap_summary, annual, anomaly_rows, observed

