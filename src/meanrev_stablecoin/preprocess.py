from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BAR_SECONDS, DAY_SECONDS, HARD_ANOMALY_TIMESTAMPS


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    out["date"] = out["datetime"].dt.floor("D")
    out["year"] = out["datetime"].dt.year.astype("int16")
    out["log_price"] = np.log(out["close"])
    out["deviation"] = out["close"] - 1.0
    out["typical_price"] = (out["high"] + out["low"] + out["close"]) / 3.0
    out["median_ohlc"] = out[["open", "high", "low", "close"]].median(axis=1)
    out["log_range"] = np.log(out["high"] / out["low"])
    out["log_volume"] = np.log1p(out["volume"])
    out["log_trades"] = np.log1p(out["trades"])
    out["delta_seconds"] = out["timestamp"].diff()
    out["delta_days"] = out["delta_seconds"] / DAY_SECONDS
    out["exact_5min"] = out["delta_seconds"].eq(BAR_SECONDS)
    delta_minutes = out["delta_seconds"] / 60
    out["gap_class"] = np.select(
        [delta_minutes.eq(5), delta_minutes.le(30), delta_minutes.le(360), delta_minutes.gt(360)],
        [0, 1, 2, 3],
        default=-1,
    ).astype("int8")
    out["flag_hard_anomaly"] = out["timestamp"].isin(HARD_ANOMALY_TIMESTAMPS)
    out["sample_baseline"] = out["datetime"].between(
        pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:55:00", tz="UTC")
    )
    out["sample_extended"] = out["datetime"].between(
        pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:55:00", tz="UTC")
    )
    out["sample_train"] = out["datetime"].between(
        pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:55:00", tz="UTC")
    )
    out["sample_validation"] = out["datetime"].between(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:55:00", tz="UTC")
    )
    out["sample_test"] = out["datetime"].between(
        pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:55:00", tz="UTC")
    )
    return out


def select_sample(df: pd.DataFrame, sample: str, exclude_anomalies: bool = False) -> pd.DataFrame:
    if sample == "baseline":
        mask = df["sample_baseline"]
    elif sample == "extended":
        mask = df["sample_extended"]
    elif sample == "full":
        mask = np.ones(len(df), dtype=bool)
    else:
        raise ValueError(f"Unknown sample: {sample}")
    if exclude_anomalies:
        mask = mask & ~df["flag_hard_anomaly"]
    return df.loc[mask].copy()


def sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "baseline": ("2021-01-01", "2025-12-31 23:55:00", "sample_baseline"),
        "extended": ("2020-01-01", "2025-12-31 23:55:00", "sample_extended"),
        "full": (None, None, None),
        "train": ("2021-01-01", "2023-12-31 23:55:00", "sample_train"),
        "validation": ("2024-01-01", "2024-12-31 23:55:00", "sample_validation"),
        "test": ("2025-01-01", "2025-12-31 23:55:00", "sample_test"),
    }
    rows = []
    for name, (start, end, col) in definitions.items():
        part = df if col is None else df[df[col]]
        if part.empty:
            continue
        if start is None:
            start_ts, end_ts = int(part["timestamp"].iloc[0]), int(part["timestamp"].iloc[-1])
        else:
            start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
            end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
        theoretical = (end_ts - start_ts) // BAR_SECONDS + 1
        rows.append(
            {
                "sample": name,
                "start_utc": part["datetime"].iloc[0],
                "end_utc": part["datetime"].iloc[-1],
                "observations": len(part),
                "theoretical_bars": theoretical,
                "coverage_pct": 100 * len(part) / theoretical,
                "exact_5min_pairs": int(np.sum(np.diff(part["timestamp"].to_numpy(dtype=np.int64)) == BAR_SECONDS)),
                "hard_anomalies": int(part["flag_hard_anomaly"].sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_ohlcvt(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    expected = minutes // 5
    indexed = df.set_index("datetime")
    result = indexed.resample(f"{minutes}min", label="right", closed="right").agg(
        timestamp=("timestamp", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trades=("trades", "sum"),
        source_bars=("close", "count"),
    )
    result = result.dropna(subset=["close"]).reset_index()
    result["timestamp"] = result["timestamp"].astype("int64")
    result["complete_bin"] = result["source_bars"].eq(expected)
    result["log_price"] = np.log(result["close"])
    return result


def write_parquet_compat(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
