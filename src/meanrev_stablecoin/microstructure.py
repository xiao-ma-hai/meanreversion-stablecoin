from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import RAW_COLUMNS


ASSETS = ("USDT", "USDC", "DAI")
FREQUENCIES = (1, 5)


def raw_file(asset: str, minutes: int, raw_dir: Path) -> Path:
    asset = asset.upper()
    if asset not in ASSETS or minutes not in FREQUENCIES:
        raise ValueError("unsupported asset or frequency")
    return raw_dir / f"{asset}USD_{minutes}.csv"


def load_raw_market(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        dtype={
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "trades": "int64",
        },
    )


def price_proxies(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "close": df["close"].to_numpy(dtype=float),
        "ohlc_average": df[["open", "high", "low", "close"]].mean(axis=1).to_numpy(dtype=float),
        "typical_price": ((df["high"] + df["low"] + df["close"]) / 3).to_numpy(dtype=float),
    }


def _increment_mode(nonzero: np.ndarray) -> float:
    if len(nonzero) == 0:
        return np.nan
    rounded = np.round(nonzero, 10)
    values, counts = np.unique(rounded, return_counts=True)
    return float(values[np.argmax(counts)])


def transition_audit_row(
    df: pd.DataFrame,
    values: np.ndarray,
    *,
    asset: str,
    minutes: int,
    proxy: str,
    period: str,
) -> dict[str, object]:
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    dt = np.diff(timestamps)
    changes = np.diff(np.asarray(values, dtype=float))
    exact = dt == minutes * 60
    exact_changes = changes[exact]
    nonzero = np.abs(exact_changes[exact_changes != 0])
    breaks = np.r_[True, ~exact]
    segment_ids = np.cumsum(breaks) - 1
    segment_lengths = np.bincount(segment_ids)
    theoretical = 0
    if len(timestamps):
        theoretical = int((timestamps[-1] - timestamps[0]) // (minutes * 60) + 1)
    return {
        "asset": asset,
        "frequency_minutes": minutes,
        "price_proxy": proxy,
        "period": period,
        "start_utc": pd.to_datetime(timestamps[0], unit="s", utc=True) if len(timestamps) else pd.NaT,
        "end_utc": pd.to_datetime(timestamps[-1], unit="s", utc=True) if len(timestamps) else pd.NaT,
        "observations": int(len(timestamps)),
        "theoretical_bars": theoretical,
        "bar_coverage": float(len(timestamps) / theoretical) if theoretical else np.nan,
        "exact_pairs": int(exact.sum()),
        "exact_pair_share_of_adjacent": float(exact.mean()) if len(exact) else np.nan,
        "tie_count": int(np.sum(exact_changes == 0)),
        "tie_rate": float(np.mean(exact_changes == 0)) if len(exact_changes) else np.nan,
        "min_nonzero_price_change": float(np.min(nonzero)) if len(nonzero) else np.nan,
        "p01_nonzero_price_change": float(np.quantile(nonzero, 0.01)) if len(nonzero) else np.nan,
        "modal_nonzero_price_change": _increment_mode(nonzero),
        "zero_volume_rate": float(np.mean(df["volume"].to_numpy(dtype=float) == 0)),
        "one_trade_or_less_rate": float(np.mean(df["trades"].to_numpy(dtype=int) <= 1)),
        "continuous_segments": int(len(segment_lengths)),
        "median_segment_bars": float(np.median(segment_lengths)),
        "p90_segment_bars": float(np.quantile(segment_lengths, 0.90)),
        "max_segment_bars": int(np.max(segment_lengths)),
    }


def annual_transition_audit(df: pd.DataFrame, asset: str, minutes: int) -> pd.DataFrame:
    years = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.year.to_numpy()
    proxies = price_proxies(df)
    rows: list[dict[str, object]] = []
    selections: list[tuple[str, np.ndarray]] = [("full", np.ones(len(df), dtype=bool))]
    selections.extend((str(year), years == year) for year in np.unique(years))
    for period, mask in selections:
        part = df.loc[mask].reset_index(drop=True)
        for proxy, values in proxies.items():
            rows.append(
                transition_audit_row(
                    part,
                    values[mask],
                    asset=asset,
                    minutes=minutes,
                    proxy=proxy,
                    period=period,
                )
            )
    return pd.DataFrame(rows)


def common_exact_pair_timestamps(frames: dict[str, pd.DataFrame], seconds: int) -> np.ndarray:
    common: set[int] | None = None
    for frame in frames.values():
        timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
        valid = timestamps[:-1][np.diff(timestamps) == seconds]
        values = set(map(int, valid))
        common = values if common is None else common.intersection(values)
    return np.asarray(sorted(common or ()), dtype=np.int64)


def timestamp_hash(timestamps: Iterable[int]) -> str:
    array = np.asarray(list(timestamps), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()
