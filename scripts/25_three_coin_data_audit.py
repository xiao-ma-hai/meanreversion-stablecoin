from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from meanrev_stablecoin.data_io import sha256_file, write_csv, write_json
from meanrev_stablecoin.microstructure import (
    ASSETS,
    common_exact_pair_timestamps,
    load_raw_market,
    raw_file,
    timestamp_hash,
    transition_audit_row,
)


START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")


def subset_window(frame: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.loc[dt.between(START, END)].reset_index(drop=True)


def lookup_close(frame: pd.DataFrame, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = frame["timestamp"].to_numpy(dtype=np.int64)
    index = np.searchsorted(time, timestamps)
    next_index = np.searchsorted(time, timestamps + 300)
    if not (np.all(time[index] == timestamps) and np.all(time[next_index] == timestamps + 300)):
        raise RuntimeError("common timestamp construction is inconsistent")
    close = frame["close"].to_numpy(dtype=float)
    return close[index], close[next_index]


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    frames = {asset: subset_window(load_raw_market(raw_file(asset, 5, raw_dir))) for asset in ASSETS}
    common = common_exact_pair_timestamps(frames, 300)
    if len(common) == 0:
        raise RuntimeError("no common exact five-minute pairs")

    audit_rows = []
    for asset, frame in frames.items():
        audit_rows.append(
            transition_audit_row(
                frame,
                frame["close"].to_numpy(dtype=float),
                asset=asset,
                minutes=5,
                proxy="close",
                period="2021-2025 asset-specific calendar",
            )
        )
        previous, following = lookup_close(frame, common)
        audit_rows.append({
            "asset": asset,
            "frequency_minutes": 5,
            "price_proxy": "close",
            "period": "2021-2025 aligned common exact pairs",
            "start_utc": pd.to_datetime(common[0], unit="s", utc=True),
            "end_utc": pd.to_datetime(common[-1] + 300, unit="s", utc=True),
            "observations": len(np.unique(np.r_[common, common + 300])),
            "theoretical_bars": int((common[-1] + 300 - common[0]) // 300 + 1),
            "bar_coverage": len(np.unique(np.r_[common, common + 300])) / int((common[-1] + 300 - common[0]) // 300 + 1),
            "exact_pairs": len(common),
            "exact_pair_share_of_adjacent": np.nan,
            "tie_count": int(np.sum(previous == following)),
            "tie_rate": float(np.mean(previous == following)),
            "min_nonzero_price_change": float(np.min(np.abs(following[previous != following] - previous[previous != following]))),
            "p01_nonzero_price_change": float(np.quantile(np.abs(following[previous != following] - previous[previous != following]), 0.01)),
            "modal_nonzero_price_change": np.nan,
            "zero_volume_rate": np.nan,
            "one_trade_or_less_rate": np.nan,
            "continuous_segments": np.nan,
            "median_segment_bars": np.nan,
            "p90_segment_bars": np.nan,
            "max_segment_bars": np.nan,
        })
    audit = pd.DataFrame(audit_rows)
    write_csv(audit, ROOT / "output" / "tables" / "three_coin_data_audit.csv")

    common_summary = pd.DataFrame([{
        "frequency_minutes": 5,
        "window_start_utc": START,
        "window_end_utc": END,
        "assets": ",".join(ASSETS),
        "common_exact_pairs": len(common),
        "common_pair_timestamp_sha256": timestamp_hash(common),
        "no_forward_fill": True,
        "quality_gate_min_common_pairs": 100_000,
        "quality_gate_pass": bool(len(common) >= 100_000),
    }])
    write_csv(common_summary, ROOT / "output" / "tables" / "common_sample_selection.csv")

    config = {
        "design": "balanced common exact-pair calendar",
        "frequency_minutes": 5,
        "window_start_utc": START.isoformat(),
        "window_end_utc": END.isoformat(),
        "assets": list(ASSETS),
        "common_exact_pairs": int(len(common)),
        "common_pair_timestamp_sha256": timestamp_hash(common),
        "selection_rule": "intersection of observed t and t+300 seconds for all assets; no imputation",
        "quality_gate": {"minimum_common_pairs": 100_000, "passed": bool(len(common) >= 100_000)},
        "raw_sha256": {asset: sha256_file(raw_file(asset, 5, raw_dir)) for asset in ASSETS},
    }
    write_json(config, ROOT / "configs" / "common_three_coin_sample.json")
    print(json.dumps({"common_exact_pairs": len(common), "gate": config["quality_gate"]}, indent=2))


if __name__ == "__main__":
    main()
