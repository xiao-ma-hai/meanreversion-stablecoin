from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from meanrev_stablecoin.data_io import write_csv, write_json
from meanrev_stablecoin.interval_research import (
    cluster_mean_summary,
    file_sha256,
    fit_interval_pair,
    fit_rows,
    json_sha256,
    score_fit,
    tick_schedule_from_audit,
)
from meanrev_stablecoin.microstructure import ASSETS, load_raw_market, raw_file


def window(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    datetime = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.loc[datetime.between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    tables = ROOT / "output" / "tables"
    models = ROOT / "output" / "models"
    tick_audit = pd.read_csv(tables / "tick_tie_audit.csv")
    full_config = ROOT / "configs" / "full_sample_structural.json"
    train_config = ROOT / "configs" / "train_oos.json"
    full_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    serialized: dict[str, object] = {}

    for asset in ASSETS:
        path = raw_file(asset, 5, raw_dir)
        raw = load_raw_market(path)
        full = window(raw, "2021-01-01T00:00:00Z", "2025-12-31T23:59:59Z")
        training = window(raw, "2021-01-01T00:00:00Z", "2023-12-31T23:59:59Z")
        periods = {
            "validation_2024": window(raw, "2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
            "post_selection_2025": window(raw, "2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z"),
        }
        schedule = tick_schedule_from_audit(tick_audit, asset, 5)
        full_result = fit_interval_pair(full, schedule, multistart=4)
        training_result = fit_interval_pair(training, schedule, multistart=4)
        data_hash = file_sha256(path)
        full_rows.extend(fit_rows(asset, "full_2021_2025", full, full_result, data_hash, json_sha256(full_config)))
        train_rows.extend(fit_rows(asset, "training_2021_2023", training, training_result, data_hash, json_sha256(train_config)))

        fits = {fit.model_name: fit for fit in training_result["fits"]}
        for period_name, part in periods.items():
            scored = {name: score_fit(fit, part, schedule) for name, fit in fits.items()}
            for name, (scores, density_equivalent, days) in scored.items():
                raw_summary = cluster_mean_summary(scores, days)
                density_summary = cluster_mean_summary(density_equivalent, days)
                score_rows.append({
                    "asset": asset,
                    "evaluation_sample": period_name,
                    "model": name,
                    "fit_sample": "training_2021_2023 frozen",
                    "mean_log_observation_probability": raw_summary["mean"],
                    "cluster_se_log_observation_probability": raw_summary["cluster_se"],
                    "ci_lower_log_observation_probability": raw_summary["ci_lower"],
                    "ci_upper_log_observation_probability": raw_summary["ci_upper"],
                    "mean_log_density_equivalent": density_summary["mean"],
                    "nobs": raw_summary["nobs"],
                    "utc_day_clusters": raw_summary["days"],
                    "comparison_scale": "discrete observation probability; density equivalent subtracts log bin width",
                })
            ou_score, _, days = scored["OU-Interval-IFM"]
            mou_score, _, _ = scored["MOU-Interval-IFM"]
            difference = cluster_mean_summary(mou_score - ou_score, days)
            z_value = difference["mean"] / difference["cluster_se"] if difference["cluster_se"] > 0 else np.nan
            score_rows.append({
                "asset": asset,
                "evaluation_sample": period_name,
                "model": "MOU-minus-OU paired difference",
                "fit_sample": "training_2021_2023 frozen",
                "mean_log_observation_probability": difference["mean"],
                "cluster_se_log_observation_probability": difference["cluster_se"],
                "ci_lower_log_observation_probability": difference["ci_lower"],
                "ci_upper_log_observation_probability": difference["ci_upper"],
                "mean_log_density_equivalent": difference["mean"],
                "nobs": difference["nobs"],
                "utc_day_clusters": difference["days"],
                "z_value": z_value,
                "comparison_scale": "paired model difference; tick-bin width cancels exactly",
            })
        serialized[asset] = {
            "annual_tick_schedule": schedule,
            "full": [fit.to_dict() for fit in full_result["fits"]],
            "training": [fit.to_dict() for fit in training_result["fits"]],
        }
        print(f"completed {asset}")

    write_csv(pd.DataFrame(full_rows), tables / "full_sample_model_estimates.csv")
    write_csv(pd.DataFrame(train_rows), tables / "training_model_estimates.csv")
    write_csv(pd.DataFrame(score_rows), tables / "oos_scores_with_uncertainty.csv")
    write_json(serialized, models / "interval_likelihood_models.json")
    print(json.dumps({"assets": list(ASSETS), "status": "complete"}))


if __name__ == "__main__":
    main()
