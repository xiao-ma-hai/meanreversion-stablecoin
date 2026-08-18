from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import ROOT
from meanrev_stablecoin.data_io import write_csv, write_json
from meanrev_stablecoin.interval_research import tick_schedule_from_audit
from meanrev_stablecoin.microstructure import ASSETS, common_exact_pair_timestamps, load_raw_market, raw_file, timestamp_hash
from meanrev_stablecoin.models.mou_interval_likelihood import (
    fit_interval_mou_ifm,
    fit_interval_ou_ifm,
    fit_rounded_normal_margin,
)


def baseline(frame: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.loc[dt.between(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))].reset_index(drop=True)


def pair_values(frame: pd.DataFrame, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = frame["timestamp"].to_numpy(dtype=np.int64)
    left = np.searchsorted(time, timestamps)
    right = np.searchsorted(time, timestamps + 300)
    price = frame["close"].to_numpy(dtype=float)
    return price[left], price[right], left, right


def nonparametric_drift(previous: np.ndarray, following: np.ndarray, asset: str, bins: int = 31) -> tuple[pd.DataFrame, float]:
    x = np.log(previous)
    increment = (np.log(following) - x) * 288
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    labels = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    violation = 0.0
    for index in range(len(edges) - 1):
        mask = labels == index
        if not np.any(mask):
            continue
        center = float(np.mean(x[mask]))
        drift = float(np.mean(increment[mask]))
        weight = float(np.mean(mask))
        violation += max(0.0, center * drift) ** 2 * weight
        rows.append({
            "asset": asset,
            "bin": index,
            "state_center": center,
            "drift_per_day": drift,
            "observations": int(mask.sum()),
            "weight": weight,
            "toward_zero": bool(center * drift < 0),
        })
    return pd.DataFrame(rows), float(violation)


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    frames = {asset: baseline(load_raw_market(raw_file(asset, 5, raw_dir))) for asset in ASSETS}
    common = common_exact_pair_timestamps(frames, 300)
    tick_audit = pd.read_csv(ROOT / "output" / "tables" / "tick_tie_audit.csv")
    parameter_rows = []
    drift_tables = []
    model_payload = {}

    for asset, frame in frames.items():
        previous, following, left, right = pair_values(frame, common)
        schedule = tick_schedule_from_audit(tick_audit, asset, 5)
        years = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.year.to_numpy()
        tick_all = np.asarray([schedule[int(year)] for year in years])
        state_index = np.unique(np.r_[left, right])
        theta, tau, margin_loglik = fit_rounded_normal_margin(
            frame["close"].to_numpy(dtype=float)[state_index], tick_all[state_index]
        )
        delta = np.full(len(common), 1 / 288)
        ou = fit_interval_ou_ifm(
            following, previous, tick_all[right], delta, theta=theta, tau=tau, multistart=4
        )
        mou = fit_interval_mou_ifm(
            following, previous, tick_all[right], delta, theta=theta, tau=tau, multistart=4
        )
        drift, violation = nonparametric_drift(previous, following, asset)
        drift_tables.append(drift)
        tail_values = np.log(frame["close"].to_numpy(dtype=float)[state_index])
        for fit in (ou, mou):
            parameter_rows.append({
                "asset": asset,
                "model": fit.model_name,
                "sample": "aligned common exact pairs 2021-2025",
                "common_pairs": len(common),
                "common_pair_timestamp_sha256": timestamp_hash(common),
                **fit.params,
                "base_half_life_minutes": fit.metadata["base_half_life_minutes"],
                "conditional_mean_half_life_minutes": fit.metadata["conditional_mean_half_life_minutes"],
                "local_drift_violation": violation,
                "tail_abs_gt_10bp": float(np.mean(np.abs(tail_values) > 0.001)),
                "tail_abs_gt_25bp": float(np.mean(np.abs(tail_values) > 0.0025)),
                "tail_abs_gt_50bp": float(np.mean(np.abs(tail_values) > 0.005)),
                "tail_abs_gt_100bp": float(np.mean(np.abs(tail_values) > 0.01)),
                "loglik": fit.loglik,
                "converged": fit.converged,
                "observation_model": "annual-tick conditional interval approximation",
                "structural_status": "provisional pending full grid-filter and synchronous-bootstrap validation",
            })
        model_payload[asset] = {"OU": ou.to_dict(), "MOU": mou.to_dict(), "margin_loglik": margin_loglik}
        print(f"completed common-sample {asset}")

    parameters = pd.DataFrame(parameter_rows)
    write_csv(parameters, ROOT / "output" / "tables" / "cross_stablecoin_parameters.csv")
    write_csv(pd.concat(drift_tables, ignore_index=True), ROOT / "output" / "tables" / "cross_stablecoin_drift_curves.csv")
    mixed = parameters[parameters["model"] == "MOU-Interval-IFM"].set_index("asset")
    pairwise_rows = []
    for first, second in itertools.combinations(ASSETS, 2):
        pairwise_rows.append({
            "first_asset": first,
            "second_asset": second,
            "delta_kappa": float(mixed.loc[first, "kappa"] - mixed.loc[second, "kappa"]),
            "delta_lambda": float(mixed.loc[first, "lambda"] - mixed.loc[second, "lambda"]),
            "delta_conditional_mean_half_life_minutes": float(
                mixed.loc[first, "conditional_mean_half_life_minutes"]
                - mixed.loc[second, "conditional_mean_half_life_minutes"]
            ),
            "synchronous_ci_lower": np.nan,
            "synchronous_ci_upper": np.nan,
            "formal_inference_gate": False,
            "inference_note": "synchronous day-block refit deferred until full filtered likelihood passes size/power validation",
        })
    write_csv(pd.DataFrame(pairwise_rows), ROOT / "output" / "tables" / "cross_stablecoin_pairwise_tests.csv")
    write_json(model_payload, ROOT / "output" / "models" / "cross_stablecoin_interval_models.json")

    drift_all = pd.concat(drift_tables, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for asset, group in drift_all.groupby("asset"):
        ax.plot(10_000 * group["state_center"], 10_000 * group["drift_per_day"], marker="o", ms=3, label=asset)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Log-price deviation (bp)")
    ax.set_ylabel("Estimated local drift (bp/day)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "output" / "figures" / "cross_stablecoin_drift_curves.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "output" / "figures" / "cross_stablecoin_drift_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(ASSETS))
    ax.bar(x - 0.18, mixed.loc[list(ASSETS), "base_half_life_minutes"] / 60, width=0.36, label="Base persistence")
    ax.bar(x + 0.18, mixed.loc[list(ASSETS), "conditional_mean_half_life_minutes"] / 60, width=0.36, label="With refresh")
    ax.set_xticks(x, ASSETS)
    ax.set_ylabel("Conditional-mean half-life (hours)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "output" / "figures" / "cross_stablecoin_half_lives.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "output" / "figures" / "cross_stablecoin_half_lives.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
