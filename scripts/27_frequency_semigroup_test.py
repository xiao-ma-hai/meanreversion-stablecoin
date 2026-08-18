from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import ROOT
from meanrev_stablecoin.data_io import write_csv, write_json
from meanrev_stablecoin.interval_research import cluster_mean_summary, tick_schedule_from_audit
from meanrev_stablecoin.microstructure import ASSETS, load_raw_market, raw_file
from meanrev_stablecoin.models.mou_interval_likelihood import (
    fit_interval_mou_ifm,
    fit_interval_ou_ifm,
    fit_rounded_normal_margin,
    mixed_ou_interval_logprob_tau,
)


MAX_PAIRS = 20_000
SEED = 20260818


def training_window(frame: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.loc[dt.between(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:59:59", tz="UTC"))].reset_index(drop=True)


def day_stratified_pairs(frame: pd.DataFrame, seconds: int, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    left = np.flatnonzero(np.diff(timestamps) == seconds)
    if len(left) <= maximum:
        selected = left
    else:
        days = pd.to_datetime(timestamps[left], unit="s", utc=True).floor("D").to_numpy()
        unique_days, starts, counts = np.unique(days, return_index=True, return_counts=True)
        per_day = int(np.ceil(maximum / len(unique_days)))
        rng = np.random.default_rng(seed)
        pieces = []
        for start, count in zip(starts, counts):
            candidates = left[start:start + count]
            take = min(per_day, len(candidates))
            pieces.append(rng.choice(candidates, size=take, replace=False))
        selected = np.sort(np.concatenate(pieces))
        if len(selected) > maximum:
            selected = np.sort(rng.choice(selected, size=maximum, replace=False))
    digest = hashlib.sha256(timestamps[selected].astype("<i8").tobytes()).hexdigest()
    return selected, selected + 1, np.asarray([digest])


def fit_frequency(
    asset: str,
    minutes: int,
    tick_audit: pd.DataFrame,
    extra_dynamics: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    print(f"starting {asset} {minutes}m", flush=True)
    frame = training_window(load_raw_market(raw_file(asset, minutes, ROOT / "data" / "raw")))
    left, right, digest = day_stratified_pairs(frame, minutes * 60, MAX_PAIRS, SEED + minutes)
    schedule = tick_schedule_from_audit(tick_audit, asset, minutes)
    years = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.year.to_numpy()
    ticks = np.asarray([schedule[int(year)] for year in years])
    price = frame["close"].to_numpy(dtype=float)
    states = np.unique(np.r_[left, right])
    # The frequency exercise is a pilot.  Use the same midpoint-moment IFM
    # margin at both frequencies and reserve the much slower rounded-margin MLE
    # for the full five-minute structural and OOS tables.
    state_log_price = np.log(price[states])
    theta = float(np.mean(state_log_price))
    tau = float(np.std(state_log_price, ddof=0))
    margin_loglik = np.nan
    print(f"margin complete {asset} {minutes}m", flush=True)
    delta = np.full(len(left), minutes / 1440)
    ou = fit_interval_ou_ifm(
        price[right], price[left], ticks[right], delta, theta=theta, tau=tau, multistart=2
    )
    print(f"OU complete {asset} {minutes}m", flush=True)
    mou = fit_interval_mou_ifm(
        price[right], price[left], ticks[right], delta, theta=theta, tau=tau,
        multistart=4, extra_starts=extra_dynamics,
    )
    print(f"MOU complete {asset} {minutes}m", flush=True)
    days = pd.to_datetime(frame["timestamp"].to_numpy(dtype=np.int64)[left], unit="s", utc=True).floor("D").to_numpy()
    return {
        "asset": asset,
        "minutes": minutes,
        "frame": frame,
        "left": left,
        "right": right,
        "ticks": ticks,
        "price": price,
        "days": days,
        "theta": theta,
        "tau": tau,
        "margin_loglik": margin_loglik,
        "OU": ou,
        "MOU": mou,
        "pair_index_sha256": str(digest[0]),
    }


def main() -> None:
    tick_audit = pd.read_csv(ROOT / "output" / "tables" / "tick_tie_audit.csv")
    results: dict[tuple[str, int], dict[str, object]] = {}
    parameter_rows = []
    model_payload = {}
    for asset in ASSETS:
        one_minute = fit_frequency(asset, 1, tick_audit)
        one_fit = one_minute["MOU"]
        ordered_results = (
            one_minute,
            fit_frequency(
                asset,
                5,
                tick_audit,
                extra_dynamics=[(one_fit.params["kappa"], one_fit.params["lambda"])],
            ),
        )
        for result in ordered_results:
            minutes = int(result["minutes"])
            results[(asset, minutes)] = result
            fit = result["MOU"]
            parameter_rows.append({
                "asset": asset,
                "frequency_minutes": minutes,
                "fit_sample": "training_2021_2023",
                "pair_sampling": f"UTC-day stratified without replacement; cap={MAX_PAIRS}",
                "pair_index_sha256": result["pair_index_sha256"],
                "pairs_used": len(result["left"]),
                **fit.params,
                "base_half_life_minutes": fit.metadata["base_half_life_minutes"],
                "conditional_mean_half_life_minutes": fit.metadata["conditional_mean_half_life_minutes"],
                "converged": fit.converged,
                "observation_model": "annual-tick conditional interval approximation",
                "margin_estimator": "log-bin-midpoint moments (frequency pilot)",
            })
            model_payload[f"{asset}_{minutes}m"] = {"OU": result["OU"].to_dict(), "MOU": fit.to_dict()}
            print(f"completed {asset} {minutes}m")

    parameters = pd.DataFrame(parameter_rows)
    comparison_rows = []
    for asset in ASSETS:
        one = results[(asset, 1)]
        five = results[(asset, 5)]
        one_fit = one["MOU"]
        five_fit = five["MOU"]
        p1, p5 = one_fit.params, five_fit.params
        price = five["price"]
        left, right = five["left"], five["right"]
        delta = np.full(len(left), 5 / 1440)
        transfer = mixed_ou_interval_logprob_tau(
            price[right], price[left], five["ticks"][right], delta,
            p5["theta"], p5["tau"], p1["kappa"], p1["lambda"],
        )
        own = mixed_ou_interval_logprob_tau(
            price[right], price[left], five["ticks"][right], delta,
            p5["theta"], p5["tau"], p5["kappa"], p5["lambda"],
        )
        difference = cluster_mean_summary(own - transfer, five["days"])
        comparison_rows.append({
            "asset": asset,
            "kappa_1m_per_day": p1["kappa"],
            "kappa_5m_per_day": p5["kappa"],
            "lambda_1m_per_day": p1["lambda"],
            "lambda_5m_per_day": p5["lambda"],
            "log_kappa_ratio_1m_to_5m": np.log(p1["kappa"] / p5["kappa"]),
            "log_lambda_ratio_1m_to_5m": np.log(p1["lambda"] / p5["lambda"]),
            "mean_5m_own_minus_1m_transfer_log_score": difference["mean"],
            "day_cluster_se": difference["cluster_se"],
            "ci_lower": difference["ci_lower"],
            "ci_upper": difference["ci_upper"],
            "nobs": difference["nobs"],
            "kernel_test_interpretation": "transfer 1m dynamics with the 5m frozen margin; observed likelihood remains midpoint interval approximation",
            "formal_semigroup_gate": False,
        })

    write_csv(parameters, ROOT / "output" / "tables" / "frequency_parameter_stability.csv")
    write_csv(pd.DataFrame(comparison_rows), ROOT / "output" / "tables" / "one_to_five_minute_kernel_test.csv")
    write_json(model_payload, ROOT / "output" / "models" / "frequency_interval_models.json")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, parameter in zip(axes, ("kappa", "lambda")):
        pivot = parameters.pivot(index="asset", columns="frequency_minutes", values=parameter).loc[list(ASSETS)]
        x = np.arange(len(ASSETS))
        ax.bar(x - 0.18, pivot[1], width=0.36, label="1 minute")
        ax.bar(x + 0.18, pivot[5], width=0.36, label="5 minutes")
        ax.set_xticks(x, ASSETS)
        ax.set_ylabel(f"{parameter} (per day)")
        ax.set_title(f"Frequency stability of {parameter}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "output" / "figures" / "kappa_lambda_by_frequency.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "output" / "figures" / "kappa_lambda_by_frequency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
