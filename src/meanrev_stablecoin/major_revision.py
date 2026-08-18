from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .data_io import write_csv, write_json
from .models.latent_noise_ou import fit_latent_noise_ou
from .models.mixed_ou import fit_mou_ifm, fit_mou_peg, fit_mou_uqml, mixed_ou_logpdf_tau
from .models.ou import fit_ou, ou_logpdf
from .pairs import build_exact_horizon_pairs


def _systematic_cap(*arrays: np.ndarray, maximum: int) -> tuple[np.ndarray, ...]:
    if not arrays:
        return ()
    n = len(arrays[0])
    if any(len(value) != n for value in arrays):
        raise ValueError("all arrays must have equal length")
    if n <= maximum:
        return tuple(np.asarray(value) for value in arrays)
    index = np.linspace(0, n - 1, maximum, dtype=int)
    return tuple(np.asarray(value)[index] for value in arrays)


def _empirical_margin_row(values: np.ndarray, sample: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    row: dict[str, Any] = {
        "sample": sample,
        "model": "Data",
        "margin_source": "empirical",
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=0)),
        "skewness": float(skew(values, bias=False)),
        "kurtosis": float(kurtosis(values, fisher=False, bias=False)),
        "ks_distance": 0.0,
        "cvm_distance": 0.0,
    }
    for bp in (1, 10, 25, 50, 100):
        row[f"tail_abs_gt_{bp}bp"] = float(np.mean(np.abs(values) > bp / 10_000))
    return row


def _normal_margin_row(
    values: np.ndarray,
    sample: str,
    model: str,
    theta: float,
    tau: float,
    source: str,
) -> dict[str, Any]:
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    fitted_cdf = norm.cdf(values, loc=theta, scale=tau)
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    ks = float(max(np.max(np.abs(upper - fitted_cdf)), np.max(np.abs(fitted_cdf - lower))))
    mid = (np.arange(1, n + 1) - 0.5) / n
    cvm = float(np.mean((fitted_cdf - mid) ** 2) + 1 / (12 * n * n))
    row: dict[str, Any] = {
        "sample": sample,
        "model": model,
        "margin_source": source,
        "mean": theta,
        "sd": tau,
        "skewness": 0.0,
        "kurtosis": 3.0,
        "ks_distance": ks,
        "cvm_distance": cvm,
    }
    for bp in (1, 10, 25, 50, 100):
        threshold = bp / 10_000
        row[f"tail_abs_gt_{bp}bp"] = float(
            norm.cdf(-threshold, loc=theta, scale=tau)
            + norm.sf(threshold, loc=theta, scale=tau)
        )
    return row


def invariant_margin_audit(
    samples: dict[str, np.ndarray],
    margins: dict[str, tuple[float, float, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample, values in samples.items():
        empirical = _empirical_margin_row(values, sample)
        rows.append(empirical)
        for model, (theta, tau, source) in margins.items():
            row = _normal_margin_row(values, sample, model, theta, tau, source)
            mean_error_sd = abs(theta - empirical["mean"]) / max(empirical["sd"], 1e-12)
            sd_ratio = tau / max(empirical["sd"], 1e-12)
            empirical_tail = empirical["tail_abs_gt_50bp"]
            model_tail = row["tail_abs_gt_50bp"]
            tail_ratio = model_tail / max(empirical_tail, 1 / len(values))
            tail_gate_applicable = empirical_tail * len(values) >= 20
            tail_gate = (0.20 <= tail_ratio <= 5.0) if tail_gate_applicable else True
            row.update({
                "mean_error_in_empirical_sd": mean_error_sd,
                "sd_ratio_to_empirical": sd_ratio,
                "tail_50bp_ratio_to_empirical": tail_ratio,
                "tail_50bp_gate_applicable": tail_gate_applicable,
                "stationary_margin_gate": bool(
                    mean_error_sd <= 0.25
                    and 0.75 <= sd_ratio <= 1.25
                    and row["ks_distance"] <= 0.25
                    and tail_gate
                ),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_period(part: pd.DataFrame, maximum: int, multistart: int) -> dict[str, Any]:
    left, right = build_exact_horizon_pairs(part, 300)
    x = part["log_price"].to_numpy(dtype=float)
    xp, xn = _systematic_cap(x[left], x[right], maximum=maximum)
    delta = np.full(len(xp), 1 / 288)
    ou = fit_ou(xn, xp, delta, model_name="OUF", multistart=min(multistart, 6))
    peg = fit_mou_peg(xn, xp, delta, x, multistart=multistart)
    ifm = fit_mou_ifm(xn, xp, delta, x, multistart=multistart)
    uqml = fit_mou_uqml(xn, xp, delta, multistart=multistart)
    return {"x": x, "xp": xp, "xn": xn, "delta": delta, "fits": [ou, peg, ifm, uqml], "pairs_available": len(left)}


def _score_fit(fit: Any, xn: np.ndarray, xp: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if fit.model_name == "OUF":
        p = fit.params
        return ou_logpdf(xn, xp, delta, p["theta"], p["kappa"], p["sigma"])
    p = fit.params
    return mixed_ou_logpdf_tau(xn, xp, delta, p["theta"], p["tau"], p["kappa"], p["lambda"])


def _invariant_logpdf(fit: Any, values: np.ndarray) -> np.ndarray:
    p = fit.params
    tau = p["sigma"] / np.sqrt(2 * p["kappa"]) if fit.model_name == "OUF" else p["tau"]
    return norm.logpdf(values, loc=p["theta"], scale=tau)


def _identification_slices(
    fit: Any,
    xn: np.ndarray,
    xp: np.ndarray,
    delta: np.ndarray,
    grid_points: int,
) -> pd.DataFrame:
    xn, xp, delta = _systematic_cap(xn, xp, delta, maximum=40_000)
    p = fit.params
    log_grid = np.linspace(-1.5, 1.5, grid_points)
    theta_grid = p["theta"] + np.linspace(-2, 2, grid_points) * p["tau"]
    rows: list[dict[str, Any]] = []

    def add_slice(name: str, first_name: str, first_values: np.ndarray, second_name: str, second_values: np.ndarray) -> None:
        for first in first_values:
            for second in second_values:
                values = dict(theta=p["theta"], tau=p["tau"], kappa=p["kappa"], lam=p["lambda"])
                values[first_name] = float(first)
                values[second_name] = float(second)
                ll = mixed_ou_logpdf_tau(
                    xn, xp, delta, values["theta"], values["tau"], values["kappa"], values["lam"]
                )
                rows.append({
                    "slice": name,
                    "first_parameter": first_name,
                    "first_value": first,
                    "second_parameter": "lambda" if second_name == "lam" else second_name,
                    "second_value": second,
                    "mean_loglik": float(np.mean(ll)),
                    "nobs": len(xn),
                    "nuisance_parameters": "held_at_MOU-UQML_estimate",
                })

    add_slice("kappa_lambda", "kappa", p["kappa"] * np.exp(log_grid), "lam", p["lambda"] * np.exp(log_grid))
    add_slice("tau_lambda", "tau", p["tau"] * np.exp(log_grid), "lam", p["lambda"] * np.exp(log_grid))
    add_slice("theta_tau", "theta", theta_grid, "tau", p["tau"] * np.exp(log_grid))
    result = pd.DataFrame(rows)
    result["relative_mean_loglik"] = result["mean_loglik"] - result.groupby("slice")["mean_loglik"].transform("max")
    return result


def _plot_margin_audit(audit: pd.DataFrame, target: Path) -> None:
    baseline = audit[audit["sample"] == "baseline_full"].copy()
    models = baseline["model"].tolist()
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(x, baseline["sd"].to_numpy() * 10_000, color=["0.35" if m == "Data" else "#4472C4" for m in models])
    axes[0].set_xticks(x, models, rotation=35, ha="right")
    axes[0].set_ylabel("Standard deviation (bp, log scale approximation)")
    axes[0].set_title("Invariant-scale audit")
    axes[1].bar(x, baseline["tail_abs_gt_50bp"].to_numpy() * 100, color=["0.35" if m == "Data" else "#C55A11" for m in models])
    axes[1].set_xticks(x, models, rotation=35, ha="right")
    axes[1].set_ylabel("Probability outside ±50 bp (%)")
    axes[1].set_title("Invariant tail audit")
    fig.tight_layout()
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_identification_slices(slices: pd.DataFrame, target: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (name, group) in zip(axes, slices.groupby("slice", sort=False)):
        table = group.pivot(index="second_value", columns="first_value", values="relative_mean_loglik")
        image = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=-0.05, vmax=0)
        ax.set_title(name.replace("_", "–"))
        ax.set_xlabel(group["first_parameter"].iloc[0])
        ax.set_ylabel(group["second_parameter"].iloc[0])
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.75, label="Relative mean log likelihood")
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.15, top=0.88, wspace=0.35)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_major_revision_empirics(
    df: pd.DataFrame,
    config: dict[str, Any],
    tables: Path,
    models: Path,
    figures: Path,
) -> dict[str, Any]:
    settings = config["major_revision"]
    baseline = df[(df["timestamp"] >= pd.Timestamp(config["data"]["baseline_start"]).timestamp()) &
                  (df["timestamp"] <= pd.Timestamp(config["data"]["baseline_end"]).timestamp())].copy()
    # Reuse the precomputed non-overlapping period flags from preprocessing.
    periods = {
        "training_2021_2023": baseline[baseline["sample_train"]].reset_index(drop=True),
        "validation_2024": baseline[baseline["sample_validation"]].reset_index(drop=True),
        "test_2025": baseline[baseline["sample_test"]].reset_index(drop=True),
    }
    full = _fit_period(baseline.reset_index(drop=True), int(settings["estimation_pairs"]), int(settings["multistart"]))
    training = _fit_period(periods["training_2021_2023"], int(settings["estimation_pairs"]), int(settings["multistart"]))

    fit_rows: list[dict[str, Any]] = []
    for fit_sample, result in (("baseline_full", full), ("training_2021_2023", training)):
        for fit in result["fits"]:
            fit_rows.append({
                "fit_sample": fit_sample, "model": fit.model_name, **fit.params,
                "loglik": fit.loglik, "nobs": fit.nobs, "aic": fit.aic, "bic": fit.bic,
                "converged": fit.converged, "gradient_norm": fit.gradient_norm,
                "hessian_condition": fit.hessian_condition,
                "repeated_best_solutions": fit.metadata.get("repeated_best_solutions"),
                "weak_identification": fit.metadata.get("weak_identification"),
                "pairs_available": result["pairs_available"],
            })
    fits_table = pd.DataFrame(fit_rows)

    score_rows: list[dict[str, Any]] = []
    for period_name, part in periods.items():
        left, right = build_exact_horizon_pairs(part, 300)
        values = part["log_price"].to_numpy(dtype=float)
        xp, xn = _systematic_cap(values[left], values[right], maximum=int(settings["evaluation_pairs"]))
        delta = np.full(len(xp), 1 / 288)
        for fit in training["fits"]:
            scores = _score_fit(fit, xn, xp, delta)
            copula_scores = scores - _invariant_logpdf(fit, xn)
            score_rows.append({
                "evaluation_sample": period_name, "model": fit.model_name,
                "mean_log_score": float(np.mean(scores)), "total_log_score": float(np.sum(scores)),
                "mean_model_copula_log_score": float(np.mean(copula_scores)),
                "copula_score_warning": "model-specific PIT; compare only jointly with margin audit",
                "nobs": len(scores), "fit_sample": "training_2021_2023",
            })
    scores_table = pd.DataFrame(score_rows)

    full_fit_by_name = {fit.model_name: fit for fit in full["fits"]}
    ou = full_fit_by_name["OUF"]
    margins: dict[str, tuple[float, float, str]] = {
        "OUF": (ou.params["theta"], ou.params["sigma"] / np.sqrt(2 * ou.params["kappa"]), "joint_transition_MLE"),
    }
    for name in ("MOU-Peg", "MOU-IFM", "MOU-UQML"):
        fit = full_fit_by_name[name]
        margins[name] = (fit.params["theta"], fit.params["tau"], fit.metadata["parameterization"])

    latent_n = int(settings["latent_noise_observations"])
    ordered = baseline.sort_values("timestamp").reset_index(drop=True)
    index = np.linspace(0, len(ordered) - 1, min(latent_n, len(ordered)), dtype=int)
    latent_x = ordered["log_price"].to_numpy(dtype=float)[index]
    latent_time = ordered["timestamp"].to_numpy(dtype=float)[index]
    latent_delta = np.diff(latent_time) / 86_400
    latent_fit = fit_latent_noise_ou(latent_x, latent_delta, multistart=int(settings["latent_noise_multistart"]))
    margins["LatentNoiseOU-observed"] = (
        latent_fit.params["theta"], latent_fit.metadata["observed_stationary_sd"], "state_space_implied_observed_margin"
    )

    margin_samples = {"baseline_full": baseline["log_price"].to_numpy(dtype=float)}
    margin_samples.update({name: part["log_price"].to_numpy(dtype=float) for name, part in periods.items()})
    margin_audit = invariant_margin_audit(margin_samples, margins)
    slices = _identification_slices(
        full_fit_by_name["MOU-UQML"], full["xn"], full["xp"], full["delta"], int(settings["identification_grid_points"])
    )

    write_csv(fits_table, tables / "major_revision_mou_estimates.csv")
    write_csv(scores_table, tables / "major_revision_mou_oos_scores.csv")
    write_csv(margin_audit, tables / "invariant_margin_audit.csv")
    write_csv(slices, tables / "mixed_ou_identification_slices.csv")
    write_json({
        "fits": {"baseline_full": [fit.to_dict() for fit in full["fits"]],
                 "training_2021_2023": [fit.to_dict() for fit in training["fits"]]},
        "latent_noise_ou": latent_fit.to_dict(),
        "estimation_design": {
            "systematic_pair_cap": int(settings["estimation_pairs"]),
            "evaluation_pair_cap": int(settings["evaluation_pairs"]),
            "latent_noise_systematic_observations": len(latent_x),
        },
    }, models / "major_revision_core_models.json")
    _plot_margin_audit(margin_audit, figures / "invariant_margin_audit")
    _plot_identification_slices(slices, figures / "mixed_ou_identification_slices")
    return {
        "models": [fit.model_name for fit in full["fits"]] + [latent_fit.model_name],
        "full_margin_gates": margin_audit[margin_audit["sample"] == "baseline_full"]
            .set_index("model").get("stationary_margin_gate", pd.Series(dtype=bool)).dropna().to_dict(),
        "best_validation_model": scores_table[scores_table["evaluation_sample"] == "validation_2024"]
            .sort_values("mean_log_score", ascending=False).iloc[0]["model"],
    }
