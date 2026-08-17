from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .audit import audit_raw_data
from .constants import DAY_SECONDS, RANDOM_SEED, project_root
from .data_io import (
    ensure_output_dirs,
    load_config,
    load_kraken_ohlcvt,
    resolve_path,
    setup_logging,
    write_csv,
    write_json,
)
from .diagnostics import descriptive_statistics, pit_diagnostics, pit_liquidity_groups
from .nonparametric import estimate_nonparametric_drift
from .pairs import build_exact_horizon_pairs, build_irregular_adjacent_pairs
from .plotting import (
    plot_asymmetry,
    plot_coverage_and_gaps,
    plot_deviation_distribution,
    plot_drift,
    plot_price_series,
)
from .preprocess import aggregate_ohlcvt, prepare_data, sample_summary, select_sample, write_parquet_compat


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {key: resolve_path(value) for key, value in config["output"].items()}


def _raw_path(config: dict[str, Any]) -> Path:
    return resolve_path(config["data"]["raw_csv"])


def _load_prepared(config: dict[str, Any]) -> pd.DataFrame:
    path = project_root() / "data/processed/full.parquet"
    if not path.exists():
        run_prepare(config, force=True)
    return pd.read_parquet(path)


def _write_generic_and_sample(df: pd.DataFrame, generic: str, sample: str, tables: Path) -> None:
    write_csv(df, tables / f"{Path(generic).stem}_{sample}.csv")
    if sample == "baseline":
        write_csv(df, tables / generic)


def _fit_rows(fits, sample: str, category: str) -> pd.DataFrame:
    rows = []
    for fit in fits:
        for parameter, estimate in fit.params.items():
            rows.append(
                {
                    "sample": sample,
                    "category": category,
                    "model": fit.model_name,
                    "parameter": parameter,
                    "estimate": estimate,
                    "ordinary_se": np.nan,
                    "sandwich_se": np.nan,
                    "bootstrap_se": np.nan,
                    "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan,
                    "converged": fit.converged,
                    "gradient_norm": fit.gradient_norm,
                    "hessian_condition": fit.hessian_condition,
                    "weak_identification": fit.metadata.get("weak_identification"),
                    "half_life_minutes": fit.metadata.get("half_life_minutes", fit.metadata.get("base_half_life_minutes")),
                    "effective_half_life_minutes": fit.metadata.get("effective_half_life_minutes"),
                    "inference_note": "SE unavailable unless populated by model-specific inference",
                }
            )
    return pd.DataFrame(rows)


def _comparison_rows(fits, sample: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample": sample,
                "model": fit.model_name,
                "nobs": fit.nobs,
                "loglik": fit.loglik,
                "aic": fit.aic,
                "bic": fit.bic,
                "converged": fit.converged,
                "gradient_norm": fit.gradient_norm,
                "hessian_condition": fit.hessian_condition,
                "weak_identification": fit.metadata.get("weak_identification"),
                "likelihood_type": fit.metadata.get("likelihood_type", "exact_transition"),
                "acceptance_pass": bool(fit.converged and np.isfinite(fit.gradient_norm) and fit.gradient_norm < 1e-4
                                         and not fit.metadata.get("weak_identification", False)),
            }
            for fit in fits
        ]
    )


def run_audit(config: dict[str, Any], force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    ensure_output_dirs(config)
    logger = setup_logging("01_audit_data", config)
    if dry_run:
        logger.info("DRY RUN: audit %s", _raw_path(config))
        return {"dry_run": True}
    df = load_kraken_ohlcvt(_raw_path(config))
    audit, gaps, annual, anomalies, observed = audit_raw_data(df, _raw_path(config), config["data"]["expected_sha256"])
    tables = _paths(config)["tables"]
    write_csv(audit, tables / "data_audit.csv")
    write_csv(gaps, tables / "gap_summary.csv")
    write_csv(annual, tables / "annual_coverage.csv")
    write_csv(anomalies, tables / "anomaly_candidates.csv")
    write_json(observed, _paths(config)["logs"] / "data_audit.json")
    gap_seconds = np.diff(df["timestamp"].to_numpy(dtype=np.int64))
    plot_coverage_and_gaps(annual, gap_seconds[gap_seconds > 300] / 60, _paths(config)["figures"] / "coverage_and_gaps")
    logger.info("Audit pass=%s rows=%d sha256=%s", observed["audit_pass"], len(df), observed["sha256"])
    if not observed["audit_pass"]:
        raise RuntimeError("Raw-data audit failed; model estimation is blocked")
    return observed


def run_prepare(config: dict[str, Any], force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    logger = setup_logging("02_prepare_data", config)
    target = project_root() / "data/processed/full.parquet"
    if dry_run:
        logger.info("DRY RUN: prepare raw data to %s", target)
        return {"dry_run": True}
    if target.exists() and not force:
        logger.info("Using validated cached prepared data: %s", target)
        return {"cached": True, "path": str(target)}
    audit_status = run_audit(config)
    if not audit_status["audit_pass"]:
        raise RuntimeError("Audit failed")
    df = prepare_data(load_kraken_ohlcvt(_raw_path(config)))
    write_parquet_compat(df, target)
    for sample in ("baseline", "extended"):
        part = select_sample(df, sample)
        write_parquet_compat(part, project_root() / f"data/processed/{sample}.parquet")
    baseline = select_sample(df, "baseline")
    for minutes in (15, 30, 60):
        write_parquet_compat(aggregate_ohlcvt(baseline, minutes), project_root() / f"data/processed/baseline_{minutes}min.parquet")
    summary = sample_summary(df)
    write_csv(summary, _paths(config)["tables"] / "sample_summary.csv")
    manifest = {
        "raw_immutable": True,
        "raw_sha256": audit_status["sha256"],
        "rows_full": len(df),
        "columns": list(df.columns),
        "utc_timezone": True,
        "state_variable": "log(close)",
        "no_imputation": True,
        "hard_anomalies_flagged_not_replaced": int(df["flag_hard_anomaly"].sum()),
    }
    write_json(manifest, _paths(config)["logs"] / "preprocess_manifest.json")
    logger.info("Prepared %d rows and sample parquet files", len(df))
    return manifest


def run_descriptive(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    logger = setup_logging("03_descriptive_analysis", config)
    if dry_run:
        logger.info("DRY RUN descriptive sample=%s", sample)
        return {"dry_run": True}
    target = _paths(config)["tables"] / f"descriptive_statistics_{sample}.csv"
    if target.exists() and not force:
        logger.info("Using cached descriptive results: %s", target)
        return {"cached": True, "path": str(target)}
    df = select_sample(_load_prepared(config), sample)
    stats = descriptive_statistics(df, sample)
    annual = df.groupby("year").agg(observations=("close", "size"), mean_price=("close", "mean"),
                                    sd_log_price=("log_price", "std"), mean_volume=("volume", "mean"),
                                    mean_trades=("trades", "mean")).reset_index()
    _write_generic_and_sample(stats, "descriptive_statistics.csv", sample, _paths(config)["tables"])
    write_csv(annual, _paths(config)["tables"] / f"annual_statistics_{sample}.csv")
    if sample == "baseline":
        full = _load_prepared(config)
        plot_price_series(full, _paths(config)["figures"] / "full_price_and_deviation")
        plot_deviation_distribution(df["log_price"].to_numpy(), _paths(config)["figures"] / "deviation_distribution")
        plot_asymmetry(df, _paths(config)["figures"] / "empirical_asymmetry")
    logger.info("Descriptive statistics complete sample=%s n=%d", sample, len(df))
    return {"sample": sample, "nobs": len(df)}


def run_nonparametric(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    logger = setup_logging("04_nonparametric_drift", config)
    if dry_run:
        logger.info("DRY RUN nonparametric sample=%s", sample)
        return {"dry_run": True}
    target = _paths(config)["tables"] / f"sign_test_{sample}.csv"
    if target.exists() and not force:
        logger.info("Using cached nonparametric results: %s", target)
        return {"cached": True, "path": str(target)}
    df = select_sample(_load_prepared(config), sample)
    i, j = build_exact_horizon_pairs(df, 300)
    values = df["log_price"].to_numpy()
    reps = config["estimation"]["bootstrap_replications_final"] if sample == "baseline" else config["estimation"]["bootstrap_replications_mvp"]
    result = estimate_nonparametric_drift(
        values[i], values[j], np.full(len(i), 1 / 288), df["date"].to_numpy()[i],
        bootstrap_replications=reps,
        max_pairs=config["estimation"]["nonparametric_max_pairs"], seed=config["project"]["random_seed"],
    )
    _write_generic_and_sample(result.grid, "drift_sign_test.csv", sample, _paths(config)["tables"])
    write_csv(result.grid, _paths(config)["tables"] / f"drift_grid_{sample}.csv")
    write_csv(result.sign_summary, _paths(config)["tables"] / f"sign_test_{sample}.csv")
    write_json(result.metadata, _paths(config)["logs"] / f"nonparametric_{sample}.json")
    logger.info("Nonparametric drift complete sample=%s pairs=%d reps=%d", sample, len(i), reps)
    return result.metadata


def _ou_daily_bootstrap(x: np.ndarray, y: np.ndarray, days: np.ndarray, replications: int, seed: int) -> pd.DataFrame:
    unique, inverse = np.unique(days, return_inverse=True)
    stats = np.zeros((len(unique), 6), dtype=float)
    np.add.at(stats[:, 0], inverse, 1)
    np.add.at(stats[:, 1], inverse, x)
    np.add.at(stats[:, 2], inverse, y)
    np.add.at(stats[:, 3], inverse, x * x)
    np.add.at(stats[:, 4], inverse, x * y)
    np.add.at(stats[:, 5], inverse, y * y)
    rng = np.random.default_rng(seed)
    rows = []
    delta = 1 / 288
    for _ in range(replications):
        count = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        n, sx, sy, sx2, sxy, sy2 = count @ stats
        xbar, ybar = sx / n, sy / n
        denom = sx2 - sx * sx / n
        phi = (sxy - sx * sy / n) / denom if denom > 0 else np.nan
        if not np.isfinite(phi) or not (1e-8 < phi < 1 - 1e-10):
            rows.append({"theta": np.nan, "kappa": np.nan, "sigma": np.nan})
            continue
        intercept = ybar - phi * xbar
        theta = intercept / (1 - phi)
        kappa = -np.log(phi) / delta
        sse = sy2 + n * intercept**2 + phi**2 * sx2 - 2 * intercept * sy - 2 * phi * sxy + 2 * intercept * phi * sx
        innovation = max(sse / n, 1e-18)
        sigma = np.sqrt(innovation * 2 * kappa / (-np.expm1(-2 * kappa * delta)))
        rows.append({"theta": theta, "kappa": kappa, "sigma": sigma})
    return pd.DataFrame(rows)


def _ou_moving_day_block_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    days: np.ndarray,
    block_lengths: tuple[int, ...],
    replications: int,
    seed: int,
) -> pd.DataFrame:
    """Moving UTC-day block bootstrap using daily AR(1) sufficient statistics."""
    unique, inverse = np.unique(days, return_inverse=True)
    stats = np.zeros((len(unique), 6), dtype=float)
    np.add.at(stats[:, 0], inverse, 1)
    np.add.at(stats[:, 1], inverse, x)
    np.add.at(stats[:, 2], inverse, y)
    np.add.at(stats[:, 3], inverse, x * x)
    np.add.at(stats[:, 4], inverse, x * y)
    np.add.at(stats[:, 5], inverse, y * y)
    rng = np.random.default_rng(seed)
    delta = 1 / 288
    draws: list[dict[str, float | int]] = []
    n_days = len(unique)
    for block_length in block_lengths:
        if block_length < 1 or block_length > n_days:
            raise ValueError("block length must be between one and the number of UTC days")
        blocks_per_draw = int(np.ceil(n_days / block_length))
        last_start = n_days - block_length
        for replication in range(replications):
            starts = rng.integers(0, last_start + 1, size=blocks_per_draw)
            sampled = np.concatenate([np.arange(start, start + block_length) for start in starts])[:n_days]
            counts = np.bincount(sampled, minlength=n_days)
            n, sx, sy, sx2, sxy, sy2 = counts @ stats
            denom = sx2 - sx * sx / n
            phi = (sxy - sx * sy / n) / denom if denom > 0 else np.nan
            theta = kappa = sigma = np.nan
            if np.isfinite(phi) and 1e-8 < phi < 1 - 1e-10:
                xbar, ybar = sx / n, sy / n
                intercept = ybar - phi * xbar
                theta = intercept / (1 - phi)
                kappa = -np.log(phi) / delta
                sse = sy2 + n * intercept**2 + phi**2 * sx2 - 2 * intercept * sy - 2 * phi * sxy + 2 * intercept * phi * sx
                innovation = max(sse / n, 1e-18)
                sigma = np.sqrt(innovation * 2 * kappa / (-np.expm1(-2 * kappa * delta)))
            draws.append({"block_length_days": block_length, "replication": replication,
                          "theta": theta, "kappa": kappa, "sigma": sigma})
    raw = pd.DataFrame(draws)
    rows = []
    for block_length, group in raw.groupby("block_length_days"):
        for parameter in ("theta", "kappa", "sigma"):
            values = group[parameter].dropna().to_numpy()
            rows.append({"block_length_days": int(block_length), "parameter": parameter,
                         "replications_requested": replications, "replications_valid": len(values),
                         "bootstrap_mean": np.mean(values), "bootstrap_se": np.std(values, ddof=1),
                         "ci_lower": np.quantile(values, 0.025), "ci_upper": np.quantile(values, 0.975)})
    return pd.DataFrame(rows)


def _populate_inference(
    table: pd.DataFrame,
    model: str,
    params: dict[str, float],
    ordinary_beta: np.ndarray | None,
    robust_beta: np.ndarray | None,
    beta_param_names: list[str],
    log_params: set[str],
    bootstrap: pd.DataFrame | None = None,
) -> None:
    if ordinary_beta is not None:
        jac = np.array([params[name] if name in log_params else 1.0 for name in beta_param_names])
        ordinary = np.diag(jac) @ ordinary_beta @ np.diag(jac)
        robust = None if robust_beta is None else np.diag(jac) @ robust_beta @ np.diag(jac)
        for idx, name in enumerate(beta_param_names):
            mask = (table["model"] == model) & (table["parameter"] == name)
            table.loc[mask, "ordinary_se"] = np.sqrt(max(ordinary[idx, idx], 0))
            if robust is not None:
                table.loc[mask, "sandwich_se"] = np.sqrt(max(robust[idx, idx], 0))
    if bootstrap is not None:
        for name in bootstrap.columns:
            if name not in params:
                continue
            values = bootstrap[name].dropna().to_numpy()
            mask = (table["model"] == model) & (table["parameter"] == name)
            table.loc[mask, "bootstrap_se"] = np.std(values, ddof=1)
            table.loc[mask, "bootstrap_ci_lower"] = np.quantile(values, 0.025)
            table.loc[mask, "bootstrap_ci_upper"] = np.quantile(values, 0.975)
    mask = table["model"] == model
    table.loc[mask, "inference_note"] = "Observed Hessian; UTC-day cluster sandwich; UTC-day block bootstrap"


def run_ou_models(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .estimation.inference import cluster_sandwich_covariance
    from .models.ou import fit_ou, ou_cdf, ou_logpdf
    from .models.threshold_ou import fit_threshold_ou

    logger = setup_logging("05_fit_ou_models", config)
    if dry_run:
        logger.info("DRY RUN OU sample=%s", sample)
        return {"dry_run": True}
    target = _paths(config)["models"] / f"ou_fits_{sample}.json"
    if target.exists() and not force:
        logger.info("Using cached OU results: %s", target)
        return {"cached": True, "path": str(target)}
    df = select_sample(_load_prepared(config), sample)
    i, j = build_exact_horizon_pairs(df, 300)
    values = df["log_price"].to_numpy(); xp, xn = values[i], values[j]
    delta = np.full(len(i), 1 / 288)
    starts = int(config["estimation"]["multistart"])
    ou0 = fit_ou(xn, xp, delta, fixed_theta=0.0, multistart=starts)
    ouf = fit_ou(xn, xp, delta, fixed_theta=None, multistart=starts)
    threshold = fit_threshold_ou(xn, xp, delta, theta=0.0, heterogeneous_sigma=False)
    threshold_het = fit_threshold_ou(xn, xp, delta, theta=0.0, heterogeneous_sigma=True)
    ixp, ixn, idelta = build_irregular_adjacent_pairs(df, maximum_gap_class=2)
    irregular = fit_ou(ixn, ixp, idelta, fixed_theta=None, model_name="OUF_irregular_G0_G2", multistart=6)
    fits = [ou0, ouf, threshold, threshold_het, irregular]

    day_codes = df["date"].to_numpy()[i]
    beta_f = np.array([ouf.params["theta"], np.log(ouf.params["kappa"]), np.log(ouf.params["sigma"])])
    logpdf_f = lambda b: ou_logpdf(xn, xp, delta, b[0], np.exp(b[1]), np.exp(b[2]))
    ordinary_f, robust_f, condition_f = cluster_sandwich_covariance(logpdf_f, beta_f, day_codes)
    beta_0 = np.log([ou0.params["kappa"], ou0.params["sigma"]])
    logpdf_0 = lambda b: ou_logpdf(xn, xp, delta, 0.0, np.exp(b[0]), np.exp(b[1]))
    ordinary_0, robust_0, condition_0 = cluster_sandwich_covariance(logpdf_0, beta_0, day_codes)
    bootstrap_reps = config["estimation"]["bootstrap_replications_final"] if sample == "baseline" else config["estimation"]["bootstrap_replications_mvp"]
    bootstrap = _ou_daily_bootstrap(xp, xn, day_codes, bootstrap_reps, config["project"]["random_seed"])
    estimates = _fit_rows(fits, sample, "OU")
    _populate_inference(estimates, "OUF", ouf.params, ordinary_f, robust_f, ["theta", "kappa", "sigma"], {"kappa", "sigma"}, bootstrap)
    bootstrap0 = bootstrap.assign(theta=0.0)
    _populate_inference(estimates, "OU0", ou0.params, ordinary_0, robust_0, ["kappa", "sigma"], {"kappa", "sigma"}, bootstrap0)
    comparison = _comparison_rows(fits, sample)

    pit0 = np.clip(ou_cdf(xn, xp, delta, **ou0.params), 1e-10, 1 - 1e-10)
    pitf = np.clip(ou_cdf(xn, xp, delta, **ouf.params), 1e-10, 1 - 1e-10)
    diagnostics = pd.DataFrame([pit_diagnostics(pit0, "OU0"), pit_diagnostics(pitf, "OUF")])
    liquidity = pd.concat([
        pit_liquidity_groups(pit0, df.iloc[j].reset_index(drop=True), "OU0"),
        pit_liquidity_groups(pitf, df.iloc[j].reset_index(drop=True), "OUF"),
    ], ignore_index=True)
    comparison = comparison.merge(diagnostics, on="model", how="left")
    tables, models = _paths(config)["tables"], _paths(config)["models"]
    write_csv(estimates, tables / f"model_estimates_ou_{sample}.csv")
    write_csv(comparison, tables / f"model_comparison_ou_{sample}.csv")
    write_csv(liquidity, tables / f"pit_liquidity_ou_{sample}.csv")
    write_csv(bootstrap, tables / f"ou_daily_bootstrap_{sample}.csv")
    write_json([fit.to_dict() for fit in fits], models / f"ou_fits_{sample}.json")
    np.savez_compressed(models / f"ou_pit_{sample}.npz", OU0=pit0, OUF=pitf)
    if sample == "baseline":
        drift_path = tables / "drift_sign_test.csv"
        if drift_path.exists():
            plot_drift(pd.read_csv(drift_path), ouf.params["theta"], ouf.params["kappa"], _paths(config)["figures"] / "nonparametric_vs_ou_drift")
    logger.info("OU models complete sample=%s exact_pairs=%d HessianCond(OUF)=%.4g", sample, len(i), condition_f)
    return {"sample": sample, "exact_pairs": len(i), "models": [fit.model_name for fit in fits]}


def run_cir_models(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .models.cir import cir_cdf, fit_cir

    logger = setup_logging("06_fit_cir_models", config)
    if dry_run:
        logger.info("DRY RUN CIR sample=%s", sample)
        return {"dry_run": True}
    target = _paths(config)["models"] / f"cir_fits_{sample}.json"
    if target.exists() and not force:
        logger.info("Using cached CIR results: %s", target)
        return {"cached": True, "path": str(target)}
    df = select_sample(_load_prepared(config), sample)
    i, j = build_exact_horizon_pairs(df, 300)
    delta = np.full(len(i), 1 / 288)
    price = df["close"].to_numpy(); p_prev, p_next = price[i], price[j]
    price_fit = fit_cir(p_next, p_prev, delta, "CIR_Price")
    pressure = np.abs(df["deviation"].to_numpy()) + 1e-8
    pressure_fit = fit_cir(pressure[j], pressure[i], delta, "CIR_DepegPressure", max_nobs=100_000)
    fits = [price_fit, pressure_fit]
    idx = np.linspace(0, len(i) - 1, min(len(i), 150_000), dtype=int)
    pit_price = np.clip(cir_cdf(p_next[idx], p_prev[idx], delta[idx], **price_fit.params), 1e-10, 1 - 1e-10)
    pit_pressure = np.clip(cir_cdf(pressure[j[idx]], pressure[i[idx]], delta[idx], **pressure_fit.params), 1e-10, 1 - 1e-10)
    diagnostics = pd.DataFrame([pit_diagnostics(pit_price, "CIR_Price"), pit_diagnostics(pit_pressure, "CIR_DepegPressure")])
    comparison = _comparison_rows(fits, sample).merge(diagnostics, on="model", how="left")
    estimates = _fit_rows(fits, sample, "CIR")
    estimates["inference_note"] = "Observed optimizer Hessian only; weak-identification flag governs interpretation"
    tables, models = _paths(config)["tables"], _paths(config)["models"]
    write_csv(estimates, tables / f"model_estimates_cir_{sample}.csv")
    write_csv(comparison, tables / f"model_comparison_cir_{sample}.csv")
    write_json([fit.to_dict() for fit in fits], models / f"cir_fits_{sample}.json")
    np.savez_compressed(models / f"cir_pit_{sample}.npz", CIR_Price=pit_price, CIR_DepegPressure=pit_pressure)
    logger.info("CIR models complete sample=%s pairs=%d", sample, len(i))
    return {"sample": sample, "models": [fit.model_name for fit in fits]}


def _fit_from_json(path: Path, model: str):
    from .estimation.optimize import FitResult

    payloads = json.loads(path.read_text(encoding="utf-8"))
    payload = next(item for item in payloads if item["model_name"] == model)
    for key in ("covariance", "robust_covariance"):
        if payload.get(key) is not None:
            payload[key] = np.asarray(payload[key])
    return FitResult(**payload)


def run_transformed_models(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .models.transformed_cir import fit_transformed_cir_profile
    from .models.transformed_ou import fit_transformed_ou

    logger = setup_logging("07_fit_transformed_models", config)
    if dry_run:
        logger.info("DRY RUN transformed models sample=%s", sample)
        return {"dry_run": True}
    target = _paths(config)["models"] / f"transformed_fits_{sample}.json"
    if target.exists() and not force:
        logger.info("Using cached transformed-model results: %s", target)
        return {"cached": True, "path": str(target)}
    models_path = _paths(config)["models"]
    if not (models_path / f"cir_fits_{sample}.json").exists():
        run_cir_models(config, sample)
    df = select_sample(_load_prepared(config), sample)
    i, j = build_exact_horizon_pairs(df, 300)
    idx = np.linspace(0, len(i) - 1, min(len(i), 120_000), dtype=int)
    i_fit, j_fit = i[idx], j[idx]
    time_days = (df["timestamp"].to_numpy() - df["timestamp"].iloc[0]) / DAY_SECONDS
    delta = np.full(len(idx), 1 / 288)
    x = df["log_price"].to_numpy(); price = df["close"].to_numpy()
    tou = fit_transformed_ou(x[j_fit], x[i_fit], time_days[j_fit], time_days[i_fit], delta)
    base_cir = _fit_from_json(models_path / f"cir_fits_{sample}.json", "CIR_Price")
    tcir, profile = fit_transformed_cir_profile(price[j_fit], price[i_fit], time_days[j_fit], time_days[i_fit], delta, base_cir)
    fits = [tou, tcir]
    estimates = _fit_rows(fits, sample, "Transformed")
    comparison = _comparison_rows(fits, sample)
    tables = _paths(config)["tables"]
    write_csv(estimates, tables / f"model_estimates_transformed_{sample}.csv")
    write_csv(comparison, tables / f"model_comparison_transformed_{sample}.csv")
    write_csv(pd.DataFrame(profile, columns=["gamma_kappa_minus_d", "profile_loglik"]), tables / f"transformed_cir_profile_{sample}.csv")
    write_json([fit.to_dict() for fit in fits], models_path / f"transformed_fits_{sample}.json")
    support_report = pd.DataFrame([{"sample": sample, "model": "AffineTransformedCIR", "support_errors": 0,
                                    "A0_fixed": True, "B0_fixed": True, "profile_boundary_hit": abs(tcir.params["gamma_kappa_minus_d"]) >= 0.001}])
    write_csv(support_report, tables / f"transformed_support_report_{sample}.csv")
    logger.info("Transformed models complete sample=%s estimation_pairs=%d", sample, len(idx))
    return {"sample": sample, "models": [fit.model_name for fit in fits]}


def run_mixed_models(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .estimation.bootstrap import parametric_mou_lr_bootstrap
    from .estimation.inference import cluster_sandwich_covariance
    from .models.mixed_cir import fit_mixed_cir_lambda
    from .models.mixed_ou import fit_mixed_ou, mixed_ou_cdf, mixed_ou_logpdf
    from .models.ou import ou_moments
    from .plotting import plot_pit_diagnostics

    logger = setup_logging("08_fit_mixed_copula_models", config)
    if dry_run:
        logger.info("DRY RUN mixed models sample=%s", sample)
        return {"dry_run": True}
    target = _paths(config)["models"] / f"mixed_fits_{sample}.json"
    if target.exists() and not force:
        logger.info("Using cached mixed-model results: %s", target)
        return {"cached": True, "path": str(target)}
    models_path = _paths(config)["models"]
    if not (models_path / f"ou_fits_{sample}.json").exists():
        run_ou_models(config, sample)
    if not (models_path / f"cir_fits_{sample}.json").exists():
        run_cir_models(config, sample)
    df = select_sample(_load_prepared(config), sample)
    i, j = build_exact_horizon_pairs(df, 300)
    x = df["log_price"].to_numpy(); xp, xn = x[i], x[j]
    delta = np.full(len(i), 1 / 288)
    ou0 = _fit_from_json(models_path / f"ou_fits_{sample}.json", "OU0")
    ouf = _fit_from_json(models_path / f"ou_fits_{sample}.json", "OUF")
    starts = int(config["estimation"]["multistart"] if sample == "baseline" else 6)
    mou0 = fit_mixed_ou(xn, xp, delta, fixed_theta=0.0, base_fit=ou0, multistart=starts)
    mouf = fit_mixed_ou(xn, xp, delta, fixed_theta=None, base_fit=ouf, multistart=starts)
    cir = _fit_from_json(models_path / f"cir_fits_{sample}.json", "CIR_Price")
    price = df["close"].to_numpy()
    mcir = fit_mixed_cir_lambda(price[j], price[i], delta, cir)
    fits = [mou0, mouf, mcir]

    estimates = _fit_rows(fits, sample, "Mixed")
    beta = np.array([mouf.params["theta"], np.log(mouf.params["kappa"]), np.log(mouf.params["sigma"]), np.log(mouf.params["lambda"])])
    logpdf = lambda b: mixed_ou_logpdf(xn, xp, delta, b[0], np.exp(b[1]), np.exp(b[2]), np.exp(b[3]))
    ordinary, robust, condition = cluster_sandwich_covariance(logpdf, beta, df["date"].to_numpy()[i])
    _populate_inference(estimates, "MOUF", mouf.params, ordinary, robust,
                        ["theta", "kappa", "sigma", "lambda"], {"kappa", "sigma", "lambda"})
    estimates.loc[estimates["model"] == "MOUF", "inference_note"] += "; lambda boundary significance uses parametric LR bootstrap"

    comparison = _comparison_rows(fits, sample)
    # Function argument is named lam while stored parameter is lambda.
    pit = np.clip(mixed_ou_cdf(xn, xp, delta, mouf.params["theta"], mouf.params["kappa"], mouf.params["sigma"], mouf.params["lambda"]), 1e-10, 1 - 1e-10)
    pit_diag = pd.DataFrame([pit_diagnostics(pit, "MOUF")])
    comparison = comparison.merge(pit_diag, on="model", how="left")

    lr_summary, lr_values = parametric_mou_lr_bootstrap(
        xp, xn,
        replications=int(config["estimation"]["lr_bootstrap_replications"]),
        bootstrap_nobs=int(config["estimation"]["lr_bootstrap_nobs"]),
        seed=int(config["project"]["random_seed"]),
    )
    lr_table = pd.DataFrame([lr_summary])

    rng = np.random.default_rng(config["project"]["random_seed"])
    state_grid = mouf.params["theta"] + np.array([-2, -1, 0, 1, 2]) * mouf.params["sigma"] / np.sqrt(2 * mouf.params["kappa"])
    validation_rows = []
    for state in state_grid:
        n_sim = 100_000
        a = np.exp(-mouf.params["lambda"] / 288)
        reset = rng.random(n_sim) > a
        base_mean, base_var = ou_moments(np.array([state]), np.array([1 / 288]), mouf.params["theta"], mouf.params["kappa"], mouf.params["sigma"])
        draws = base_mean[0] + np.sqrt(base_var[0]) * rng.standard_normal(n_sim)
        stationary_sd = mouf.params["sigma"] / np.sqrt(2 * mouf.params["kappa"])
        draws[reset] = mouf.params["theta"] + stationary_sd * rng.standard_normal(reset.sum())
        theoretical_mean = mouf.params["theta"] + np.exp(-(mouf.params["kappa"] + mouf.params["lambda"]) / 288) * (state - mouf.params["theta"])
        component_mean = base_mean[0]
        mixture_mean = theoretical_mean
        theoretical_var = a * (base_var[0] + (component_mean - mixture_mean) ** 2) + (1 - a) * (stationary_sd**2 + (mouf.params["theta"] - mixture_mean) ** 2)
        validation_rows.append({"state": state, "simulated_mean": draws.mean(), "theoretical_mean": theoretical_mean,
                                "mean_error": draws.mean() - theoretical_mean, "simulated_variance": draws.var(),
                                "theoretical_variance": theoretical_var, "variance_error": draws.var() - theoretical_var})
    validation = pd.DataFrame(validation_rows)

    tables = _paths(config)["tables"]
    write_csv(estimates, tables / f"model_estimates_mixed_{sample}.csv")
    write_csv(comparison, tables / f"model_comparison_mixed_{sample}.csv")
    _write_generic_and_sample(lr_table, "mixed_lr_bootstrap.csv", sample, tables)
    write_csv(validation, tables / f"mou_generator_validation_{sample}.csv")
    write_json([fit.to_dict() for fit in fits], models_path / f"mixed_fits_{sample}.json")
    np.savez_compressed(models_path / f"mixed_pit_{sample}.npz", MOUF=pit, lr_bootstrap=lr_values)
    if sample == "baseline":
        ou_pit = np.load(models_path / "ou_pit_baseline.npz")
        subsample = slice(None, None, max(len(pit) // 150_000, 1))
        plot_pit_diagnostics({"OUF": ou_pit["OUF"][subsample], "MOUF": pit[subsample]}, _paths(config)["figures"] / "pit_and_residual_acf")
    logger.info("Mixed models complete sample=%s pairs=%d MOUF lambda=%.6g sandwich_condition=%.4g", sample, len(i), mouf.params["lambda"], condition)
    return {"sample": sample, "models": [fit.model_name for fit in fits], "lr": lr_summary}


def run_semiparametric_copula(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .models.copula import (
        EmpiricalMarginal,
        copula_conditional_mean,
        fit_multihorizon_composite_likelihood,
        rank_pseudo_observations,
    )
    from .plotting import plot_copula_contours, plot_horizon_dependence

    logger = setup_logging("09_fit_semiparametric_copula", config)
    if dry_run:
        logger.info("DRY RUN semiparametric copula sample=%s", sample)
        return {"dry_run": True}
    target = _paths(config)["models"] / f"semiparametric_copula_{sample}.json"
    if target.exists() and not force:
        logger.info("Using cached semiparametric-copula results: %s", target)
        return {"cached": True, "path": str(target)}
    df = select_sample(_load_prepared(config), sample)
    x = df["log_price"].to_numpy()
    u_all = rank_pseudo_observations(x)
    horizon_pairs = {}
    pair_rows = []
    for minutes in config["estimation"]["horizons_minutes"]:
        i, j = build_exact_horizon_pairs(df, int(minutes * 60))
        horizon_pairs[int(minutes)] = (u_all[i], u_all[j])
        pair_rows.append({"sample": sample, "horizon_minutes": minutes, "exact_pairs": len(i)})
    fit = fit_multihorizon_composite_likelihood(
        horizon_pairs, random_seed=config["project"]["random_seed"],
        max_pairs_per_horizon=config["estimation"]["copula_max_pairs_per_horizon"],
    )
    horizon_table = pd.DataFrame(fit.pop("horizon_rows"))
    horizon_table.insert(0, "sample", sample)
    marginal = EmpiricalMarginal.fit(x)
    grid = np.quantile(x, np.arange(1, 100) / 100)
    means = {}
    for nodes in (64, 128, 256):
        means[nodes] = copula_conditional_mean(grid, marginal, 5 / 1440, fit["kappa"], fit["lambda"], nodes)
    drift = pd.DataFrame({"sample": sample, "x_grid": grid, "conditional_mean_128": means[128],
                          "drift_per_day_128": (means[128] - grid) / (5 / 1440),
                          "mean_abs_difference_64_128": np.abs(means[64] - means[128]),
                          "mean_abs_difference_256_128": np.abs(means[256] - means[128]),
                          "sign_match_anchor": (-grid * (means[128] - grid)) > 0})
    train = df[df["sample_train"]]
    test = df[df["sample_test"]]
    training_marginal = EmpiricalMarginal.fit(train["log_price"].to_numpy())
    test_u = training_marginal.cdf(test["log_price"].to_numpy())
    mapping_audit = pd.DataFrame([{"sample": sample, "training_rows": len(train), "test_rows": len(test),
                                   "test_mapping_source": "training_2021_2023_empirical_CDF",
                                   "test_u_min": test_u.min(), "test_u_max": test_u.max(),
                                   "test_reranked": False}])
    tables = _paths(config)["tables"]
    _write_generic_and_sample(horizon_table, "copula_horizon_fit.csv", sample, tables)
    write_csv(pd.DataFrame(pair_rows), tables / f"copula_pair_counts_{sample}.csv")
    write_csv(drift, tables / f"copula_conditional_drift_{sample}.csv")
    write_csv(mapping_audit, tables / f"copula_no_lookahead_audit_{sample}.csv")
    write_json(fit, _paths(config)["models"] / f"semiparametric_copula_{sample}.json")
    if sample == "baseline":
        plot_horizon_dependence(horizon_table, _paths(config)["figures"] / "multihorizon_dependence")
        plot_copula_contours(fit["kappa"], fit["lambda"], 60, _paths(config)["figures"] / "gaussian_vs_mixed_copula")
    logger.info("Semiparametric copula complete sample=%s kappa=%.6g lambda=%.6g", sample, fit["kappa"], fit["lambda"])
    return fit


def run_article_core_extensions(config: dict[str, Any], force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .estimation.bootstrap import matched_joint_mou_lr_bootstrap
    from .models.copula import (
        EmpiricalMarginal,
        evaluate_copula_models,
        fit_copula_model_comparison,
        rank_pseudo_observations,
    )
    from .plotting import plot_extended_horizon_dependence

    logger = setup_logging("13_article_core_extensions", config)
    tables, models, figures = _paths(config)["tables"], _paths(config)["models"], _paths(config)["figures"]
    bootstrap_target = tables / "matched_joint_mou_lr_bootstrap.csv"
    copula_target = tables / "copula_scale_model_comparison.csv"
    if dry_run:
        logger.info("DRY RUN article core extensions")
        return {"dry_run": True}
    if bootstrap_target.exists() and copula_target.exists() and not force:
        logger.info("Using cached article core extensions")
        return {"cached": True, "bootstrap": str(bootstrap_target), "copula": str(copula_target)}

    df = select_sample(_load_prepared(config), "baseline")
    i, j = build_exact_horizon_pairs(df, 300)
    x = df["log_price"].to_numpy(); xp, xn = x[i], x[j]
    extension = config["article_extension"]
    summary, draws, observed = matched_joint_mou_lr_bootstrap(
        xp, xn,
        replications=int(extension["matched_lr_bootstrap_replications"]),
        matched_pairs=int(extension["matched_lr_pairs"]),
        multistart=int(extension["matched_lr_multistart"]),
        parallel_jobs=int(extension["matched_lr_parallel_jobs"]),
        seed=int(config["project"]["random_seed"]),
    )
    summary_table = pd.DataFrame([{**summary, **{f"observed_{key}": value for key, value in observed.items()}}])
    write_csv(summary_table, bootstrap_target)
    write_csv(pd.DataFrame(list(draws)), tables / "matched_joint_mou_lr_bootstrap_draws.csv")
    write_json({"summary": summary, "observed_fit": observed}, models / "matched_joint_mou_lr_bootstrap.json")

    horizons = [int(value) for value in config["estimation"]["horizons_minutes"]]

    def horizon_pairs(part: pd.DataFrame, pseudo: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        result = {}
        for minutes in horizons:
            left, right = build_exact_horizon_pairs(part, minutes * 60)
            result[minutes] = (pseudo[left], pseudo[right])
        return result

    full_u = rank_pseudo_observations(x)
    full_pairs = horizon_pairs(df, full_u)
    full_models, horizon_rows = fit_copula_model_comparison(
        full_pairs,
        random_seed=int(config["project"]["random_seed"]),
        max_pairs_per_horizon=int(extension["two_scale_copula_max_pairs_per_horizon"]),
    )

    train = df[df["sample_train"]].reset_index(drop=True)
    validation = df[df["sample_validation"]].reset_index(drop=True)
    test = df[df["sample_test"]].reset_index(drop=True)
    train_x = train["log_price"].to_numpy()
    train_marginal = EmpiricalMarginal.fit(train_x)
    train_pairs = horizon_pairs(train, rank_pseudo_observations(train_x))
    training_models, _ = fit_copula_model_comparison(
        train_pairs,
        random_seed=int(config["project"]["random_seed"]),
        max_pairs_per_horizon=int(extension["two_scale_copula_max_pairs_per_horizon"]),
    )
    validation_pairs = horizon_pairs(validation, train_marginal.cdf(validation["log_price"].to_numpy()))
    test_pairs = horizon_pairs(test, train_marginal.cdf(test["log_price"].to_numpy()))
    oos_rows = []
    oos_rows.extend(evaluate_copula_models(training_models, train_pairs, "training_2021_2023"))
    oos_rows.extend(evaluate_copula_models(training_models, validation_pairs, "validation_2024"))
    oos_rows.extend(evaluate_copula_models(training_models, test_pairs, "test_2025"))

    model_rows = []
    for model in full_models:
        row = {key: value for key, value in model.items() if key != "params"}
        row.update(model["params"])
        model_rows.append(row)
    comparison = pd.DataFrame(model_rows).sort_values("objective_mean_log_copula", ascending=False)
    write_csv(comparison, copula_target)
    write_csv(pd.DataFrame(horizon_rows), tables / "copula_scale_horizon_fit.csv")
    write_csv(pd.DataFrame(oos_rows), tables / "copula_scale_oos_scores.csv")
    write_json({"full_sample_models": full_models, "training_models": training_models}, models / "copula_scale_models.json")
    plot_extended_horizon_dependence(pd.DataFrame(horizon_rows), figures / "single_vs_two_scale_dependence")

    hierarchy = pd.DataFrame([
        {"tier": "core", "models": "OUF; MOUF", "article_relation": "exact stationary OU and constant-intensity mixed-OU kernel", "claim_scope": "primary structural comparison"},
        {"tier": "main_extension", "models": "ThresholdOU_Heteroskedastic", "article_relation": "empirical asymmetry extension", "claim_scope": "conditional volatility robustness"},
        {"tier": "diagnostic", "models": "SingleGaussian; SingleMixed; TwoScaleGaussian; TwoScaleMixed", "article_relation": "copula dependence diagnostics; two-scale observed state is not one-dimensional Markov", "claim_scope": "multi-horizon fit and out-of-sample copula score"},
        {"tier": "forecast_benchmark", "models": "RandomWalk; NestedOU_proxy", "article_relation": "not article propositions", "claim_scope": "predictive benchmarks only"},
        {"tier": "appendix", "models": "CIR; MCIR; affine transformed OU/CIR", "article_relation": "article examples with restricted calibration", "claim_scope": "feasibility and weak-identification diagnostics"},
    ])
    write_csv(hierarchy, tables / "article_model_hierarchy.csv")
    logger.info(
        "Article extensions complete: matched bootstrap B=%d p=%.6g; best copula=%s",
        summary["replications"], summary["finite_replication_corrected_pvalue"], comparison.iloc[0]["model"],
    )
    return {
        "bootstrap_replications": summary["replications"],
        "matched_pairs": summary["matched_pairs"],
        "bootstrap_pvalue": summary["finite_replication_corrected_pvalue"],
        "best_full_sample_copula": comparison.iloc[0]["model"],
    }


def run_forecasts(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .forecasting import rolling_forecast_evaluation
    from .plotting import plot_reliability

    logger = setup_logging("10_forecast_comparison", config)
    if dry_run:
        logger.info("DRY RUN forecasts")
        return {"dry_run": True}
    target = _paths(config)["tables"] / "forecast_scores.csv"
    if target.exists() and not force:
        logger.info("Using cached forecast results: %s", target)
        return {"cached": True, "path": str(target)}
    df = select_sample(_load_prepared(config), "baseline")
    scores, reliability, fit_audit = rolling_forecast_evaluation(
        df, list(config["forecast"]["horizons_minutes"]), [0.001, 0.0025, 0.005, 0.01],
        refit_days=int(config["forecast"]["refit_frequency_days"]),
        window_days=int(config["forecast"]["rolling_window_days"]),
    )
    leakage = pd.to_datetime(fit_audit["fit_end_exclusive_utc"], utc=True) <= pd.to_datetime(fit_audit["refit_utc"], utc=True)
    fit_audit["no_lookahead"] = leakage
    if not leakage.all():
        raise RuntimeError("Forecast look-ahead audit failed")
    validation = scores[scores["period"] == "validation_2024"].groupby("model").agg(
        mean_NLS=("NLS", "mean"), mean_CRPS=("CRPS", "mean"), mean_RMSE=("RMSE", "mean")
    ).sort_values("mean_NLS").reset_index()
    validation["validation_rank_NLS"] = np.arange(1, len(validation) + 1)
    validation["selected_by_validation"] = validation["validation_rank_NLS"].eq(1)
    validation["test_hyperparameter_retuning"] = False
    tables = _paths(config)["tables"]
    write_csv(scores, tables / "forecast_scores.csv")
    write_csv(reliability, tables / "forecast_reliability_bins.csv")
    write_csv(fit_audit, tables / "forecast_refit_audit.csv")
    write_csv(validation, tables / "validation_model_selection.csv")
    plot_data = reliability[(reliability["period"] == "test_2025") & (reliability["horizon_minutes"] == 60)]
    plot_reliability(plot_data, _paths(config)["figures"] / "test_depeg_reliability")
    logger.info("Forecast evaluation complete rows=%d selected=%s", len(scores), validation.loc[validation["selected_by_validation"], "model"].tolist())
    return {"score_rows": len(scores), "selected_model": validation.loc[validation["selected_by_validation"], "model"].iloc[0]}


def _simulate_first_passage(fit, initial_states: np.ndarray, model: str, paths: int, max_steps: int, seed: int) -> np.ndarray:
    from .models.ou import ou_moments

    rng = np.random.default_rng(seed)
    theta, kappa, sigma = (fit.params[k] for k in ("theta", "kappa", "sigma"))
    states = rng.choice(initial_states, size=paths, replace=True)
    active = np.ones(paths, dtype=bool)
    durations = np.full(paths, np.nan)
    lam = fit.params.get("lambda", 0.0)
    stationary_sd = sigma / np.sqrt(2 * kappa)
    for step in range(1, max_steps + 1):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        mean, variance = ou_moments(states[idx], np.full(len(idx), 1 / 288), theta, kappa, sigma)
        next_values = mean + np.sqrt(variance) * rng.standard_normal(len(idx))
        if model == "MOUF":
            reset = rng.random(len(idx)) > np.exp(-lam / 288)
            next_values[reset] = theta + stationary_sd * rng.standard_normal(reset.sum())
        states[idx] = next_values
        recovered = np.abs(np.exp(next_values) - 1) <= 0.0025
        durations[idx[recovered]] = step * 5
        active[idx[recovered]] = False
    return durations


def run_rolling_and_events(config: dict[str, Any], sample: str = "baseline", force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .events import identify_depeg_events, rolling_parameter_estimates
    from .forecasting import fast_ou_parameters, fit_conditional_lambda
    from .models.mixed_ou import mixed_ou_logpdf
    from .plotting import plot_depeg_paths, plot_rolling_parameters

    logger = setup_logging("11_rolling_and_events", config)
    if dry_run:
        logger.info("DRY RUN rolling/events")
        return {"dry_run": True}
    target = _paths(config)["tables"] / "rolling_parameters.csv"
    if target.exists() and not force:
        logger.info("Using cached rolling/event results: %s", target)
        return {"cached": True, "path": str(target)}
    df = select_sample(_load_prepared(config), "baseline")
    event_specs = [(0.005, 0.0025, "baseline_50_25bp"), (0.0025, 0.001, "25_10bp"), (0.01, 0.005, "100_50bp")]
    event_tables = []
    for b_out, b_in, label in event_specs:
        event = identify_depeg_events(df, b_out=b_out, b_in=b_in)
        event.insert(0, "specification", label)
        event_tables.append(event)
    events = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    rolling = rolling_parameter_estimates(df, [30, 90, 180, 365], 30)
    tables = _paths(config)["tables"]
    write_csv(events, tables / "depeg_events.csv")
    write_csv(rolling, tables / "rolling_parameters.csv")
    plot_rolling_parameters(rolling, _paths(config)["figures"] / "rolling_parameters")
    base_events = events[events["specification"] == "baseline_50_25bp"]
    plot_depeg_paths(df, base_events, _paths(config)["figures"] / "depeg_event_paths")

    model_path = _paths(config)["models"]
    if not (model_path / "mixed_fits_baseline.json").exists():
        run_mixed_models(config, "baseline")
    ouf = _fit_from_json(model_path / "ou_fits_baseline.json", "OUF")
    mouf = _fit_from_json(model_path / "mixed_fits_baseline.json", "MOUF")
    initial = np.log(base_events["start_price"].to_numpy()) if len(base_events) else np.array([np.log(0.994), np.log(1.006)])
    first_passage_rows = []
    empirical = base_events["recovery_minutes"].dropna().to_numpy()
    for label, values in [("Empirical", empirical),
                          ("OUF_simulated", _simulate_first_passage(ouf, initial, "OUF", 3000, 2016, RANDOM_SEED)),
                          ("MOUF_simulated", _simulate_first_passage(mouf, initial, "MOUF", 3000, 2016, RANDOM_SEED + 1))]:
        observed = values[np.isfinite(values)]
        first_passage_rows.append({"distribution": label, "n": len(values), "recovered_within_7d": len(observed) / max(len(values), 1),
                                   "median_minutes": np.median(observed) if len(observed) else np.nan,
                                   "p75_minutes": np.quantile(observed, 0.75) if len(observed) else np.nan,
                                   "p90_minutes": np.quantile(observed, 0.90) if len(observed) else np.nan,
                                   "p95_minutes": np.quantile(observed, 0.95) if len(observed) else np.nan})
    write_csv(pd.DataFrame(first_passage_rows), tables / "first_passage_comparison.csv")

    last = df[df["datetime"] >= df["datetime"].max() - pd.Timedelta(days=365)]
    i, j = build_exact_horizon_pairs(last, 300)
    x = last["log_price"].to_numpy(); xp, xn = x[i], x[j]
    if len(i) > 75_000:
        sel = np.linspace(0, len(i) - 1, 75_000, dtype=int); xp, xn = xp[sel], xn[sel]
    pars = fast_ou_parameters(xp, xn); lam, _ = fit_conditional_lambda(xp, xn, np.full(len(xp), 1 / 288), pars)
    k_grid = pars.kappa * np.exp(np.linspace(-1.2, 1.2, 31))
    l_grid = lam * np.exp(np.linspace(-2, 2, 31))
    surface_rows = []
    for kappa in k_grid:
        for reset in l_grid:
            ll = mixed_ou_logpdf(xn, xp, np.full(len(xp), 1 / 288), pars.theta, kappa, pars.sigma, reset)
            surface_rows.append({"kappa": kappa, "lambda": reset, "mean_loglik": np.mean(ll)})
    write_csv(pd.DataFrame(surface_rows), tables / "rolling_likelihood_surface_final365d.csv")
    logger.info("Rolling/events complete events=%d rolling_rows=%d", len(events), len(rolling))
    return {"events": len(events), "rolling_rows": len(rolling)}


def run_robustness(config: dict[str, Any], force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .forecasting import fast_ou_parameters, fit_conditional_lambda

    logger = setup_logging("robustness", config)
    if dry_run:
        logger.info("DRY RUN robustness")
        return {"dry_run": True}
    target = _paths(config)["tables"] / "robustness_estimates.csv"
    block_target = _paths(config)["tables"] / "ou_block_length_robustness.csv"
    if target.exists() and block_target.exists() and not force:
        logger.info("Using cached robustness results: %s", target)
        return {"cached": True, "path": str(target)}
    full = _load_prepared(config)
    rows = []
    for sample in ("baseline", "extended"):
        df = select_sample(full, sample)
        for minutes in (5, 15, 30, 60):
            if minutes == 5:
                frequency_df = df
            else:
                frequency_df = aggregate_ohlcvt(df, minutes)
                frequency_df = frequency_df[frequency_df["complete_bin"]].reset_index(drop=True)
            i, j = build_exact_horizon_pairs(frequency_df, minutes * 60)
            x = frequency_df["log_price"].to_numpy(); xp, xn = x[i], x[j]
            delta = minutes / 1440
            ou = fast_ou_parameters(xp, xn, delta)
            lam, _ = fit_conditional_lambda(xp, xn, np.full(len(i), delta), ou, max_nobs=75_000)
            rows.append({"sample": sample, "anomaly_policy": "flagged_not_present_or_retained", "frequency_minutes": minutes,
                         "n_pairs": len(i), "aggregation_complete_bins_only": True,
                         "theta": ou.theta, "kappa": ou.kappa, "sigma": ou.sigma, "lambda": lam,
                         "half_life_minutes": 1440 * np.log(2) / ou.kappa,
                         "effective_half_life_minutes": 1440 * np.log(2) / (ou.kappa + lam)})
    for policy, exclude in (("retain_raw_flagged", False), ("exclude_flagged", True)):
        df = select_sample(full, "full", exclude_anomalies=exclude)
        i, j = build_exact_horizon_pairs(df, 300)
        x = df["log_price"].to_numpy(); xp, xn = x[i], x[j]
        ou = fast_ou_parameters(xp, xn)
        lam, _ = fit_conditional_lambda(xp, xn, np.full(len(i), 1 / 288), ou, max_nobs=75_000)
        rows.append({"sample": "full", "anomaly_policy": policy, "frequency_minutes": 5,
                     "n_pairs": len(i), "aggregation_complete_bins_only": True,
                     "theta": ou.theta, "kappa": ou.kappa, "sigma": ou.sigma, "lambda": lam,
                     "half_life_minutes": 1440 * np.log(2) / ou.kappa,
                     "effective_half_life_minutes": 1440 * np.log(2) / (ou.kappa + lam)})
    result = pd.DataFrame(rows)
    write_csv(result, _paths(config)["tables"] / "robustness_estimates.csv")
    write_csv(result[result["sample"] == "full"], _paths(config)["tables"] / "full_sample_anomaly_robustness.csv")
    baseline = select_sample(full, "baseline")
    i, j = build_exact_horizon_pairs(baseline, 300)
    values = baseline["log_price"].to_numpy()
    block_summary = _ou_moving_day_block_bootstrap(
        values[i], values[j], baseline["date"].to_numpy()[i], (1, 3, 7),
        int(config["estimation"]["bootstrap_replications_mvp"]), int(config["project"]["random_seed"]),
    )
    write_csv(block_summary, block_target)
    logger.info("Robustness complete rows=%d block_rows=%d", len(result), len(block_summary))
    return {"rows": len(result), "block_rows": len(block_summary)}


def run_report(config: dict[str, Any], force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from .reporting import build_empirical_report

    logger = setup_logging("12_build_empirical_report", config)
    if dry_run:
        logger.info("DRY RUN report")
        return {"dry_run": True}
    md, html, manifest = build_empirical_report(config)
    logger.info("Report built: %s and %s", md, html)
    return {"markdown": str(md), "html": str(html), "manifest_artifacts": len(manifest["artifacts"])}
