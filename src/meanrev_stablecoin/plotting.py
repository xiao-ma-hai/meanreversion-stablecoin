from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .data_io import resolve_path
from .models.copula import gaussian_copula_logpdf


def save_dual(fig: plt.Figure, stem: str | Path) -> tuple[Path, Path]:
    stem = resolve_path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_price_series(df: pd.DataFrame, stem: str | Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df["datetime"], df["close"], lw=0.35, color="#174a7e")
    axes[0].axhline(1, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("USDT/USD")
    axes[1].plot(df["datetime"], 10_000 * df["deviation"], lw=0.35, color="#b23a48")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Deviation (bp)")
    axes[1].set_xlabel("UTC")
    fig.suptitle("Kraken USDT/USD price and peg deviation")
    save_dual(fig, stem)


def plot_coverage_and_gaps(annual: pd.DataFrame, gap_minutes: np.ndarray, stem: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(annual["year"].astype(str), annual["coverage_pct"], color="#3b7a57")
    axes[0].set_ylim(0, 102); axes[0].set_ylabel("Coverage (%)"); axes[0].tick_params(axis="x", rotation=45)
    positive = np.asarray(gap_minutes); positive = positive[positive > 0]
    axes[1].hist(np.log10(positive), bins=50, color="#7f6a93")
    axes[1].set_xlabel("log10(gap minutes)"); axes[1].set_ylabel("Count")
    fig.suptitle("Annual coverage and missing-bar gap distribution")
    save_dual(fig, stem)


def plot_deviation_distribution(values: np.ndarray, stem: str | Path) -> None:
    values = np.asarray(values)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(values * 10_000, bins=150, density=True, color="#2f6690")
    axes[0].set_xlabel("Log deviation (bp)"); axes[0].set_ylabel("Density")
    q = np.linspace(0.005, 0.995, 199)
    z = norm.ppf(q); empirical = np.quantile(values, q)
    axes[1].scatter(z, empirical * 10_000, s=7, alpha=0.6)
    axes[1].plot([z.min(), z.max()], [np.mean(values) * 1e4 + np.std(values) * z.min() * 1e4,
                                      np.mean(values) * 1e4 + np.std(values) * z.max() * 1e4], color="black", lw=0.8)
    axes[1].set_xlabel("Normal quantile"); axes[1].set_ylabel("Empirical quantile (bp)")
    axes[2].hist(values * 10_000, bins=300, density=True, color="#b56576")
    axes[2].set_xlim(np.quantile(values, 0.001) * 1e4, np.quantile(values, 0.999) * 1e4)
    axes[2].set_xlabel("Central 99.8% (bp)")
    fig.suptitle("Peg-deviation distribution, normal QQ, and central-tail zoom")
    save_dual(fig, stem)


def plot_drift(drift: pd.DataFrame, theta: float, kappa: float | None, stem: str | Path) -> None:
    x = drift["x_grid"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x * 1e4, drift["simultaneous_lower"], drift["simultaneous_upper"], color="#aac9e6", alpha=0.5, label="95% simultaneous band")
    ax.plot(x * 1e4, drift["mu_hat_per_day"], color="#174a7e", lw=1.5, label="Local-linear drift")
    if kappa is not None:
        ax.plot(x * 1e4, kappa * (theta - x), color="#b23a48", ls="--", label="OU fitted drift")
    ax.axvline(theta * 1e4, color="black", lw=0.8); ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Log-price state (bp)"); ax.set_ylabel("Drift per day"); ax.legend()
    ax.set_title("Nonparametric and OU-implied conditional drift")
    save_dual(fig, stem)


def plot_asymmetry(df: pd.DataFrame, stem: str | Path) -> None:
    work = df[df["exact_5min"]].copy()
    work["change_bp"] = work["log_price"].diff() * 1e4
    work["state_bp"] = work["log_price"].shift() * 1e4
    work = work.dropna(subset=["change_bp", "state_bp"])
    work["bin"] = pd.qcut(work["state_bp"], 30, duplicates="drop")
    grouped = work.groupby("bin", observed=True).agg(state_bp=("state_bp", "mean"), change_bp=("change_bp", "mean"), n=("change_bp", "size"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = np.where(grouped["state_bp"] >= 0, "#b23a48", "#2f6690")
    ax.scatter(grouped["state_bp"], grouped["change_bp"], c=colors, s=25)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Lagged log deviation (bp)"); ax.set_ylabel("Mean next 5-min change (bp)")
    ax.set_title("Empirical adjustment above and below the peg")
    save_dual(fig, stem)


def plot_pit_diagnostics(pit_map: dict[str, np.ndarray], stem: str | Path) -> None:
    models = list(pit_map)
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3.2 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        pit = np.clip(pit_map[model], 1e-8, 1 - 1e-8)
        axes[row, 0].hist(pit, bins=30, density=True, alpha=0.7)
        axes[row, 0].axhline(1, color="black", ls="--", lw=0.8); axes[row, 0].set_title(f"{model}: PIT")
        z = norm.ppf(pit); z -= z.mean(); denom = np.dot(z, z)
        acf = [1.0] + [np.dot(z[k:], z[:-k]) / denom for k in range(1, 31)]
        axes[row, 1].stem(np.arange(31), acf, basefmt=" ")
        axes[row, 1].axhline(1.96 / np.sqrt(len(z)), color="red", ls="--", lw=0.7)
        axes[row, 1].axhline(-1.96 / np.sqrt(len(z)), color="red", ls="--", lw=0.7)
        axes[row, 1].set_title(f"{model}: Gaussianized PIT ACF")
    fig.tight_layout()
    save_dual(fig, stem)


def plot_copula_contours(kappa: float, lam: float, horizon_minutes: int, stem: str | Path) -> None:
    grid = np.linspace(0.01, 0.99, 80)
    u, v = np.meshgrid(grid, grid)
    h = horizon_minutes / 1440
    rho = np.exp(-kappa * h)
    base = np.exp(gaussian_copula_logpdf(u, v, rho))
    mixed = np.exp(-lam * h) * base + 1 - np.exp(-lam * h)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, density, title in zip(axes, [base, mixed], ["Gaussian", "Mixed Gaussian"]):
        contour = ax.contourf(u, v, np.minimum(density, np.quantile(density, 0.98)), levels=20, cmap="viridis")
        fig.colorbar(contour, ax=ax); ax.set_xlabel("u"); ax.set_ylabel("v"); ax.set_title(title)
    fig.suptitle(f"{horizon_minutes}-minute temporal copula densities")
    save_dual(fig, stem)


def plot_horizon_dependence(table: pd.DataFrame, stem: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["horizon_minutes"], table["empirical_gaussian_rho"], "o", label="Empirical Gaussian rho")
    ax.plot(table["horizon_minutes"], table["fitted_base_rho"], "-", label="Base decay")
    ax.plot(table["horizon_minutes"], table["fitted_mixed_linear_dependence"], "--", label="Mixed effective decay")
    ax.set_xscale("log"); ax.set_xlabel("Horizon (minutes)"); ax.set_ylabel("Dependence"); ax.legend()
    ax.set_title("Multi-horizon temporal dependence")
    save_dual(fig, stem)


def plot_extended_horizon_dependence(table: pd.DataFrame, stem: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["horizon_minutes"], table["empirical_gaussian_rho"], "o", ms=7,
            color="black", label="Empirical Gaussian rho")
    styles = {
        "SingleGaussian": ("-", "#1f77b4"),
        "SingleMixed": ("--", "#ff7f0e"),
        "TwoScaleGaussian": ("-.", "#2ca02c"),
        "TwoScaleMixed": (":", "#d62728"),
    }
    for column, (linestyle, color) in styles.items():
        if column in table:
            ax.plot(table["horizon_minutes"], table[column], linestyle=linestyle,
                    color=color, lw=2, label=column)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Gaussian-rank dependence")
    ax.set_title("Single- versus two-scale temporal copula dependence")
    ax.legend(fontsize=8)
    save_dual(fig, stem)


def plot_rolling_parameters(table: pd.DataFrame, stem: str | Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for window, group in table.groupby("window_days"):
        axes[0].plot(pd.to_datetime(group["end_utc"]), group["theta"] * 1e4, label=f"{window}d")
        axes[1].plot(pd.to_datetime(group["end_utc"]), group["half_life_minutes"], label=f"{window}d")
        axes[2].plot(pd.to_datetime(group["end_utc"]), group["lambda"], label=f"{window}d")
    axes[0].set_ylabel("theta (bp)"); axes[1].set_ylabel("OU half-life (min)"); axes[2].set_ylabel("lambda / day")
    axes[2].set_xlabel("Window end (UTC)"); axes[0].legend(ncol=4)
    fig.suptitle("Rolling parameter estimates and reset intensity")
    save_dual(fig, stem)


def plot_depeg_paths(df: pd.DataFrame, events: pd.DataFrame, stem: str | Path, max_events: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for _, event in events.head(max_events).iterrows():
        start = pd.Timestamp(event["start_utc"])
        end = start + pd.Timedelta(hours=24)
        part = df[df["datetime"].between(start, end)].copy()
        if part.empty:
            continue
        hours = (part["datetime"] - start).dt.total_seconds() / 3600
        ax.plot(hours, part["deviation"] * 1e4, lw=0.9, alpha=0.75)
    ax.axhline(50, color="red", ls="--", lw=0.7); ax.axhline(-50, color="red", ls="--", lw=0.7)
    ax.axhline(25, color="black", ls=":", lw=0.7); ax.axhline(-25, color="black", ls=":", lw=0.7)
    ax.set_xlabel("Hours since event start"); ax.set_ylabel("Deviation (bp)")
    ax.set_title("Observed paths after selected 50-bp depeg events")
    save_dual(fig, stem)


def plot_reliability(table: pd.DataFrame, stem: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for (model, threshold), group in table.groupby(["model", "threshold"]):
        ax.plot(group["mean_predicted_probability"], group["observed_frequency"], marker="o", ms=3, lw=0.8, label=f"{model}, {threshold:.4f}")
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Test-period depeg probability calibration"); ax.legend(fontsize=6, ncol=2)
    save_dual(fig, stem)
