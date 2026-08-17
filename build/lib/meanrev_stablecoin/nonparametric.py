from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftResult:
    grid: pd.DataFrame
    sign_summary: pd.DataFrame
    metadata: dict


def _local_stats(x: np.ndarray, y: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    stats = np.empty((len(grid), 6), dtype=float)
    for g, point in enumerate(grid):
        z = x - point
        w = np.exp(-0.5 * (z / bandwidth) ** 2)
        stats[g] = [
            w.sum(), np.dot(w, z), np.dot(w, z * z),
            np.dot(w, y), np.dot(w, z * y), np.dot(w, y * y),
        ]
    return stats


def _solve_local(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s0, s1, s2, t0, t1, y2 = np.moveaxis(stats, -1, 0)
    denom = s0 * s2 - s1 * s1
    valid = np.abs(denom) > 1e-18
    intercept = np.full_like(s0, np.nan)
    slope = np.full_like(s0, np.nan)
    intercept[valid] = (s2[valid] * t0[valid] - s1[valid] * t1[valid]) / denom[valid]
    slope[valid] = (s0[valid] * t1[valid] - s1[valid] * t0[valid]) / denom[valid]
    sse = y2 - 2 * intercept * t0 - 2 * slope * t1 + intercept**2 * s0 + 2 * intercept * slope * s1 + slope**2 * s2
    return intercept, slope, sse


def _evaluate_local_sse(stats: np.ndarray, intercept: np.ndarray, slope: np.ndarray) -> np.ndarray:
    s0, s1, _s2, t0, t1, y2 = np.moveaxis(stats, -1, 0)
    s2 = stats[..., 2]
    return y2 - 2 * intercept * t0 - 2 * slope * t1 + intercept**2 * s0 + 2 * intercept * slope * s1 + slope**2 * s2


def _blocked_bandwidth_cv(
    x: np.ndarray,
    y: np.ndarray,
    day_codes: np.ndarray,
    grid: np.ndarray,
    candidates: np.ndarray,
) -> tuple[float, list[dict]]:
    unique_days = np.unique(day_codes)
    folds = np.array_split(unique_days, 5)
    rows = []
    for bandwidth in candidates:
        fold_losses = []
        for fold_days in folds:
            valid = np.isin(day_codes, fold_days)
            train_stats = _local_stats(x[~valid], y[~valid], grid, bandwidth)
            mu, _, _ = _solve_local(train_stats)
            predictions = np.interp(x[valid], grid, mu, left=mu[0], right=mu[-1])
            fold_losses.append(float(np.mean((y[valid] - predictions) ** 2)))
        rows.append({"bandwidth": float(bandwidth), "blocked_cv_mse": float(np.mean(fold_losses))})
    best = min(rows, key=lambda row: row["blocked_cv_mse"])["bandwidth"]
    return float(best), rows


def estimate_nonparametric_drift(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    delta_days: np.ndarray,
    day_codes: np.ndarray,
    bootstrap_replications: int = 1000,
    max_pairs: int = 200_000,
    seed: int = 20260806,
) -> DriftResult:
    x_prev = np.asarray(x_prev, dtype=float)
    x_next = np.asarray(x_next, dtype=float)
    delta_days = np.asarray(delta_days, dtype=float)
    day_codes = np.asarray(day_codes)
    if len(x_prev) > max_pairs:
        idx = np.linspace(0, len(x_prev) - 1, max_pairs, dtype=int)
        x_prev, x_next, delta_days, day_codes = x_prev[idx], x_next[idx], delta_days[idx], day_codes[idx]
    y = (x_next - x_prev) / delta_days
    grid = np.quantile(x_prev, np.arange(1, 100) / 100)
    robust_scale = max(float(np.subtract(*np.quantile(x_prev, [0.75, 0.25])) / 1.349), 1e-8)
    silverman = 1.06 * robust_scale * len(x_prev) ** (-1 / 5)
    candidates = silverman * np.array([0.6, 0.9, 1.3, 1.9, 2.7, 3.8, 5.4, 7.6, 10.8, 15.0, 22.0])
    bandwidth, cv_rows = _blocked_bandwidth_cv(x_prev, y, day_codes, grid, candidates)
    full_stats = _local_stats(x_prev, y, grid, bandwidth)
    mu, slope, _ = _solve_local(full_stats)

    unique_days, inverse = np.unique(day_codes, return_inverse=True)
    day_stats = np.zeros((len(unique_days), len(candidates), len(grid), 6), dtype=float)
    day_sum_x = np.zeros(len(unique_days))
    day_count = np.zeros(len(unique_days), dtype=int)
    for day_idx in range(len(unique_days)):
        mask = inverse == day_idx
        day_sum_x[day_idx] = x_prev[mask].sum()
        day_count[day_idx] = mask.sum()
        for h_idx, candidate in enumerate(candidates):
            day_stats[day_idx, h_idx] = _local_stats(x_prev[mask], y[mask], grid, candidate)

    rng = np.random.default_rng(seed)
    bootstrap_mu = np.empty((bootstrap_replications, len(grid)))
    bootstrap_theta = np.empty(bootstrap_replications)
    selected_bandwidths = np.empty(bootstrap_replications)
    for start in range(0, bootstrap_replications, 25):
        size = min(25, bootstrap_replications - start)
        draws = rng.integers(0, len(unique_days), size=(size, len(unique_days)))
        split = len(unique_days) // 2
        counts_train = np.stack([np.bincount(row[:split], minlength=len(unique_days)) for row in draws])
        counts_valid = np.stack([np.bincount(row[split:], minlength=len(unique_days)) for row in draws])
        aggregate_train = np.tensordot(counts_train, day_stats, axes=(1, 0))
        aggregate_valid = np.tensordot(counts_valid, day_stats, axes=(1, 0))
        aggregate = aggregate_train + aggregate_valid
        counts = counts_train + counts_valid
        for offset in range(size):
            losses, mus = [], []
            for h_idx in range(len(candidates)):
                train_mu, train_slope, _ = _solve_local(aggregate_train[offset, h_idx])
                losses.append(np.nansum(_evaluate_local_sse(aggregate_valid[offset, h_idx], train_mu, train_slope)))
                full_mu, _, _ = _solve_local(aggregate[offset, h_idx])
                mus.append(full_mu)
            chosen = int(np.nanargmin(losses))
            bootstrap_mu[start + offset] = mus[chosen]
            selected_bandwidths[start + offset] = candidates[chosen]
            bootstrap_theta[start + offset] = np.dot(counts[offset], day_sum_x) / np.dot(counts[offset], day_count)

    lower, upper = np.nanquantile(bootstrap_mu, [0.025, 0.975], axis=0)
    simultaneous_radius = float(np.nanquantile(np.nanmax(np.abs(bootstrap_mu - mu[None, :]), axis=1), 0.95))
    theta_hat = float(np.mean(x_prev))
    density_weights = np.maximum(full_stats[:, 0], 0)
    density_weights /= density_weights.sum()
    table = pd.DataFrame(
        {
            "x_grid": grid,
            "mu_hat_per_day": mu,
            "local_slope": slope,
            "ci_lower": lower,
            "ci_upper": upper,
            "simultaneous_lower": mu - simultaneous_radius,
            "simultaneous_upper": mu + simultaneous_radius,
            "expected_sign_theta0": np.sign(-grid),
            "sign_match_theta0": (-grid * mu) > 0,
            "expected_sign_theta_hat": np.sign(theta_hat - grid),
            "sign_match_theta_hat": ((theta_hat - grid) * mu) > 0,
            "density_weight": density_weights,
        }
    )
    summaries = []
    for label, theta in (("anchor_theta_0", 0.0), ("sample_theta_hat", theta_hat)):
        product = (theta - grid) * mu
        violation = np.minimum(0, product)
        scr = float(np.mean(product > 0))
        t_equal = float(np.mean(violation**2))
        t_density = float(np.sum(density_weights * violation**2))
        scr_boot = np.mean((bootstrap_theta[:, None] - grid[None, :]) * bootstrap_mu > 0, axis=1) if label == "sample_theta_hat" else np.mean((-grid[None, :]) * bootstrap_mu > 0, axis=1)
        summaries.append(
            {
                "theta_definition": label,
                "theta": theta,
                "SCR": scr,
                "SCR_bootstrap_ci_lower": float(np.quantile(scr_boot, 0.025)),
                "SCR_bootstrap_ci_upper": float(np.quantile(scr_boot, 0.975)),
                "Tn_equal_weight": t_equal,
                "Tn_density_weight": t_density,
                "grid_points": len(grid),
            }
        )
    return DriftResult(
        grid=table,
        sign_summary=pd.DataFrame(summaries),
        metadata={
            "bandwidth": bandwidth,
            "bandwidth_candidates": candidates.tolist(),
            "blocked_cv": cv_rows,
            "bootstrap_replications": bootstrap_replications,
            "block_unit": "UTC day",
            "bootstrap_bandwidth_median": float(np.median(selected_bandwidths)),
            "estimation_pairs": len(x_prev),
            "simultaneous_radius": simultaneous_radius,
        },
    )
