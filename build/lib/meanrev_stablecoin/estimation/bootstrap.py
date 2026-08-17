from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize

from ..forecasting import fast_ou_parameters, fit_conditional_lambda
from ..models.mixed_ou import mixed_ou_logpdf
from ..models.ou import ou_logpdf, simulate_ou


def block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or block_length <= 0:
        raise ValueError("n and block_length must be positive")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, max(n - block_length + 1, 1), size=n_blocks)
    indices = np.concatenate([np.arange(start, min(start + block_length, n)) for start in starts])
    if len(indices) < n:
        indices = np.resize(indices, n)
    return indices[:n]


def daily_block_resample_indices(day_codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    day_codes = np.asarray(day_codes)
    unique = np.unique(day_codes)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    groups = {day: np.flatnonzero(day_codes == day) for day in unique}
    return np.concatenate([groups[day] for day in sampled])


def parametric_mou_lr_bootstrap(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    replications: int = 200,
    bootstrap_nobs: int = 25_000,
    seed: int = 20260806,
) -> tuple[dict, np.ndarray]:
    """Boundary LR bootstrap with OU re-estimation and conditional lambda fitting.

    The computational bootstrap uses a fixed five-minute design and a capped
    simulation length, both reported explicitly in the returned metadata.
    """
    delta = 1 / 288
    null = fast_ou_parameters(np.asarray(x_prev), np.asarray(x_next), delta)
    observed_null_ll = float(np.sum(ou_logpdf(x_next, x_prev, np.full(len(x_prev), delta), null.theta, null.kappa, null.sigma)))
    observed_lam, _ = fit_conditional_lambda(
        np.asarray(x_prev), np.asarray(x_next), np.full(len(x_prev), delta), null, max_nobs=min(len(x_prev), 100_000)
    )
    observed_alt_ll = float(np.sum(mixed_ou_logpdf(
        x_next, x_prev, np.full(len(x_prev), delta), null.theta, null.kappa, null.sigma, observed_lam
    )))
    observed_lr = max(0.0, 2 * (observed_alt_ll - observed_null_ll))
    n = min(int(bootstrap_nobs), len(x_prev))
    rng = np.random.default_rng(seed)
    statistics = np.empty(replications)
    for b in range(replications):
        path = simulate_ou(n + 1, delta, null.theta, null.kappa, null.sigma, rng)
        xp, xn = path[:-1], path[1:]
        fitted_null = fast_ou_parameters(xp, xn, delta)
        null_ll = float(np.sum(ou_logpdf(xn, xp, np.full(n, delta), fitted_null.theta, fitted_null.kappa, fitted_null.sigma)))
        _, alt_ll = fit_conditional_lambda(xp, xn, np.full(n, delta), fitted_null, max_nobs=n)
        statistics[b] = max(0.0, 2 * (alt_ll - null_ll))
    raw_p = float(np.mean(statistics >= observed_lr))
    corrected_p = float((1 + np.sum(statistics >= observed_lr)) / (replications + 1))
    return {
        "null_model": "OUF",
        "alternative_model": "MOUF_conditional_lambda",
        "observed_lambda": observed_lam,
        "observed_lr": observed_lr,
        "replications": replications,
        "bootstrap_nobs": n,
        "raw_pvalue": raw_p,
        "finite_sample_corrected_pvalue": corrected_p,
        "critical_90": float(np.quantile(statistics, 0.90)),
        "critical_95": float(np.quantile(statistics, 0.95)),
        "critical_99": float(np.quantile(statistics, 0.99)),
        "fixed_interval_design": True,
        "parameters_reestimated_each_replication": True,
        "alternative_base_parameters_reestimated_then_fixed_for_lambda_profile": True,
    }, statistics


def _joint_mou_fixed_design_fit(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    delta: float,
    multistart: int,
) -> dict[str, float | bool]:
    """Fast joint MOUF conditional MLE used by the matched-design bootstrap."""
    null = fast_ou_parameters(x_prev, x_next, delta)
    null_ll = float(np.sum(ou_logpdf(
        x_next, x_prev, np.full(len(x_prev), delta), null.theta, null.kappa, null.sigma
    )))
    lambda_starts = np.array([0.0, 0.05, 0.5, 2.0, 8.0, 30.0])[: max(multistart, 1)]
    state_sd = max(float(np.std(x_prev)), 1e-5)
    theta_bound = max(0.01, 25 * state_sd)
    starts = []
    for lam in lambda_starts:
        scale = 0.05 if lam >= 0.5 else 1.0
        sigma_scale = 0.75 if lam >= 0.5 else 1.0
        starts.append(np.array([
            null.theta,
            np.log(max(null.kappa * scale, 1e-6)),
            np.log(max(null.sigma * sigma_scale, 1e-8)),
            lam * delta,
        ]))

    bounds = [
        (-theta_bound, theta_bound),
        (-10, 12),
        (-16, 1),
        (0.0, 100.0 * delta),
    ]
    dd = np.full(len(x_prev), delta)

    def objective(beta: np.ndarray) -> float:
        theta = float(beta[0])
        kappa, sigma = map(float, np.exp(beta[1:3]))
        lam = float(beta[3] / delta)
        values = mixed_ou_logpdf(x_next, x_prev, dd, theta, kappa, sigma, lam)
        return float(-np.mean(values)) if np.isfinite(values).all() else 1e100

    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 220, "ftol": 1e-10, "gtol": 2e-6, "maxls": 30},
        )
        for start in starts
    ]
    best = min((result for result in results if np.isfinite(result.fun)), key=lambda result: result.fun)
    theta = float(best.x[0])
    kappa, sigma = map(float, np.exp(best.x[1:3]))
    lam = float(best.x[3] / delta)
    alt_ll = -float(best.fun) * len(x_prev)
    # The null is nested at lambda=0. Numerical optimization must never produce
    # a negative LR merely because the boundary is represented on a log scale.
    if alt_ll < null_ll:
        theta, kappa, sigma, lam, alt_ll = null.theta, null.kappa, null.sigma, 0.0, null_ll
    return {
        "null_loglik": null_ll,
        "alternative_loglik": alt_ll,
        "lr": max(0.0, 2 * (alt_ll - null_ll)),
        "theta_null": null.theta,
        "kappa_null": null.kappa,
        "sigma_null": null.sigma,
        "theta_alt": theta,
        "kappa_alt": kappa,
        "sigma_alt": sigma,
        "lambda_alt": lam,
        "alternative_converged": bool(best.success and np.isfinite(alt_ll)),
        "alternative_gradient_norm": float(np.linalg.norm(best.jac)),
    }


def matched_joint_mou_lr_bootstrap(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    replications: int = 500,
    matched_pairs: int = 25_000,
    multistart: int = 4,
    parallel_jobs: int = 1,
    seed: int = 20260806,
) -> tuple[dict, np.ndarray, dict]:
    """Matched-size fixed-design bootstrap of OUF versus jointly fitted MOUF.

    A deterministic systematic subset is used for both the observed statistic
    and every bootstrap replication. Conditional outcomes are simulated from
    the refitted OU null at the observed starting states. Thus sample size,
    design points, null refitting, and joint alternative fitting are identical
    on the observed and simulated sides of the test.
    """
    x_prev = np.asarray(x_prev, dtype=float)
    x_next = np.asarray(x_next, dtype=float)
    if len(x_prev) != len(x_next) or len(x_prev) < 500:
        raise ValueError("matched bootstrap requires equal arrays and at least 500 pairs")
    n = min(int(matched_pairs), len(x_prev))
    selected = np.linspace(0, len(x_prev) - 1, n, dtype=int)
    xp, xn = x_prev[selected], x_next[selected]
    delta = 1 / 288
    observed = _joint_mou_fixed_design_fit(xp, xn, delta, multistart)
    null = fast_ou_parameters(xp, xn, delta)
    from ..models.ou import ou_moments

    conditional_mean, conditional_variance = ou_moments(
        xp, np.full(n, delta), null.theta, null.kappa, null.sigma
    )
    seeds = np.random.SeedSequence(seed).spawn(replications)

    def one_replication(index: int) -> dict[str, float | bool | int]:
        rng = np.random.default_rng(seeds[index])
        simulated_next = conditional_mean + np.sqrt(conditional_variance) * rng.standard_normal(n)
        result = _joint_mou_fixed_design_fit(xp, simulated_next, delta, multistart)
        result["replication"] = index
        return result

    # The standard-library executor avoids joblib's Windows multiprocessing
    # pipes, which are unavailable in the managed workspace sandbox.
    if parallel_jobs == 1:
        rows = [one_replication(index) for index in range(replications)]
    else:
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            rows = list(executor.map(one_replication, range(replications)))
    statistics = np.array([float(row["lr"]) for row in rows])
    observed_lr = float(observed["lr"])
    exceedances = int(np.sum(statistics >= observed_lr))
    summary = {
        "test_name": "matched_fixed_design_joint_MOUF_boundary_bootstrap",
        "null_model": "OUF",
        "alternative_model": "MOUF_joint_theta_kappa_sigma_lambda",
        "total_available_pairs": len(x_prev),
        "matched_pairs": n,
        "pair_selection": "deterministic_systematic_over_full_baseline",
        "bootstrap_design": "fixed_observed_x_prev_conditional_OU_simulation",
        "replications": replications,
        "same_sample_size_observed_and_bootstrap": True,
        "same_null_refit_observed_and_bootstrap": True,
        "same_joint_alternative_fit_observed_and_bootstrap": True,
        "observed_lr": observed_lr,
        "raw_pvalue": float(exceedances / replications),
        "finite_replication_corrected_pvalue": float((1 + exceedances) / (replications + 1)),
        "critical_90": float(np.quantile(statistics, 0.90)),
        "critical_95": float(np.quantile(statistics, 0.95)),
        "critical_99": float(np.quantile(statistics, 0.99)),
        "bootstrap_alternative_convergence_rate": float(np.mean([row["alternative_converged"] for row in rows])),
        "bootstrap_lambda_boundary_share": float(np.mean([float(row["lambda_alt"]) < 1e-5 for row in rows])),
        "multistart": multistart,
        "parallel_jobs": parallel_jobs,
        "random_seed": seed,
        "interpretation_scope": "conditional composite likelihood on the prespecified matched systematic design",
    }
    return summary, np.array(rows, dtype=object), observed
