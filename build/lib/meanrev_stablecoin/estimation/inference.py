from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def finite_difference_hessian(fun, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    hessian = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = step
            ej = np.zeros(n); ej[j] = step
            value = (fun(x + ei + ej) - fun(x + ei - ej) - fun(x - ei + ej) + fun(x - ei - ej)) / (4 * step * step)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def ljung_box_summary(values: np.ndarray, lag: int = 20) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    correlations = np.array([np.dot(x[k:], x[:-k]) / denom for k in range(1, lag + 1)])
    q = n * (n + 2) * np.sum(correlations * correlations / (n - np.arange(1, lag + 1)))
    return float(q), float(chi2.sf(q, lag))


def cluster_sandwich_covariance(
    logpdf_from_beta,
    beta: np.ndarray,
    cluster_codes: np.ndarray,
    step: float = 1e-5,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Observed-Hessian and cluster-sandwich covariance for mean log likelihood."""
    beta = np.asarray(beta, dtype=float)
    clusters = np.asarray(cluster_codes)
    n, p = len(clusters), len(beta)

    def mean_nll(value):
        ll = np.asarray(logpdf_from_beta(value), dtype=float)
        return float(-np.mean(ll))

    hessian = finite_difference_hessian(mean_nll, beta, step=max(step, 1e-4))
    condition = float(np.linalg.cond(hessian))
    try:
        bread = np.linalg.inv(hessian)
        ordinary = bread / n
    except np.linalg.LinAlgError:
        return None, None, condition
    scores = np.empty((n, p), dtype=float)
    for j in range(p):
        direction = np.zeros(p); direction[j] = step
        scores[:, j] = (np.asarray(logpdf_from_beta(beta + direction)) - np.asarray(logpdf_from_beta(beta - direction))) / (2 * step)
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_scores = np.zeros((len(unique), p))
    np.add.at(cluster_scores, inverse, scores)
    meat = cluster_scores.T @ cluster_scores
    robust = bread @ meat @ bread / (n * n)
    return ordinary, robust, condition
