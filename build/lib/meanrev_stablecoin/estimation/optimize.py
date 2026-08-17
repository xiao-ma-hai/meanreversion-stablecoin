from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize


@dataclass
class FitResult:
    model_name: str
    params: dict[str, float]
    loglik: float
    nobs: int
    aic: float
    bic: float
    converged: bool
    optimizer_message: str
    gradient_norm: float
    hessian_condition: float | None
    covariance: np.ndarray | None
    robust_covariance: np.ndarray | None
    start_values: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("covariance", "robust_covariance"):
            value = payload[key]
            payload[key] = None if value is None else np.asarray(value).tolist()
        return payload


def multistart_minimize(
    objective: Callable[[np.ndarray], float],
    starts: Sequence[np.ndarray],
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    method: str = "L-BFGS-B",
    options: dict[str, Any] | None = None,
) -> tuple[OptimizeResult, list[OptimizeResult]]:
    results: list[OptimizeResult] = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method=method,
            bounds=bounds,
            options=options or {"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
        )
        results.append(result)
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError("All optimization starts returned non-finite objectives")
    best = min(finite, key=lambda result: result.fun)
    return best, results


def optimizer_diagnostics(result: OptimizeResult, nobs: int) -> tuple[float, float | None, np.ndarray | None]:
    jac = np.asarray(getattr(result, "jac", np.array([np.nan])), dtype=float)
    gradient_norm = float(np.linalg.norm(jac[np.isfinite(jac)])) if np.isfinite(jac).any() else np.nan
    condition = None
    covariance = None
    hess_inv = getattr(result, "hess_inv", None)
    if hess_inv is not None:
        try:
            dense = np.asarray(hess_inv.todense() if hasattr(hess_inv, "todense") else hess_inv, dtype=float)
            condition = float(np.linalg.cond(dense))
            covariance = dense / max(nobs, 1)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return gradient_norm, condition, covariance


def repeated_solution_count(results: Sequence[OptimizeResult], tolerance: float = 1e-8) -> int:
    converged = [r for r in results if bool(r.success) and np.isfinite(r.fun)]
    if not converged:
        return 0
    best = min(r.fun for r in converged)
    return sum(abs(r.fun - best) <= tolerance for r in converged)

