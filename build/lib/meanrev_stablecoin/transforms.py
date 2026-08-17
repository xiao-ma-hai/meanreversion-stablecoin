from __future__ import annotations

import numpy as np


def affine_forward(r: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if np.any(a <= 0):
        raise ValueError("Affine scale must be positive")
    return a * np.asarray(r, dtype=float) + np.asarray(b, dtype=float)


def affine_inverse(y: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if np.any(a <= 0):
        raise ValueError("Affine scale must be positive")
    return (np.asarray(y, dtype=float) - np.asarray(b, dtype=float)) / a


def affine_inverse_jacobian(a: np.ndarray | float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if np.any(a <= 0):
        raise ValueError("Affine scale must be positive")
    return 1.0 / a

