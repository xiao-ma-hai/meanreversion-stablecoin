from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .constants import RAW_COLUMNS, project_root


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (root or project_root()) / path


def load_config(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("inherits", None)
    if parent:
        base = load_config(path.parent / parent)
        return _deep_merge(base, config)
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_kraken_ohlcvt(path: str | Path) -> pd.DataFrame:
    """Read the immutable headerless Kraken OHLCVT file."""
    path = resolve_path(path)
    dtype = {
        "timestamp": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "trades": "int64",
    }
    return pd.read_csv(path, header=None, names=RAW_COLUMNS, dtype=dtype)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_dirs(config: dict[str, Any]) -> None:
    root = project_root()
    for value in config["output"].values():
        resolve_path(value, root).mkdir(parents=True, exist_ok=True)
    for value in ("data/interim", "data/processed"):
        resolve_path(value, root).mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8", **kwargs)
    return target


def write_json(payload: Any, path: str | Path) -> Path:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return target


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def setup_logging(name: str, config: dict[str, Any]) -> logging.Logger:
    ensure_output_dirs(config)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
        file_handler = logging.FileHandler(
            resolve_path(config["output"]["logs"]) / f"{name}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger
