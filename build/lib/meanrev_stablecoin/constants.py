from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 20260806
DAY_SECONDS = 86_400
BAR_SECONDS = 300
DELTA_5_DAY = BAR_SECONDS / DAY_SECONDS
EXPECTED_SHA256 = (
    "417ea3aa71f86c4476c30568f251f3f8"
    "73ae8b0d44c5f480f5ad2f35fc6acc22"
)
EXPECTED_ROWS = 797_178
EXPECTED_FIRST_TIMESTAMP = 1_490_807_100
EXPECTED_LAST_TIMESTAMP = 1_767_225_300
RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
HARD_ANOMALY_TIMESTAMPS = (1_557_591_900, 1_568_645_400, 1_568_645_700)


def project_root() -> Path:
    """Return the repository root without relying on the caller's CWD."""
    return Path(__file__).resolve().parents[2]

