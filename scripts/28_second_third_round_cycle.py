from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT


SCRIPTS = (
    "20_audit_tick_and_ties.py",
    "25_three_coin_data_audit.py",
    "21_interval_likelihood_research.py",
    "26_cross_stablecoin_comparison.py",
    "27_frequency_semigroup_test.py",
    "19_build_submission_paper.py",
)


def main() -> None:
    for name in SCRIPTS:
        path = ROOT / "scripts" / name
        print(f"[second-third-round] running {name}", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)
    print("[second-third-round] complete", flush=True)


if __name__ == "__main__":
    main()
