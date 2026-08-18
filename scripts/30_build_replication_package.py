from __future__ import annotations

import json
import zipfile
from pathlib import Path

from _bootstrap import ROOT
from meanrev_stablecoin.data_io import sha256_file


TARGET = ROOT / "submission" / "replication" / "copula_mean_reversion_2026-08-18_replication.zip"
RAW_FILES = (
    "USDTUSD_1.csv",
    "USDTUSD_5.csv",
    "USDCUSD_1.csv",
    "USDCUSD_5.csv",
    "DAIUSD_1.csv",
    "DAIUSD_5.csv",
)
OUTPUT_FILES = (
    "output/tables/tick_tie_audit.csv",
    "output/tables/three_coin_data_audit.csv",
    "output/tables/common_sample_selection.csv",
    "output/tables/full_sample_model_estimates.csv",
    "output/tables/training_model_estimates.csv",
    "output/tables/oos_scores_with_uncertainty.csv",
    "output/tables/cross_stablecoin_parameters.csv",
    "output/tables/cross_stablecoin_pairwise_tests.csv",
    "output/tables/cross_stablecoin_drift_curves.csv",
    "output/tables/frequency_parameter_stability.csv",
    "output/tables/one_to_five_minute_kernel_test.csv",
    "output/models/interval_likelihood_models.json",
    "output/models/cross_stablecoin_interval_models.json",
    "output/models/frequency_interval_models.json",
    "output/figures/tie_rate_by_year.pdf",
    "output/figures/tie_rate_by_year.png",
    "output/figures/cross_stablecoin_drift_curves.pdf",
    "output/figures/cross_stablecoin_drift_curves.png",
    "output/figures/cross_stablecoin_half_lives.pdf",
    "output/figures/cross_stablecoin_half_lives.png",
    "output/figures/kappa_lambda_by_frequency.pdf",
    "output/figures/kappa_lambda_by_frequency.png",
)


def main() -> None:
    files: set[Path] = set()
    for relative in (
        "README.md",
        "pyproject.toml",
        "environment.yml",
        "environment.from-history.yml",
        "environment.lock.yml",
        "docs/2026-08-18_second_third_round_assessment.md",
        "docs/worklog/2026-08-18.md",
    ):
        files.add(ROOT / relative)
    for directory in ("configs", "scripts", "src", "tests", "manuscript", "submission/overleaf"):
        files.update(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.update(ROOT / relative for relative in OUTPUT_FILES)
    missing = [str(path.relative_to(ROOT)) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"replication inputs missing: {missing}")

    data_manifest = {
        "raw_data_included": False,
        "placement": "Copy the six immutable files to data/raw/ before running scripts/28_second_third_round_cycle.py.",
        "files": {},
    }
    for name in RAW_FILES:
        path = ROOT / "data" / "raw" / name
        data_manifest["files"][name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("data/raw/DATA_MANIFEST.json", json.dumps(data_manifest, indent=2))
    print(json.dumps({"target": str(TARGET), "files": len(files) + 1, "size_bytes": TARGET.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
