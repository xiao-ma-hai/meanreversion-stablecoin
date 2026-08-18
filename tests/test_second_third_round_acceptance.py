from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_three_coin_common_sample_is_fixed_and_nontrivial() -> None:
    config = json.loads((ROOT / "configs/common_three_coin_sample.json").read_text(encoding="utf-8"))
    assert config["assets"] == ["USDT", "USDC", "DAI"]
    assert config["common_exact_pairs"] == 229_484
    assert config["quality_gate"]["passed"]
    assert config["selection_rule"].endswith("no imputation")


def test_ties_are_material_for_every_asset() -> None:
    audit = pd.read_csv(ROOT / "output/tables/three_coin_data_audit.csv")
    aligned = audit[audit["period"].str.contains("aligned common")].set_index("asset")
    assert set(aligned.index) == {"USDT", "USDC", "DAI"}
    assert (aligned["tie_rate"] > 0.10).all()


def test_full_and_training_estimates_are_separate() -> None:
    full = pd.read_csv(ROOT / "output/tables/full_sample_model_estimates.csv")
    training = pd.read_csv(ROOT / "output/tables/training_model_estimates.csv")
    assert set(full["fit_sample"]) == {"full_2021_2025"}
    assert set(training["fit_sample"]) == {"training_2021_2023"}
    assert (full["pairs_used"] == full["exact_pairs_available"]).all()
    assert (training["pairs_used"] == training["exact_pairs_available"]).all()
    assert set(full["likelihood_type"]) == {"conditional_interval_approximation"}


def test_oos_scores_use_frozen_training_parameters_and_report_uncertainty() -> None:
    scores = pd.read_csv(ROOT / "output/tables/oos_scores_with_uncertainty.csv")
    assert set(scores["fit_sample"]) == {"training_2021_2023 frozen"}
    paired = scores[scores["model"] == "MOU-minus-OU paired difference"]
    assert len(paired) == 6
    assert np.isfinite(paired["cluster_se_log_observation_probability"]).all()
    assert (paired["ci_lower_log_observation_probability"] < paired["ci_upper_log_observation_probability"]).all()


def test_frequency_pretest_is_retained_as_a_failed_formal_gate() -> None:
    transfer = pd.read_csv(ROOT / "output/tables/one_to_five_minute_kernel_test.csv")
    assert set(transfer["asset"]) == {"USDT", "USDC", "DAI"}
    assert not transfer["formal_semigroup_gate"].astype(bool).any()
    assert (transfer["ci_lower"] > 0).all()


def test_revised_theory_contains_second_round_quantifier_and_spectral_repairs() -> None:
    theory = (ROOT / "manuscript/source/meanreversion_article_theory_revised.tex").read_text(encoding="utf-8")
    for marker in (
        "rev:def-initial-law-mr",
        "rev:R-contraction",
        "rev:prop-mixed-weight-necessity",
        "rev:thm-mixed-spectrum",
        "Right continuity alone does not exclude",
        "Strict acceleration of conditional-mean decay",
    ):
        assert marker in theory


def test_current_submission_withdraws_old_continuous_bootstrap_evidence() -> None:
    empirical = (ROOT / "submission/overleaf/sections/empirical.tex").read_text(encoding="utf-8")
    main = (ROOT / "submission/overleaf/main.tex").read_text(encoding="utf-8")
    assert "Empirical Evidence from Dollar-Pegged Stablecoins" in empirical
    assert "continuous-density LR calculation is therefore withdrawn" in empirical
    assert "tables/path_bootstrap.tex" not in empirical
    assert "USDC/USD" in main and "DAI/USD" in main
