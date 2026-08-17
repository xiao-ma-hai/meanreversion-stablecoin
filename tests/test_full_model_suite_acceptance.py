from __future__ import annotations

from pathlib import Path

import pandas as pd

from meanrev_stablecoin.models.full_registry import MODEL_REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def test_full_registry_is_unique_and_covers_article_nonlinear_families() -> None:
    names = [spec.model for spec in MODEL_REGISTRY]
    assert len(names) == 25
    assert len(names) == len(set(names))
    assert {
        "ExponentialTransformedOU",
        "SpecialFunctionTransformedOU",
        "QuadraticTransformedCIR",
        "ExponentialTransformedCIR",
        "SpecialFunctionTransformedCIR",
        "GaussianGammaStationary",
        "GaussianGammaTimeVarying",
        "MixedCopulaBondPDE",
    }.issubset(names)


def test_updated_stage_gate_and_terminal_statuses() -> None:
    gates = pd.read_csv(ROOT / "output/tables/updated_stage_gate.csv")
    assert gates["pass"].astype(bool).all()
    required = {
        "all_methodology_models_implemented",
        "complete_margin_copula_score_grid",
        "paired_selection_uncertainty_available",
        "failed_models_retained",
        "no_complexity_preselection",
        "test_not_used_for_selection",
    }
    assert required.issubset(set(gates["gate"]))

    status = pd.read_csv(ROOT / "output/tables/full_model_implementation_status.csv")
    assert len(status) == len(MODEL_REGISTRY)
    assert status["registry_covered"].astype(bool).all()
    assert status["implementation_status"].notna().all()
    assert status["implementation_status"].str.contains("failed|constraint|not_observable").any()
    assert status["weak_identification"].astype(bool).any()


def test_validation_selection_is_complete_and_test_free() -> None:
    ranking = pd.read_csv(ROOT / "output/tables/full_model_validation_ranking.csv")
    selected = ranking[ranking["selected_for_emphasis"].astype(bool)]
    assert len(selected) == 1
    assert selected.iloc[0]["model"] == "MarkovSwitchingOU"
    assert not ranking["test_used_for_selection"].astype(bool).any()
    assert ranking["finite_score_coverage"].min() >= 0.999

    pairwise = pd.read_csv(ROOT / "output/tables/full_model_pairwise_validation.csv")
    assert {"MOUF", "OUF", "RandomWalk"}.issubset(set(pairwise["reference_model"]))
    assert (pairwise["bootstrap_replications"] == 2000).all()

