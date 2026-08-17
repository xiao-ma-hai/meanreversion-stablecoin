from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_theory_symbolic_checks_meet_their_expected_outcomes() -> None:
    checks = json.loads(
        (ROOT / "output/models/theory_symbolic_checks.json").read_text(encoding="utf-8")
    )
    assert checks
    assert all(item["pass"] for item in checks.values())
    assert checks["cir_quadratic"]["residuals"] == ["0", "0", "0"]
    assert checks["cir_secondary_change_of_variables"]["counterexample_k2_sigma1_theta1_x1"] == "4"


def test_theory_audit_retains_invalid_and_conditional_results() -> None:
    audit = pd.read_csv(ROOT / "output/tables/theory_derivation_audit.csv")
    by_object = audit.set_index("object")
    assert len(audit) == 19
    assert by_object.loc["cir_secondary_change_of_variables", "verdict"] == "incorrect"
    assert by_object.loc["cir_quadratic", "verdict"] == "finite_horizon_only_for_standard_cir"
    assert by_object.loc["cir_special_kummer", "verdict"] == "under_specified_but_some_valid_branches"
    assert by_object.loc["gamma_stationary_case", "verdict"] == "insufficient"


def test_standalone_theory_audit_states_k_negative_distinction() -> None:
    text = (ROOT / "theoretical_derivation_audit.tex").read_text(encoding="utf-8")
    assert "\\min\\{B_0,B_0+\\eta\\}\\geq0" in text
    assert "超临界平方根扩散" in text
    assert "u_\\tau=zu_{zz}" in text
    assert "“Proofs” 章节为空" in text
