import numpy as np

from meanrev_stablecoin.major_revision import invariant_margin_audit


def test_invariant_margin_gate_rejects_gross_scale_mismatch():
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 0.001, 10_000)
    audit = invariant_margin_audit(
        {"sample": data},
        {
            "matched": (0.0, 0.001, "test"),
            "too_wide": (0.0, 0.006, "test"),
        },
    ).set_index("model")
    assert bool(audit.loc["matched", "stationary_margin_gate"])
    assert not bool(audit.loc["too_wide", "stationary_margin_gate"])
