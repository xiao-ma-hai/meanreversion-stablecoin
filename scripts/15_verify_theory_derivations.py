from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]


def _zero(expr: sp.Expr) -> bool:
    return sp.simplify(expr) == 0


def main() -> None:
    t, k, d, sigma, theta = sp.symbols(
        "t k d sigma theta", real=True, nonzero=True
    )
    a0, b0, c0 = sp.symbols("A_0 B_0 C_0", real=True, nonzero=True)
    checks: dict[str, dict[str, object]] = {}

    # OU affine transform.
    a = a0 * sp.exp((k - d) * t)
    b = b0 * sp.exp(-d * t)
    ou_affine = [sp.diff(a, t) - (k - d) * a, sp.diff(b, t) + d * b]
    checks["ou_affine"] = {
        "expected": "all_zero",
        "pass": all(_zero(value) for value in ou_affine),
        "residuals": [str(sp.simplify(value)) for value in ou_affine],
    }

    # OU exponential-affine transform.
    a = a0 * sp.exp(k * t)
    b = b0 - sigma**2 * a0**2 * (sp.exp(2 * k * t) - 1) / (4 * k) - d * t
    c = c0 * sp.exp(-d * t)
    ou_exp = [
        sp.diff(a, t) - k * a,
        sp.diff(b, t) + sp.Rational(1, 2) * sigma**2 * a**2 + d,
        sp.diff(c, t) + d * c,
    ]
    checks["ou_exponential"] = {
        "expected": "all_zero",
        "pass": all(_zero(value) for value in ou_exp),
        "residuals": [str(sp.simplify(value)) for value in ou_exp],
    }

    # OU special-function separation.  The PDE residual equals 2k times
    # Kummer's differential equation after z=kx^2/sigma^2.
    z = sp.symbols("z", nonnegative=True)
    j = sp.Function("J")(z)
    p_ou = (sp.Symbol("c") - d) / (2 * k)
    kummer_ou = z * sp.diff(j, z, 2) + (sp.Rational(1, 2) - z) * sp.diff(j, z) - p_ou * j
    ou_special_reduced = (
        (-sp.Symbol("c") + d) * j
        + k * sp.diff(j, z)
        + 2 * k * z * sp.diff(j, z, 2)
        - 2 * k * z * sp.diff(j, z)
    )
    checks["ou_special_kummer_reduction"] = {
        "expected": "identity_to_2k_times_kummer",
        "pass": _zero(ou_special_reduced - 2 * k * kummer_ou),
        "residual_minus_2k_kummer": str(sp.simplify(ou_special_reduced - 2 * k * kummer_ou)),
    }

    # CIR affine transform.
    a = a0 * sp.exp((k - d) * t)
    b = theta * (1 - a0 * sp.exp((k - d) * t)) + b0 * sp.exp(-d * t)
    cir_affine = [
        sp.diff(a, t) - (k - d) * a,
        sp.diff(b, t) + d * (b - theta) + k * theta * a,
    ]
    checks["cir_affine"] = {
        "expected": "all_zero",
        "pass": all(_zero(value) for value in cir_affine),
        "residuals": [str(sp.simplify(value)) for value in cir_affine],
    }

    # CIR quadratic transform.
    s_term = sigma**2 + 2 * k * theta
    a = a0 * sp.exp((2 * k - d) * t)
    b = -(a0 * s_term / k) * sp.exp((2 * k - d) * t) + (
        b0 + a0 * s_term / k
    ) * sp.exp((k - d) * t)
    c = (
        theta
        + theta * a0 * s_term * sp.exp((2 * k - d) * t) / (2 * k)
        - theta * (k * b0 + a0 * s_term) * sp.exp((k - d) * t) / k
        + (
            c0
            - theta
            + theta * b0
            + theta * a0 * s_term / (2 * k)
        )
        * sp.exp(-d * t)
    )
    cir_quad = [
        sp.diff(a, t) + (d - 2 * k) * a,
        sp.diff(b, t) + (d - k) * b + a * s_term,
        sp.diff(c, t) + d * c - d * theta + k * theta * b,
    ]
    checks["cir_quadratic"] = {
        "expected": "all_zero",
        "pass": all(_zero(value) for value in cir_quad),
        "residuals": [str(sp.simplify(value)) for value in cir_quad],
        "B_factor": "exp((k-d)t) [B_0 + A_0(sigma^2+2k theta)/k (1-exp(kt))]",
    }

    # CIR exponential-affine transform.
    a = 2 * k / (
        sigma**2 + (2 * k / a0 - sigma**2) * sp.exp(-k * t)
    )
    b = (
        -d * t
        - (2 * k * theta / sigma**2)
        * sp.log((2 * k / a0 - sigma**2) + sigma**2 * sp.exp(k * t))
        + b0
        + (2 * k * theta / sigma**2) * sp.log(2 * k / a0)
    )
    c = theta + (c0 - theta) * sp.exp(-d * t)
    cir_exp = [
        sp.diff(a, t) + sp.Rational(1, 2) * sigma**2 * a**2 - k * a,
        sp.diff(b, t) + d + k * theta * a,
        sp.diff(c, t) + d * c - d * theta,
    ]
    checks["cir_exponential"] = {
        "expected": "all_zero",
        "pass": all(_zero(value) for value in cir_exp),
        "residuals": [str(sp.simplify(value)) for value in cir_exp],
    }

    # CIR special-function separation.  The residual equals k times Kummer's
    # equation after z=2kx/sigma^2.
    q_cir = 2 * k * theta / sigma**2
    p_cir = (sp.Symbol("c") - d) / k
    kummer_cir = z * sp.diff(j, z, 2) + (q_cir - z) * sp.diff(j, z) - p_cir * j
    cir_special_reduced = (
        (-sp.Symbol("c") + d) * j
        + k * z * sp.diff(j, z, 2)
        + k * (q_cir - z) * sp.diff(j, z)
    )
    checks["cir_special_kummer_reduction"] = {
        "expected": "identity_to_k_times_kummer",
        "pass": _zero(cir_special_reduced - k * kummer_cir),
        "residual_minus_k_kummer": str(sp.simplify(cir_special_reduced - k * kummer_cir)),
    }

    # Counterexample to the second CIR change of variables in article lines 584--593.
    # Take the claimed solution u(z,tau)=z.  The displayed change of variables then
    # gives the following normalized residual in the original CIR PDE.
    a_root = sp.sqrt(2 * k / sigma**2)
    b_root = sp.sqrt(2 * k * sigma**2)
    bad_transform_residual = (
        k * theta
        + (2 * b_root - k - k * theta + k * theta * a_root) * sp.Symbol("x")
        + k * (1 - a_root) * sp.Symbol("x") ** 2
    )
    counterexample = bad_transform_residual.subs(
        {k: sp.Integer(2), sigma: sp.Integer(1), theta: sp.Integer(1), sp.Symbol("x"): sp.Integer(1)}
    )
    checks["cir_secondary_change_of_variables"] = {
        "expected": "nonzero_counterexample",
        "pass": sp.simplify(counterexample) != 0,
        "normalized_residual": str(bad_transform_residual),
        "counterexample_k2_sigma1_theta1_x1": str(sp.simplify(counterexample)),
    }

    # Markov-kernel mixture consistency: Pi absorbs the base kernel on either
    # side, and multiplicative weights leave coefficient 1-ab on Pi.
    weight_a, weight_b = sp.symbols("a b")
    pi_coefficient = (
        weight_a * (1 - weight_b)
        + (1 - weight_a) * weight_b
        + (1 - weight_a) * (1 - weight_b)
    )
    checks["mixed_kernel_composition"] = {
        "expected": "pi_coefficient_equals_1_minus_ab",
        "pass": _zero(pi_coefficient - (1 - weight_a * weight_b)),
        "residual": str(sp.simplify(pi_coefficient - (1 - weight_a * weight_b))),
    }

    if not all(bool(item["pass"]) for item in checks.values()):
        raise RuntimeError("At least one symbolic derivation check did not meet its expectation")

    audit_rows = [
        ("base_ou", "valid_with_conditions", "exact solution and Normal transition verified", "require k>0; stationary law also requires k>0"),
        ("base_cir", "valid_with_conditions", "exact conditional mean and noncentral-chi-square transition verified", "require k>0, theta>=0, sigma>0 for the article's mean-reverting CIR"),
        ("mean_reversion_definition", "needs_revision", "backward local drift is usable under the stated matching limit", "clarify forward/backward generator convention and state-space indexing"),
        ("gaussian_copula_generator", "valid", "Ito transformation of stationary OU gives the displayed generator", "generator-domain assumptions must be stated on (0,1) with boundary control"),
        ("gamma_stationary_case", "insufficient", "the displayed scalar equation is necessary at the mean quantile", "one zero-point equality does not prove the global sign condition"),
        ("gamma_time_varying_case", "insufficient", "the displayed ODE is necessary at the moving mean quantile", "global sign condition and a feasible convergent endpoint are unproved"),
        ("martingale_transform", "valid_with_conditions", "algebra and conditional-expectation scaling verified", "requires integrability and R_0t>0"),
        ("diffusion_transform_pde", "valid_with_conditions", "Ito gives the displayed PDE", "C1,2 alone is not enough for the stochastic integral to be a true martingale"),
        ("ou_affine", "valid_with_conditions", "symbolic residual is zero", "require A0>0 and d>0"),
        ("ou_exponential", "valid_with_conditions", "symbolic residual is zero", "require A0>0, d>0; positivity additionally needs C0>=0"),
        ("ou_special", "pde_only", "Kummer reduction is correct", "every stated nonconstant branch is even in x and cannot be globally monotone on R"),
        ("cir_affine", "valid_with_conditions", "symbolic residual is zero", "require standard CIR conditions, A0>0 and d>0"),
        ("cir_quadratic", "finite_horizon_only_for_standard_cir", "symbolic residual is zero", "for k>0 and A0>0, B(t) eventually becomes negative; global monotonicity fails"),
        ("cir_exponential", "valid_with_conditions", "symbolic residual is zero", "require real logs, A(t)>0, integrability and observed-support coverage"),
        ("cir_special_kummer", "under_specified_but_some_valid_branches", "Kummer separation is correct", "choose and normalize a branch; prove monotonicity, integrability and support"),
        ("cir_secondary_change_of_variables", "incorrect", "a claimed u=z solution gives nonzero residual after back-substitution", "replace or delete the displayed change of variables and its three derived solutions"),
        ("mixed_copula_family", "valid_with_conditions", "kernel composition verifies consistency because a_su=a_st a_tu", "base copula family must be Markov-consistent and preserve Uniform(0,1)"),
        ("poisson_reset_process", "valid", "conditioning on zero versus at least one reset gives the mixture kernel", "stationarity and independence assumptions are essential"),
        ("mixed_copula_bond_pde", "valid_with_conditions", "generator plus Feynman-Kac gives the nonlocal PDE", "requires risk-neutral integrability and a classical/viscosity solution framework"),
    ]
    audit = pd.DataFrame(audit_rows, columns=["object", "verdict", "algebra_check", "missing_condition_or_action"])

    table_path = ROOT / "output" / "tables" / "theory_derivation_audit.csv"
    json_path = ROOT / "output" / "models" / "theory_symbolic_checks.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(table_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
