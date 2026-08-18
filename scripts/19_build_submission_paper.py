from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT


PAPER = ROOT / "submission" / "overleaf"
SUBMISSION_ZIP = ROOT / "submission" / "copula_mean_reversion_2026-08-18_overleaf.zip"
THEORY_SOURCE = ROOT / "manuscript" / "source" / "meanreversion_article_theory_revised.tex"
ARTICLE_BIBLIOGRAPHY = ROOT / "manuscript" / "source" / "article_references.bib"


def between(text: str, start: str, stop: str) -> str:
    return text[text.index(start):text.index(stop)].strip() + "\n"


def demote_headings(text: str) -> str:
    text = text.replace("\\subsection{", "@@SUBSECTION@@{")
    text = text.replace("\\section{", "\\subsection{")
    return text.replace("@@SUBSECTION@@{", "\\subsubsection{")


def format_number(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def write_tables() -> None:
    tables = PAPER / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    estimates = pd.read_csv(ROOT / "output/tables/major_revision_mou_estimates.csv")
    full = estimates[estimates["fit_sample"] == "baseline_full"]
    parameter_lines = []
    for _, row in full.iterrows():
        tau = row.get("tau")
        if pd.isna(tau):
            tau = float(row["sigma"]) / (2 * float(row["kappa"])) ** 0.5
        parameter_lines.append(
            f'{row["model"]} & {format_number(row["theta"], 6)} & {format_number(tau, 6)} & '
            f'{format_number(row["kappa"], 3)} & {format_number(row.get("lambda"), 3)} & '
            f'{format_number(row["hessian_condition"], 1)} & {str(row["converged"])}' + r' \tabularnewline'
        )
    (tables / "mou_estimates.tex").write_text(
        "\\begin{tabular}{lrrrrrl}\n\\toprule\n"
        "Model & $\\theta$ & $\\tau$ & $\\kappa$ & $\\lambda$ & Hessian cond. & Conv.\\\\\n"
        "\\midrule\n" + "\n".join(parameter_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    audit = pd.read_csv(ROOT / "output/tables/invariant_margin_audit.csv")
    audit = audit[audit["sample"] == "baseline_full"]
    audit_lines = []
    for _, row in audit.iterrows():
        audit_lines.append(
            f'{row["model"]} & {format_number(10000 * row["mean"], 3)} & '
            f'{format_number(10000 * row["sd"], 3)} & {format_number(row["ks_distance"], 3)} & '
            f'{format_number(100 * row["tail_abs_gt_50bp"], 3)} & '
            f'{("Pass" if row.get("stationary_margin_gate") is True else "Fail") if row["model"] != "Data" else "--"}' + r' \tabularnewline'
        )
    (tables / "invariant_audit.tex").write_text(
        "\\begin{tabular}{lrrrrl}\n\\toprule\n"
        "Model & Mean (bp) & SD (bp) & KS & $\\Pr(|X|>50\\mathrm{bp})$ (\\%) & Gate\\\\\n"
        "\\midrule\n" + "\n".join(audit_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    scores = pd.read_csv(ROOT / "output/tables/major_revision_mou_oos_scores.csv")
    score_lines = []
    for period in ("validation_2024", "test_2025"):
        part = scores[scores["evaluation_sample"] == period]
        for _, row in part.iterrows():
            score_lines.append(
                f'{period.replace("_", " ")} & {row["model"]} & '
                f'{format_number(row["mean_log_score"], 5)} & '
                f'{format_number(row.get("mean_model_copula_log_score"), 5)}' + r' \tabularnewline'
            )
    (tables / "oos_scores.tex").write_text(
        "\\begin{tabular}{llrr}\n\\toprule\n"
        "Evaluation period & Model & Mean full log score & Mean model-Copula score\\\\\n"
        "\\midrule\n" + "\n".join(score_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    bootstrap = pd.read_csv(ROOT / "output/tables/path_level_mou_bootstrap_summary.csv")
    bootstrap_lines = []
    for _, row in bootstrap.iterrows():
        bootstrap_lines.append(
            f'{row["initialization"].replace("_", " ")} & {int(row["replications"])} & '
            f'{format_number(row["observed_lr"], 2)} & {format_number(row["critical_95_finite"], 4)} & '
            f'{format_number(row["conservative_corrected_pvalue"], 3)} & '
            f'{("Yes" if bool(row["formal_requirement_met"]) else "No")}' + r' \tabularnewline'
        )
    (tables / "path_bootstrap.tex").write_text(
        "\\begin{tabular}{lrrrrl}\n\\toprule\n"
        "Initialization & $B$ & Observed LR & Null 95\\% & Corrected $p$ & Formal gate\\\\\n"
        "\\midrule\n" + "\n".join(bootstrap_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    data_audit = pd.read_csv(ROOT / "output/tables/three_coin_data_audit.csv")
    specific = data_audit[data_audit["period"].str.contains("asset-specific")].set_index("asset")
    aligned = data_audit[data_audit["period"].str.contains("aligned common")].set_index("asset")
    audit_lines = []
    for asset in ("USDT", "USDC", "DAI"):
        audit_lines.append(
            f'{asset} & {format_number(100 * specific.loc[asset, "bar_coverage"], 1)} & '
            f'{int(aligned.loc[asset, "exact_pairs"]):,} & '
            f'{format_number(100 * aligned.loc[asset, "tie_rate"], 1)} & '
            f'{format_number(aligned.loc[asset, "min_nonzero_price_change"], 5)}' + r' \tabularnewline'
        )
    (tables / "three_coin_data_audit.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Asset & Own-calendar coverage (\\%) & Common pairs & Ties (\\%) & Min. increment\\\\\n"
        "\\midrule\n" + "\n".join(audit_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    for source_name, target_name in (
        ("full_sample_model_estimates.csv", "full_interval_estimates.tex"),
        ("training_model_estimates.csv", "training_interval_estimates.tex"),
    ):
        estimates = pd.read_csv(ROOT / "output/tables" / source_name)
        estimate_lines = []
        for _, row in estimates.iterrows():
            estimate_lines.append(
                f'{row["asset"]} & {row["model"].replace("-Interval-IFM", "")} & '
                f'{format_number(10000 * row["theta"], 3)} & {format_number(10000 * row["tau"], 3)} & '
                f'{format_number(row["kappa"], 3)} & {format_number(row["lambda"], 3)} & '
                f'{format_number(row["conditional_mean_half_life_minutes"], 1)}' + r' \tabularnewline'
            )
        (tables / target_name).write_text(
            "\\begin{tabular}{llrrrrr}\n\\toprule\n"
            "Asset & Model & $\\theta$ (bp) & $\\tau$ (bp) & $\\kappa$ & $\\lambda$ & Mean half-life (min)\\\\\n"
            "\\midrule\n" + "\n".join(estimate_lines) + "\n\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )

    cross = pd.read_csv(ROOT / "output/tables/cross_stablecoin_parameters.csv")
    cross = cross[cross["model"] == "MOU-Interval-IFM"]
    cross_lines = []
    for _, row in cross.iterrows():
        cross_lines.append(
            f'{row["asset"]} & {format_number(10000 * row["theta"], 3)} & '
            f'{format_number(10000 * row["tau"], 3)} & {format_number(row["kappa"], 3)} & '
            f'{format_number(row["lambda"], 3)} & {format_number(row["base_half_life_minutes"] / 60, 2)} & '
            f'{format_number(row["conditional_mean_half_life_minutes"] / 60, 2)} & '
            f'{format_number(100 * row["tail_abs_gt_50bp"], 3)}' + r' \tabularnewline'
        )
    (tables / "cross_stablecoin_parameters.tex").write_text(
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Asset & $\\theta$ (bp) & $\\tau$ (bp) & $\\kappa$ & $\\lambda$ & Base $H$ (h) & Mixed $H$ (h) & $|X|>50$bp (\\%)\\\\\n"
        "\\midrule\n" + "\n".join(cross_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    oos = pd.read_csv(ROOT / "output/tables/oos_scores_with_uncertainty.csv")
    oos = oos[oos["model"] == "MOU-minus-OU paired difference"]
    oos_lines = []
    for _, row in oos.iterrows():
        period = "Validation 2024" if row["evaluation_sample"] == "validation_2024" else "Post-selection 2025"
        oos_lines.append(
            f'{row["asset"]} & {period} & {format_number(row["mean_log_observation_probability"], 4)} & '
            f'{format_number(row["cluster_se_log_observation_probability"], 4)} & '
            f'[{format_number(row["ci_lower_log_observation_probability"], 4)}, '
            f'{format_number(row["ci_upper_log_observation_probability"], 4)}] & {int(row["nobs"]):,}' + r' \tabularnewline'
        )
    (tables / "oos_interval_differences.tex").write_text(
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Asset & Window & Mean $\\Delta$ score & Day-cluster SE & 95\\% CI & Pairs\\\\\n"
        "\\midrule\n" + "\n".join(oos_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    frequency = pd.read_csv(ROOT / "output/tables/frequency_parameter_stability.csv")
    frequency_lines = []
    for _, row in frequency.iterrows():
        frequency_lines.append(
            f'{row["asset"]} & {int(row["frequency_minutes"])} & {format_number(row["kappa"], 3)} & '
            f'{format_number(row["lambda"], 3)} & {format_number(row["base_half_life_minutes"] / 60, 2)} & '
            f'{format_number(row["conditional_mean_half_life_minutes"] / 60, 2)}' + r' \tabularnewline'
        )
    (tables / "frequency_parameters.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Asset & Minutes & $\\kappa$ & $\\lambda$ & Base $H$ (h) & Mixed $H$ (h)\\\\\n"
        "\\midrule\n" + "\n".join(frequency_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    transfer = pd.read_csv(ROOT / "output/tables/one_to_five_minute_kernel_test.csv")
    transfer_lines = []
    for _, row in transfer.iterrows():
        transfer_lines.append(
            f'{row["asset"]} & {format_number(row["mean_5m_own_minus_1m_transfer_log_score"], 4)} & '
            f'{format_number(row["day_cluster_se"], 4)} & '
            f'[{format_number(row["ci_lower"], 4)}, {format_number(row["ci_upper"], 4)}]' + r' \tabularnewline'
        )
    (tables / "frequency_transfer_test.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Asset & Five-minute own minus one-minute transfer & Day-cluster SE & 95\\% CI\\\\\n"
        "\\midrule\n" + "\n".join(transfer_lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )


def write_submission_zip() -> None:
    """Create an Overleaf-ready ZIP with main.tex at the archive root."""

    SUBMISSION_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path in sorted(path for path in PAPER.rglob("*") if path.is_file()):
            archive.write(source_path, source_path.relative_to(PAPER).as_posix())


def prune_unreferenced_assets() -> None:
    """Keep the upload package free of stale tables and figures from prior rounds."""

    source = "\n".join(path.read_text(encoding="utf-8") for path in PAPER.rglob("*.tex"))
    used_tables = {
        f"{name}.tex" if not name.endswith(".tex") else name
        for name in re.findall(r"\\input\{tables/([^}]+)\}", source)
    }
    used_figures = {
        Path(name).name
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}", source)
    }
    for path in (PAPER / "tables").glob("*"):
        if path.is_file() and path.name not in used_tables:
            path.unlink()
    for path in (PAPER / "figures").glob("*"):
        if path.is_file() and path.name not in used_figures:
            path.unlink()


def main() -> None:
    (PAPER / "sections").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    source = THEORY_SOURCE.read_text(encoding="utf-8")
    first = between(
        source,
        "\\section{Probability space, supports, and notation}",
        "\\section{Mean-reverting monotone transforms of diffusions}",
    )
    first = (
        "\\section{Copula-Based Mean-Reverting Markov Processes}\n"
        "\\label{sec:copula-foundations}\n\n"
        + demote_headings(first)
    )
    transforms = between(
        source,
        "\\section{Mean-reverting monotone transforms of diffusions}",
        "\\section{Mixed copula families as stationary resets}",
    ).replace(
        "\\section{Mean-reverting monotone transforms of diffusions}",
        "\\section{Mean-Reverting Processes from Monotone Diffusion Transforms}",
        1,
    )
    mixed = between(
        source,
        "\\section{Mixed copula families as stationary resets}",
        "\\section{Risk-neutral bond pricing with reset risk}",
    ).replace(
        "\\section{Mixed copula families as stationary resets}",
        "\\section{Mean-Reverting Processes from Mixed Copula Families}",
        1,
    )
    generated_notice = "% GENERATED by scripts/19_build_submission_paper.py; edit the canonical theory source.\n"
    (PAPER / "sections/theory.tex").write_text(
        generated_notice + first + "\n" + transforms + "\n" + mixed,
        encoding="utf-8",
    )
    proofs = between(
        source,
        "\\section{Detailed proofs}",
        "\\clearpage\n\\bibliographystyle",
    )
    (PAPER / "sections/proofs.tex").write_text(generated_notice + proofs, encoding="utf-8")

    shutil.copy2(ARTICLE_BIBLIOGRAPHY, PAPER / "references.bib")
    for stem in ("invariant_margin_audit", "mixed_ou_identification_slices",
                 "nonparametric_drift", "single_vs_two_scale_dependence",
                 "tie_rate_by_year", "cross_stablecoin_drift_curves",
                 "cross_stablecoin_half_lives", "kappa_lambda_by_frequency"):
        for suffix in (".pdf", ".png"):
            source_figure = ROOT / "output/figures" / f"{stem}{suffix}"
            if source_figure.exists():
                shutil.copy2(source_figure, PAPER / "figures" / source_figure.name)
    write_tables()
    prune_unreferenced_assets()
    manifest = {
        "canonical_theory_source": "manuscript/source/meanreversion_article_theory_revised.tex",
        "canonical_article_bibliography": "manuscript/source/article_references.bib",
        "generated_sections": ["sections/theory.tex", "sections/proofs.tex"],
        "empirical_sources": [
            "output/tables/tick_tie_audit.csv",
            "output/tables/three_coin_data_audit.csv",
            "output/tables/full_sample_model_estimates.csv",
            "output/tables/training_model_estimates.csv",
            "output/tables/oos_scores_with_uncertainty.csv",
            "output/tables/cross_stablecoin_parameters.csv",
            "output/tables/frequency_parameter_stability.csv",
            "output/tables/one_to_five_minute_kernel_test.csv",
        ],
        "paper_entry_point": "main.tex",
    }
    (PAPER / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_submission_zip()


if __name__ == "__main__":
    main()
