from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "archive" / "legacy_manuscripts" / "empirical_results_article.tex"


def _without_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def test_empirical_tex_structure_and_references() -> None:
    text = TEX.read_text(encoding="utf-8")
    source = _without_comments(text)

    depth = 0
    for match in re.finditer(r"(?<!\\)[{}]", source):
        depth += 1 if match.group() == "{" else -1
        assert depth >= 0, f"closing brace without opening brace near offset {match.start()}"
    assert depth == 0, "unbalanced TeX braces"

    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", source):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        else:
            assert stack and stack[-1] == environment, (
                f"environment mismatch: expected {stack[-1] if stack else None}, got {environment}"
            )
            stack.pop()
    assert not stack, f"unclosed environments: {stack}"

    labels = re.findall(r"\\label\{([^}]+)\}", source)
    assert len(labels) == len(set(labels)), "duplicate TeX labels"

    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    assert figures, "expected explicit figure references"
    assert r"\graphicspath" not in source, "figure paths should be explicit"
    for figure in figures:
        assert figure.startswith("output/figures/"), f"non-explicit project figure path: {figure}"
        assert (ROOT / figure).exists(), f"missing figure: {figure}"

    referenced_outputs = set(re.findall(r"\\path\{(output/[^}]+)\}", source))
    assert referenced_outputs, "expected explicit output provenance paths"
    for relative in referenced_outputs:
        assert (ROOT / relative).exists(), f"missing referenced output: {relative}"

    required_equations = [
        r"Q_h(x,\dd y)",
        r"e^{-\lambda h}P_h(x,\dd y)",
        r"e^{-(\kappa+\lambda)h}(x-\theta)",
        r"c_h^{\mathrm{mix}}(u,v)",
    ]
    for expression in required_equations:
        assert expression in source, f"missing theory-alignment expression: {expression}"

    required_full_suite = [
        "Full-model registry and validation selection",
        r"\widehat S_m^{V}",
        "exact hidden-state predictive density",
        "output/tables/full_model_validation_ranking.csv",
        "output/tables/full_model_pairwise_validation.csv",
        "output/tables/full_mixture_incremental_scores.csv",
        "output/tables/updated_stage_gate.csv",
    ]
    for item in required_full_suite:
        assert item in source, f"missing full-suite reporting item: {item}"
