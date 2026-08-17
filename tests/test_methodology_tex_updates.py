from __future__ import annotations

import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
METHODOLOGY = WORKSPACE / "meanreversion_stablecoin_codex_methodology.tex"
DESIGN = WORKSPACE / "stablecoin_mean_reversion_empirical_design.tex"


def _without_comments(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        cleaned.append(line[: match.start()] if match else line)
    return "\n".join(cleaned)


def _assert_balanced_tex(path: Path) -> str:
    source = _without_comments(path.read_text(encoding="utf-8"))
    depth = 0
    for match in re.finditer(r"(?<!\\)[{}]", source):
        depth += 1 if match.group() == "{" else -1
        assert depth >= 0, f"closing brace without opening brace in {path}"
    assert depth == 0, f"unbalanced braces in {path}"
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", source):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        else:
            assert stack and stack[-1] == environment, f"environment mismatch in {path}: {environment}"
            stack.pop()
    assert not stack, f"unclosed environments in {path}: {stack}"
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    assert len(labels) == len(set(labels)), f"duplicate labels in {path}"
    return source


def test_codex_methodology_contains_full_model_gate() -> None:
    source = _assert_balanced_tex(METHODOLOGY)
    for item in (
        "全模型第二阶段：不得按复杂度预先排除",
        r"\label{tab:full-model-gate}",
        "14_full_model_suite.py",
        "full_model_validation_ranking.csv",
        "full_model_pairwise_validation.csv",
        "updated_stage_gate.csv",
    ):
        assert item in source


def test_empirical_design_requires_full_registry_before_selection() -> None:
    source = _assert_balanced_tex(DESIGN)
    for item in (
        "第四步：全模型实现",
        "第五步：失败留痕与统一选择",
        "模型不得因参数多或计算慢而预先删除",
        "理论贡献归属与预测优胜模型可以不同",
    ):
        assert item in source
