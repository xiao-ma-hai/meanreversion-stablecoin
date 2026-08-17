"""Static integrity checks for the corrected theory deliverables.

This script is intentionally dependency-free.  It verifies coverage of every
label in the original theory range, TeX environment balance, internal
cross-references, unique labels, one end-of-document marker, the complete
25-model empirical embedding inventory, and several source corruptions that
ordinary brace/environment checks miss (control characters, lost command
backslashes, and double superscripts).  It also checks that every citation in
the revised theory resolves to exactly one entry in the companion BibTeX file.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT.parent / "meanreversion_article_main.tex"
REPORT = PROJECT / "theory_correction_comparison.tex"
REVISED = PROJECT / "meanreversion_article_theory_revised.tex"
BIBLIOGRAPHY = PROJECT / "meanreversion_theory_references.bib"

MODELS = (
    "OUF",
    "ThresholdOU_Heteroskedastic",
    "CIR_Price",
    "CIR_DepegPressure",
    "AffineTransformedOU",
    "ExponentialTransformedOU",
    "SpecialFunctionTransformedOU",
    "AffineTransformedCIR",
    "QuadraticTransformedCIR",
    "ExponentialTransformedCIR",
    "SpecialFunctionTransformedCIR",
    "MOUF",
    "MCIR",
    "SeasonalIntensityMOU",
    "JumpOU",
    "MarkovSwitchingOU",
    "NestedOU",
    "RandomWalk",
    "GaussianGammaStationary",
    "GaussianGammaTimeVarying",
    "ParametricCopulaGrid",
    "TwoScaleGaussian",
    "TwoScaleMixed",
    "FullyTimeVaryingMarginalCopula",
    "MixedCopulaBondPDE",
)


def uncommented(tex: str) -> str:
    """Remove unescaped TeX comments while preserving line structure."""

    cleaned: list[str] = []
    for line in tex.splitlines():
        match = re.search(r"(?<!\\)%", line)
        cleaned.append(line[: match.start()] if match else line)
    return "\n".join(cleaned)


def labels(tex: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", uncommented(tex))


def refs(tex: str) -> list[str]:
    return re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", uncommented(tex))


def citations(tex: str) -> list[str]:
    matches = re.findall(
        r"\\cite\w*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}", uncommented(tex)
    )
    return [key.strip() for group in matches for key in group.split(",")]


def bibliography_keys(bib: str) -> list[str]:
    return re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,", bib)


def searchable_plain(tex: str) -> str:
    """Normalize harmless TeX break hints before literal inventory checks."""

    plain = uncommented(tex)
    plain = re.sub(r"\\allowbreak(?:\{\})?\s*", "", plain)
    return plain.replace(r"\_", "_")


def environment_errors(tex: str) -> list[str]:
    tokens = re.finditer(r"\\(begin|end)\{([^}]+)\}", uncommented(tex))
    stack: list[tuple[str, int]] = []
    errors: list[str] = []
    for token in tokens:
        kind, environment = token.group(1), token.group(2)
        line = tex.count("\n", 0, token.start()) + 1
        if kind == "begin":
            stack.append((environment, line))
        elif not stack:
            errors.append(f"orphan end{{{environment}}} at line {line}")
        else:
            opened, opened_line = stack.pop()
            if opened != environment:
                errors.append(
                    f"end{{{environment}}} at line {line} closes "
                    f"begin{{{opened}}} from line {opened_line}"
                )
    errors.extend(f"unclosed begin{{{env}}} at line {line}" for env, line in stack)
    return errors


def source_corruption_errors(tex: str) -> list[str]:
    """Find byte-level and token-level defects known to break TeX compilation."""

    errors: list[str] = []
    for position, character in enumerate(tex):
        code = ord(character)
        if code < 32 and code not in {9, 10, 13}:
            line = tex.count("\n", 0, position) + 1
            errors.append(f"U+{code:04X} control character at line {line}")

    suspicious_patterns = {
        "literal qquad (missing backslash)": r"(?<!\\)\bqquad\b",
        "literal quad (missing backslash)": r"(?<!\\)\bquad\b",
        "literal label command (missing backslash)": r"(?<!\\)\blabel\{",
        "literal frac command (missing backslash)": r"(?<!\\)\b(?:dfrac|tfrac|frac)\{",
        "prime followed by a second superscript": r"'\s*\^",
    }
    clean = uncommented(tex)
    for description, pattern in suspicious_patterns.items():
        for match in re.finditer(pattern, clean):
            line = clean.count("\n", 0, match.start()) + 1
            errors.append(f"{description} at line {line}")

    if r"\mathscr" in clean and "mathrsfs" not in clean and "unicode-math" not in clean:
        errors.append(r"\mathscr used without mathrsfs or unicode-math")
    return errors


def file_checks(path: Path) -> dict[str, object]:
    tex = path.read_text(encoding="utf-8")
    found_labels = labels(tex)
    duplicate_labels = sorted(
        label for label, count in Counter(found_labels).items() if count > 1
    )
    undefined_refs = sorted(set(refs(tex)) - set(found_labels))
    end_marker = r"\end{document}"
    after_end = tex.split(end_marker, 1)[1].strip() if end_marker in tex else None
    return {
        "labels": len(found_labels),
        "duplicate_labels": duplicate_labels,
        "undefined_refs": undefined_refs,
        "environment_errors": environment_errors(tex),
        "source_corruption_errors": source_corruption_errors(tex),
        "end_document_count": tex.count(end_marker),
        "content_after_end_document": bool(after_end),
    }


def main() -> None:
    original_lines = ORIGINAL.read_text(encoding="utf-8").splitlines()
    original_theory = "\n".join(original_lines[153:674])
    original_labels = labels(original_theory)
    report_text = REPORT.read_text(encoding="utf-8")
    revised_text = REVISED.read_text(encoding="utf-8")
    bibliography_text = BIBLIOGRAPHY.read_text(encoding="utf-8")
    report_plain = searchable_plain(report_text)
    revised_plain = searchable_plain(revised_text)
    cited_keys = citations(revised_text)
    bib_keys = bibliography_keys(bibliography_text)

    results: dict[str, object] = {
        "original_theory_labels": len(original_labels),
        "original_theory_unique_labels": len(set(original_labels)),
        "original_labels_missing_from_report": sorted(
            label for label in original_labels if label not in report_plain
        ),
        "classification_terms_missing": [
            term
            for term in (
                "A：正确",
                "B：可由中文原稿完整修正",
                "C：须补充定义或条件修正",
                "D：现有框架中难以修正",
            )
            if term not in report_text
        ],
        "empirical_models_missing_from_report": [
            model for model in MODELS if model not in report_plain
        ],
        "empirical_models_missing_from_revised_theory": [
            model for model in MODELS if model not in revised_plain
        ],
        "citation_keys": sorted(set(cited_keys)),
        "citations_missing_from_bibliography": sorted(set(cited_keys) - set(bib_keys)),
        "duplicate_bibliography_keys": sorted(
            key for key, count in Counter(bib_keys).items() if count > 1
        ),
        "uncited_bibliography_keys": sorted(set(bib_keys) - set(cited_keys)),
        REPORT.name: file_checks(REPORT),
        REVISED.name: file_checks(REVISED),
    }

    failures: list[str] = []
    for key, value in results.items():
        if key.endswith("missing_from_report") or key.startswith("classification_terms_missing"):
            if value:
                failures.append(key)
        if key == "empirical_models_missing_from_revised_theory" and value:
            failures.append(key)
        if key in {"citations_missing_from_bibliography", "duplicate_bibliography_keys"} and value:
            failures.append(key)
    for filename in (REPORT.name, REVISED.name):
        checks = results[filename]
        assert isinstance(checks, dict)
        if (
            checks["duplicate_labels"]
            or checks["undefined_refs"]
            or checks["environment_errors"]
            or checks["source_corruption_errors"]
        ):
            failures.append(filename)
        if checks["end_document_count"] != 1 or checks["content_after_end_document"]:
            failures.append(filename)

    results["passed"] = not failures
    results["failures"] = failures
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
