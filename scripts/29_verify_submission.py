from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from _bootstrap import ROOT


PAPER = ROOT / "submission" / "overleaf"
PACKAGE = ROOT / "submission" / "copula_mean_reversion_2026-08-18_overleaf.zip"


def uncommented(text: str) -> str:
    lines = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def main() -> None:
    tex_files = sorted(PAPER.rglob("*.tex"))
    sources = {path: uncommented(path.read_text(encoding="utf-8")) for path in tex_files}
    combined = "\n".join(sources.values())
    failures: list[str] = []

    depth = 0
    for match in re.finditer(r"(?<!\\)[{}]", combined):
        depth += 1 if match.group() == "{" else -1
        if depth < 0:
            failures.append(f"closing brace without opening near offset {match.start()}")
            break
    if depth != 0:
        failures.append(f"unbalanced brace depth: {depth}")

    for path, source in sources.items():
        stack: list[str] = []
        for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", source):
            action, environment = match.groups()
            if action == "begin":
                stack.append(environment)
            elif not stack or stack.pop() != environment:
                failures.append(f"environment mismatch in {path.relative_to(PAPER)}")
                break
        if stack:
            failures.append(f"unclosed environments in {path.relative_to(PAPER)}: {stack}")

    labels = re.findall(r"\\label\{([^}]+)\}", combined)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    references = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", combined))
    undefined_references = sorted(references - set(labels))
    if duplicate_labels:
        failures.append(f"duplicate labels: {duplicate_labels}")
    if undefined_references:
        failures.append(f"undefined references: {undefined_references}")

    bibliography = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
    citations: set[str] = set()
    for group in re.findall(r"\\cite\w*(?:\[[^]]*\])?\{([^}]+)\}", combined):
        citations.update(key.strip() for key in group.split(","))
    missing_citations = sorted(citations - bib_keys)
    if missing_citations:
        failures.append(f"citations missing from bibliography: {missing_citations}")

    for relative in re.findall(r"\\(?:input|include)\{([^}]+)\}", combined):
        path = PAPER / (relative if relative.endswith(".tex") else f"{relative}.tex")
        if not path.exists():
            failures.append(f"missing input: {relative}")
    for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", combined):
        if not (PAPER / relative).exists():
            failures.append(f"missing figure: {relative}")

    with zipfile.ZipFile(PACKAGE) as archive:
        names = set(archive.namelist())
    required_zip = {path.relative_to(PAPER).as_posix() for path in PAPER.rglob("*") if path.is_file()}
    missing_zip = sorted(required_zip - names)
    extra_zip = sorted(names - required_zip)
    if missing_zip:
        failures.append(f"ZIP missing files: {missing_zip}")
    if extra_zip:
        failures.append(f"ZIP has stale files: {extra_zip}")
    if "main.tex" not in names:
        failures.append("main.tex is not at ZIP root")

    report = {
        "tex_files": len(tex_files),
        "labels": len(labels),
        "references": len(references),
        "citations": len(citations),
        "bibliography_entries": len(bib_keys),
        "zip_entries": len(names),
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
