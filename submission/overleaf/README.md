# Current Overleaf/XeLaTeX package

Upload the contents of this directory to Overleaf and compile main.tex with
XeLaTeX.  The numbered article structure is:

1. Introduction
2. Copula-Based Mean-Reverting Markov Processes
3. Mean-Reverting Processes from Monotone Diffusion Transforms
4. Mean-Reverting Processes from Mixed Copula Families
5. Empirical Evidence from Dollar-Pegged Stablecoins
6. Conclusion

Theory and proof sections are generated from the canonical workspace source
`manuscript/source/meanreversion_article_theory_revised.tex` by
`scripts/19_build_submission_paper.py`. Do not edit `sections/theory.tex` or
`sections/proofs.tex` directly. Empirical table fragments are generated from
the versioned CSV outputs listed in `BUILD_MANIFEST.json`.

When copying only this directory to Overleaf, no workspace-relative paths are
needed: `main.tex`, `references.bib`, all section files, tables, and figures
are self-contained here.
