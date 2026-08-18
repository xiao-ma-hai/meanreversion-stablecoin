# Copula-based stablecoin mean-reversion project

This project implements the two methodology specifications in the parent
directory. All paths are resolved relative to this project root; the raw CSV is
treated as immutable and audited before any estimation step.

## Environment

The user authorized one new named environment while protecting all existing
environments. The verified interpreter is:

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe
```

The local Conda solvers stalled at index resolution, so the environment was
created by an offline, read-only copy of an existing Python 3.11 scientific
stack into the new target, followed by installing PyYAML, pytest, and arch only
in the new target. `environment.yml` is the canonical clean rebuild
specification; `environment.from-history.yml` and `environment.lock.yml` are
snapshots of the verified runtime. The project package is installed only in
the new target. No pre-existing environment was modified or deleted.

## Reproduce

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\99_run_all.py --config configs\base.yml --sample baseline
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\99_run_all.py --config configs\robustness.yml --sample extended
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\99_run_all.py --config configs\robustness.yml --sample full
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\15_verify_theory_derivations.py
```

The one-click baseline entry runs `pytest` first, audits the immutable raw
file, reuses validated cached estimates unless `--force` is supplied, and
writes the joint test/pipeline result to `output/manifest.json`.

Every numbered script supports `--force`, `--dry-run`, and
`--sample baseline|extended|full`. Final tables are CSV, figures are emitted as
both PNG and PDF, and UTF-8 logs are kept under `output/logs`.

The standalone mathematical audit is
`manuscript/audits/theoretical_derivation_audit.tex`.
Its symbolic zero-residual checks and retained counterexample are written to
`output/models/theory_symbolic_checks.json`; the model-by-model verdicts are in
`output/tables/theory_derivation_audit.csv`.

The 2026-08-17 major-revision cycle is reproduced with:

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\17_major_revision_cycle.py --force
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\18_major_revision_path_bootstrap.py --force
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\19_build_submission_paper.py
```

The 2026-08-18 tick-discrete, three-stablecoin cycle supersedes the old
continuous-density LR/bootstrap evidence and is reproduced with:

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\28_second_third_round_cycle.py
```

The one-click command runs the following stages in order:

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\20_audit_tick_and_ties.py
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\25_three_coin_data_audit.py
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\21_interval_likelihood_research.py
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\26_cross_stablecoin_comparison.py
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\27_frequency_semigroup_test.py
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\19_build_submission_paper.py
```

The active likelihood is the bounded nearest-tick conditional interval
approximation. A full grid filter is implemented for validation samples, but
production-scale filtered estimation remains an explicit stage gate.

Build the separate code-and-output replication archive with:

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\30_build_replication_package.py
```

The resulting `submission/replication/` ZIP excludes the six large raw CSVs
but contains their required filenames, byte sizes, and SHA-256 hashes.

## Submission files

The upload-ready Overleaf package is
`submission/copula_mean_reversion_2026-08-18_overleaf.zip`. After upload,
select XeLaTeX (TeX Live 2025) and compile `main.tex`. The identical unpacked
source is under `submission/overleaf/`; this is the only active submission
directory. This machine has no local XeLaTeX installation, so the 2026-08-18
revision has static TeX/package verification but must be compiled on Overleaf;
the preceding compiled PDF is retained only in `archive/legacy_pdfs/`.

The generated theory and proof sections trace to
`manuscript/source/meanreversion_article_theory_revised.tex`. The active
article bibliography is `manuscript/source/article_references.bib`.

## Directory map

- `submission/`: files to upload or inspect as the current complete article.
- `manuscript/`: canonical theory sources and mathematical audit reports.
- `src/`, `scripts/`, `configs/`, `tests/`: empirical code and verification.
- `data/`: immutable raw input plus processed/interim data.
- `output/`: reproducible empirical tables, figures, model objects, and logs.
- `build/`: temporary compilation artifacts.
- `archive/`: superseded manuscripts and legacy Overleaf packages; retained
  for provenance and not used by the current build.
