# USDT/USD stablecoin mean-reversion empirical project

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

The standalone mathematical audit is `theoretical_derivation_audit.tex`.
Its symbolic zero-residual checks and retained counterexample are written to
`output/models/theory_symbolic_checks.json`; the model-by-model verdicts are in
`output/tables/theory_derivation_audit.csv`.
