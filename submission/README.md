# Current submission

Upload `copula_mean_reversion_2026-08-18_overleaf.zip` to Overleaf, choose
XeLaTeX (TeX Live 2025), and compile `main.tex`.

- `overleaf/` is the identical unpacked, self-contained source package.
- `pdf/` is reserved for the PDF downloaded after the new Overleaf compile;
  the prior PDF was archived because it predates the 2026-08-18 revision.
- Do not upload anything from `archive/`; it contains superseded versions only.

The workspace build command below regenerates both `overleaf/` and the ZIP:

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\19_build_submission_paper.py
```

The separate `replication/` directory contains a code-and-output archive plus
an immutable raw-data hash manifest; it is not an Overleaf upload.
