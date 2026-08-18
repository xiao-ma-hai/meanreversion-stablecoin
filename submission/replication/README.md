# Replication package

`copula_mean_reversion_2026-08-18_replication.zip` contains the environment
specifications, source code, scripts, tests, configurations, active outputs,
canonical theory, and the complete Overleaf source.

The six raw Kraken CSV files are not duplicated in the ZIP because they total
hundreds of megabytes. Their required names, byte sizes, and SHA-256 hashes are
stored at `data/raw/DATA_MANIFEST.json` inside the archive. After placing the
verified files under `data/raw/`, run:

```powershell
D:\Anaconda\envs\meanreversion-stablecoin\python.exe scripts\28_second_third_round_cycle.py
```
