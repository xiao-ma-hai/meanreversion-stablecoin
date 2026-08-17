from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import traceback

from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.data_io import write_json
from meanrev_stablecoin.workflow import (
    run_article_core_extensions,
    run_audit,
    run_cir_models,
    run_descriptive,
    run_forecasts,
    run_full_model_suite,
    run_mixed_models,
    run_nonparametric,
    run_ou_models,
    run_prepare,
    run_report,
    run_robustness,
    run_rolling_and_events,
    run_semiparametric_copula,
    run_transformed_models,
)


def main() -> None:
    args, config = parse_script_args("Run the complete mean-reversion empirical pipeline")
    def run_tests():
        if args.dry_run:
            return {"dry_run": True}
        junit = ROOT / "output/logs/pytest.xml"
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", f"--junitxml={junit}"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        (ROOT / "output/logs/pytest.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(f"pytest failed with return code {completed.returncode}")
        return {"returncode": completed.returncode, "junit": str(junit)}

    common = [
        ("00_pytest", run_tests),
        ("01_audit_data", lambda: run_audit(config, args.force, args.dry_run)),
        ("02_prepare_data", lambda: run_prepare(config, args.force, args.dry_run)),
        ("03_descriptive_analysis", lambda: run_descriptive(config, args.sample, args.force, args.dry_run)),
        ("04_nonparametric_drift", lambda: run_nonparametric(config, args.sample, args.force, args.dry_run)),
        ("05_fit_ou_models", lambda: run_ou_models(config, args.sample, args.force, args.dry_run)),
    ]
    if args.sample == "baseline":
        steps = common + [
            ("06_fit_cir_models", lambda: run_cir_models(config, args.sample, args.force, args.dry_run)),
            ("07_fit_transformed_models", lambda: run_transformed_models(config, args.sample, args.force, args.dry_run)),
            ("08_fit_mixed_copula_models", lambda: run_mixed_models(config, args.sample, args.force, args.dry_run)),
            ("09_fit_semiparametric_copula", lambda: run_semiparametric_copula(config, args.sample, args.force, args.dry_run)),
            ("09b_article_core_extensions", lambda: run_article_core_extensions(config, args.force, args.dry_run)),
            ("09c_full_model_suite", lambda: run_full_model_suite(config, args.force, args.dry_run)),
            ("10_forecast_comparison", lambda: run_forecasts(config, args.sample, args.force, args.dry_run)),
            ("11_rolling_and_events", lambda: run_rolling_and_events(config, args.sample, args.force, args.dry_run)),
            ("robustness", lambda: run_robustness(config, args.force, args.dry_run)),
            ("12_build_empirical_report", lambda: run_report(config, args.force, args.dry_run)),
        ]
    else:
        steps = common + [
            ("09_fit_semiparametric_copula", lambda: run_semiparametric_copula(config, args.sample, args.force, args.dry_run)),
        ]
    status = {"sample": args.sample, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "steps": [], "status": "RUNNING"}
    try:
        for name, function in steps:
            started = dt.datetime.now(dt.timezone.utc)
            result = function()
            ended = dt.datetime.now(dt.timezone.utc)
            status["steps"].append({"step": name, "status": "PASS", "seconds": (ended - started).total_seconds(), "result": result})
            write_json(status, ROOT / "output/run_status.json")
        status["status"] = "DRY_RUN" if args.dry_run else "PASS"
    except Exception as exc:
        status["status"] = "FAIL"
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        write_json(status, ROOT / "output/run_status.json")
        raise
    finally:
        status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(status, ROOT / "output/run_status.json")
        manifest_path = ROOT / "output/manifest.json"
        if manifest_path.exists() and not args.dry_run:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pytest_pass = any(step["step"] == "00_pytest" and step["status"] == "PASS" for step in status["steps"])
            manifest["pytest_status"] = "PASS" if pytest_pass else "FAIL"
            manifest["pipeline_status"] = status["status"]
            manifest["completion_status"] = "PASS" if pytest_pass and status["status"] == "PASS" else "FAIL"
            write_json(manifest, manifest_path)
    print(status["status"])


if __name__ == "__main__":
    main()
