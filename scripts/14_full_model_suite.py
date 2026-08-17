from _bootstrap import ROOT  # noqa: F401
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_full_model_suite


if __name__ == "__main__":
    args, config = parse_script_args("Run all finite-dimensional article and methodology models")
    run_full_model_suite(config, force=args.force, dry_run=args.dry_run)
