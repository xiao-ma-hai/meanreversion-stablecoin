from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_prepare

if __name__ == "__main__":
    args, config = parse_script_args("Prepare UTC state variables and samples")
    run_prepare(config, args.force, args.dry_run)

