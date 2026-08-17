from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_descriptive

if __name__ == "__main__":
    args, config = parse_script_args("Descriptive stablecoin analysis")
    run_descriptive(config, args.sample, args.force, args.dry_run)

