from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_nonparametric

if __name__ == "__main__":
    args, config = parse_script_args("Nonparametric drift and sign test")
    run_nonparametric(config, args.sample, args.force, args.dry_run)

