from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_mixed_models

if __name__ == "__main__":
    args, config = parse_script_args("Fit parametric mixed OU/CIR reset models")
    run_mixed_models(config, args.sample, args.force, args.dry_run)

