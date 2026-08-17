from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_transformed_models

if __name__ == "__main__":
    args, config = parse_script_args("Fit identifiable affine transformed OU/CIR models")
    run_transformed_models(config, args.sample, args.force, args.dry_run)

