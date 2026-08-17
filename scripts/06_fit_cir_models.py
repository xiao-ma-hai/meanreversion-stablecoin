from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_cir_models

if __name__ == "__main__":
    args, config = parse_script_args("Fit exact CIR price and pressure models")
    run_cir_models(config, args.sample, args.force, args.dry_run)

