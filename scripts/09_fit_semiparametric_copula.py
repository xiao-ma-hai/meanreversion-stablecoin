from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_semiparametric_copula

if __name__ == "__main__":
    args, config = parse_script_args("Fit multi-horizon semiparametric mixed Gaussian copula")
    run_semiparametric_copula(config, args.sample, args.force, args.dry_run)

