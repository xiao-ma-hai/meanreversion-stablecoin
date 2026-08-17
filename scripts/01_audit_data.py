from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_audit

if __name__ == "__main__":
    args, config = parse_script_args("Audit immutable Kraken OHLCVT data")
    run_audit(config, args.force, args.dry_run)

