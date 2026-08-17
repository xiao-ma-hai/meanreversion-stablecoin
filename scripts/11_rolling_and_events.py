from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_rolling_and_events

if __name__ == "__main__":
    args, config = parse_script_args("Rolling parameters and first-passage events")
    run_rolling_and_events(config, args.sample, args.force, args.dry_run)

