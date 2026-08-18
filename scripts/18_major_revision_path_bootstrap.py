from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_major_revision_path_bootstrap


if __name__ == "__main__":
    args, config = parse_script_args("Run continuous-segment OU-null path bootstrap pilot")
    run_major_revision_path_bootstrap(config, args.force, args.dry_run)
