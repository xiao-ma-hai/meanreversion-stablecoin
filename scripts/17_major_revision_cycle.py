from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_major_revision_cycle


if __name__ == "__main__":
    args, config = parse_script_args(
        "Run margin-constrained mixed-OU, invariant-margin audit, and latent-noise benchmark"
    )
    run_major_revision_cycle(config, args.force, args.dry_run)
