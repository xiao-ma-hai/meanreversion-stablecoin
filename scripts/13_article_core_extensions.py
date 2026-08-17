from _bootstrap import ROOT
from meanrev_stablecoin.cli import parse_script_args
from meanrev_stablecoin.workflow import run_article_core_extensions


if __name__ == "__main__":
    args, config = parse_script_args("Run matched MOU bootstrap and two-scale copula comparisons")
    run_article_core_extensions(config, args.force, args.dry_run)
