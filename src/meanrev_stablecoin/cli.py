from __future__ import annotations

import argparse
from dataclasses import dataclass

from .data_io import load_config


@dataclass(frozen=True)
class ScriptArgs:
    config_path: str
    sample: str
    force: bool
    dry_run: bool


def parse_script_args(description: str) -> tuple[ScriptArgs, dict]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/base.yml")
    parser.add_argument("--sample", choices=["baseline", "extended", "full"], default="baseline")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args()
    args = ScriptArgs(ns.config, ns.sample, ns.force, ns.dry_run)
    return args, load_config(ns.config)

