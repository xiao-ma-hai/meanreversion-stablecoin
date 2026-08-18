from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import ROOT
from meanrev_stablecoin.data_io import write_csv
from meanrev_stablecoin.microstructure import ASSETS, FREQUENCIES, annual_transition_audit, load_raw_market, raw_file


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    tables = ROOT / "output" / "tables"
    figures = ROOT / "output" / "figures"
    rows = []
    for minutes in FREQUENCIES:
        for asset in ASSETS:
            path = raw_file(asset, minutes, raw_dir)
            if not path.exists():
                raise FileNotFoundError(path)
            frame = load_raw_market(path)
            rows.append(annual_transition_audit(frame, asset, minutes))
    audit = pd.concat(rows, ignore_index=True)
    write_csv(audit, tables / "tick_tie_audit.csv")

    annual = audit[(audit["period"] != "full") & (audit["price_proxy"] == "close")].copy()
    annual["year"] = annual["period"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, minutes in zip(axes, FREQUENCIES):
        part = annual[annual["frequency_minutes"] == minutes]
        for asset, group in part.groupby("asset"):
            ax.plot(group["year"], 100 * group["tie_rate"], marker="o", label=asset)
        ax.set_title(f"{minutes}-minute close")
        ax.set_xlabel("Year")
        ax.set_ylabel("Exact-transition tie rate (%)")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "tie_rate_by_year.pdf", bbox_inches="tight")
    fig.savefig(figures / "tie_rate_by_year.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {len(audit)} audit rows")


if __name__ == "__main__":
    main()
