from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(summary: pd.DataFrame, tx_log: pd.DataFrame, fb_log: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    def bar(metric, name):
        ax = summary.pivot_table(index="sim_id", columns="olla_rule", values=metric).plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel(metric); ax.figure.tight_layout(); ax.figure.savefig(out / name); plt.close(ax.figure)
    bar("pdcch_miss_rate", "pdcch_miss_rate_by_rule.png")
    bar("average_al", "average_al_by_rule.png")
    bar("false_pdcch_penalty_rate", "false_pdcch_penalty_rate_by_rule.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    for (sim, rule), g in fb_log.groupby(["sim_id", "olla_rule"]):
        ax.plot(g["time_ms"] / 1000.0, g["cqi_olla_after"], label=f"{sim}:{rule}", alpha=0.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("CQI OLLA"); ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout(); fig.savefig(out / "cqi_olla_trace.png"); plt.close(fig)
    al_cols = ["al1_fraction", "al2_fraction", "al4_fraction", "al8_fraction", "al16_fraction"]
    ax = summary.groupby("olla_rule")[al_cols].mean().plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_ylabel("fraction"); ax.figure.tight_layout(); ax.figure.savefig(out / "al_distribution_by_rule.png"); plt.close(ax.figure)
