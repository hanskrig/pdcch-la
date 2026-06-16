from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pandas as pd
from config import ChannelConfig, OllaConfig, PdcchCurveConfig, SimConfig
from olla import NaiveDtxAckOlla, OracleOlla, SuspiciousCombinationOlla
from plots import generate_plots
from simulator import run_simulation


def scenarios() -> list[SimConfig]:
    base = SimConfig()
    return [
        replace(base, sim_id="A_stable", channel=ChannelConfig(mode="stable", mean_cqi=8, cqi_noise_std=1.0), pdsch_fail_prob=0.10),
        replace(base, sim_id="B_good", channel=ChannelConfig(mode="stable", mean_cqi=11, cqi_noise_std=0.5), pdsch_fail_prob=0.10),
        replace(base, sim_id="C_bad", channel=ChannelConfig(mode="stable", mean_cqi=5, cqi_noise_std=1.0), pdsch_fail_prob=0.10),
        replace(base, sim_id="D_decreasing", channel=ChannelConfig(mode="decreasing", start_cqi=12, degradation_rate_per_second=0.8, cqi_noise_std=0.5), pdsch_fail_prob=0.10),
        replace(base, sim_id="E_pure_pdsch", channel=ChannelConfig(mode="stable", mean_cqi=8, cqi_noise_std=0.5), pdsch_fail_prob=0.30, pdcch_curve=PdcchCurveConfig().shifted(-6.0)),
        replace(base, sim_id="F_pure_pdcch", channel=ChannelConfig(mode="stable", mean_cqi=7, cqi_noise_std=1.0), pdsch_fail_prob=0.0),
    ]


def strategies(cfg: OllaConfig):
    yield NaiveDtxAckOlla(cfg)
    yield NaiveDtxAckOlla(replace(cfg, naive_nack_is_failure=True))
    yield SuspiciousCombinationOlla(cfg)
    yield OracleOlla(cfg)


def run_all(output_dir: str = "results", seed: int = 1) -> pd.DataFrame:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    all_tx, all_fb, all_sum = [], [], []
    for cfg in scenarios():
        for strat in strategies(cfg.olla):
            if isinstance(strat, NaiveDtxAckOlla) and strat.config.naive_nack_is_failure:
                strat.name = "naive_nack_failure"
            tx, fb, summ = run_simulation(cfg, strat, seed)
            all_tx.append(tx); all_fb.append(fb); all_sum.append(summ)
    tx_df = pd.concat(all_tx, ignore_index=True); fb_df = pd.concat(all_fb, ignore_index=True); summary = pd.concat(all_sum, ignore_index=True)
    tx_df.to_csv(out / "transmission_log.csv", index=False)
    fb_df.to_csv(out / "feedback_log.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    generate_plots(summary, tx_df, fb_df, out)
    print(summary[["sim_id", "olla_rule", "pdcch_miss_rate", "average_al", "false_pdcch_penalty_rate", "throughput_proxy"]].to_string(index=False))
    return summary

if __name__ == "__main__":
    run_all()
