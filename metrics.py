from __future__ import annotations

import pandas as pd
from config import AL_SET, SimConfig


def compute_summary(sim_id: str, rule_name: str, config: SimConfig, tx_log: pd.DataFrame, fb_log: pd.DataFrame) -> dict:
    total = len(tx_log)
    decoded = int(tx_log["pdcch_ok"].sum())
    miss = int((tx_log["hidden_outcome"] == "PDCCH_MISS").sum())
    nack = int((tx_log["hidden_outcome"] == "PDSCH_NACK").sum())
    succ = int((tx_log["hidden_outcome"] == "ACK").sum())
    neg = fb_log[fb_log["olla_delta"] < 0]
    false_neg = neg[(neg["group_has_pdsch_nack"] == True) & (neg["group_has_pdcch_miss"] == False)]
    observed_bits = int(fb_log["num_ack"].sum() + fb_log["num_nack"].sum())
    out = {
        "sim_id": sim_id,
        "olla_rule": rule_name,
        "channel_mode": config.channel.mode,
        "seed": None,
        "total_dl_tx": total,
        "total_feedback_reports": len(fb_log),
        "pdcch_miss_rate": miss / total if total else 0.0,
        "pdsch_bler_given_pdcch_ok": nack / decoded if decoded else 0.0,
        "total_block_error_rate": (miss + nack) / total if total else 0.0,
        "observed_dtx_rate": (fb_log["feedback_type"] == "DTX").mean() if len(fb_log) else 0.0,
        "observed_nack_rate": fb_log["num_nack"].sum() / observed_bits if observed_bits else 0.0,
        "average_al": tx_log["selected_al"].mean() if total else 0.0,
        "average_cce_per_dl": tx_log["selected_al"].mean() if total else 0.0,
        "average_cqi_olla": tx_log["cqi_olla_before"].mean() if total else 0.0,
        "final_cqi_olla": fb_log["cqi_olla_after"].iloc[-1] if len(fb_log) else 0.0,
        "false_pdcch_penalty_rate": len(false_neg) / len(neg) if len(neg) else 0.0,
        "throughput_proxy": succ - config.cce_cost_weight * tx_log["selected_al"].sum(),
    }
    for al in AL_SET:
        out[f"al{al}_fraction"] = (tx_log["selected_al"] == al).mean() if total else 0.0
    return out
