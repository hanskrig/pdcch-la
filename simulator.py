from __future__ import annotations

import numpy as np
import pandas as pd
from config import SimConfig
from channel import generate_true_cqi, update_illa_cqi
from feedback import ACK, PDCCH_MISS, PDSCH_NACK, build_dl_groups, generate_feedback_observation
from metrics import compute_summary
from pdcch import pdcch_error_prob, select_aggregation_level
from pdsch import pdsch_decode


def run_simulation(config: SimConfig, strategy, seed: int = 1):
    rng = np.random.default_rng(seed)
    true_cqi = generate_true_cqi(config, rng)
    cqi_illa = update_illa_cqi(config, true_cqi, rng)
    groups = build_dl_groups(config.tdd_pattern, config.num_slots)
    tx_rows, fb_rows = [], []
    by_group = {}
    pattern_len = len(config.tdd_pattern)
    for group in groups:
        txs = []
        for pos, slot in enumerate(group.dl_slots):
            before = strategy.cqi_olla
            eff = float(np.clip(cqi_illa[slot] + before, 0.0, 15.0))
            al = select_aggregation_level(eff, config.pdcch_curve, config.olla.target_pdcch_error)
            p_err = pdcch_error_prob(true_cqi[slot], al, config.pdcch_curve)
            pdcch_ok = bool(rng.random() > p_err)
            if pdcch_ok:
                pdsch_ok = pdsch_decode(rng, config.pdsch_fail_prob)
                outcome = ACK if pdsch_ok else PDSCH_NACK
            else:
                pdsch_ok = None
                outcome = PDCCH_MISS
            row = {"sim_id": config.sim_id, "slot": slot, "time_ms": slot * config.slot_duration_ms, "tdd_pattern_index": slot % pattern_len,
                   "dl_group_id": group.group_id, "dl_pos_in_group": pos, "true_cqi": true_cqi[slot], "cqi_illa": cqi_illa[slot],
                   "cqi_olla_before": before, "cqi_eff": eff, "selected_al": al, "pdcch_error_prob": p_err,
                   "pdcch_ok": pdcch_ok, "pdsch_ok": pdsch_ok, "hidden_outcome": outcome}
            tx_rows.append(row); txs.append(row)
        by_group[group.group_id] = txs
        obs = generate_feedback_observation(group, txs, config.olla)
        update = strategy.update(obs, txs if getattr(strategy, "uses_hidden_truth", False) else None)
        fb_rows.append({"sim_id": config.sim_id, "ul_slot": group.ul_slot, "time_ms": group.ul_slot * config.slot_duration_ms,
                        "dl_group_id": group.group_id, "group_size": obs.group_size, "feedback_type": obs.feedback_type,
                        "bits_string": obs.bits_string, "num_ack": obs.num_ack, "num_nack": obs.num_nack,
                        "suspicious_score": obs.suspicious_score, "olla_rule": strategy.name, "cqi_olla_before": update.before,
                        "olla_delta": update.delta, "cqi_olla_after": update.after, "update_reason": update.reason,
                        "group_has_pdcch_miss": any(t["hidden_outcome"] == PDCCH_MISS for t in txs),
                        "group_has_pdsch_nack": any(t["hidden_outcome"] == PDSCH_NACK for t in txs)})
    tx_df, fb_df = pd.DataFrame(tx_rows), pd.DataFrame(fb_rows)
    summary = compute_summary(config.sim_id, strategy.name, config, tx_df, fb_df)
    summary["seed"] = seed
    return tx_df, fb_df, pd.DataFrame([summary])
