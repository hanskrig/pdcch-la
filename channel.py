from __future__ import annotations

import numpy as np
from config import ChannelConfig, SimConfig


def snr_to_cqi(snr_db: np.ndarray) -> np.ndarray:
    return np.clip((snr_db + 6.0) / 2.0, 0.0, 15.0)


def generate_true_cqi(config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    n = config.num_slots
    t = np.arange(n) * config.slot_duration_ms / 1000.0
    ch = config.channel
    if ch.mode == "stable":
        cqi = ch.mean_cqi + rng.normal(0.0, ch.cqi_noise_std, n)
    elif ch.mode == "decreasing":
        cqi = ch.start_cqi - ch.degradation_rate_per_second * t + rng.normal(0.0, ch.cqi_noise_std, n)
    elif ch.mode == "snr":
        path_loss = ch.path_loss_db_start + ch.path_loss_increase_db_per_second * t
        snr = ch.tx_power_dbm - path_loss - ch.noise_power_dbm + rng.normal(0.0, ch.fading_noise_std_db, n)
        cqi = snr_to_cqi(snr)
    else:
        raise ValueError(f"Unsupported channel mode {ch.mode}")
    return np.clip(cqi, 0.0, 15.0)


def update_illa_cqi(config: SimConfig, true_cqi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = config.num_slots
    illa = np.zeros(n, dtype=float)
    ul_slots = [i for i, x in enumerate((config.tdd_pattern * ((n // len(config.tdd_pattern)) + 1))[:n]) if x == "U"]
    report_times_ms = np.arange(0.0, config.total_time_s * 1000.0 + 1e-9, config.csi_period_ms)
    report_slots = []
    for ms in report_times_ms:
        desired = int(np.ceil(ms / config.slot_duration_ms)) + config.csi_report_delay_slots
        candidates = [u for u in ul_slots if u >= desired]
        if candidates:
            report_slots.append(candidates[0])
    current = round(true_cqi[0])
    report_set = set(report_slots)
    for slot in range(n):
        if slot in report_set:
            measured = true_cqi[slot] + rng.normal(0.0, config.csi_error_std)
            current = round(float(np.clip(measured, 0.0, 15.0)))
        illa[slot] = current
    return illa
