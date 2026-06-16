from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict

AL_SET = [1, 2, 4, 8, 16]


@dataclass(frozen=True)
class ChannelConfig:
    mode: str = "stable"  # stable, decreasing, snr
    mean_cqi: float = 8.0
    start_cqi: float = 12.0
    degradation_rate_per_second: float = 0.8
    cqi_noise_std: float = 1.0
    tx_power_dbm: float = 23.0
    path_loss_db_start: float = 120.0
    path_loss_increase_db_per_second: float = 0.0
    noise_power_dbm: float = -100.0
    fading_noise_std_db: float = 2.0


@dataclass(frozen=True)
class PdcchCurveConfig:
    cqi_1pct: Dict[int, float] = field(default_factory=lambda: {1: 13.0, 2: 10.0, 4: 7.0, 8: 4.0, 16: 1.0})
    target: float = 0.01
    slope: float = 1.0
    p_min: float = 1e-5
    p_max: float = 0.95

    def shifted(self, delta: float) -> "PdcchCurveConfig":
        return replace(self, cqi_1pct={al: v + delta for al, v in self.cqi_1pct.items()})


@dataclass(frozen=True)
class OllaConfig:
    target_pdcch_error: float = 0.01
    down_step: float = 0.05
    cqi_olla_min: float = -6.0
    cqi_olla_max: float = 6.0
    naive_nack_is_failure: bool = False
    suspicious_nack_before_ack: float = 0.4
    suspicious_last_nack_with_ack: float = 0.1
    suspicious_all_nack: float = 0.5

    @property
    def up_step(self) -> float:
        return self.down_step * self.target_pdcch_error / (1.0 - self.target_pdcch_error)


@dataclass(frozen=True)
class SimConfig:
    sim_id: str = "sim"
    slot_duration_ms: float = 0.5
    total_time_s: float = 10.0
    tdd_pattern: str = "DDDUDDUDDU"
    csi_period_ms: float = 50.0
    csi_report_delay_slots: int = 0
    csi_error_std: float = 0.0
    pdsch_fail_prob: float = 0.10
    cce_cost_weight: float = 0.0
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    olla: OllaConfig = field(default_factory=OllaConfig)
    pdcch_curve: PdcchCurveConfig = field(default_factory=PdcchCurveConfig)

    @property
    def num_slots(self) -> int:
        return int(round(self.total_time_s * 1000.0 / self.slot_duration_ms))
