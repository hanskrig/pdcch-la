from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence
from config import OllaConfig

ACK = "ACK"
NACK = "NACK"
DTX = "DTX"
PDCCH_MISS = "PDCCH_MISS"
PDSCH_NACK = "PDSCH_NACK"


@dataclass(frozen=True)
class DlGroup:
    group_id: int
    ul_slot: int
    dl_slots: List[int]


@dataclass(frozen=True)
class FeedbackObservation:
    ul_slot: int
    dl_group_id: int
    group_size: int
    feedback_type: str
    bits: List[str]
    num_ack: int
    num_nack: int
    suspicious_score: float = 0.0
    update_reason: str = ""

    @property
    def bits_string(self) -> str:
        if self.feedback_type == DTX:
            return DTX
        return "".join("A" if b == ACK else "N" for b in self.bits)


def build_dl_groups(tdd_pattern: str, num_slots: int) -> List[DlGroup]:
    full = (tdd_pattern * ((num_slots // len(tdd_pattern)) + 1))[:num_slots]
    groups: List[DlGroup] = []
    pending: List[int] = []
    gid = 0
    for slot, sym in enumerate(full):
        if sym == "D":
            pending.append(slot)
        elif sym == "U" and pending:
            groups.append(DlGroup(gid, slot, pending))
            gid += 1
            pending = []
    return groups


def generate_feedback_observation(group: DlGroup, transmissions: Sequence[Dict[str, Any]], olla_config: OllaConfig | None = None) -> FeedbackObservation:
    if not transmissions:
        raise ValueError("transmissions must not be empty")
    pdcch_ok = [bool(tx["pdcch_ok"]) for tx in transmissions]
    if not any(pdcch_ok) or not pdcch_ok[-1]:
        obs = FeedbackObservation(group.ul_slot, group.group_id, len(transmissions), DTX, [], 0, 0)
        return classify_feedback(obs, olla_config or OllaConfig())
    bits = [ACK if tx["hidden_outcome"] == ACK else NACK for tx in transmissions]
    obs = FeedbackObservation(group.ul_slot, group.group_id, len(bits), "HARQ_BITS", bits, bits.count(ACK), bits.count(NACK))
    return classify_feedback(obs, olla_config or OllaConfig())


def classify_feedback(obs: FeedbackObservation, cfg: OllaConfig) -> FeedbackObservation:
    if obs.feedback_type == DTX:
        return replace(obs, suspicious_score=1.0, update_reason="DTX: all PDCCH missed or last PDCCH missed in simplified model")
    if obs.num_nack == 0:
        return replace(obs, suspicious_score=0.0, update_reason="clean ACK group")
    if obs.num_ack == 0:
        return replace(obs, suspicious_score=cfg.suspicious_all_nack, update_reason="all NACK: could be multiple PDSCH failures or reconstructed PDCCH misses")
    if obs.bits[-1] == ACK:
        return replace(obs, suspicious_score=cfg.suspicious_nack_before_ack, update_reason="NACK before later ACK: possible earlier DCI miss reconstructed by later DAI, but also could be PDSCH failure")
    return replace(obs, suspicious_score=cfg.suspicious_last_nack_with_ack, update_reason="last DCI was decoded because feedback bit exists; likely PDSCH failure, weak PDCCH evidence")
