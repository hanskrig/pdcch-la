from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import numpy as np
from config import OllaConfig
from feedback import ACK, DTX, PDCCH_MISS, FeedbackObservation


@dataclass
class OllaUpdate:
    before: float
    delta: float
    after: float
    reason: str


class BaseOllaStrategy:
    name = "base"
    uses_hidden_truth = False

    def __init__(self, config: OllaConfig):
        self.config = config
        self.cqi_olla = 0.0

    def _apply(self, delta: float, reason: str) -> OllaUpdate:
        before = self.cqi_olla
        after = float(np.clip(before + delta, self.config.cqi_olla_min, self.config.cqi_olla_max))
        self.cqi_olla = after
        return OllaUpdate(before, after - before, after, reason)

    def update(self, obs: FeedbackObservation, hidden_group: Sequence[Dict[str, Any]] | None = None) -> OllaUpdate:
        raise NotImplementedError


class NaiveDtxAckOlla(BaseOllaStrategy):
    name = "naive_dtx_ack"

    def update(self, obs: FeedbackObservation, hidden_group: Sequence[Dict[str, Any]] | None = None) -> OllaUpdate:
        if obs.feedback_type == DTX:
            return self._apply(-self.config.down_step, obs.update_reason)
        delta = obs.num_ack * self.config.up_step
        if self.config.naive_nack_is_failure:
            delta -= obs.num_nack * self.config.down_step
        return self._apply(delta, obs.update_reason)


class SuspiciousCombinationOlla(BaseOllaStrategy):
    name = "suspicious_combination"

    def update(self, obs: FeedbackObservation, hidden_group: Sequence[Dict[str, Any]] | None = None) -> OllaUpdate:
        if obs.feedback_type == DTX:
            return self._apply(-self.config.down_step, obs.update_reason)
        if obs.num_nack == 0:
            return self._apply(obs.group_size * self.config.up_step, obs.update_reason)
        delta = obs.num_ack * self.config.up_step - self.config.down_step * obs.suspicious_score * obs.num_nack / obs.group_size
        return self._apply(delta, obs.update_reason)


class OracleOlla(BaseOllaStrategy):
    name = "oracle"
    uses_hidden_truth = True

    def update(self, obs: FeedbackObservation, hidden_group: Sequence[Dict[str, Any]] | None = None) -> OllaUpdate:
        if hidden_group is None:
            raise ValueError("OracleOlla requires hidden_group")
        delta = 0.0
        for tx in hidden_group:
            if tx["hidden_outcome"] == PDCCH_MISS:
                delta -= self.config.down_step
            elif tx["hidden_outcome"] == ACK:
                delta += self.config.up_step
        return self._apply(delta, "oracle: hidden PDCCH misses penalized, ACKs rewarded")
