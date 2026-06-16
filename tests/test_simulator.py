from dataclasses import replace
import pandas as pd

from config import OllaConfig, PdcchCurveConfig, SimConfig
from feedback import DlGroup, generate_feedback_observation, build_dl_groups, ACK, PDCCH_MISS
from olla import NaiveDtxAckOlla, SuspiciousCombinationOlla
from pdcch import pdcch_error_prob, select_aggregation_level
from simulator import run_simulation


def test_tdd_pattern_groups():
    assert [len(g.dl_slots) for g in build_dl_groups("DDDUDDUDDU", 10)] == [3, 2, 2]


def test_al_selection_monotonic():
    curve = PdcchCurveConfig()
    als = [select_aggregation_level(cqi, curve) for cqi in range(16)]
    assert all(later <= earlier for earlier, later in zip(als, als[1:]))


def test_pdcch_error_probability_monotonic():
    curve = PdcchCurveConfig()
    assert pdcch_error_prob(8, 4, curve) < pdcch_error_prob(6, 4, curve)
    assert pdcch_error_prob(7, 8, curve) < pdcch_error_prob(7, 4, curve)


def _tx(ok, outcome):
    return {"pdcch_ok": ok, "hidden_outcome": outcome}


def test_feedback_cases():
    g = DlGroup(0, 3, [0, 1, 2])
    assert generate_feedback_observation(g, [_tx(False, PDCCH_MISS)] * 3).feedback_type == "DTX"
    assert generate_feedback_observation(g, [_tx(True, ACK), _tx(True, ACK), _tx(False, PDCCH_MISS)]).feedback_type == "DTX"
    obs = generate_feedback_observation(g, [_tx(False, PDCCH_MISS), _tx(True, ACK), _tx(True, ACK)])
    assert obs.feedback_type == "HARQ_BITS"
    assert obs.bits == ["NACK", "ACK", "ACK"]


def test_olla_updates():
    cfg = OllaConfig()
    dtx = generate_feedback_observation(DlGroup(0, 1, [0]), [_tx(False, PDCCH_MISS)], cfg)
    ack = generate_feedback_observation(DlGroup(0, 1, [0]), [_tx(True, ACK)], cfg)
    strat = NaiveDtxAckOlla(cfg)
    assert strat.update(dtx).delta < 0
    assert strat.update(ack).delta > 0
    susp = SuspiciousCombinationOlla(cfg)
    g2 = DlGroup(0, 2, [0, 1])
    ack_nack = generate_feedback_observation(g2, [_tx(True, ACK), _tx(True, "PDSCH_NACK")], cfg)
    nack_ack = generate_feedback_observation(g2, [_tx(False, PDCCH_MISS), _tx(True, ACK)], cfg)
    assert susp.update(ack_nack).delta > dtx.suspicious_score * -cfg.down_step
    d1 = SuspiciousCombinationOlla(cfg).update(ack_nack).delta
    d2 = SuspiciousCombinationOlla(cfg).update(nack_ack).delta
    assert d2 < d1


def test_reproducibility_same_seed():
    cfg = replace(SimConfig(total_time_s=0.1), sim_id="repro")
    _, _, s1 = run_simulation(cfg, SuspiciousCombinationOlla(cfg.olla), seed=7)
    _, _, s2 = run_simulation(cfg, SuspiciousCombinationOlla(cfg.olla), seed=7)
    pd.testing.assert_frame_equal(s1, s2)
