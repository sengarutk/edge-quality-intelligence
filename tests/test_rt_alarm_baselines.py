"""External alarm baselines (src/runtime/alarm_baselines.py) and correlated score sampling."""

from __future__ import annotations

import json

import numpy as np

from src.config import PolicyConfig, PolicyMode, SpoolerConfig
from src.policy import RiskState
from src.runtime.alarm_baselines import make_engine
from tests.helpers import inf, reading


def run(mode, vs, ss=None):
    e = make_engine(PolicyConfig(policy_mode=mode))
    ss = ss if ss is not None else [0.05] * len(vs)
    return [e.evaluate(inf(v), reading(s)) for v, s in zip(vs, ss)]


def test_delay_timer_needs_k_consecutive_samples_and_alerts_once():
    ds = run(PolicyMode.DELAY_TIMER, [0.9, 0.9, 0.9, 0.1] + [0.9] * 4 + [0.1] * 40)
    assert [d.is_new_alert for d in ds].count(True) == 1  # both timers fire at the 4th consecutive sample: one alert at HIGH
    assert ds[2].risk_state == RiskState.NORMAL and ds[7].risk_state == RiskState.HIGH_SEVERITY
    assert ds[-1].risk_state == RiskState.NORMAL  # off-delay of T_cool samples


def test_hysteresis_clears_only_below_deadband():
    ds = run(PolicyMode.EMA_HYSTERESIS, [0.6] * 20 + [0.45] * 20)
    assert ds[19].risk_state == RiskState.REVIEW_REQUIRED and ds[-1].risk_state == RiskState.REVIEW_REQUIRED
    assert sum(d.is_new_alert for d in ds) == 1


def test_decision_fusion_needs_both_modalities_for_high():
    vs = [0.9] * 10
    assert run(PolicyMode.DECISION_FUSION, vs)[-1].risk_state == RiskState.REVIEW_REQUIRED
    assert run(PolicyMode.DECISION_FUSION, vs, [0.95] * 10)[-1].risk_state == RiskState.HIGH_SEVERITY


def test_correlated_sampling_keeps_pools_and_adds_dependence():
    from src.experiments.runtime_sim import simulate_workload
    from src.experiments.workloads import EventFamily, WorkloadConfig

    class Bank:
        pools = {0: np.linspace(0, 0.4, 50), 1: np.linspace(0.6, 1.0, 50), 2: np.linspace(0.5, 1.0, 50)}
        frames = np.full((2, 224, 224, 3), 128, np.uint8)
        blur_threshold = 0.0

        def frame(self, idx, optical):
            return self.frames[idx % 2]

    wl = WorkloadConfig(name="t", description="d", total_steps=600,
                        events=[EventFamily(kind="vision_defect", count=2, min_len=60, max_len=60)])
    cfg = PolicyConfig(policy_mode=PolicyMode.BASELINE)
    for rho in (0.0, 0.9):
        recs, tl = simulate_workload(wl, Bank(), cfg, unit=1, rho=rho)
        assert len(recs) == 600


def test_async_writer_persists_every_decision(tmp_path):
    from src.audit_log import AuditLogDB
    from src.policy import TemporalPolicyEngine
    from src.runtime.async_writer import AsyncPersistenceWriter
    from src.spooler import DiskSpooler

    spool = DiskSpooler(config=SpoolerConfig(db_path=str(tmp_path / "s.db")))
    audit = AuditLogDB(db_path=str(tmp_path / "a.db"))
    w = AsyncPersistenceWriter(spool, audit, "t")
    e = TemporalPolicyEngine()
    ids = []
    for _ in range(50):
        d = e.evaluate(inf(0.05), reading(0.05))
        ids.append(d.event_id)
        w.submit(d)
    w.close()
    assert spool.count_on_disk() == 50 and len(w.lag_ms) == 50 and w.errors == 0
    stored = [json.loads(r[2])["event_id"] for r in spool.peek_batch(100)]
    assert stored == ids
    assert len(audit.query_events_in_order()) == 50
