"""Operational metrics for a stream of policy decisions with ground-truth episodes.

Definitions (used everywhere in the repository and in the paper):

* Episode: a maximal run of consecutive ground-truth defect steps.
* Alert: a decision with ``is_new_alert = True``. Alerts are what enter the operator queue.
* An alert at step t is *attributed* to episode e if ``start_e <= t < end_e + grace``;
  otherwise it is a *false alert*.
* Re-alerts: attributed alerts beyond the first one of their episode.
* False critical escalations: false alerts raised at HIGH_SEVERITY (automatic line stop).
* Routing recall: fraction of episodes with at least one non-NORMAL decision in
  ``[start_e, start_e + max_delay]``. Delay is measured to the first such decision.
* Rates per hour use the nominal sampling interval; false alerts are normalized by the
  time spent outside episode windows.

When a stream has no ground-truth episode, recall and delay are ``None`` (undefined),
never a default value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Episode:
    start: int
    end: int  # exclusive


def episodes_from_steps(defect_steps: Sequence[int]) -> List[Episode]:
    """Group ground-truth step indices into maximal consecutive runs."""
    steps = sorted(set(int(s) for s in defect_steps))
    episodes: List[Episode] = []
    for s in steps:
        if episodes and s == episodes[-1].end:
            episodes[-1] = Episode(episodes[-1].start, s + 1)
        else:
            episodes.append(Episode(s, s + 1))
    return episodes


def compute_stream_metrics(
    records: Sequence[Dict[str, Any]],
    defect_steps: Sequence[int] = (),
    frame_interval_ms: float = 1000.0 / 30.0,
    grace_frames: int = 15,
    max_delay_frames: int = 15,
    deadline_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute alert-level and latency metrics.

    Args:
        records: One dict per step, in step order, with keys ``risk_state`` (str),
            ``is_new_alert`` (bool) and optionally ``latency_ms`` (float).
        defect_steps: Ground-truth defect step indices.
        frame_interval_ms: Sampling interval used to convert counts into hourly rates.
        grace_frames: Frames after an episode ends during which alerts are still attributed to it.
        max_delay_frames: Detection window for routing recall.
        deadline_ms: Optional per-cycle deadline for the deadline-miss rate.
    """
    n = len(records)
    if n == 0:
        raise ValueError("compute_stream_metrics needs at least one record")
    hours_per_step = frame_interval_ms / 1000.0 / 3600.0
    episodes = episodes_from_steps(defect_steps)

    covered = np.zeros(n, dtype=bool)
    for ep in episodes:
        covered[ep.start:min(n, ep.end + grace_frames)] = True

    alert_steps = [i for i, r in enumerate(records) if r.get("is_new_alert")]
    non_normal = np.array([r.get("risk_state", "NORMAL") != "NORMAL" for r in records])
    false_alerts = [t for t in alert_steps if not covered[t]]
    false_high = [t for t in false_alerts if records[t].get("risk_state") == "HIGH_SEVERITY"]

    per_episode_alerts: List[int] = []
    delays: List[int] = []
    detected = 0
    for ep in episodes:
        window_end = min(n, ep.end + grace_frames)
        per_episode_alerts.append(sum(1 for t in alert_steps if ep.start <= t < window_end))
        hits = np.flatnonzero(non_normal[ep.start:min(n, ep.start + max_delay_frames + 1)])
        if hits.size:
            detected += 1
            delays.append(int(hits[0]))

    nominal_hours = float((~covered).sum()) * hours_per_step
    total_hours = n * hours_per_step
    re_alerts = int(sum(max(0, c - 1) for c in per_episode_alerts))

    out: Dict[str, Any] = {
        "total_steps": n,
        "total_hours": total_hours,
        "n_episodes": len(episodes),
        "alerts": len(alert_steps),
        "alerts_per_hour": len(alert_steps) / total_hours,
        "false_alerts": len(false_alerts),
        "false_alerts_per_hour": (len(false_alerts) / nominal_hours) if nominal_hours > 0 else None,
        "false_high_alerts": len(false_high),
        "false_high_alerts_per_hour": (len(false_high) / nominal_hours) if nominal_hours > 0 else None,
        "re_alerts": re_alerts,
        "re_alerts_per_episode": (re_alerts / len(episodes)) if episodes else None,
        "alerts_per_episode": float(np.mean(per_episode_alerts)) if episodes else None,
        "routing_recall": (detected / len(episodes)) if episodes else None,
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "median_delay_frames": float(np.median(delays)) if delays else None,
        "non_normal_fraction": float(non_normal.mean()),
    }

    lat = np.array([r["latency_ms"] for r in records if r.get("latency_ms") is not None], dtype=np.float64)
    if lat.size:
        out.update({
            "latency_mean_ms": float(lat.mean()),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_p99_ms": float(np.percentile(lat, 99)),
            "latency_max_ms": float(lat.max()),
        })
        if deadline_ms is not None:
            out["deadline_miss_rate"] = float(np.mean(lat > deadline_ms))
    return out
