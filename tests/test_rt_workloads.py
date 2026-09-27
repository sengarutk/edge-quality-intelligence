"""Tests for workload timelines (src/experiments/workloads.py)."""

from pathlib import Path

import numpy as np
import pytest

from src.experiments.workloads import EventFamily, WorkloadConfig, build_timeline, load_workload

SCENARIOS = sorted(Path("configs/scenarios").glob("*.yaml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_every_workload_builds_for_all_units(path):
    wl = load_workload(path)
    for unit in (0, 6, 100, 106, 200, 206):
        tl = build_timeline(wl, unit)
        assert tl.n == wl.total_steps


def test_timeline_is_deterministic_per_unit():
    wl = load_workload("configs/scenarios/multimodal_faults.yaml")
    a, b = build_timeline(wl, 5), build_timeline(wl, 5)
    assert np.array_equal(a.actionable, b.actionable) and np.array_equal(a.vision_source, b.vision_source)
    assert not np.array_equal(a.actionable, build_timeline(wl, 6).actionable)


def test_glitches_are_never_ground_truth():
    tl = build_timeline(load_workload("configs/scenarios/transient_glitches.yaml"), 0)
    assert (tl.vision_source == 2).sum() > 0 and not tl.actionable.any()


def test_events_inside_maintenance_are_not_actionable():
    tl = build_timeline(load_workload("configs/scenarios/state_transitions.yaml"), 0)
    maint = tl.machine_state == "MAINTENANCE"
    assert tl.mechanical[maint].any() and not tl.actionable[maint].any()


def test_unplaceable_workload_raises():
    wl = WorkloadConfig(name="x", description="d", total_steps=200,
                        events=[EventFamily(kind="vision_defect", count=5, min_len=150, max_len=150)])
    with pytest.raises(RuntimeError):
        build_timeline(wl, 0)
