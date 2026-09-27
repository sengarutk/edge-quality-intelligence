import numpy as np
import pytest

from src.experiments.decision_changes import (
    compute_decision_change_matrix,
    run_decision_change_analysis
)


def test_compute_decision_change_matrix_math():
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    mat = compute_decision_change_matrix(scores, labels, tau_baseline=0.45, tau_cct=0.65)

    assert mat["total_flips"] == 2
    assert mat["nominal_relief_count"] == 1
    assert mat["defect_escape_count"] == 1
    assert mat["defect_catch_count"] == 0

    assert mat["nominal_relief_count"] + mat["defect_escape_count"] + mat["defect_catch_count"] == mat["total_flips"]
    assert mat["nominal_relief_rate"] == 1.0 / 4.0
    assert mat["defect_escape_rate"] == 1.0 / 4.0


def test_decision_change_matrix_empty():
    mat = compute_decision_change_matrix(np.array([]), np.array([]), tau_baseline=0.5, tau_cct=0.6)
    assert mat["total_flips"] == 0
    assert mat["nominal_relief_count"] == 0


def test_run_decision_change_analysis_cost_ratio_sweep(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)

    np.savez(
        scores_dir / "bottle_patchcore_42.npz",
        image_labels=np.array([0]*80 + [1]*20),
        image_scores=np.concatenate([np.random.normal(0.2, 0.05, 80), np.random.normal(0.8, 0.05, 20)])
    )

    df = run_decision_change_analysis(
        scores_dir=str(scores_dir),
        output_dir=str(out_dir),
        cost_ratios=[5.0, 10.0, 20.0, 50.0]
    )

    assert len(df) == 4
    assert set(df["cost_ratio"].unique()) == {5.0, 10.0, 20.0, 50.0}
    assert "total_flips" in df.columns
    assert "nominal_relief_count" in df.columns
    assert "defect_escape_count" in df.columns
    assert "cwe" in df.columns