import os
import numpy as np
import pandas as pd
import pytest

from src.visualization.publication_plots import (
    plot_fa_vs_md_tradeoff,
    plot_tpr_vs_alert_budget,
    plot_cost_weighted_error_curves,
    plot_operator_review_overload
)


@pytest.fixture
def mock_plot_data():
    np.random.seed(42)
    y = np.array([0]*100 + [1]*50)
    s_pc = np.concatenate([np.random.normal(0.2, 0.05, 100), np.random.normal(0.8, 0.05, 50)])
    s_pd = np.concatenate([np.random.normal(0.3, 0.08, 100), np.random.normal(0.7, 0.08, 50)])
    s_ae = np.concatenate([np.random.normal(0.4, 0.15, 100), np.random.normal(0.6, 0.15, 50)])
    return {
        "patchcore": (y, s_pc),
        "padim": (y, s_pd),
        "autoencoder": (y, s_ae)
    }


def test_plot_fa_vs_md_tradeoff(tmp_path, mock_plot_data):
    out_file = str(tmp_path / "fa_vs_md.png")
    plot_fa_vs_md_tradeoff(mock_plot_data, out_file)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0


def test_plot_tpr_vs_alert_budget(tmp_path, mock_plot_data):
    out_file = str(tmp_path / "tpr_budget.png")
    plot_tpr_vs_alert_budget(mock_plot_data, out_file)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0


def test_plot_cost_weighted_error_curves(tmp_path, mock_plot_data):
    out_file = str(tmp_path / "cwe_curves.png")
    plot_cost_weighted_error_curves(mock_plot_data, out_file)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0


def test_plot_operator_review_overload(tmp_path):
    overload_df = pd.DataFrame([
        {"method": "patchcore", "defect_prior": 0.01, "overload_probability": 0.02},
        {"method": "patchcore", "defect_prior": 0.05, "overload_probability": 0.05},
        {"method": "padim", "defect_prior": 0.01, "overload_probability": 0.10},
        {"method": "padim", "defect_prior": 0.05, "overload_probability": 0.20},
        {"method": "autoencoder", "defect_prior": 0.01, "overload_probability": 0.40},
        {"method": "autoencoder", "defect_prior": 0.05, "overload_probability": 0.80},
    ])
    out_file = str(tmp_path / "overload.png")
    plot_operator_review_overload(overload_df, out_file)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0
