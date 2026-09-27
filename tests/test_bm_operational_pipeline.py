import os
import glob
import numpy as np
import pandas as pd
import pytest

from scripts.run_operational_eval import main as run_operational_eval_main
from scripts.generate_operational_plots import main as generate_operational_plots_main


def test_operational_pipeline_e2e(tmp_path, monkeypatch):
    scores_dir = str(tmp_path / "scores")
    output_dir = str(tmp_path / "output")
    figures_dir = str(tmp_path / "output" / "figures" / "operational")
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Create synthetic .npz score archives for 2 categories x 3 methods x 2 seeds
    np.random.seed(42)
    categories = ["bottle", "cable"]
    methods = ["patchcore", "padim", "autoencoder"]
    seeds = [42, 123]

    for cat in categories:
        for m in methods:
            for s in seeds:
                y = np.array([0]*60 + [1]*20)
                if m == "patchcore":
                    scores = np.concatenate([np.random.normal(0.2, 0.05, 60), np.random.normal(0.85, 0.05, 20)])
                elif m == "padim":
                    scores = np.concatenate([np.random.normal(0.3, 0.08, 60), np.random.normal(0.75, 0.08, 20)])
                else:
                    scores = np.concatenate([np.random.normal(0.4, 0.12, 60), np.random.normal(0.65, 0.12, 20)])

                pixel_amaps = np.zeros((80, 256, 256), dtype=np.float32)
                gt_masks = np.zeros((80, 256, 256), dtype=np.float32)
                gt_masks[60:, 50:100, 50:100] = 1.0

                npz_path = os.path.join(scores_dir, f"{cat}_{m}_{s}.npz")
                np.savez_compressed(
                    npz_path,
                    image_labels=y,
                    image_scores=scores,
                    pixel_amaps=pixel_amaps,
                    ground_truth_masks=gt_masks
                )

    # 2. Run run_operational_eval
    monkeypatch.setattr("sys.argv", [
        "run_operational_eval.py",
        "--scores-dir", scores_dir,
        "--output-dir", output_dir,
        "--n-stream", "2000"
    ])
    run_operational_eval_main()

    tables_dir = os.path.join(output_dir, "tables")
    out_csv = os.path.join(tables_dir, "operational_results.csv")
    out_md = os.path.join(tables_dir, "operational_results.md")
    out_tex = os.path.join(tables_dir, "operational_results.tex")

    assert os.path.exists(out_csv)
    assert os.path.exists(out_md)
    assert os.path.exists(out_tex)

    df = pd.read_csv(out_csv)
    assert len(df) == len(categories) * len(methods)
    assert "tpr_at_5_mean" in df.columns
    assert "cwe_r10_mean" in df.columns

    # 3. Run generate_operational_plots
    monkeypatch.setattr("sys.argv", [
        "generate_operational_plots.py",
        "--scores-dir", scores_dir,
        "--tables-dir", tables_dir,
        "--output-dir", figures_dir
    ])
    generate_operational_plots_main()

    for fig_name in [
        "fa_vs_md_tradeoff.png",
        "tpr_vs_alert_budget.png",
        "cost_weighted_error_curves.png",
        "operator_review_overload.png"
    ]:
        fig_path = os.path.join(figures_dir, fig_name)
        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 0
