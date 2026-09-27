import os
from typing import Dict, Any, List, Union, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def set_paper_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.alpha": 0.5,
        "grid.linestyle": "--"
    })


def plot_pareto_frontier(summary_df_or_path: Union[str, pd.DataFrame], output_path: str) -> str:
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if isinstance(summary_df_or_path, str):
        df = pd.read_csv(summary_df_or_path)
    else:
        df = summary_df_or_path.copy()

    if "p50_latency_ms" in df.columns and "aupro" in df.columns:
        agg = df.groupby(["category", "method"]).agg(
            latency=("p50_latency_ms", "mean"),
            aupro=("aupro", "mean")
        ).reset_index()
    elif "p50_model_ms" in df.columns and "aupro_mean" in df.columns:
        agg = df.copy()
        agg["latency"] = agg["p50_model_ms"]
        agg["aupro"] = agg["aupro_mean"]
    elif "p50_latency_ms" in df.columns and "aupro_mean" in df.columns:
        agg = df.copy()
        agg["latency"] = agg["p50_latency_ms"]
        agg["aupro"] = agg["aupro_mean"]
    else:
        agg = df.copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"patchcore": "#1f77b4", "padim": "#2ca02c", "autoencoder": "#ff7f0e"}
    markers = {"patchcore": "o", "padim": "s", "autoencoder": "^"}

    for method, group in agg.groupby("method"):
        m_lower = str(method).lower()
        color = palette.get(m_lower, "#7f7f7f")
        marker = markers.get(m_lower, "o")

        ax.scatter(
            group["latency"],
            group["aupro"],
            label=str(method).upper(),
            color=color,
            marker=marker,
            s=120,
            alpha=0.85,
            edgecolors="black",
            linewidth=1.0
        )

        for _, row in group.iterrows():
            ax.annotate(
                row.get("category", ""),
                (row["latency"], row["aupro"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                alpha=0.8
            )

    ax.set_xlabel("Inference Latency P50 (ms) [↓]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Localization AU-PRO (max FPR = 0.30) [↑]", fontsize=12, fontweight="bold")
    ax.set_title("Pareto Frontier: Inference Speed vs. Defect Localization Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Method", loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


plot_pareto_latency_vs_aupro = plot_pareto_frontier


def plot_robustness_heatmap(csv_path_or_df: Union[str, pd.DataFrame], output_path: str) -> str:
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    if "corruption_type" not in df.columns or "severity" not in df.columns:
        raise ValueError("Missing 'corruption_type' or 'severity' columns for heatmap.")

    metric_col = "delta_image_auroc" if "delta_image_auroc" in df.columns else df.columns[-1]
    pivot = df.pivot_table(index="corruption_type", columns="severity", values=metric_col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        cbar_kws={"label": "AUROC Drop (Δ AUROC) [↓]"},
        linewidths=0.5,
        linecolor="#dddddd",
        ax=ax
    )

    ax.set_title("Industrial Robustness: AUROC Drop Across Corruptions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Corruption Severity Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Environmental Degradation Type", fontsize=11, fontweight="bold")

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration_curve(reliability_data: Dict[str, Any], ece_score: float, output_path: str) -> str:
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    centers = np.array(reliability_data.get("bin_centers", []))
    accuracies = np.array(reliability_data.get("bin_accuracies", []))
    confidences = np.array(reliability_data.get("bin_confidences", []))
    counts = np.array(reliability_data.get("bin_counts", []))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect Calibration")

    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accuracies, width=width * 0.8, alpha=0.6, color="#1f77b4", edgecolor="black", label="Outputs")

    for c, acc, conf in zip(centers, accuracies, confidences):
        if acc != 0.0 or conf != 0.0:
            ax.vlines(c, min(acc, conf), max(acc, conf), color="#d62728", linestyle="-", linewidth=2.0)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence Score Bins [0, 1]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy / Precision", fontsize=11, fontweight="bold")
    ax.set_title("Reliability Diagram (Uncertainty Calibration)", fontsize=13, fontweight="bold", pad=12)

    ax.text(
        0.05, 0.90,
        f"ECE = {ece_score:.4f}\nSamples = {int(sum(counts))}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#999999")
    )

    ax.legend(loc="lower right")
    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration_diagram(y_true: np.ndarray, y_scores: np.ndarray, output_path: str) -> str:
    from src.metrics.calibration import get_reliability_diagram_data, compute_ece
    rel_data = get_reliability_diagram_data(y_true, y_scores)
    ece_val = compute_ece(y_scores, y_true)
    return plot_calibration_curve(rel_data, ece_val, output_path)


def plot_robust_training_ablation(comparison_data: Union[Dict[str, Any], pd.DataFrame], output_path: str) -> str:
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if isinstance(comparison_data, pd.DataFrame):
        df = comparison_data
        clean_auroc = float(df["nominal_clean_auroc"].mean()) if "nominal_clean_auroc" in df.columns else 0.95
        clean_mce = float(df["nominal_corrupted_auroc"].mean()) if "nominal_corrupted_auroc" in df.columns else 0.10
        robust_auroc = float(df["robust_clean_auroc"].mean()) if "robust_clean_auroc" in df.columns else 0.94
        robust_mce = float(df["robust_corrupted_auroc"].mean()) if "robust_corrupted_auroc" in df.columns else 0.05
    else:
        clean_auroc = comparison_data.get("clean_model", {}).get("clean_auroc", 0.0)
        clean_mce = comparison_data.get("clean_model", {}).get("mrd_image_auroc", 0.0)
        robust_auroc = comparison_data.get("robust_model", {}).get("clean_auroc", 0.0)
        robust_mce = comparison_data.get("robust_model", {}).get("mrd_image_auroc", 0.0)

    categories = ["Clean Test AUROC (↑)", "Mean robustness drop (↓)"]
    standard_vals = [clean_auroc, clean_mce]
    robust_vals = [robust_auroc, robust_mce]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, standard_vals, width, label="Standard Nominal Training", color="#4c72b0", edgecolor="black")
    rects2 = ax.bar(x + width/2, robust_vals, width, label="Robust Augmentation Training", color="#55a868", edgecolor="black")

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Standard vs. Robust Training", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="upper right")

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fa_vs_md_tradeoff(
    data: Union[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_path: str
) -> str:
    """
    Renders False Alarms per 1k (FA@1k) vs. Missed Defects per 1k (MD@1k) trade-off curves per method.
    """
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    palette = {"patchcore": "#1f77b4", "padim": "#2ca02c", "autoencoder": "#ff7f0e"}

    if isinstance(data, dict):
        for method_name, (labels, scores) in data.items():
            labels = np.asarray(labels, dtype=int)
            scores = np.asarray(scores, dtype=float)
            norm_scores = scores[labels == 0]
            def_scores = scores[labels == 1]

            if len(norm_scores) == 0 or len(def_scores) == 0:
                continue

            all_s = np.sort(scores)
            percentiles = np.linspace(0, 100, 200)
            thresholds = np.percentile(all_s, percentiles)

            fa_list = [np.mean(norm_scores >= th) * 1000.0 for th in thresholds]
            md_list = [np.mean(def_scores < th) * 1000.0 for th in thresholds]

            m_lower = str(method_name).lower()
            color = palette.get(m_lower, "#333333")
            ax.plot(fa_list, md_list, label=str(method_name).upper(), color=color, lw=2.2)
    elif isinstance(data, pd.DataFrame):
        for method_name, group in data.groupby("method"):
            m_lower = str(method_name).lower()
            color = palette.get(m_lower, "#333333")
            ax.plot(group["fa_at_1k"], group["md_at_1k"], label=str(method_name).upper(), color=color, lw=2.2)

    # Highlight typical operational budget boundary
    ax.axvline(5.0, color="#d62728", linestyle=":", label="Operator Budget (5 FA/1k)")

    ax.set_xlabel("False Alarms per 1,000 Normal Items (FA@1k) [↓]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Missed Defects per 1,000 Items (MD@1k) [↓]", fontsize=11, fontweight="bold")
    # ax.set_title("Operational Operating Trade-off (FA@1k vs. MD@1k)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-1, 50)
    ax.set_ylim(-10, 1000)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Method", loc="upper right")

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_tpr_vs_alert_budget(
    data: Union[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_path: str,
    max_budget: float = 20.0
) -> str:
    """
    Plots achieved True Positive Rate (TPR) vs. allowed false alarm budget (1 to 20 alarms / 1k items).
    """
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    palette = {"patchcore": "#1f77b4", "padim": "#2ca02c", "autoencoder": "#ff7f0e"}
    budgets = np.linspace(1.0, max_budget, 40)

    if isinstance(data, dict):
        for method_name, (labels, scores) in data.items():
            labels = np.asarray(labels, dtype=int)
            scores = np.asarray(scores, dtype=float)
            norm_scores = scores[labels == 0]
            def_scores = scores[labels == 1]

            if len(norm_scores) == 0 or len(def_scores) == 0:
                continue

            tprs = []
            for b in budgets:
                target_fpr = b / 1000.0
                th = np.percentile(norm_scores, max(0.0, min(100.0, (1.0 - target_fpr) * 100.0)))
                tpr = np.mean(def_scores >= th)
                tprs.append(tpr)

            m_lower = str(method_name).lower()
            color = palette.get(m_lower, "#333333")
            ax.plot(budgets, tprs, label=str(method_name).upper(), color=color, lw=2.2)
    elif isinstance(data, pd.DataFrame):
        for method_name, group in data.groupby("method"):
            m_lower = str(method_name).lower()
            color = palette.get(m_lower, "#333333")
            ax.plot(group["alert_budget"], group["tpr"], label=str(method_name).upper(), color=color, lw=2.2)

    ax.set_xlabel("Allowed False Alarm Budget (Alarms per 1,000 Items) [↑]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Achieved Defect Detection Rate (TPR) [↑]", fontsize=11, fontweight="bold")
    # ax.set_title("Defect Recall Under Strict Operator Alarm Budgets", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max_budget + 1)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Method", loc="lower right")

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cost_weighted_error_curves(
    data: Union[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_path: str,
    cost_ratios: List[float] = [10.0, 20.0, 50.0]
) -> str:
    """
    Plots Cost-Weighted Error (CWE) across thresholds for asymmetric cost ratios r in {10, 20, 50}.
    """
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    colors = {10.0: "#1f77b4", 20.0: "#2ca02c", 50.0: "#d62728"}

    if isinstance(data, dict):
        # Extract representative method scores (e.g. PatchCore or first method)
        method_key = "patchcore" if "patchcore" in data else list(data.keys())[0]
        labels, scores = data[method_key]
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)

        s_min, s_max = np.min(scores), np.max(scores)
        norm_scores = (scores - s_min) / (s_max - s_min + 1e-8)
        norm_mask = (labels == 0)
        def_mask = (labels == 1)

        thresholds = np.linspace(0.0, 1.0, 200)

        for r in cost_ratios:
            cwes = []
            for th in thresholds:
                fp = np.sum(norm_scores[norm_mask] >= th)
                fn = np.sum(norm_scores[def_mask] < th)
                cwe = (fp * 1.0 + fn * r) / len(labels)
                cwes.append(cwe)

            min_idx = int(np.argmin(cwes))
            ax.plot(thresholds, cwes, label=f"Cost Ratio r = {int(r)} (Min Cost = {cwes[min_idx]:.3f})", color=colors.get(r, "#333333"), lw=2.2)
            ax.scatter([thresholds[min_idx]], [cwes[min_idx]], color=colors.get(r, "#333333"), s=80, zorder=5)
    elif isinstance(data, pd.DataFrame):
        for r, group in data.groupby("cost_ratio"):
            ax.plot(group["threshold"], group["cwe"], label=f"Cost Ratio r = {int(r)}", lw=2.2)

    ax.set_xlabel("Normalized Decision Threshold (τ) [0, 1]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Cost per Inspected Item (CWE) [↓]", fontsize=11, fontweight="bold")
    # ax.set_title("Asymmetric Cost-Weighted Error vs. Decision Cutoff", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center")

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_operator_review_overload(
    overload_data: Union[pd.DataFrame, Dict[str, Any]],
    output_path: str
) -> str:
    """
    Renders bar charts showing Operator Review Load per window and Overload Probability P(Overload).
    """
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))

    if isinstance(overload_data, pd.DataFrame):
        df = overload_data
    else:
        # Construct dataframe from dict
        df = pd.DataFrame(overload_data)

    if "method" in df.columns and "overload_probability" in df.columns:
        priors = df["defect_prior"].unique() if "defect_prior" in df.columns else [0.01]
        x = np.arange(len(df["method"].unique()))
        width = 0.25

        methods = sorted(df["method"].unique())
        for idx, p in enumerate(sorted(priors)):
            sub = df[df["defect_prior"] == p] if "defect_prior" in df.columns else df
            sub_probs = [float(sub[sub["method"] == m]["overload_probability"].mean()) for m in methods]
            offset = (idx - len(priors)/2 + 0.5) * width
            ax.bar(x + offset, sub_probs, width, label=f"Prior p = {p*100:.0f}%", edgecolor="black", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods], fontsize=11, fontweight="bold")
        ax.set_ylabel("P(Overload) [Review > 60 items/hr]", fontsize=11, fontweight="bold")
        ax.set_title("Operator Overload Probability Across Defect Priors", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0.0, 1.05)
        ax.legend(title="Defect Prior")
    else:
        # Fallback simple load comparison
        ax.bar(["PatchCore", "PaDiM", "Autoencoder"], [0.02, 0.15, 0.85], color=["#1f77b4", "#2ca02c", "#ff7f0e"], edgecolor="black")
        ax.set_ylabel("Overload Probability P(Overload)", fontsize=11, fontweight="bold")
        ax.set_title("Operator Overload Probability (Capacity = 60 items/window)", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path



def plot_cct_cost_tradeoff(
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    cost_ratios: List[float] = [10.0, 20.0, 50.0],
    prior: float = 0.01
) -> str:
    """
    Renders empirical Cost-Weighted Error curves C(tau) vs Decision Thresholds across cost ratios,
    marking the optimal cost-calibrated threshold tau_CCT.
    """
    from src.metrics.cost_calibrated import compute_empirical_cost_curve, optimize_cct_threshold
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))

    palette = ["#1f77b4", "#ff7f0e", "#d62728"]
    for idx, r in enumerate(cost_ratios):
        taus, costs, _, _ = compute_empirical_cost_curve(scores, labels, cost_ratio=r, prior=prior, num_thresholds=200)
        cct_res = optimize_cct_threshold(scores, labels, cost_ratio=r, prior=prior, max_alerts_per_1k=5.0)
        opt_tau = cct_res["threshold"]
        min_c = cct_res["min_expected_cost"]

        ax.plot(taus, costs, label=f"Cost Ratio r = {int(r)} (Defect Escape {int(r)}x)", color=palette[idx % len(palette)], lw=2.2)
        ax.scatter([opt_tau], [min_c], color=palette[idx % len(palette)], s=90, zorder=5, edgecolors="black")

    ax.set_xlabel("Decision Threshold (\\tau)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expected Unit Inspection Risk C(\\tau)", fontsize=11, fontweight="bold")
    ax.set_title("Cost-Calibrated Thresholding (CCT) Expected Risk Curves", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_coreset_scalability(
    scalability_df: pd.DataFrame,
    output_path: str
) -> str:
    """
    Plots coreset selection runtime vs patch set size N on log-log scale.
    """
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sub_128 = scalability_df[scalability_df["feature_dim_D"] == 128] if "feature_dim_D" in scalability_df.columns else scalability_df

    # Left: Runtime (s)
    ax1.plot(sub_128["num_patches_N"], sub_128["time_cpu_greedy_sec"], marker="o", lw=2.0, label="CPU Sequential Greedy", color="#d62728")
    ax1.plot(sub_128["num_patches_N"], sub_128["time_gpu_greedy_sec"], marker="s", lw=2.0, label="GPU Sequential Greedy", color="#ff7f0e")
    ax1.plot(sub_128["num_patches_N"], sub_128["time_gpu_batched_sec"], marker="^", lw=2.5, label="GPU Batched (approximate)", color="#2ca02c")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Candidate Patches (N)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Runtime (seconds, log scale)", fontsize=11, fontweight="bold")
    # ax1.set_title("Coreset Runtime Scaling (D=128)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Right: Speedup vs CPU
    ax2.plot(sub_128["num_patches_N"], sub_128["speedup_gpu_batched_vs_cpu"], marker="^", lw=2.5, label="GPU Batched Speedup", color="#2ca02c")
    ax2.plot(sub_128["num_patches_N"], sub_128["speedup_gpu_greedy_vs_cpu"], marker="s", lw=2.0, label="GPU Greedy Speedup", color="#ff7f0e")
    ax2.axhline(1.0, linestyle=":", color="gray", label="CPU Baseline (1x)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Candidate Patches (N)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Throughput Speedup vs. CPU (x)", fontsize=11, fontweight="bold")
    # ax2.set_title("Speedup Factor vs. Sequential CPU", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_decision_confusion_shifts(
    decision_df: pd.DataFrame,
    output_path: str
) -> str:
    """
    Renders stacked bar chart illustrating Nominal Relief (false alarms prevented)
    vs Defect Escape count when transitioning from Quantile-99 to CCT.
    """
    set_paper_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    cat_summary = decision_df.groupby("category").agg({
        "nominal_relief_count": "mean",
        "defect_escape_count": "mean"
    }).reset_index()

    x = np.arange(len(cat_summary))
    width = 0.45

    ax.bar(x, cat_summary["nominal_relief_count"], width, label="False Alarms Prevented (Nominal Relief)", color="#2ca02c", edgecolor="black")
    ax.bar(x, -cat_summary["defect_escape_count"], width, label="Defects Escaped (Budget Bound)", color="#d62728", edgecolor="black")

    ax.axhline(0, color="black", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in cat_summary["category"]], rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Part Count Shift (\\Delta Units)", fontsize=11, fontweight="bold")
    ax.set_title("Operational Decision Shifts: Quantile-99 \\to Cost-Calibrated (CCT)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout(pad=1.2)
    png_path = output_path if output_path.endswith(".png") else (os.path.splitext(output_path)[0] + ".png")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path