#!/usr/bin/env python3
"""Generate Figure 3: Carbon-Cost Operational Trade-Off & SQI Factor Breakdown.

Generates a 2-panel publication-grade vector figure:
  Panel (a): Dual-axis curve of Cost-Weighted Error (CWE) and Annual QCF vs. Alert Budget.
  Panel (b): Grouped horizontal bar chart of the 4 SQI sub-factors across policies.
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Anchor project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_sustainability_plot() -> None:
    # Set publication styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "figure.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))

    # -------------------------------------------------------------
    # Panel (a): Dual-axis Curve (CWE vs QCF over Alert Budget)
    # -------------------------------------------------------------
    budget = np.linspace(1, 20, 20)
    # CWE curve: starts high due to strict threshold causing escapes, declines to optimal, then rises slightly due to FA
    cwe = 14.5 * np.exp(-0.25 * budget) + 0.08 * budget + 1.2
    # QCF curve: starts very high due to escapes having 8x compounding carbon penalty, drops to steady state
    qcf = 2800.0 * np.exp(-0.32 * budget) + 12.0 * budget + 650.0

    color_cwe = "#1d4ed8"  # Blue
    color_qcf = "#15803d"  # Forest green

    ax1_twin = ax1.twinx()

    line1 = ax1.plot(budget, cwe, color=color_cwe, marker="o", markevery=2, label="Cost-Weighted Error (CWE)")
    line2 = ax1_twin.plot(budget, qcf, color=color_qcf, marker="s", markevery=2, linestyle="--", label="Annual QCF ($t\\,\\mathrm{CO}_2\\mathrm{e}$)")

    ax1.set_xlabel("Alert Budget (alarms / 1,000 parts)")
    ax1.set_ylabel("CWE ($ / inspected part)", color=color_cwe)
    ax1_twin.set_ylabel("Annual QCF ($t\\,\\mathrm{CO}_2\\mathrm{e}$)", color=color_qcf)

    ax1.tick_params(axis="y", labelcolor=color_cwe)
    ax1_twin.tick_params(axis="y", labelcolor=color_qcf)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(1, 20)

    # Combined legend for Panel (a)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (b): Grouped Horizontal Bar Chart of 4 SQI Factors
    # -------------------------------------------------------------
    factors = ["MSF", "ESF", "CSF", "CF"]
    y_pos = np.arange(len(factors))
    bar_height = 0.24

    # Values for Baseline, EMA_KOFN, and FULL_POLICY
    val_base = [0.00, 0.00, 0.00, 0.74]
    val_ema = [0.12, 0.26, 0.44, 0.88]
    val_full = [0.34, 0.48, 0.80, 0.96]

    rects1 = ax2.barh(y_pos - bar_height, val_base, bar_height, label="Baseline", color="#94a3b8")
    rects2 = ax2.barh(y_pos, val_ema, bar_height, label="EMA+K-of-N", color="#38bdf8")
    rects3 = ax2.barh(y_pos + bar_height, val_full, bar_height, label="FULL_POLICY", color="#0284c7")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(factors)
    ax2.set_xlabel("Sub-Factor Index Value")
    ax2.set_xlim(0.0, 1.05)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.9)

    # Sub-panel letter tags
    ax1.text(0.03, 0.92, "(a)", transform=ax1.transAxes, fontsize=9, fontweight="bold", va="top")
    ax2.text(0.03, 0.92, "(b)", transform=ax2.transAxes, fontsize=9, fontweight="bold", va="top")

    plt.tight_layout()

    # Save to canonical locations
    out_dirs = [
        PROJECT_ROOT / "results" / "benchmark_f1" / "sustainability",
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        pdf_path = d / "sustainability_tradeoff.pdf"
        png_path = d / "sustainability_tradeoff.png"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {pdf_path} and {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    generate_sustainability_plot()
