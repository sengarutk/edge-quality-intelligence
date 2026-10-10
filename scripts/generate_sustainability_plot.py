#!/usr/bin/env python3
"""Plot the computed carbon comparison of scripts/03_compute_sustainability.py.

Every plotted value is read from results/benchmark_f1/sustainability/sustainability_summary.json;
the script fails if that file is missing. Panel (a): annual QCF of the nominal 99th-percentile
threshold and of the cost-calibrated threshold (CCT) per category. Panel (b): the SQI sub-factors
of CCT relative to the baseline threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "benchmark_f1" / "sustainability"


def generate_sustainability_plot() -> None:
    src = OUT / "sustainability_summary.json"
    if not src.is_file():
        sys.exit(f"{src} is missing; run scripts/03_compute_sustainability.py first")
    rows = json.loads(src.read_text(encoding="utf-8"))["categories"]
    names = [r["category"] + (" (a)" if r["params"] == "assumed" else "") for r in rows]
    x = np.arange(len(rows))

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.bar(x - 0.2, [r["base_qcf"] for r in rows], 0.4, label="q99 threshold", color="#2a78d6")
    ax1.bar(x + 0.2, [r["policy_qcf"] for r in rows], 0.4, label="CCT threshold", color="#eb6834")
    ax1.set_yscale("log")
    ax1.set_ylabel("Annual QCF (t CO$_2$e)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right")
    ax1.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    ax1.set_title("(a) computed footprint", fontsize=8, pad=18)

    factors = ("msf", "esf", "csf", "cf")
    for i, f in enumerate(factors):
        ax2.bar(x + (i - 1.5) * 0.2, [r[f] for r in rows], 0.2, label=f.upper())
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("SQI sub-factor of CCT vs. q99")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right")
    ax2.legend(frameon=False, ncol=4, fontsize=7)
    ax2.set_title("(b) SQI sub-factors", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"sustainability_tradeoff.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"Saved {OUT / 'sustainability_tradeoff.pdf'}")


if __name__ == "__main__":
    generate_sustainability_plot()
