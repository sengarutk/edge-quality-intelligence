#!/usr/bin/env bash
# Regenerate every result, table, figure and number of paper/main.tex.
# Requires: the MVTec AD categories in data/mvtec_ad, a mosquitto binary, pdflatex and bibtex.
# Approximate time on a laptop GPU: score banks 4 min, ablation (independent and correlated
# scores) 8 min, short defects 5 min, sensitivity 5 min, latency 15 min, broker durability 8 min.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${PYTHON:-python}"

echo "[1/9] tests";                     "${PY}" -m pytest tests/ -q
echo "[2/9] PatchCore score banks";     "${PY}" scripts/build_score_bank.py
echo "[3/9] optical-check audit";       "${PY}" scripts/analyze_optical_check.py > /dev/null
echo "[4/9] policy ablation";           "${PY}" scripts/run_ablation_study.py
                                        "${PY}" scripts/run_ablation_study.py --rho 0.9
                                        "${PY}" scripts/run_short_defect_recall.py
echo "[5/9] sensitivity sweep";         "${PY}" scripts/run_sensitivity_analysis.py
echo "[6/9] synthetic trace replay";    "${PY}" scripts/run_real_trace_benchmark.py
echo "[7/9] latency";                   "${PY}" scripts/benchmark_latency.py
echo "[8/9] broker durability";         "${PY}" scripts/benchmark_spooler_resilience.py
echo "[9/9] paper assets and PDF"
"${PY}" scripts/write_environment_manifest.py > /dev/null
"${PY}" scripts/build_paper_assets.py
bash paper/compile_paper.sh
echo "Done: paper/main.pdf"
