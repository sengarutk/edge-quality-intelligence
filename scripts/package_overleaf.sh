#!/usr/bin/env bash
# Package the manuscript sources (tex, bib, generated macros, tables, figures) for Overleaf.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/paper"
rm -f "${ROOT}/paper_overleaf.zip"
zip -qr "${ROOT}/paper_overleaf.zip" main.tex references.bib IEEEtran.bst generated_metrics.tex tables figures
unzip -l "${ROOT}/paper_overleaf.zip"
