#!/usr/bin/env bash
# Package the manuscript sources (tex, bib, generated macros, tables, figures) for Overleaf or arXiv.
# main.bbl is included because arXiv does not run BibTeX; compile the paper first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/paper"
test -f main.bbl || { echo "paper/main.bbl missing: run paper/compile_paper.sh first" >&2; exit 1; }
rm -f "${ROOT}/paper_overleaf.zip"
zip -qr "${ROOT}/paper_overleaf.zip" main.tex main.bbl references.bib IEEEtran.bst generated_metrics.tex tables figures
unzip -l "${ROOT}/paper_overleaf.zip"
