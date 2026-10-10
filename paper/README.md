# Paper: publication files

Everything needed to submit or upload the manuscript is in this folder.

| File | Purpose |
|---|---|
| `main.tex` | Manuscript source (IEEEtran conference format) |
| `references.bib` | Bibliography |
| `IEEEtran.bst` | IEEE bibliography style |
| `generated_metrics.tex` | Every number used in the text; generated, do not edit by hand |
| `tables/*.tex` | Tables I-VII; generated |
| `figures/*.pdf` | Figures 1-2; generated |
| `main.pdf` | Compiled paper |
| `compile_paper.sh` | Builds `main.pdf` with pdflatex + bibtex |

Build the PDF: `bash paper/compile_paper.sh`.
For Overleaf, upload the whole folder (or run `bash scripts/package_overleaf.sh` to get
`paper_overleaf.zip`) and set `main.tex` as the main document.

The numbers, tables and figures are produced from `results/` by
`scripts/build_paper_assets.py`; `scripts/reproduce_all_paper_results.sh` reruns every
experiment first.
