#!/usr/bin/env bash
# Build paper/main.pdf with a local TeX installation and fail on unresolved references.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
rm -f main.aux main.bbl main.blg main.log main.out
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
if grep -qE "undefined|Citation .* undefined" main.log; then
    grep -E "undefined" main.log
    echo "Unresolved references" >&2
    exit 1
fi
grep -o "Output written on main.pdf ([0-9]* pages" main.log
