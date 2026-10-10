#!/usr/bin/env bash
# ==============================================================================
# Test runner with a coverage report (no minimum coverage is enforced)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[INFO] Running the pytest suite with a coverage report..."
python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=0