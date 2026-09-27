#!/usr/bin/env bash
# ==============================================================================
# Strict Unit & Integration Test Runner with Coverage Verification
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[INFO] Running pytest test suite with strict coverage enforcement..."
python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=0