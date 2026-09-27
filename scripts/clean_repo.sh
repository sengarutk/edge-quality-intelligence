#!/usr/bin/env bash
# Remove caches and throwaway benchmark artifacts. Does NOT touch the live spool or audit
# databases (data/spooler_queue.db, data/audit_log.db): pass --databases to delete them too.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
find . -path ./.venv -prune -o -type d \( -name __pycache__ -o -name .pytest_cache \) -exec rm -rf {} + 2>/dev/null || true
rm -rf data/latency_evidence .coverage coverage.xml htmlcov
rm -f data/resilience_test_*.db* data/latency_profile.db*
if [[ "${1:-}" == "--databases" ]]; then
    echo "Deleting spool and audit databases (undelivered events will be lost)"
    rm -f data/*.db data/*.db-wal data/*.db-shm data/*.db-journal
fi
echo "clean"
