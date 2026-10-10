#!/usr/bin/env bash
# Install and verify the official MVTec AD archive (it cannot be downloaded automatically).
# Usage: bash scripts/download_data.sh /path/to/mvtec_anomaly_detection.tar.xz [data_root]
set -euo pipefail
ARCHIVE=${1:?"pass the path of the official mvtec_anomaly_detection.tar.xz (see scripts/download_dataset.py)"}
DATA_DIR=${2:-"data/mvtec_ad"}
python scripts/download_dataset.py --archive "$ARCHIVE" --data-root "$DATA_DIR"
