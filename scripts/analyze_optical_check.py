#!/usr/bin/env python3
"""How often the calibrated blur check fires on real MVTec AD test images.

The blur threshold of each category is calibrated on held-out *training* images
(scripts/build_score_bank.py). This script applies it to every test image and reports the
flag rate per defect type. Frames flagged here would bypass visual scoring in the runtime.
Output: results/score_bank/optical_check_on_test.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_service import focus_measure  # noqa: E402


def main() -> None:
    out = {}
    for bank_path in sorted((PROJECT_ROOT / "results" / "score_bank").glob("*.npz")):
        b = np.load(bank_path)
        lap = np.array([focus_measure(cv2.imread(str(PROJECT_ROOT / p))) for p in b["test_paths"]])
        flagged = lap < float(b["blur_threshold"])
        types = b["test_defect_types"]
        out[bank_path.stem] = {
            "blur_threshold": float(b["blur_threshold"]),
            "flag_rate_nominal": float(flagged[b["test_labels"] == 0].mean()),
            "flag_rate_defective": float(flagged[b["test_labels"] == 1].mean()),
            "flagged_by_defect_type": dict(Counter(types[flagged].tolist())),
            "n_test": int(len(lap)),
        }
    dest = PROJECT_ROOT / "results" / "score_bank" / "optical_check_on_test.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
