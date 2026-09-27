#!/usr/bin/env python3
"""Record the software and hardware environment used to produce results/ (results/environment_manifest.json)."""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("numpy", "scipy", "pandas", "pydantic", "opencv-python-headless", "torch", "torchvision",
            "scikit-learn", "paho-mqtt", "matplotlib", "streamlit", "onnxruntime")


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        try:
            mod = __import__({"scikit-learn": "sklearn", "opencv-python-headless": "cv2"}.get(pkg, pkg))
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            return "not installed"


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return "unavailable"


def main() -> None:
    import torch

    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sqlite": sqlite3.sqlite_version,
        "mosquitto": (run(["mosquitto", "-h"]).splitlines() or ["unavailable"])[0],
        "packages": {p: version(p) for p in PACKAGES},
    }
    out = ROOT / "results" / "environment_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
