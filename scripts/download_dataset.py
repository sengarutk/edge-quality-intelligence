#!/usr/bin/env python3
"""Install the official MVTec AD dataset and verify it.

MVTec AD cannot be downloaded automatically: obtain ``mvtec_anomaly_detection.tar.xz`` from
https://www.mvtec.com/company/research/datasets/mvtec-ad (license CC BY-NC-SA 4.0), then run

    python scripts/download_dataset.py --archive /path/to/mvtec_anomaly_detection.tar.xz

The archive's MD5 is checked against the official release, it is extracted, and every
requested category is checked against the official image counts and resolutions before
``data/mvtec_ad/<category>`` is linked to it. This script never generates images: experiments
must not run on synthetic stand-ins for MVTec categories.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_MD5 = "eefca59f2cede9c3fc5b6befbfec275e"

# category: (train good, test total, image side, PIL mode) from the official release
OFFICIAL = {
    "bottle": (209, 83, 900, "RGB"),
    "cable": (224, 150, 1024, "RGB"),
    "capsule": (219, 132, 1000, "RGB"),
    "carpet": (280, 117, 1024, "RGB"),
    "grid": (264, 78, 1024, "L"),
    "hazelnut": (391, 110, 1024, "RGB"),
    "leather": (245, 124, 1024, "RGB"),
    "metal_nut": (220, 115, 700, "RGB"),
    "pill": (267, 167, 800, "RGB"),
    "screw": (320, 160, 1024, "L"),
    "tile": (230, 117, 840, "RGB"),
    "toothbrush": (60, 42, 1024, "RGB"),
    "transistor": (213, 100, 1024, "RGB"),
    "wood": (247, 79, 1024, "RGB"),
    "zipper": (240, 151, 1024, "L"),
}
PAPER_CATEGORIES = ["bottle", "cable", "carpet", "grid", "hazelnut", "leather", "metal_nut"]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify_category(cat_dir: Path, category: str) -> list[str]:
    """Return a list of deviations from the official release (empty if the category matches)."""
    n_train, n_test, side, mode = OFFICIAL[category]
    problems = []
    train = sorted((cat_dir / "train" / "good").glob("*.png"))
    test = sorted((cat_dir / "test").glob("*/*.png"))
    if len(train) != n_train:
        problems.append(f"{len(train)} train images (official {n_train})")
    if len(test) != n_test:
        problems.append(f"{len(test)} test images (official {n_test})")
    if train:
        with Image.open(train[0]) as im:
            if im.size != (side, side) or im.mode != mode:
                problems.append(f"images are {im.size} {im.mode} (official {(side, side)} {mode})")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="official mvtec_anomaly_detection.tar.xz")
    ap.add_argument("--extract-dir", type=Path, default=PROJECT_ROOT / "data" / "mvtec_ad_full")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mvtec_ad")
    ap.add_argument("--categories", nargs="+", default=PAPER_CATEGORIES)
    ap.add_argument("--verify-only", action="store_true", help="only check --data-root")
    args = ap.parse_args()

    if not args.verify_only:
        if args.archive is None or not args.archive.is_file():
            sys.exit("Pass --archive with the official archive (see the module docstring).")
        digest = md5sum(args.archive)
        if digest != OFFICIAL_MD5:
            sys.exit(f"MD5 mismatch: {digest} (official {OFFICIAL_MD5}); refusing to use this archive.")
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive) as tar:
            tar.extractall(args.extract_dir, filter="data")
        args.data_root.mkdir(parents=True, exist_ok=True)
        for cat in args.categories:
            link = args.data_root / cat
            if link.exists() or link.is_symlink():
                sys.exit(f"{link} already exists; move it away first.")
            link.symlink_to((args.extract_dir / cat).resolve())

    failed = False
    for cat in args.categories:
        problems = verify_category(args.data_root / cat, cat)
        print(f"{cat}: {'OK' if not problems else 'NOT OFFICIAL - ' + '; '.join(problems)}")
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
