"""Workload generation for the runtime ablation study.

A workload (configs/scenarios/*.yaml) declares randomized event families; for each seed
they are placed on a timeline without overlap. Visual scores are drawn from PatchCore
score banks (scripts/build_score_bank.py): real image-level distances on MVTec AD test
images, normalized with a reference computed on held-out *training* images only.

Event kinds
-----------
vision_defect     a defective part is in view; ground truth = actionable.
                  ``with_mechanical: true`` also injects a mechanical fault (sensor evidence).
mechanical_fault  vibration, current and heat rise without a visual defect; actionable.
vision_glitch     a nominal part with a specular glare artifact (glare score pool); not actionable.
optical_blur / optical_dark / optical_bright   camera degradation; not actionable.
sensor_dropout    listed channels stop reporting (zero-order hold); not actionable.
sensor_drift      temperature measurement drift ramp; not actionable.
idle / maintenance  machine state windows; events placed inside them are not actionable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.config import resolve_path

EVENT_KINDS = (
    "vision_defect", "mechanical_fault", "vision_glitch", "optical_blur", "optical_dark", "optical_bright",
    "sensor_dropout", "sensor_drift", "idle", "maintenance",
)


class EventFamily(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="One of EVENT_KINDS.")
    count: int = Field(..., ge=0)
    min_len: int = Field(..., ge=1)
    max_len: int = Field(..., ge=1)
    with_mechanical: bool = False
    channels: List[str] = Field(default_factory=list)
    inside: Optional[str] = Field(default=None, description="Place inside windows of this kind (idle/maintenance).")


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    total_steps: int = Field(default=9000, gt=0)
    events: List[EventFamily] = Field(default_factory=list)


def load_workload(path: str | Path) -> WorkloadConfig:
    data = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    cfg = WorkloadConfig.model_validate(data)
    for ev in cfg.events:
        if ev.kind not in EVENT_KINDS:
            raise ValueError(f"{cfg.name}: unknown event kind {ev.kind!r}")
        if ev.min_len > ev.max_len:
            raise ValueError(f"{cfg.name}: min_len > max_len for {ev.kind}")
    return cfg


@dataclass
class Timeline:
    """Per-step ground truth and fault configuration."""
    n: int
    vision_source: np.ndarray            # 0 nominal, 1 defect, 2 glare
    actionable: np.ndarray               # ground truth (bool)
    mechanical: np.ndarray               # bool
    optical: np.ndarray                  # 0 ok, 1 blur, 2 dark, 3 bright
    machine_state: np.ndarray            # object array of MachineState values
    drift_c: np.ndarray                  # temperature measurement offset
    dropout: List[List[str]] = field(default_factory=list)

    def defect_steps(self) -> List[int]:
        return np.flatnonzero(self.actionable).tolist()


def _place(rng: np.random.Generator, occupied: np.ndarray, length: int, lo: int, hi: int, tries: int = 500) -> Optional[int]:
    """Random start in [lo, hi - length] whose window (with a 30-frame margin) is free."""
    if hi - lo < length:
        return None
    for _ in range(tries):
        s = int(rng.integers(lo, hi - length + 1))
        a, b = max(0, s - 30), min(occupied.size, s + length + 30)
        if not occupied[a:b].any():
            return s
    return None


def build_timeline(cfg: WorkloadConfig, seed: int) -> Timeline:
    """Place every event family on a timeline. Raises if an event cannot be placed."""
    rng = np.random.default_rng(seed)
    n = cfg.total_steps
    tl = Timeline(
        n=n,
        vision_source=np.zeros(n, dtype=np.int8),
        actionable=np.zeros(n, dtype=bool),
        mechanical=np.zeros(n, dtype=bool),
        optical=np.zeros(n, dtype=np.int8),
        machine_state=np.array(["RUNNING"] * n, dtype=object),
        drift_c=np.zeros(n, dtype=np.float64),
        dropout=[[] for _ in range(n)],
    )
    windows: Dict[str, List[tuple]] = {"idle": [], "maintenance": []}
    occupied = np.zeros(n, dtype=bool)

    # Machine-state windows first; other events may be placed inside them.
    for fam in (f for f in cfg.events if f.kind in ("idle", "maintenance")):
        for _ in range(fam.count):
            length = int(rng.integers(fam.min_len, fam.max_len + 1))
            s = _place(rng, occupied, length, 60, n - 60)
            if s is None:
                raise RuntimeError(f"{cfg.name}: cannot place {fam.kind} window")
            tl.machine_state[s:s + length] = fam.kind.upper()
            occupied[s:s + length] = True
            windows[fam.kind].append((s, s + length))

    running_occupied = occupied.copy()
    inner_occupied = np.zeros(n, dtype=bool)
    # Sensor overlays are background conditions: they may overlap other events but not each other.
    overlay_occupied = {k: np.zeros(n, dtype=bool) for k in ("sensor_drift", "sensor_dropout")}
    for fam in (f for f in cfg.events if f.kind not in ("idle", "maintenance")):
        for _ in range(fam.count):
            length = int(rng.integers(fam.min_len, fam.max_len + 1))
            if fam.inside:
                spans = windows[fam.inside]
                if not spans:
                    raise RuntimeError(f"{cfg.name}: no {fam.inside} window for {fam.kind}")
                lo, hi = spans[int(rng.integers(len(spans)))]
                s = _place(rng, inner_occupied, length, lo, hi)
                occ = inner_occupied
            elif fam.kind in overlay_occupied:
                occ = overlay_occupied[fam.kind]
                s = _place(rng, occ, length, 60, n - 60)
            else:
                s = _place(rng, running_occupied, length, 60, n - 60)
                occ = running_occupied
            if s is None:
                raise RuntimeError(f"{cfg.name}: cannot place {fam.kind} (length {length})")
            occ[s:s + length] = True
            sl = slice(s, s + length)
            in_running = fam.inside is None
            if fam.kind == "vision_defect":
                tl.vision_source[sl] = 1
                tl.actionable[sl] = in_running
                if fam.with_mechanical:
                    tl.mechanical[sl] = True
            elif fam.kind == "mechanical_fault":
                tl.mechanical[sl] = True
                tl.actionable[sl] = in_running
            elif fam.kind == "vision_glitch":
                tl.vision_source[sl] = 2
            elif fam.kind.startswith("optical_"):
                tl.optical[sl] = {"optical_blur": 1, "optical_dark": 2, "optical_bright": 3}[fam.kind]
            elif fam.kind == "sensor_dropout":
                for t in range(s, s + length):
                    tl.dropout[t] = list(fam.channels or ["current"])
            elif fam.kind == "sensor_drift":
                tl.drift_c[sl] = 12.0 * np.arange(1, length + 1) / length
    return tl


class ScoreBank:
    """Normalized PatchCore scores and calibration frames for one MVTec category."""

    def __init__(self, path: str | Path) -> None:
        z = np.load(resolve_path(path), allow_pickle=False)
        self.category = Path(path).stem
        self.d_ref = float(z["d_ref"])
        labels = z["test_labels"]
        norm = lambda d: np.clip(0.5 * np.asarray(d, dtype=np.float64) / self.d_ref, 0.0, 1.0)  # noqa: E731
        self.pools = {0: norm(z["test_distances"][labels == 0]), 1: norm(z["test_distances"][labels == 1]),
                      2: norm(z["glare_distances"])}
        self.blur_threshold = float(z["blur_threshold"])
        self.frames = z["frames_bgr"]

    def frame(self, idx: int, optical: int) -> np.ndarray:
        f = self.frames[idx % len(self.frames)]
        if optical == 1:
            return cv2.GaussianBlur(f, (0, 0), 3.0)
        if optical == 2:
            return (f.astype(np.float32) * 0.03).astype(np.uint8)
        if optical == 3:
            return np.full_like(f, 255)
        return f
