"""Split-aware scan of the YOLO label files on disk.

`DatasetStats` (`src/yolov8/dataset_report.py`) is accumulated during conversion, before
the split exists, so it cannot answer the question that actually catches problems: which
classes have no ground truth in the split the model was measured on. The label files in
`<dataset_dir>/{train,valid,test}/labels` are post-filter, post-split and already carry
the polygons, so reading them is both cheaper and strictly more informative than
threading split identity back through the converter.

Everything accumulated here is a fixed-size histogram or a `Tally`. Nothing is a list
that grows with dataset size, and nothing logs inside the per-file or per-line loop --
one summary is emitted at the end.
"""

from __future__ import annotations
import math

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import Tally, get_logger


logger = get_logger(__name__)

SPLIT_DIRS = ("train", "valid", "val", "test")
# The three splits the report shows, and the directory names each can appear under.
SPLIT_ALIASES = {"train": ("train",), "valid": ("valid", "val"), "test": ("test",)}

AREA_BINS = 50
AR_BINS = 50
FILL_BINS = 50
HEAT_BINS = 32
MAX_OBJECTS_KEY = 200
MAX_VERTEX_KEY = 100

# A box shorter than this many pixels at `imgsz` cannot survive the model's stride.
SMALL_SIDE_PX = 8
BORDER_EPS = 0.002
DUPLICATE_IOU = 0.9
AR_OUTLIER = 3.0
RECTANGLE_FILL = 0.95


class SplitScan:
    """One split's accumulated composition."""

    def __init__(self) -> None:
        self.images = 0
        self.instances: Counter[int] = Counter()
        self.empty_images = 0

    def as_dict(self) -> dict[str, Any]:
        return {"images": self.images, "instances": dict(self.instances)}


class DatasetScan:
    """Everything the dataset section needs, accumulated in fixed-size structures."""

    def __init__(self, names: list[str], imgsz: int = 640) -> None:
        self.names = names
        self.imgsz = int(imgsz) or 640
        self.splits: dict[str, SplitScan] = {}
        self.objects_per_image: Counter[int] = Counter()
        self.area_hist = np.zeros(AREA_BINS, dtype=np.int64)
        self.ar_hist = np.zeros(AR_BINS, dtype=np.int64)
        self.fill_hist = np.zeros(FILL_BINS, dtype=np.int64)
        self.vertex_hist: Counter[int] = Counter()
        self.heat = np.zeros((HEAT_BINS, HEAT_BINS), dtype=np.int64)
        self.flags = Tally()
        self.total_instances = 0
        self.polygon_instances = 0
        self.rectangle_polygons = 0
        self.scanned_files = 0
        self.errors = Tally()

    # --- accumulation ---------------------------------------------------------------
    def note_label_file(self, split: str, path: Path) -> None:
        """Read one label file. Called once per image; must not log."""
        scan = self.splits.setdefault(split, SplitScan())
        scan.images += 1
        self.scanned_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            self.errors.add("unreadable label file", e)
            return

        boxes: list[tuple[int, float, float, float, float]] = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                self.errors.add("unparsable label line", line[:60])
                continue
            box = self._note_annotation(vals)
            if box is not None:
                boxes.append((cls, *box))
            scan.instances[cls] += 1
            self.total_instances += 1

        n = len(boxes)
        self.objects_per_image[min(n, MAX_OBJECTS_KEY)] += 1
        if n == 0:
            scan.empty_images += 1
            self.flags.add("images with no annotations", path.name)
        self._note_duplicates(boxes, path.name)

    def _note_annotation(self, vals: list[float]) -> tuple | None:
        """Fold one annotation into the histograms; return its normalised xywh."""
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:
            # Polygon: x1 y1 x2 y2 ... normalised. Its bounding box is what the box
            # metrics see, and the fill ratio against it is the rectangle-polygon test.
            xs = np.asarray(vals[0::2], dtype=float)
            ys = np.asarray(vals[1::2], dtype=float)
            k = min(xs.size, ys.size)
            if k < 3:
                return None
            xs, ys = xs[:k], ys[:k]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = x1 - x0, y1 - y0
            self.polygon_instances += 1
            self.vertex_hist[min(k, MAX_VERTEX_KEY)] += 1
            area = 0.5 * abs(
                float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
            )
            bbox_area = max(w * h, 1e-12)
            fill = min(area / bbox_area, 1.0)
            self.fill_hist[int(np.clip(fill * FILL_BINS, 0, FILL_BINS - 1))] += 1
            if fill > RECTANGLE_FILL:
                self.rectangle_polygons += 1

        if w <= 0 or h <= 0:
            self.flags.add("zero-area annotations")
            return None

        self.area_hist[int(np.clip(math.sqrt(w * h) * AREA_BINS, 0, AREA_BINS - 1))] += 1
        lr = math.log(max(w, 1e-9) / max(h, 1e-9))
        self.ar_hist[int(np.clip((lr + 3.0) / 6.0 * AR_BINS, 0, AR_BINS - 1))] += 1
        self.heat[
            int(np.clip(cy * HEAT_BINS, 0, HEAT_BINS - 1)),
            int(np.clip(cx * HEAT_BINS, 0, HEAT_BINS - 1)),
        ] += 1

        if min(w, h) * self.imgsz < SMALL_SIDE_PX:
            self.flags.add(f"boxes under {SMALL_SIDE_PX}px at imgsz={self.imgsz}")
        if (
            cx - w / 2 <= BORDER_EPS
            or cy - h / 2 <= BORDER_EPS
            or cx + w / 2 >= 1 - BORDER_EPS
            or cy + h / 2 >= 1 - BORDER_EPS
        ):
            self.flags.add("boxes touching the image border")
        if abs(lr) > AR_OUTLIER:
            self.flags.add("extreme aspect ratios (|log(w/h)| > 3)")
        return (cx, cy, w, h)

    def _note_duplicates(self, boxes: list[tuple], name: str) -> None:
        """Flag near-duplicate same-class boxes inside one image."""
        if len(boxes) < 2 or len(boxes) > 60:
            return
        arr = np.asarray([b[1:] for b in boxes], dtype=float)
        cls = np.asarray([b[0] for b in boxes], dtype=int)
        xy = np.stack(
            [
                arr[:, 0] - arr[:, 2] / 2,
                arr[:, 1] - arr[:, 3] / 2,
                arr[:, 0] + arr[:, 2] / 2,
                arr[:, 1] + arr[:, 3] / 2,
            ],
            axis=1,
        )
        from src.report.capture import box_iou_np

        iou = box_iou_np(xy, xy)
        np.fill_diagonal(iou, 0.0)
        same = cls[:, None] == cls[None, :]
        if bool(((iou > DUPLICATE_IOU) & same).any()):
            self.flags.add(
                f"images with near-duplicate same-class boxes (IoU > {DUPLICATE_IOU})",
                name,
            )

    # --- read side ------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return self.total_instances == 0 and self.scanned_files == 0

    def split_counts(self) -> dict[str, int]:
        return {k: v.images for k, v in self.splits.items() if v.images}

    def class_name(self, idx: int) -> str:
        if 0 <= idx < len(self.names):
            return self.names[idx]
        return f"class_{idx}"

    def terciles(self) -> tuple[float, float]:
        """Return the two sqrt(area) boundaries that split this dataset's boxes in three.

        Deliberately relative to *this* dataset rather than COCO's 32²/96² absolutes:
        a dataset whose objects are all large has no "small" objects in COCO's sense,
        and a stratified panel that says every bucket is "large" says nothing.
        """
        total = int(self.area_hist.sum())
        if total == 0:
            return (1 / 3, 2 / 3)
        cum = np.cumsum(self.area_hist)
        edges = []
        for q in (1 / 3, 2 / 3):
            i = int(np.searchsorted(cum, q * total))
            edges.append(min(i + 1, AREA_BINS) / AREA_BINS)
        return (edges[0], edges[1])

    def log_summary(self) -> None:
        """One line for the whole scan. Never called from inside the loop."""
        logger.info(
            "dataset scan: %d label files, %d instances, splits %s%s",
            self.scanned_files,
            self.total_instances,
            self.split_counts() or "none",
            f", flags: {self.flags.summary()}" if self.flags else "",
        )
        if self.errors:
            logger.warning("dataset scan skipped: %s", self.errors.summary(True))


def scan_dataset(
    dataset_dir: str | Path | None,
    names: list[str],
    imgsz: int = 640,
) -> DatasetScan | None:
    """Scan every split's label directory under `dataset_dir`.

    Returns None when there is nothing to scan, which the report renders as a banner
    rather than an error -- a dataset directory cleaned up before the report ran is a
    degradation, not a failure.
    """
    scan = DatasetScan(names, imgsz)
    if not dataset_dir:
        return None
    root = Path(dataset_dir)
    if not root.is_dir():
        return None

    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            labels = root / alias / "labels"
            if not labels.is_dir():
                continue
            for path in sorted(labels.glob("*.txt")):
                scan.note_label_file(split, path)
            break

    if scan.is_empty:
        return None
    scan.log_summary()
    return scan
