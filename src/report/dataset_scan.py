"""Split-aware scan of the YOLO label files and image headers on disk.

`DatasetStats` (`src/yolov8/dataset_report.py`) is accumulated during conversion, before
the split exists, so it cannot answer the question that actually catches problems: which
classes have no ground truth in the split the model was measured on. The label files in
`<dataset_dir>/{train,valid,test}/labels` are post-filter, post-split and already carry
the polygons, so reading them is both cheaper and strictly more informative than
threading split identity back through the converter.

A second pass reads the *images* under `<split>/images`, header only: `Image.open` is
lazy, so `.size`, `.mode` and `.format` cost a few hundred bytes of file and no pixel
decode. That is cheap enough to cover every image rather than a sample, which is the
whole reason the image figures can be read as facts about the dataset instead of about a
subsample.

Everything accumulated here is a fixed-size histogram, a capped `Counter` or a `Tally`.
Nothing is a list that grows with dataset size, and nothing logs inside the per-file or
per-line loop -- one summary is emitted at the end.
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

# --- image pass ---------------------------------------------------------------------
# Suffixes ultralytics itself accepts, plus the two TIFF spellings.
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dng"}
)
MP_BINS = 44
# Megapixels above this land in the top bin. 44 MP covers every consumer sensor and a
# 45 MP satellite tile is still counted, just not resolved past the edge.
MP_MAX = 44.0
IMG_AR_BINS = 50
# Past this many distinct sizes the counter stops admitting keys and tallies the rest as
# "other": a dataset of uniquely-sized crops must not turn the report into a size list.
MAX_RESOLUTION_KEYS = 4000
# Modes and formats come from PIL's own fixed vocabularies, so these are belt and braces.
MAX_MODE_KEYS = 32
MAX_FORMAT_KEYS = 32
OTHER_KEY = "other"

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

        # --- image headers, all fixed-size ---------------------------------------
        self.images_scanned = 0
        self.mp_hist = np.zeros(MP_BINS, dtype=np.int64)
        self.img_ar_hist = np.zeros(IMG_AR_BINS, dtype=np.int64)
        self.resolutions: Counter[str] = Counter()
        self.resolutions_other = 0
        self.resolutions_truncated = False
        self.modes: Counter[str] = Counter()
        self.formats: Counter[str] = Counter()
        self.orientation: Counter[str] = Counter()
        self.fit: Counter[str] = Counter()
        self.pixels_total = 0
        self.pixels_min = 0
        self.pixels_max = 0

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

    def note_image_file(self, path: Path) -> None:
        """Read one image *header*. Called once per image; must not log.

        `Image.open` parses the header and stops, so this never decodes pixels -- the
        cost is one seek per file, which is what makes covering every image affordable.
        The import is inside the function because a report must survive a broken PIL,
        and after the first call it is a `sys.modules` lookup.

        The size is EXIF-corrected exactly as `ultralytics.data.utils.exif_size` does
        it. Without that, a split of phone photos -- stored landscape with an
        orientation tag -- is reported as landscape while the loader trains on portrait,
        and every number below it describes a dataset nobody has.
        """
        try:
            from PIL import Image

            with Image.open(path) as im:
                width, height = _exif_size(im)
                mode = str(im.mode or "?")
                fmt = str(im.format or "?")
        except Exception as e:
            self.errors.add("unreadable image file", e)
            return
        if width <= 0 or height <= 0:
            self.errors.add("image with no dimensions", path.name)
            return

        self.images_scanned += 1
        pixels = width * height
        self.pixels_total += pixels
        self.pixels_min = pixels if self.pixels_min == 0 else min(self.pixels_min, pixels)
        self.pixels_max = max(self.pixels_max, pixels)

        mp = pixels / 1e6
        self.mp_hist[int(np.clip(mp / MP_MAX * MP_BINS, 0, MP_BINS - 1))] += 1
        lr = math.log(width / height)
        self.img_ar_hist[
            int(np.clip((lr + 3.0) / 6.0 * IMG_AR_BINS, 0, IMG_AR_BINS - 1))
        ] += 1
        if width == height:
            self.orientation["square"] += 1
        else:
            self.orientation["landscape" if width > height else "portrait"] += 1

        key = f"{width}x{height}"
        if key in self.resolutions or len(self.resolutions) < MAX_RESOLUTION_KEYS:
            self.resolutions[key] += 1
        else:
            self.resolutions_other += 1
            self.resolutions_truncated = True
        _admit(self.modes, mode, MAX_MODE_KEYS)
        _admit(self.formats, fmt, MAX_FORMAT_KEYS)

        long_side, short_side = max(width, height), min(width, height)
        if long_side > self.imgsz:
            self.fit["downscaled"] += 1
        elif long_side < self.imgsz:
            self.fit["upscaled"] += 1
        else:
            self.fit["native"] += 1
        if short_side < self.imgsz:
            self.fit["short side below imgsz"] += 1

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

    def instances_by_class(self, n_classes: int) -> list[int]:
        """Instances per class index, summed over every split that was scanned."""
        totals = [0] * max(n_classes, 0)
        for split in self.splits.values():
            for cls, count in split.instances.items():
                if 0 <= cls < len(totals):
                    totals[cls] += int(count)
        return totals

    # --- image read side ------------------------------------------------------------
    @property
    def has_images(self) -> bool:
        return self.images_scanned > 0

    def megapixel_percentile(self, q: float) -> float | None:
        """Return a percentile of the megapixel histogram, to one bin's resolution."""
        total = int(self.mp_hist.sum())
        if total == 0:
            return None
        cum = np.cumsum(self.mp_hist)
        i = int(np.searchsorted(cum, q * total))
        return round(min(i + 0.5, MP_BINS) / MP_BINS * MP_MAX, 2)

    def top_resolutions(self, limit: int) -> tuple[list[tuple[str, int]], int, int]:
        """Return the `limit` commonest `WxH` keys, then the rest as one number.

        Returns `(rows, other_images, other_sizes)`, where `other_sizes` is 0 when the
        counter hit its key cap -- past that point the distinct count is unknowable
        without a set that grows with the dataset, and printing a guess would be worse
        than printing nothing.
        """
        rows = self.resolutions.most_common(max(int(limit), 0))
        counted = sum(n for _, n in rows)
        other_images = self.images_scanned - counted
        other_sizes = (
            0 if self.resolutions_truncated else max(len(self.resolutions) - len(rows), 0)
        )
        return rows, max(other_images, 0), other_sizes

    def log_summary(self) -> None:
        """One line for the whole scan. Never called from inside the loop."""
        logger.info(
            "dataset scan: %d label files, %d instances, splits %s%s",
            self.scanned_files,
            self.total_instances,
            self.split_counts() or "none",
            f", flags: {self.flags.summary()}" if self.flags else "",
        )
        if self.has_images:
            median = self.megapixel_percentile(0.5)
            logger.info(
                "dataset scan: %d image headers, %d distinct sizes%s, median %s MP, "
                "%d with a short side under imgsz=%d",
                self.images_scanned,
                len(self.resolutions),
                "+" if self.resolutions_truncated else "",
                median,
                self.fit.get("short side below imgsz", 0),
                self.imgsz,
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
        directory = _first_dir(root, aliases, "labels")
        if directory is None:
            continue
        for path in sorted(directory.glob("*.txt")):
            scan.note_label_file(split, path)

    if scan.is_empty:
        return None

    for aliases in SPLIT_ALIASES.values():
        directory = _first_dir(root, aliases, "images")
        if directory is not None:
            _scan_images(scan, directory)

    scan.log_summary()
    return scan


def _first_dir(root: Path, aliases: tuple[str, ...], leaf: str) -> Path | None:
    """Return the first `<root>/<alias>/<leaf>` that exists. One split, one name."""
    for alias in aliases:
        directory = root / alias / leaf
        if directory.is_dir():
            return directory
    return None


def _scan_images(scan: DatasetScan, directory: Path) -> None:
    """Header-read every image in one split directory. Iterates, never lists.

    `iterdir` rather than `sorted(glob(...))`: order changes nothing in a histogram, and
    a sorted list of 200,000 paths is memory spent for no answer.
    """
    try:
        entries = directory.iterdir()
    except OSError as e:
        scan.errors.add("unreadable image directory", e)
        return
    for path in entries:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            scan.note_image_file(path)


# The EXIF key for the orientation tag, and the two values that transpose the frame.
EXIF_ORIENTATION = 274
EXIF_TRANSPOSED = frozenset({6, 8})


def _exif_size(im: Any) -> tuple[int, int]:
    """Return the size the *loader* will see, not the size stored in the file.

    A straight copy of `ultralytics.data.utils.exif_size`, deliberately duplicated
    rather than imported: this module is read by the report, which must not drag an
    ultralytics import onto a path that runs after training. JPEG only, because that is
    the only format ultralytics corrects -- disagreeing with it would be worse than the
    bug.
    """
    width, height = (int(v) for v in im.size)
    if getattr(im, "format", None) != "JPEG":
        return width, height
    try:
        exif = im.getexif()
        if exif and exif.get(EXIF_ORIENTATION) in EXIF_TRANSPOSED:
            return height, width
    except Exception as e:
        logger.debug("exif orientation unreadable: %s", e)
    return width, height


def _admit(counter: Counter[str], key: str, limit: int) -> None:
    """Count `key`, folding everything past `limit` distinct keys into `other`."""
    if key in counter or len(counter) < limit:
        counter[key] += 1
    else:
        counter[OTHER_KEY] += 1
