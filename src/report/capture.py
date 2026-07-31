"""Per-image capture for the HTML evaluation report.

`DetectionValidator.update_metrics` calls `self._process_batch(predn, pbatch)` once per
image, before `update_stats` and before the confusion matrix, and it runs regardless of
`args.plots`. That call is handed the full post-NMS predictions at the 0.001 NMS floor,
the ground truth, `im_file`, and it returns the `(n, 10)` IoU sweep -- everything the
report needs and nothing else in ultralytics exposes. Wrapping it on the *instance*
shadows the class method, so `SegmentationValidator._process_batch` (which calls
`super()` and then adds `tp_m`) still runs intact and the mask sweep is visible too.

This is deliberately not `confusion_matrix.process_batch`, which the pinning wrapper in
`callbacks.py` already owns: that one sees only detections above the display threshold,
carries no IoU values, and needs `plots=True`.

Every bound in here is applied **before** accumulation -- a reservoir over images and
five fixed-size heaps, never "keep everything and trim afterwards" -- so peak memory is
flat in dataset size. Nothing here logs: it runs once per image.
"""

from __future__ import annotations
import heapq
import math
import random

from collections import Counter
from typing import Any

import numpy as np

from src.utils.logging import Tally, get_logger


logger = get_logger(__name__)

# After this many exceptions the wrapper uninstalls itself and hands `_process_batch`
# back. A capture that is failing on every image is worth nothing and its try/except
# is not free at 20k images.
MAX_CAPTURE_ERRORS = 50

# Ground truth per image is unbounded in principle; detections are already capped by
# `max_det`. A crowd image with thousands of boxes would otherwise dominate the
# reservoir's memory on its own.
MAX_GT_PER_IMAGE = 300
MAX_BOXES_PER_ITEM = 20

# Fixed-size accumulators. None of these grows with dataset size.
IOU_BINS = 50
AREA_BINS = 40
AR_BINS = 40
MASK_LEVELS = 11  # tp_m.sum(1) is 0..10
IOU_COARSE_BINS = 20
MAX_CONFUSED_PAIRS = 4000

MATCH_IOU = 0.5


def _np(value: Any, dtype: Any = np.float32) -> np.ndarray:
    """Convert a torch tensor / numpy array / sequence to a numpy array on the CPU."""
    if value is None:
        return np.zeros(0, dtype=dtype)
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if numpy_fn is not None and not isinstance(value, np.ndarray):
        value = numpy_fn()
    return np.asarray(value, dtype=dtype)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in `a` against every box in `b`, shape (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32).reshape(-1, 4)
    b = b.astype(np.float32).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def to_native(
    boxes: np.ndarray,
    imgsz: tuple[int, int],
    ori_shape: tuple[int, int],
    ratio_pad: Any,
) -> np.ndarray:
    """Undo the letterbox, turning model-input xyxy into original-image pixels.

    Mirrors `ultralytics.utils.ops.scale_boxes` arithmetic rather than calling it, so
    the capture path holds no torch tensors and works on plain numpy in tests. The
    transform is a uniform scale plus a translation, so IoU is unchanged by it -- which
    is why everything downstream can work in native pixels without re-deriving anything.
    """
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.astype(np.float32).reshape(-1, 4).copy()
    h0, w0 = float(ori_shape[0]), float(ori_shape[1])
    ih, iw = float(imgsz[0]), float(imgsz[1])
    gain, pad_x, pad_y = None, 0.0, 0.0
    if ratio_pad:
        try:
            raw_gain = ratio_pad[0]
            gain = float(raw_gain[0] if hasattr(raw_gain, "__len__") else raw_gain)
            pad_x, pad_y = (float(v) for v in ratio_pad[1])
        except Exception:
            gain = None
    if not gain:
        gain = min(ih / max(h0, 1.0), iw / max(w0, 1.0))
        pad_x = (iw - w0 * gain) / 2
        pad_y = (ih - h0 * gain) / 2
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= max(gain, 1e-9)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w0)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h0)
    return out


def greedy_match(
    iou: np.ndarray,
    det_cls: np.ndarray,
    gt_cls: np.ndarray,
    order: np.ndarray,
    thr: float = MATCH_IOU,
) -> dict[int, int]:
    """Class-aware greedy assignment, detections in the given order, each GT once.

    Returns `{det index: gt index}`. Deliberately a re-implementation rather than a read
    of ultralytics' `tp` column: the TIDE oracles need the *assignment*, not the boolean
    that survives it.
    """
    taken: set[int] = set()
    matched: dict[int, int] = {}
    if iou.size == 0:
        return matched
    for d in order:
        candidates = np.where((gt_cls == det_cls[d]) & (iou[:, d] >= thr))[0]
        best, best_iou = -1, -1.0
        for g in candidates:
            if g in taken:
                continue
            if iou[g, d] > best_iou:
                best, best_iou = int(g), float(iou[g, d])
        if best >= 0:
            taken.add(best)
            matched[int(d)] = best
    return matched


class _BoundedTop:
    """Keep the highest-scoring N items seen, evicting as it goes.

    A heap of fixed capacity, not a sorted list truncated at the end: the whole point is
    that memory does not track dataset size. The counter breaks score ties so payload
    dicts are never compared.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 0)
        self._heap: list[tuple[float, int, dict]] = []
        self._seq = 0

    def push(self, score: float, payload: dict) -> None:
        if self.capacity == 0:
            return
        self._seq += 1
        entry = (float(score), -self._seq, payload)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def items(self) -> list[dict]:
        """Highest score first."""
        return [e[2] for e in sorted(self._heap, key=lambda e: (-e[0], -e[1]))]

    def __len__(self) -> int:
        return len(self._heap)


class ReportCapture:
    """Per-image capture installed on `validator._process_batch`.

    Module-level in `callbacks.py` for the same reason `ValStatsAccumulator` is:
    the data is gathered across many batches and consumed once, long afterwards, and
    callbacks only ever run in the main process.
    """

    def __init__(self) -> None:
        self.gallery_per_grid = 24
        self.tide_max_images = 4000
        self.display_conf = 0.25
        self.high_conf_fp = 0.7
        self.seed = 0
        self.reset()

    # --- lifecycle -----------------------------------------------------------------
    def reset(self) -> None:
        """Drop everything captured so far, before a new validation pass."""
        self.available = False
        self.installed = False
        self._validator: Any = None
        self._original: Any = None
        self.errors = Tally()
        self.n_images = 0
        self.n_dets = 0
        self.n_gt = 0
        self.gt_truncated = 0
        self.names: list[str] = []
        self.is_seg = False

        self._rng = random.Random(self.seed)
        self._gallery_rng = random.Random(self.seed + 1)
        self.records: list[dict] = []
        self._seen_images = 0
        self._random_seen = 0
        self.random_items: list[dict] = []

        self.grids: dict[str, _BoundedTop] = {
            name: _BoundedTop(self.gallery_per_grid)
            for name in ("g_worst", "g_hiconf_fp", "g_missed", "g_lowconf_tp")
        }

        self.tp_by_class: Counter[int] = Counter()
        self.fp_by_class: Counter[int] = Counter()
        self.fn_by_class: Counter[int] = Counter()
        self.gt_by_class: Counter[int] = Counter()
        self.confused: Counter[tuple[int, int]] = Counter()

        self.iou_hist = np.zeros(IOU_BINS, dtype=np.int64)
        self.area_total = np.zeros(AREA_BINS, dtype=np.int64)
        self.area_matched = np.zeros(AREA_BINS, dtype=np.int64)
        self.ar_total = np.zeros(AR_BINS, dtype=np.int64)
        self.ar_matched = np.zeros(AR_BINS, dtype=np.int64)
        self.box_mask_hist = np.zeros((IOU_COARSE_BINS, MASK_LEVELS), dtype=np.int64)

    def new_pass(self) -> None:
        """Drop the previous pass's data but keep the wrapper installed.

        Ultralytics reuses `trainer.validator` for every epoch, so the wrapper is
        installed once and would otherwise pool 300 epochs of images into one reservoir.
        `seen == 0` in `on_val_batch_start` is the start of a pass.
        """
        validator, original = self._validator, self._original
        installed, available = self.installed, self.available
        self.reset()
        self._validator, self._original = validator, original
        self.installed, self.available = installed, available

    def configure(self, args_visualization: dict | None, seed: int = 0) -> None:
        """Read the caps out of `8_Visualization` before the pass starts."""
        cfg = args_visualization or {}
        self.gallery_per_grid = int(cfg.get("report_gallery_per_grid", 24))
        self.tide_max_images = int(cfg.get("report_tide_max_images", 4000))
        self.display_conf = float(cfg.get("confusion_matrix_conf", 0.25))
        self.high_conf_fp = float(cfg.get("report_high_conf_fp_threshold", 0.7))
        self.seed = int(seed)
        self.reset()

    def install(self, validator: Any) -> bool:
        """Wrap `validator._process_batch`. Idempotent; never raises."""
        if getattr(validator, "_report_capture_installed", False):
            return True
        original = getattr(validator, "_process_batch", None)
        if original is None or not callable(original):
            logger.debug("no _process_batch on this validator; per-image capture off")
            return False

        def _process_batch_capturing(preds, batch):
            out = original(preds, batch)
            self.note(preds, batch, out)
            return out

        self._original = original
        self._validator = validator
        validator._process_batch = _process_batch_capturing
        validator._report_capture_installed = True
        self.installed = True
        self.available = True
        names = getattr(validator, "names", None)
        if isinstance(names, dict):
            self.names = [str(names[k]) for k in sorted(names)]
        elif names:
            self.names = [str(n) for n in names]
        return True

    def uninstall(self) -> None:
        """Hand `_process_batch` back. Safe to call when nothing was installed."""
        validator = getattr(self, "_validator", None)
        original = getattr(self, "_original", None)
        if validator is not None and original is not None:
            try:
                validator._process_batch = original
                validator._report_capture_installed = False
            except Exception as e:
                logger.debug("could not restore _process_batch: %s", e)
        self.installed = False
        self._validator = None
        self._original = None

    # --- the per-image hook ---------------------------------------------------------
    def note(self, preds: Any, batch: Any, out: Any) -> None:
        """Fold one image into the bounded sinks. Never raises, never logs."""
        try:
            self._note(preds, batch, out)
        except Exception as e:  # never let a report break validation
            self.errors.add(type(e).__name__, e)
            if self.errors.total() >= MAX_CAPTURE_ERRORS:
                self.available = False
                self.uninstall()

    def _note(self, preds: Any, batch: Any, out: Any) -> None:
        det_box = _np(preds.get("bboxes")).reshape(-1, 4)
        det_conf = _np(preds.get("conf")).ravel()
        det_cls = _np(preds.get("cls"), np.int64).ravel()
        gt_box = _np(batch.get("bboxes")).reshape(-1, 4)
        gt_cls = _np(batch.get("cls"), np.int64).ravel()

        n = min(det_box.shape[0], det_conf.shape[0], det_cls.shape[0])
        det_box, det_conf, det_cls = det_box[:n], det_conf[:n], det_cls[:n]
        m = min(gt_box.shape[0], gt_cls.shape[0])
        if m > MAX_GT_PER_IMAGE:
            self.gt_truncated += 1
            m = MAX_GT_PER_IMAGE
        gt_box, gt_cls = gt_box[:m], gt_cls[:m]

        imgsz = tuple(int(v) for v in (batch.get("imgsz") or (640, 640)))
        ori = batch.get("ori_shape") or imgsz
        ori_shape = (int(ori[0]), int(ori[1]))
        ratio_pad = batch.get("ratio_pad")

        # `reshape(0, -1)` is ambiguous, so an image with no detections -- which is
        # common on a strict model -- has to be handled before the reshape, not after.
        tp = np.asarray(out.get("tp") if isinstance(out, dict) else out)
        tp = tp.reshape(n, -1) if (n and tp.size) else np.zeros((n, 10), dtype=bool)
        tp_m = out.get("tp_m") if isinstance(out, dict) else None
        if tp_m is not None:
            tp_m = np.asarray(tp_m)
            tp_m = tp_m.reshape(n, -1) if (n and tp_m.size) else np.zeros((n, 10), bool)
            self.is_seg = True

        rec = {
            "im_file": str(batch.get("im_file") or ""),
            "ori_shape": ori_shape,
            "det_box": to_native(det_box, imgsz, ori_shape, ratio_pad),
            "det_conf": det_conf.astype(np.float32),
            "det_cls": det_cls.astype(np.int16),
            "gt_box": to_native(gt_box, imgsz, ori_shape, ratio_pad),
            "gt_cls": gt_cls.astype(np.int16),
            "tp0": tp[:, 0].astype(bool) if tp.size else np.zeros(n, dtype=bool),
            "tp_lvl": tp.sum(1).astype(np.uint8) if tp.size else np.zeros(n, np.uint8),
            "tpm_lvl": (
                tp_m.sum(1).astype(np.uint8) if tp_m is not None and tp_m.size else None
            ),
        }

        self.n_images += 1
        self.n_dets += n
        self.n_gt += m
        for c in gt_cls.tolist():
            self.gt_by_class[int(c)] += 1

        self._tally(rec)
        self._feed_grids(rec)
        self._reservoir(rec)

    # --- sinks ----------------------------------------------------------------------
    def _reservoir(self, rec: dict) -> None:
        """Vitter reservoir over whole images.

        Whole records, not individual detections: the TIDE oracles re-match detections
        against ground truth, so a half-sampled image would invent both false positives
        and misses that the model never made.
        """
        self._seen_images += 1
        if len(self.records) < self.tide_max_images:
            self.records.append(rec)
            return
        j = self._rng.randrange(self._seen_images)
        if j < self.tide_max_images:
            self.records[j] = rec

    def _tally(self, rec: dict) -> None:
        """Fold one image into the fixed-size scalar accumulators."""
        iou = box_iou_np(rec["gt_box"], rec["det_box"])
        keep = np.where(rec["det_conf"] >= self.display_conf)[0]
        order = keep[np.argsort(-rec["det_conf"][keep])] if keep.size else keep
        matched = greedy_match(iou, rec["det_cls"], rec["gt_cls"], order)

        matched_gt = set(matched.values())
        for d in order.tolist():
            cls = int(rec["det_cls"][d])
            if d in matched:
                self.tp_by_class[cls] += 1
                g = matched[d]
                b = int(np.clip(iou[g, d] * IOU_BINS, 0, IOU_BINS - 1))
                self.iou_hist[b] += 1
                if rec["tpm_lvl"] is not None:
                    cb = int(np.clip(iou[g, d] * IOU_COARSE_BINS, 0, IOU_COARSE_BINS - 1))
                    lvl = int(np.clip(rec["tpm_lvl"][d], 0, MASK_LEVELS - 1))
                    self.box_mask_hist[cb, lvl] += 1
            else:
                self.fp_by_class[cls] += 1
                if iou.size:
                    g = int(np.argmax(iou[:, d]))
                    if iou[g, d] >= MATCH_IOU and len(self.confused) < MAX_CONFUSED_PAIRS:
                        self.confused[(int(rec["gt_cls"][g]), cls)] += 1

        h, w = rec["ori_shape"]
        diag = math.sqrt(max(float(h) * float(w), 1.0))
        for g in range(rec["gt_box"].shape[0]):
            cls = int(rec["gt_cls"][g])
            hit = g in matched_gt
            if not hit:
                self.fn_by_class[cls] += 1
            x0, y0, x1, y1 = rec["gt_box"][g]
            bw, bh = max(float(x1 - x0), 0.0), max(float(y1 - y0), 0.0)
            frac = math.sqrt(bw * bh) / diag
            ab = int(np.clip(frac * AREA_BINS, 0, AREA_BINS - 1))
            self.area_total[ab] += 1
            self.area_matched[ab] += int(hit)
            if bw > 0 and bh > 0:
                # log(w/h) folded onto [-3, 3]; a square is the centre bin.
                lr = math.log(bw / bh)
                rb = int(np.clip((lr + 3.0) / 6.0 * AR_BINS, 0, AR_BINS - 1))
                self.ar_total[rb] += 1
                self.ar_matched[rb] += int(hit)

    def _overlay(self, rec: dict, iou: np.ndarray, order: np.ndarray, matched: dict):
        """Build the whole-image overlay for one record, at the display threshold."""
        matched_gt = set(matched.values())
        boxes: list[list] = []
        classes: set[str] = set()
        pairs: set[str] = set()
        n_fp = n_fn = 0
        for d in order.tolist():
            cls = int(rec["det_cls"][d])
            name = self._name(cls)
            classes.add(name)
            conf = float(rec["det_conf"][d])
            if d in matched:
                g = matched[d]
                boxes.append(
                    self._box("tp", rec["det_box"][d], cls, conf, float(iou[g, d]))
                )
                continue
            n_fp += 1
            best = float(iou[:, d].max()) if iou.size else 0.0
            boxes.append(self._box("fp", rec["det_box"][d], cls, conf, best))
            if iou.size:
                g = int(np.argmax(iou[:, d]))
                if iou[g, d] >= MATCH_IOU:
                    pairs.add(f"{self._name(int(rec['gt_cls'][g]))}>{name}")
        for g in range(rec["gt_box"].shape[0]):
            cls = int(rec["gt_cls"][g])
            classes.add(self._name(cls))
            if g not in matched_gt:
                n_fn += 1
                boxes.append(self._box("fn", rec["gt_box"][g], cls, None, None))
        return boxes, sorted(classes)[:12], sorted(pairs)[:6], n_fp, n_fn

    def _feed_grids(self, rec: dict) -> None:
        """Push this image's candidates into the five fixed-size gallery heaps."""
        iou = box_iou_np(rec["gt_box"], rec["det_box"])
        keep = np.where(rec["det_conf"] >= self.display_conf)[0]
        order = keep[np.argsort(-rec["det_conf"][keep])] if keep.size else keep
        matched = greedy_match(iou, rec["det_cls"], rec["gt_cls"], order)
        matched_gt = set(matched.values())

        boxes, classes, pairs, n_fp, n_fn = self._overlay(rec, iou, order, matched)
        base = {
            "im_file": rec["im_file"],
            "ori_shape": rec["ori_shape"],
            "classes": classes,
            "pairs": pairs,
        }
        name = _basename(rec["im_file"])

        if n_fp or n_fn:
            self.grids["g_worst"].push(
                n_fp + n_fn,
                {
                    **base,
                    "focus": None,
                    "boxes": boxes[:MAX_BOXES_PER_ITEM],
                    "outcome": "mixed",
                    "label": f"{name}  FP {n_fp}  FN {n_fn}",
                },
            )

        self._feed_instance_grids(rec, iou, order, matched, matched_gt, base, name)

        self._random_seen += 1
        item = {
            **base,
            "focus": None,
            "boxes": boxes[:MAX_BOXES_PER_ITEM],
            "outcome": "mixed",
            "label": f"{name}  {len(rec['gt_cls'])} GT",
        }
        if len(self.random_items) < self.gallery_per_grid:
            self.random_items.append(item)
        else:
            j = self._gallery_rng.randrange(self._random_seen)
            if j < self.gallery_per_grid:
                self.random_items[j] = item

    def _feed_instance_grids(
        self,
        rec: dict,
        iou: np.ndarray,
        order: np.ndarray,
        matched: dict,
        matched_gt: set,
        base: dict,
        name: str,
    ) -> None:
        """Fill the three crop grids: lowest TPs, confident FPs, biggest misses."""
        for d in order.tolist():
            conf = float(rec["det_conf"][d])
            cls = int(rec["det_cls"][d])
            box = rec["det_box"][d]
            if d in matched:
                g = matched[d]
                # Ascending key: negate so the bounded *top* heap keeps the lowest.
                self.grids["g_lowconf_tp"].push(
                    -conf,
                    self._instance(base, box, "tp", cls, conf, float(iou[g, d]), name),
                )
            elif conf >= self.high_conf_fp:
                best = float(iou[:, d].max()) if iou.size else 0.0
                self.grids["g_hiconf_fp"].push(
                    conf, self._instance(base, box, "fp", cls, conf, best, name)
                )
        for g in range(rec["gt_box"].shape[0]):
            if g in matched_gt:
                continue
            x0, y0, x1, y1 = rec["gt_box"][g]
            area = max(float(x1 - x0), 0.0) * max(float(y1 - y0), 0.0)
            self.grids["g_missed"].push(
                area,
                self._instance(
                    base, rec["gt_box"][g], "fn", int(rec["gt_cls"][g]), None, None, name
                ),
            )

    # --- helpers --------------------------------------------------------------------
    def _name(self, idx: int) -> str:
        if 0 <= idx < len(self.names):
            return self.names[idx]
        return f"class_{idx}"

    def _box(
        self,
        outcome: str,
        box: np.ndarray,
        cls: int,
        conf: float | None,
        iou: float | None,
    ) -> list:
        """One overlay box: geometry in native pixels, then outcome, class, score, IoU."""
        return [
            round(float(box[0]), 1),
            round(float(box[1]), 1),
            round(float(box[2]), 1),
            round(float(box[3]), 1),
            outcome,
            self._name(cls),
            None if conf is None else round(conf, 3),
            None if iou is None else round(float(iou), 3),
        ]

    def _instance(
        self,
        base: dict,
        box: np.ndarray,
        outcome: str,
        cls: int,
        conf: float | None,
        iou: float | None,
        name: str,
    ) -> dict:
        label = f"{name}  {self._name(cls)}"
        if conf is not None:
            label += f"  conf {conf:.2f}"
        if iou is not None:
            label += f"  IoU {iou:.2f}"
        return {
            **base,
            "focus": [round(float(v), 1) for v in box[:4]],
            "boxes": [self._box(outcome, box, cls, conf, iou)],
            "outcome": outcome,
            "label": label,
            "classes": [self._name(cls)],
        }

    # --- read side ------------------------------------------------------------------
    def grid_items(self, grid_id: str) -> list[dict]:
        """Return one grid's items, best-scoring first."""
        if grid_id == "g_random":
            return list(self.random_items)
        top = self.grids.get(grid_id)
        return top.items() if top else []

    def degradations(self) -> list[str]:
        """Human-readable sentences for whatever the capture could not do."""
        out: list[str] = []
        if not self.available:
            out.append(
                "Per-image capture was unavailable, so the error decomposition, the "
                "stratified panels and the galleries were not built."
            )
        if self.errors:
            out.append(
                f"Per-image capture skipped {self.errors.total()} image(s): "
                f"{self.errors.summary(with_examples=True)}."
            )
        if self.gt_truncated:
            out.append(
                f"{self.gt_truncated} image(s) had more than {MAX_GT_PER_IMAGE} ground "
                "truth boxes and were truncated for the error analysis."
            )
        if self.records and self._seen_images > len(self.records):
            out.append(
                f"The error decomposition ran on a seeded sample of "
                f"{len(self.records):,} of {self._seen_images:,} images."
            )
        return out

    def log_summary(self) -> None:
        """One line, after the pass. Never called from inside the per-image loop."""
        logger.info(
            "report capture: %d images, %d detections, %d GT, %d sampled for TIDE, "
            "%d capture errors",
            self.n_images,
            self.n_dets,
            self.n_gt,
            len(self.records),
            self.errors.total(),
        )


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1] or path
