"""Fixtures for the HTML evaluation report tests.

Everything synthetic here is driven through the *real* code path: the capture is fed by
calling `ReportCapture.note()` with the same dict shapes `validator._process_batch`
receives, the metrics are real `DetMetrics` / `SegmentMetrics` objects rather than mocks,
and the images are real JPEGs on disk. Mocks are how the PR-curve bug survived a full
test suite -- a MagicMock answers to any attribute, including ones ultralytics has never
had.
"""

import json
import re
import tempfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from PIL import Image, ImageDraw
from ultralytics.utils.metrics import DetMetrics, SegmentMetrics

from src.report.blob import ReportContext, build_blob
from src.report.capture import ReportCapture


# Six real classes plus two the split never sees. `absent_a` has no ground truth at all,
# which is the `ap_class_index` trap: ultralytics drops it from every per-class array, so
# a naive names[:n] slice mislabels every row after it.
CLASS_NAMES = [
    "ripe",
    "unripe",
    "flower",
    "leaf",
    "stem",
    "rot",
    "rare_class",
    "absent_class",
]
PRESENT = 6  # classes 0..5 carry ground truth; 6 is rare, 7 is absent
IMG_W, IMG_H = 320, 240


def names_for(n_classes: int) -> list[str]:
    """`n_classes` class names, the first eight real, the rest generated."""
    if n_classes <= len(CLASS_NAMES):
        return CLASS_NAMES[:n_classes]
    return CLASS_NAMES + [f"class_{i}" for i in range(len(CLASS_NAMES), n_classes)]


def present_count(n_classes: int) -> int:
    """How many classes carry ground truth: the last two never do."""
    return max(n_classes - 2, 1)


def _load_metrics_suite():
    """Import `tests/yolov8/test_metrics_utils.py` by path rather than by name.

    A `tests` package exists in site-packages and, being a regular package, wins over
    this repository's `tests/` directory whatever the path order -- so
    `import tests.yolov8.test_metrics_utils` silently resolves to something else. The
    file path is unambiguous, and reusing that module's `_build_metrics` is the point:
    the report has to be pinned to the same real metrics objects the metrics suite is,
    absent class and all.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "yolov8" / "test_metrics_utils.py"
    spec = importlib.util.spec_from_file_location("_yolov8_metrics_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_METRICS_SUITE = _load_metrics_suite()
suite_metrics = _METRICS_SUITE._build_metrics
SUITE_NAMES = _METRICS_SUITE.NAMES

DEFAULT_CFG = {
    "html_report": True,
    "confusion_matrix_conf": 0.25,
    "report_gallery_per_grid": 24,
    "report_max_thumbnails": 200,
    "report_thumbnail_px": 320,
    "report_high_conf_fp_threshold": 0.7,
    "report_low_support_threshold": 30,
    "report_tide": True,
    "report_tide_max_images": 4000,
    "report_scan_labels": True,
    "report_cm_max_classes": 60,
    "report_split_bytes": 5_000_000,
    "report_max_bytes": 15_000_000,
}


# --- ClearML ------------------------------------------------------------------------


@pytest.fixture
def task():
    """Return a mock ClearML task recording every upload and every logger call.

    Same shape as `tests/yolov8/test_callbacks.py`'s fixture, plus the artifact map the
    upload helper reads the URL back out of.
    """
    mock = MagicMock()
    mock.name = "yolo-train"
    mock.id = "abcdef0123456789"
    mock.get_logger.return_value = MagicMock()
    mock.get_output_log_web_page.return_value = "https://clearml.example/projects/x/y"
    mock.data.script.version_num = "9df844cdeadbeef"
    mock.data.project = "YOLO/Training"
    mock.artifacts = {
        "evaluation_report": SimpleNamespace(url="https://files.example/report.html"),
        "evaluation_report_galleries": SimpleNamespace(
            url="https://files.example/galleries.html"
        ),
    }
    return mock


# --- images -------------------------------------------------------------------------


def make_images(directory: Path, n: int, seed: int = 0) -> list[str]:
    """Write `n` small JPEGs with drawn shapes, so thumbnails have something to show."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        img = Image.new("RGB", (IMG_W, IMG_H), tuple(rng.integers(40, 90, 3).tolist()))
        draw = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(2, 6))):
            x0, y0 = rng.integers(0, IMG_W - 60), rng.integers(0, IMG_H - 60)
            draw.ellipse(
                [x0, y0, x0 + rng.integers(20, 60), y0 + rng.integers(20, 60)],
                fill=tuple(rng.integers(90, 240, 3).tolist()),
            )
        path = directory / f"img_{i:05d}.jpg"
        img.save(path, quality=85)
        paths.append(str(path))
    return paths


# --- capture ------------------------------------------------------------------------


def _batch_and_preds(rng, path: str, n_classes: int, seg: bool):
    """One image's worth of ground truth and post-NMS predictions, in letterbox space."""
    n_present = present_count(n_classes)
    m = int(rng.integers(1, 6))
    gt_cls = rng.integers(0, n_present, size=m)
    gt = np.zeros((m, 4), dtype=np.float32)
    for i in range(m):
        w, h = rng.uniform(30, 120), rng.uniform(30, 120)
        x0, y0 = rng.uniform(0, IMG_W - w), rng.uniform(0, IMG_H - h)
        gt[i] = (x0, y0, x0 + w, y0 + h)

    # Most detections are jittered ground truth; the rest are invented, which is what
    # gives the error decomposition something other than a single type to report.
    found = rng.random(m) < 0.8
    keep, keep_cls = gt[found], gt_cls[found]
    jitter = rng.normal(0, 7, size=keep.shape).astype(np.float32)
    det = np.clip(keep + jitter, 0, max(IMG_W, IMG_H)).reshape(-1, 4)
    det_cls = keep_cls.copy()
    swap = rng.random(len(det_cls)) < 0.15
    det_cls[swap] = (det_cls[swap] + 1) % n_present

    n_extra = int(rng.integers(0, 3))
    if n_extra:
        extra = np.zeros((n_extra, 4), dtype=np.float32)
        for i in range(n_extra):
            w, h = rng.uniform(20, 90), rng.uniform(20, 90)
            x0, y0 = rng.uniform(0, IMG_W - w), rng.uniform(0, IMG_H - h)
            extra[i] = (x0, y0, x0 + w, y0 + h)
        det = np.vstack([det, extra])
        det_cls = np.concatenate([det_cls, rng.integers(0, n_present, size=n_extra)])

    n = len(det)
    conf = np.clip(rng.beta(2.2, 1.6, size=n), 0.001, 0.999).astype(np.float32)
    tp = (rng.random((n, 10)) < conf[:, None] * 0.85) if n else np.zeros((0, 10), bool)
    out = {"tp": tp}
    if seg:
        out["tp_m"] = (
            (rng.random((n, 10)) < conf[:, None] * 0.6) if n else np.zeros((0, 10), bool)
        )
    preds = {
        "bboxes": det.reshape(-1, 4),
        "conf": conf,
        "cls": det_cls.astype(np.int64),
    }
    batch = {
        "bboxes": gt,
        "cls": gt_cls.astype(np.int64),
        "ori_shape": (IMG_H, IMG_W),
        "imgsz": (IMG_H, IMG_W),
        "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        "im_file": path,
    }
    return preds, batch, out


def make_capture(
    image_dir: Path,
    n_images: int = 60,
    n_classes: int = PRESENT,
    seg: bool = False,
    seed: int = 7,
    cfg: dict | None = None,
    n_files: int = 40,
) -> ReportCapture:
    """Populate a `ReportCapture` through `note()`, exactly as the wrapper does.

    `n_files` is deliberately smaller than `n_images` so a large synthetic run reuses a
    bounded pool of real JPEGs -- writing 20,000 files to prove a cap is not worth the
    disk.
    """
    paths = make_images(image_dir, min(n_files, n_images), seed)
    capture = ReportCapture()
    capture.configure({**DEFAULT_CFG, **(cfg or {})}, seed=seed)
    capture.available = True
    capture.names = names_for(n_classes)
    rng = np.random.default_rng(seed)
    for i in range(n_images):
        preds, batch, out = _batch_and_preds(rng, paths[i % len(paths)], n_classes, seg)
        capture.note(preds, batch, out)
    return capture


# --- metrics ------------------------------------------------------------------------


def build_metrics(cls=DetMetrics, n_classes: int = PRESENT, seed: int = 0):
    """Build a real, processed metrics object over `n_classes` classes.

    Class `n_classes - 1` gets a handful of instances so the low-support dimming has
    something to dim, and the two trailing names never appear at all.
    """
    rng = np.random.default_rng(seed)
    n_pred = 900
    conf = rng.random(n_pred)
    n_present = present_count(n_classes)
    present = list(range(n_present))
    weights = (
        np.array([1.0] * (n_present - 1) + [0.02]) if n_present > 1 else np.array([1.0])
    )
    weights = weights / weights.sum()
    target = rng.choice(present, size=700, p=weights)
    stats = {
        "tp": [rng.random((n_pred, 10)) < conf[:, None] * 0.9],
        "conf": [conf],
        "pred_cls": [rng.choice(present, size=n_pred, p=weights)],
        "target_cls": [target],
        "target_img": [np.unique(target)],
    }
    if cls is SegmentMetrics:
        stats["tp_m"] = [rng.random((n_pred, 10)) < conf[:, None] * 0.6]

    metrics = cls()
    metrics.names = dict(enumerate(names_for(n_classes)))
    metrics.stats = stats
    with tempfile.TemporaryDirectory() as td:
        metrics.process(save_dir=Path(td), plot=False)
    return metrics


def make_validator(
    metrics,
    split: str = "test",
    seen: int = 60,
    matrix: bool = True,
    n_classes: int = PRESENT,
):
    """Return a validator-shaped namespace: metrics, args and a confusion matrix."""
    names = names_for(n_classes)
    n = len(names)
    cm = None
    if matrix:
        rng = np.random.default_rng(3)
        raw = rng.integers(0, 12, size=(n + 1, n + 1)).astype(float)
        np.fill_diagonal(raw, rng.integers(40, 200, size=n + 1))
        raw[n, n] = 0.0
        cm = SimpleNamespace(matrix=raw, names=dict(enumerate(names)), nc=n)
    return SimpleNamespace(
        metrics=metrics,
        seen=seen,
        names=dict(enumerate(names)),
        confusion_matrix=cm,
        args=SimpleNamespace(
            conf=0.001,
            iou=0.7,
            max_det=300,
            batch=32,
            imgsz=640,
            half=True,
            rect=False,
            split=split,
            plots=matrix,
        ),
    )


def make_trainer(save_dir: Path, epochs: int = 20, seg: bool = False):
    """Return a trainer-shaped namespace with a real `results.csv` of N rows."""
    save_dir.mkdir(parents=True, exist_ok=True)
    header = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss"]
    header.append("metrics/mAP50-95(B)")
    if seg:
        header += ["train/seg_loss", "metrics/mAP50-95(M)"]
    rows = [",".join(header)]
    for e in range(epochs):
        values = [
            str(e),
            f"{2.0 - e * 0.02:.4f}",
            f"{1.4 - e * 0.01:.4f}",
            f"{1.1 - e * 0.008:.4f}",
            f"{min(0.05 + e * 0.02, 0.82):.4f}",
        ]
        if seg:
            values += [f"{1.6 - e * 0.015:.4f}", f"{min(0.03 + e * 0.017, 0.71):.4f}"]
        rows.append(",".join(values))
    (save_dir / "results.csv").write_text("\n".join(rows) + "\n")
    return SimpleNamespace(
        save_dir=save_dir,
        args=SimpleNamespace(
            model="yolo11n-seg.pt" if seg else "yolo11n.pt",
            imgsz=640,
            epochs=epochs,
            batch=64,
            lr0=0.001,
            box=7.5,
            task="segment" if seg else "detect",
            conf=None,
            single_cls=False,
        ),
    )


# --- dataset on disk ----------------------------------------------------------------


# Two landscape sizes, one portrait and one square, one of them greyscale: enough for
# the header pass to have a resolution ranking, an orientation split, a mode other than
# RGB and images on both sides of imgsz.
SPLIT_IMAGE_SHAPES = (
    (320, 240, "RGB"),
    (320, 240, "RGB"),
    (240, 320, "RGB"),
    (800, 600, "RGB"),
    (128, 128, "L"),
)


def make_dataset(root: Path, n_classes: int = PRESENT, per_split=(40, 12, 8), seg=False):
    """Write a YOLO dataset tree with real label files, for the split scan."""
    rng = np.random.default_rng(11)
    for split, count in zip(("train", "valid", "test"), per_split, strict=True):
        _make_split_images(root / split / "images", count)
        labels = root / split / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            lines = []
            for _ in range(int(rng.integers(0, 7))):
                # The last classes are deliberately absent from valid and test, which
                # is what the split-composition table is for.
                limit = present_count(n_classes) if split != "train" else n_classes
                cls = int(rng.integers(0, max(limit, 1)))
                w, h = rng.uniform(0.03, 0.4), rng.uniform(0.03, 0.4)
                cx = rng.uniform(w / 2, 1 - w / 2)
                cy = rng.uniform(h / 2, 1 - h / 2)
                if seg:
                    xs = [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2]
                    ys = [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2]
                    coords = " ".join(
                        f"{v:.5f}" for pair in zip(xs, ys, strict=True) for v in pair
                    )
                    lines.append(f"{cls} {coords}")
                else:
                    lines.append(f"{cls} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}")
            (labels / f"img_{i:05d}.txt").write_text("\n".join(lines) + "\n")
    return root


def _make_split_images(directory: Path, count: int) -> None:
    """Write the image side of the tree, which `dataset_scan` header-reads.

    Tiny and flat-coloured: this pass never decodes a pixel, so the only thing that has
    to be real about these files is their header.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        w, h, mode = SPLIT_IMAGE_SHAPES[i % len(SPLIT_IMAGE_SHAPES)]
        Image.new(mode, (w, h), 128 if mode == "L" else (90, 110, 130)).save(
            directory / f"img_{i:05d}.jpg", quality=70
        )


# --- one-call context ---------------------------------------------------------------


def make_context(
    tmp_path: Path,
    *,
    seg: bool = False,
    n_images: int = 60,
    n_classes: int = PRESENT,
    epochs: int = 20,
    capture: ReportCapture | None = "auto",
    with_val_stats: bool = True,
    with_dataset: bool = True,
    with_trainer: bool = True,
    with_matrix: bool = True,
    task=None,
    cfg: dict | None = None,
    seed: int = 7,
) -> ReportContext:
    """Assemble a full `ReportContext` from synthetic-but-real parts."""
    metrics = build_metrics(SegmentMetrics if seg else DetMetrics, n_classes, seed)
    validator = make_validator(
        metrics, seen=n_images, matrix=with_matrix, n_classes=n_classes
    )
    if capture == "auto":
        capture = make_capture(
            tmp_path / "images",
            n_images=n_images,
            n_classes=n_classes,
            seg=seg,
            seed=seed,
            cfg=cfg,
        )
    val_stats = None
    if with_val_stats:
        from src.yolov8.metrics_utils import ValStatsAccumulator

        val_stats = ValStatsAccumulator()
        val_stats.update(validator)
    return ReportContext(
        task=task,
        validator=validator,
        final_metrics=metrics,
        trainer=make_trainer(tmp_path / "run", epochs, seg) if with_trainer else None,
        val_stats=val_stats,
        capture=capture,
        dataset_dir=str(make_dataset(tmp_path / "ds", n_classes, seg=seg))
        if with_dataset
        else None,
        class_2_idx={n: i for i, n in enumerate(names_for(n_classes))},
        task_yolo="segment" if seg else "detect",
        split_label="Test ",
        args_train={"imgsz": 640, "seed": 0, "epochs": epochs},
        args_val={"conf": 0.001, "split": "test"},
        args_visualization={**DEFAULT_CFG, **(cfg or {})},
    )


def make_blob(tmp_path: Path, **kwargs) -> dict:
    """Build a blob from a synthetic context in one call."""
    return build_blob(make_context(tmp_path, **kwargs))


def markup_only(document: str) -> str:
    """Strip every `<script>` and `<style>` body, leaving the markup.

    Needed because the inlined assets are text like any other: `report.js` documents its
    own contract in comments containing `data-thumb="<key>"`, and the plotly bundle
    carries a plotly.com link in its mode bar. Neither is a reference the *page* makes,
    and a test that scans the whole file cannot tell the difference.
    """
    return re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", r"<\1></\1>", document, flags=re.DOTALL
    )


def inline_css(document: str) -> str:
    """Return the first `<style>` body -- the report's own stylesheet."""
    match = re.search(r"<style>(.*?)</style>", document, re.DOTALL)
    return match.group(1) if match else ""


def extract_blob(document: str) -> dict:
    """Pull the JSON payload back out of a rendered page."""
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>',
        document,
        re.DOTALL,
    )
    assert match, "the report-data script element is missing"
    return json.loads(match.group(1).replace("<\\/", "</"))


@pytest.fixture
def det_blob(tmp_path):
    return make_blob(tmp_path)


@pytest.fixture
def seg_blob(tmp_path):
    return make_blob(tmp_path, seg=True)
