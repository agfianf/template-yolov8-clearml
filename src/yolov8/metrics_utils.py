"""Utility functions for extracting metrics from Ultralytics trainer/validator.

This module provides helper functions to extract various metrics data in formats
suitable for ClearML logging, including per-class metrics, confusion matrix data,
loss components, learning rates, and speed metrics.
"""

from typing import Any

import numpy as np
import pandas as pd

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator

from src.utils.logging import get_logger


logger = get_logger(__name__)

# Ultralytics computes every curve over 1000 x-points, per class. That is 1000 * n_classes
# floats per curve family, and there are four families per domain -- eight on a
# segmentation run. At 80 classes the payload runs to megabytes, which the ClearML UI
# renders as a blank panel rather than an error. 250 points is visually indistinguishable
# for a monotone-ish curve and keeps the payload flat.
MAX_CURVE_POINTS = 250


def _downsample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evenly thin a curve to MAX_CURVE_POINTS, always keeping both endpoints."""
    n = x.shape[0]
    if n <= MAX_CURVE_POINTS:
        return x, y
    idx = np.linspace(0, n - 1, MAX_CURVE_POINTS).round().astype(int)
    idx = np.unique(idx)
    return x[idx], y[:, idx]


def _get_metrics(validator: BaseValidator):
    """Return the validator's metrics object, or None when it is unusable."""
    metrics = getattr(validator, "metrics", None)
    if metrics is None or getattr(metrics, "box", None) is None:
        return None
    return metrics


def _resolved_class_names(metrics) -> list[str]:
    """Return class names ordered to match ``ap_class_index``.

    Every per-class array on a ``Metric`` is indexed by ``ap_class_index``, which holds
    only the classes that had ground truth in this split. This is the one correct way to
    label those rows.
    """
    names = getattr(metrics, "names", None)
    ap_index = getattr(metrics, "ap_class_index", None)
    # `or []` is wrong here: ap_class_index is a numpy array, whose truth value raises.
    if ap_index is None:
        return []

    resolved = []
    for c in np.asarray(ap_index).ravel().tolist():
        cid = int(c)
        if isinstance(names, dict):
            resolved.append(str(names.get(cid, f"class_{cid}")))
        elif names is not None and cid < len(names):
            resolved.append(str(names[cid]))
        else:
            resolved.append(f"class_{cid}")
    return resolved


def _seg_metric(metrics):
    """Return the mask ``Metric`` for a segmentation run, else None.

    A ``SegmentMetrics`` carries a populated ``.seg``; ``DetMetrics`` has no such
    attribute at all. ``p`` being non-empty is the check that it actually ran.
    """
    seg = getattr(metrics, "seg", None)
    if seg is None or len(getattr(seg, "p", [])) == 0:
        return None
    return seg


def extract_per_class_metrics(
    validator: BaseValidator,
    class_names: list[str] | None = None,  # noqa: ARG001
) -> pd.DataFrame | None:
    """Extract per-class metrics, including mask metrics on a segmentation run.

    Built on ultralytics' own ``metrics.summary()``, which resolves class names via
    ``names[ap_class_index[i]]``. That indirection matters: ``box.p`` and friends are
    indexed by ``ap_class_index``, which contains **only classes with ground truth
    present**. Slicing a plain ``class_names`` list against it shifts every name after
    an absent class -- the bug this replaces.

    ``summary()`` gives Box-P/R/F1 + mAP50/mAP50-95 and, on ``SegmentMetrics``,
    Mask-P/R/F1 -- but no per-class mask *mAP*. Those come from ``seg.ap50`` / ``seg.ap``,
    which are indexed the same way, plus a mask-minus-box delta that isolates classes
    whose masks lag their boxes.

    Args:
        validator: The Ultralytics validator instance after validation.
        class_names: Unused; retained for call-site compatibility. Names come from
            ``metrics.names``, which is the only index-safe source.

    Returns:
        DataFrame of per-class metrics, or None if unavailable.

    """
    metrics = _get_metrics(validator)
    if metrics is None:
        return None

    try:
        summary = metrics.summary()
    except Exception as e:
        logger.warning("metrics.summary() failed, no per-class table: %s", e)
        return None

    if not summary:
        return None

    df = pd.DataFrame(summary)

    # summary() names the box mAP columns bare; be explicit so a segmentation table
    # cannot be misread as describing masks.
    df = df.rename(columns={"mAP50": "Box-mAP50", "mAP50-95": "Box-mAP50-95"})

    seg = _seg_metric(metrics)
    if seg is not None and len(seg.ap) == len(df):
        df["Mask-mAP50"] = np.asarray(seg.ap50, dtype=float)
        df["Mask-mAP50-95"] = np.asarray(seg.ap, dtype=float)
        df["Mask-Box-mAP50-95-delta"] = df["Mask-mAP50-95"] - df["Box-mAP50-95"]

    return df


def extract_confusion_matrix_data(
    validator: BaseValidator,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray | None, list[str], np.ndarray | None]:
    """Extract confusion matrix data from validator.

    Args:
        validator: The Ultralytics validator instance after validation.
        class_names: Optional list of class names.

    Returns:
        Tuple of (normalized_matrix, labels, counts_matrix) or (None, [], None).

    """
    if not hasattr(validator, "confusion_matrix") or validator.confusion_matrix is None:
        return None, [], None

    cm = validator.confusion_matrix
    matrix = cm.matrix if hasattr(cm, "matrix") else None

    if matrix is None:
        return None, [], None

    # Get labels
    if class_names is not None:
        labels = list(class_names)
    elif hasattr(cm, "names") and cm.names:
        labels = list(cm.names.values()) if isinstance(cm.names, dict) else list(cm.names)
    else:
        labels = [f"class_{i}" for i in range(matrix.shape[0])]

    # Add background class if matrix is larger
    if len(labels) < matrix.shape[0]:
        labels.append("background")

    # Ultralytics stores matrix[predicted, ground_truth] (utils/metrics.py:467) and
    # normalizes by COLUMN -- `matrix.sum(0)` at utils/metrics.py:549 -- so each column
    # sums to 1 and the diagonal reads as per-class recall. Normalizing by row instead
    # gives precision-like numbers, which is a different plot wearing the same title as
    # the confusion_matrix_normalized.png we upload alongside it.
    with np.errstate(divide="ignore", invalid="ignore"):
        col_sums = matrix.sum(axis=0, keepdims=True)
        normalized_matrix = np.where(col_sums > 0, matrix / col_sums, 0)

    # Transpose so rows become ground truth and columns become predictions, matching
    # the yaxis="Actual" / xaxis="Predicted" labels the reporter applies.
    return normalized_matrix.T, labels, matrix.T


def extract_loss_components(trainer: BaseTrainer) -> dict[str, float]:
    """Extract individual loss components from trainer.

    Args:
        trainer: The Ultralytics trainer instance during training.

    Returns:
        Dictionary with loss component names and values.

    """
    losses = {}

    # Ultralytics >=8.4 made loss_items and tloss dicts of {name: value}; before that
    # loss_items was a tensor of values zipped against loss_names, and tloss a scalar.
    # Iterating a dict yields its keys, so the old zip silently paired names with names
    # and float() blew up on 'box_loss'. Handle both shapes.
    loss_items = getattr(trainer, "loss_items", None)
    if isinstance(loss_items, dict):
        losses.update({name: float(value) for name, value in loss_items.items()})
    elif loss_items is not None:
        loss_names = getattr(trainer, "loss_names", None) or (
            "box_loss",
            "cls_loss",
            "dfl_loss",
        )
        for name, value in zip(loss_names, loss_items, strict=False):
            losses[name] = float(value)

    tloss = getattr(trainer, "tloss", None)
    if isinstance(tloss, dict):
        losses["total_loss"] = float(sum(tloss.values()))
    elif tloss is not None:
        losses["total_loss"] = float(tloss)

    return losses


def extract_learning_rate(trainer: BaseTrainer) -> dict[str, float]:
    """Extract learning rate from trainer optimizer.

    Args:
        trainer: The Ultralytics trainer instance during training.

    Returns:
        Dictionary with parameter group indices and their learning rates.

    """
    learning_rates = {}

    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            lr = param_group.get("lr", 0.0)
            learning_rates[f"param_group_{i}"] = float(lr)

    return learning_rates


def extract_speed_metrics(validator: BaseValidator) -> dict[str, float]:
    """Extract inference speed metrics from validator.

    Args:
        validator: The Ultralytics validator instance after validation.

    Returns:
        Dictionary with speed metrics (preprocess, inference, postprocess) in ms.

    """
    speeds = {}

    if hasattr(validator, "speed") and validator.speed is not None:
        speed_dict = validator.speed
        if isinstance(speed_dict, dict):
            speeds["preprocess_ms"] = float(speed_dict.get("preprocess", 0))
            speeds["inference_ms"] = float(speed_dict.get("inference", 0))
            speeds["postprocess_ms"] = float(speed_dict.get("postprocess", 0))
            speeds["total_ms"] = sum(speeds.values())

    return speeds


def extract_curve_data(
    validator: BaseValidator,
    class_names: list[str] | None = None,  # noqa: ARG001
) -> list[dict[str, Any]] | None:
    """Extract every per-class curve family ultralytics computed.

    Uses ``metrics.curves_results``, which returns ``[x, y, xlabel, ylabel]`` per family
    and is paired positionally with ``metrics.curves`` for the family name. On a
    ``SegmentMetrics`` the list is box curves followed by mask curves
    (utils/metrics.py:1387), so names carry a ``(B)`` / ``(M)`` suffix and the two are
    reported as separate series rather than silently overlaid.

    This replaces a version that read ``box_metrics.py``. That attribute does not exist
    in ultralytics 8.4.110 -- it is ``prec_values`` -- so the old function returned None
    on every run and the PR-curve plot was never reported at all.

    Args:
        validator: The Ultralytics validator instance after validation.
        class_names: Unused; names come from ``metrics.names`` via ``ap_class_index``.

    Returns:
        List of dicts with keys: name, xlabel, ylabel, x, series (list of
        ``{class_name, y}``), or None if unavailable.

    """
    metrics = _get_metrics(validator)
    if metrics is None:
        return None

    try:
        curves_results = metrics.curves_results
        curve_names = list(metrics.curves)
    except Exception as e:
        logger.warning("metrics.curves_results failed, no curve plots: %s", e)
        return None

    if not curves_results:
        return None

    resolved = _resolved_class_names(metrics)

    curves = []
    for i, entry in enumerate(curves_results):
        if entry is None or len(entry) != 4:
            continue
        x, y, xlabel, ylabel = entry
        y_arr = np.atleast_2d(np.asarray(y, dtype=float))
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape[0] == y_arr.shape[1]:
            x_arr, y_arr = _downsample(x_arr, y_arr)
        x_list = x_arr.tolist()

        series = [
            {
                "class_name": resolved[j] if j < len(resolved) else f"class_{j}",
                "y": y_arr[j].tolist(),
            }
            for j in range(y_arr.shape[0])
        ]

        curves.append(
            {
                "name": curve_names[i] if i < len(curve_names) else f"curve_{i}",
                "xlabel": str(xlabel),
                "ylabel": str(ylabel),
                "x": x_list,
                "series": series,
            }
        )

    return curves or None


class ValStatsAccumulator:
    """Capture per-detection (confidence, was-matched) pairs during validation.

    Needed because ``DetectionValidator.get_stats()`` calls ``metrics.process()`` and
    then ``metrics.clear_stats()`` on the very next line (detect/val.py:277-278). By the
    time ``on_val_end`` or ``on_train_end`` runs, ``metrics.stats`` is empty -- so the raw
    material for a TP-vs-FP confidence split and a calibration curve is gone unless it is
    taken during the run.

    ``on_val_batch_end`` fires after ``update_metrics`` and before ``get_stats``
    (engine/validator.py:271 vs :276), which is the only window available. Only
    already-allocated arrays are referenced here and nothing is logged, so this obeys the
    project rule that work inside a per-item loop must not emit output.
    """

    def __init__(self) -> None:
        self._consumed = 0
        self.conf: list[np.ndarray] = []
        self.tp_box: list[np.ndarray] = []
        self.tp_mask: list[np.ndarray] = []

    def reset(self) -> None:
        """Drop everything captured so far, before a new validation pass."""
        self._consumed = 0
        self.conf.clear()
        self.tp_box.clear()
        self.tp_mask.clear()

    def update(self, validator: BaseValidator) -> None:
        """Consume whatever ``update_metrics`` appended since the last call."""
        stats = getattr(getattr(validator, "metrics", None), "stats", None)
        if not stats or "conf" not in stats:
            return

        conf_list = stats["conf"]
        total = len(conf_list)
        if total <= self._consumed:
            # A new validation pass reset the lists under us; start over.
            if total < self._consumed:
                self.reset()
            else:
                return

        for i in range(self._consumed, total):
            try:
                self.conf.append(np.asarray(conf_list[i], dtype=float).ravel())
                # tp is (n_preds, n_iou); column 0 is the IoU=0.50 decision, which is
                # what precision/recall and the per-image metrics both use.
                self.tp_box.append(_first_iou_column(stats["tp"][i]))
                if "tp_m" in stats:
                    self.tp_mask.append(_first_iou_column(stats["tp_m"][i]))
            except Exception as e:  # never let a report break validation
                logger.debug("val stats capture skipped one item: %s", e)
        self._consumed = total

    def confidence_split(self, prefer_mask: bool = True) -> dict[str, np.ndarray] | None:
        """Return ``{"tp": conf_of_matched, "fp": conf_of_unmatched, "basis": ...}``."""
        if not self.conf:
            return None

        tp_source = self.tp_mask if (prefer_mask and self.tp_mask) else self.tp_box
        if not tp_source or len(tp_source) != len(self.conf):
            return None

        conf = np.concatenate(self.conf) if self.conf else np.empty(0)
        matched = np.concatenate(tp_source) if tp_source else np.empty(0)
        if conf.size == 0 or conf.size != matched.size:
            return None

        matched = matched.astype(bool)
        return {
            "tp": conf[matched],
            "fp": conf[~matched],
            "basis": "mask" if tp_source is self.tp_mask else "box",
        }


def _first_iou_column(tp: Any) -> np.ndarray:
    """Return the IoU=0.50 column of a (n_preds, n_iou) true-positive array."""
    arr = np.asarray(tp)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr[:, 0]


def extract_calibration_bins(
    split: dict[str, np.ndarray],
    n_bins: int = 10,
) -> dict[str, Any] | None:
    """Bin detections by confidence and measure observed precision against it.

    A well-calibrated detector's precision inside a confidence bin equals the bin's mean
    confidence, so the curve tracks the diagonal. The summary number is the
    detection expected calibration error,
    ``ECE = sum_b (n_b / N) * |precision(b) - confidence(b)|``.

    Deliberate limitation, worth stating wherever this is displayed: it is computed over
    *predictions only*, so it is blind to false negatives. A model that confidently finds
    one easy object per image and misses everything else scores well here. Always read it
    next to recall.

    Args:
        split: Output of ``ValStatsAccumulator.confidence_split``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        Dict with bin centres, precision, mean confidence, counts and ``ece``, or None.

    """
    if not split:
        return None

    tp_conf = np.asarray(split.get("tp", []), dtype=float)
    fp_conf = np.asarray(split.get("fp", []), dtype=float)
    conf = np.concatenate([tp_conf, fp_conf])
    if conf.size == 0:
        return None
    is_tp = np.concatenate(
        [
            np.ones(tp_conf.size, dtype=bool),
            np.zeros(fp_conf.size, dtype=bool),
        ]
    )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True so the top bin includes conf == 1.0 exactly.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)

    centres, precisions, mean_confs, counts = [], [], [], []
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        centres.append(float((edges[b] + edges[b + 1]) / 2))
        counts.append(n)
        if n == 0:
            precisions.append(float("nan"))
            mean_confs.append(float("nan"))
            continue
        prec = float(is_tp[sel].mean())
        mconf = float(conf[sel].mean())
        precisions.append(prec)
        mean_confs.append(mconf)
        ece += (n / conf.size) * abs(prec - mconf)

    return {
        "bin_center": centres,
        "precision": precisions,
        "mean_confidence": mean_confs,
        "count": counts,
        "ece": float(ece),
        "basis": split.get("basis", "box"),
    }


def extract_mask_box_gap(validator: BaseValidator) -> dict[str, float]:
    """Return the mask-vs-box mAP gap for a segmentation run, else an empty dict.

    A gap that widens over training is the canonical sign that the mask head, not the
    detector, is the problem -- the boxes are found but the masks fitted inside them are
    poor. Reported as a negative number when masks lag, which they normally do.
    """
    metrics = _get_metrics(validator)
    if metrics is None:
        return {}

    seg = _seg_metric(metrics)
    if seg is None:
        return {}

    try:
        return {
            "mask_minus_box_mAP50-95": float(seg.map - metrics.box.map),
            "mask_minus_box_mAP50": float(seg.map50 - metrics.box.map50),
        }
    except Exception as e:
        logger.debug("mask/box gap unavailable: %s", e)
        return {}


def extract_optimal_confidence(validator: BaseValidator) -> dict[str, Any] | None:
    """Find the confidence threshold that maximises F1, globally and per class.

    This is the number that actually ships, and it is not the ``conf`` used to compute
    mAP (0.001) nor the ``conf`` used for the confusion matrix (0.25). Ultralytics picks
    the global operating point as ``smooth(f1_curve.mean(0), 0.1).argmax()``
    (utils/metrics.py:883); we reuse its own ``smooth`` so our number matches the P/R/F1
    scalars it prints rather than being a second, subtly different answer.

    Returns:
        Dict with ``global_conf``, ``global_f1`` and ``per_class`` (a list of
        ``{class_name, conf, f1}``), or None when curve data is unavailable.

    """
    metrics = _get_metrics(validator)
    if metrics is None:
        return None

    box = metrics.box
    f1_curve = getattr(box, "f1_curve", None)
    px = getattr(box, "px", None)
    if f1_curve is None or px is None or len(np.asarray(px)) == 0:
        return None

    f1_curve = np.atleast_2d(np.asarray(f1_curve, dtype=float))
    px = np.asarray(px, dtype=float)
    if f1_curve.size == 0 or f1_curve.shape[1] != px.shape[0]:
        return None

    try:
        from ultralytics.utils.metrics import smooth

        mean_curve = smooth(f1_curve.mean(0), 0.1)
    except Exception:  # pragma: no cover - smooth is stable, but never fail a report
        mean_curve = f1_curve.mean(0)

    gi = int(np.argmax(mean_curve))

    resolved = _resolved_class_names(metrics)

    per_class = []
    for j in range(f1_curve.shape[0]):
        ji = int(np.argmax(f1_curve[j]))
        per_class.append(
            {
                "class_name": resolved[j] if j < len(resolved) else f"class_{j}",
                "conf": float(px[ji]),
                "f1": float(f1_curve[j][ji]),
            }
        )

    return {
        "global_conf": float(px[gi]),
        "global_f1": float(mean_curve[gi]),
        "per_class": per_class,
    }


def extract_worst_images(
    validator: BaseValidator,
    limit: int = 16,
) -> pd.DataFrame | None:
    """Rank validated images worst-first by F1, from ultralytics' per-image metrics.

    ``Metric.image_metrics`` (utils/metrics.py:1085) is a dict keyed by image filename
    holding ``{precision, recall, f1, tp, fp, fn}`` at IoU 0.50. On a segmentation run we
    prefer ``metrics.seg``, whose true positives come from ``mask_iou``
    (segment/val.py:164), so the ranking is genuinely mask-aware rather than a box
    ranking wearing a mask label.

    Only populated when ultralytics ran a full validation pass; empty otherwise.

    An image with no ground truth *and* no prediction is scored 1.0 deliberately by
    ultralytics (utils/metrics.py:1078) as a trivially correct call. Those sort to the
    bottom and so never reach a worst-N list, which is the behaviour we want -- but it
    does mean any *mean* over ``image_metrics`` is inflated by empty images. Do not
    compute one from this table.

    The returned ``Rank`` column is what ties this table to the uploaded error gallery,
    whose ClearML series names are rank-only.

    Args:
        validator: The Ultralytics validator instance after validation.
        limit: How many of the worst images to return.

    Returns:
        DataFrame sorted by F1 ascending, or None when no per-image data exists.

    """
    metrics = _get_metrics(validator)
    if metrics is None:
        return None

    seg = _seg_metric(metrics)
    source = seg if seg is not None and getattr(seg, "image_metrics", None) else None
    if source is None:
        source = metrics.box
    basis = "mask" if source is seg else "box"

    image_metrics = getattr(source, "image_metrics", None)
    if not image_metrics:
        return None

    rows = [
        {
            "Image": name,
            "F1": float(m.get("f1", 0.0)),
            "Precision": float(m.get("precision", 0.0)),
            "Recall": float(m.get("recall", 0.0)),
            "TP": int(m.get("tp", 0)),
            "FP": int(m.get("fp", 0)),
            "FN": int(m.get("fn", 0)),
        }
        for name, m in image_metrics.items()
    ]
    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Break F1 ties by absolute error count so the noisiest images float to the top.
    df = df.sort_values(
        by=["F1", "FN", "FP"], ascending=[True, False, False], kind="stable"
    )
    df = df.head(limit).reset_index(drop=True)
    df.insert(0, "Rank", [f"{i:02d}" for i in range(len(df))])
    df.attrs["basis"] = basis
    return df


def collect_prediction_confidences(
    results: list,
    class_names: list[str] | None = None,
) -> tuple[list[float], dict[str, list[float]]]:
    """Collect confidence scores from prediction results.

    Args:
        results: List of Ultralytics prediction results.
        class_names: Optional list of class names.

    Returns:
        Tuple of (all_confidences, per_class_confidences).

    """
    all_confidences = []
    per_class_confidences: dict[str, list[float]] = {}

    for r in results:
        boxes = r.boxes if hasattr(r, "boxes") else None
        if boxes is None:
            continue

        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []

        for conf, cls_id in zip(confs, classes, strict=False):
            conf_value = float(conf)
            all_confidences.append(conf_value)

            cls_idx = int(cls_id)
            if class_names and cls_idx < len(class_names):
                cls_name = class_names[cls_idx]
            else:
                cls_name = f"class_{cls_idx}"

            if cls_name not in per_class_confidences:
                per_class_confidences[cls_name] = []
            per_class_confidences[cls_name].append(conf_value)

    return all_confidences, per_class_confidences
