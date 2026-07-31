"""TIDE-style error decomposition over the captured per-image records.

Six error types -- Cls, Loc, Both, Dupe, Bkg, Miss -- plus the ΔAP each one costs,
measured by an independent oracle that fixes only that error and re-runs AP. The six
ΔAP values are *not* additive: each is measured against the same baseline with every
other error left in place, which is the whole point (fixing localisation first changes
what fixing duplication is worth). Two ceilings are reported alongside them: removing
every false positive, and removing every missed ground truth.

Box IoU only, even on a segmentation run. The pairwise mask IoU matrix is computed and
then discarded inside ultralytics' segmentation `_process_batch`, so retaining it would
mean carrying full mask tensors per image. The mask story is told by the box-vs-mask
section instead, and both captions say so.

`compute_ap` is ultralytics' own, so the baseline AP50 here is directly comparable to
the headline mAP50 rather than a second, subtly different answer.
"""

from __future__ import annotations
from typing import Any

import numpy as np

from src.report.capture import MATCH_IOU, box_iou_np, greedy_match
from src.utils.logging import get_logger


logger = get_logger(__name__)

T_F = 0.5  # foreground IoU: at or above this a detection is on the object
T_B = 0.1  # background IoU: below this a detection is not on any object at all

# The order the stacked bar draws them in, which is also palette slot order.
ERROR_TYPES = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")

RECOMMENDATIONS = {
    "Loc": "raise `imgsz` or the `box` loss gain; check `4_Training/box`",
    "Cls": "classes are confusable -- see the top confused pairs and the class-map order",
    "Both": "both class and localisation are wrong -- usually a very rare class",
    "Dupe": "raise `5_Testing/iou` (NMS)",
    "Bkg": (
        "detections on unannotated regions -- suspect **missing annotations** before "
        "blaming the model"
    ),
    "Miss": (
        "recall levers: lower the deploy threshold, raise `max_det`, or add data for "
        "the weakest classes"
    ),
}


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    from ultralytics.utils.metrics import compute_ap

    return float(compute_ap(recall.tolist(), precision.tolist())[0])


def _ap50(
    det_cls: np.ndarray,
    det_conf: np.ndarray,
    det_tp: np.ndarray,
    keep: np.ndarray,
    n_gt: dict[int, int],
) -> float:
    """Mean AP50 over classes with ground truth, from a detection list and a tp vector."""
    aps = []
    for c, n in n_gt.items():
        if n <= 0:
            continue
        sel = keep & (det_cls == c)
        if not sel.any():
            aps.append(0.0)
            continue
        order = np.argsort(-det_conf[sel])
        tp = det_tp[sel][order].astype(np.float64)
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1.0 - tp)
        recall = ctp / n
        precision = ctp / np.maximum(ctp + cfp, 1e-9)
        aps.append(_compute_ap(recall, precision))
    return float(np.mean(aps)) if aps else 0.0


def _type_fp(
    d: int,
    iou: np.ndarray,
    det_cls: np.ndarray,
    gt_cls: np.ndarray,
    consumed: set[int],
    m: int,
) -> tuple[str, int, int]:
    """Type one unmatched detection, and name the ground truth an oracle would give it.

    The order of the tests is the definition, not an optimisation: a detection sitting
    on an already-claimed same-class object is a duplicate *before* it is anything else,
    and only a detection touching nothing at all is a background error.
    """
    same = np.where(gt_cls == det_cls[d])[0]
    iou_same, g_same = 0.0, -1
    if same.size and iou.size:
        k = int(same[int(np.argmax(iou[same, d]))])
        iou_same, g_same = float(iou[k, d]), k
    iou_any, g_any = 0.0, -1
    if m and iou.size:
        k = int(np.argmax(iou[:, d]))
        iou_any, g_any = float(iou[k, d]), k

    if iou_same >= T_F and g_same in consumed:
        return "Dupe", int(det_cls[d]), g_same
    if T_B <= iou_same < T_F:
        return "Loc", int(det_cls[d]), g_same
    if iou_any >= T_F:
        return "Cls", int(gt_cls[g_any]), g_any
    if T_B <= iou_any < T_F:
        return "Both", int(gt_cls[g_any]), g_any
    return "Bkg", int(det_cls[d]), -1


def _assemble(records: list[dict]) -> dict[str, Any]:
    """Flatten the sampled records into detection and ground-truth arrays.

    IoU is recomputed here per image rather than retained by the capture, so the
    reservoir stays at boxes-and-scores and never holds an `n x m` matrix per image.
    """
    d_img, d_cls, d_conf = [], [], []
    d_type, d_alt_cls, d_ogt = [], [], []
    d_tp = []
    g_cls, g_missed = [], []
    gt_offset = 0
    d_gt_index: list[int] = []

    for img_id, rec in enumerate(records):
        det_box, gt_box = rec["det_box"], rec["gt_box"]
        det_cls = rec["det_cls"].astype(np.int64)
        gt_cls = rec["gt_cls"].astype(np.int64)
        conf = rec["det_conf"].astype(np.float64)
        n, m = det_box.shape[0], gt_box.shape[0]
        iou = box_iou_np(gt_box, det_box)
        order = np.argsort(-conf)
        matched = greedy_match(iou, det_cls, gt_cls, order, T_F)
        consumed = set(matched.values())

        for d in range(n):
            d_img.append(img_id)
            d_cls.append(int(det_cls[d]))
            d_conf.append(float(conf[d]))
            if d in matched:
                d_tp.append(True)
                d_type.append("TP")
                d_alt_cls.append(int(det_cls[d]))
                d_ogt.append(gt_offset + matched[d])
                d_gt_index.append(gt_offset + matched[d])
                continue
            d_tp.append(False)
            d_gt_index.append(-1)
            kind, alt, ogt = _type_fp(d, iou, det_cls, gt_cls, consumed, m)
            d_type.append(kind)
            d_alt_cls.append(alt)
            d_ogt.append(gt_offset + ogt if ogt >= 0 else -1)

        for g in range(m):
            g_cls.append(int(gt_cls[g]))
            g_missed.append(g not in consumed)
        gt_offset += m

    return {
        "img": np.asarray(d_img, dtype=np.int64),
        "cls": np.asarray(d_cls, dtype=np.int64),
        "conf": np.asarray(d_conf, dtype=np.float64),
        "tp": np.asarray(d_tp, dtype=bool),
        "type": np.asarray(d_type, dtype=object),
        "alt_cls": np.asarray(d_alt_cls, dtype=np.int64),
        "ogt": np.asarray(d_ogt, dtype=np.int64),
        "gt_index": np.asarray(d_gt_index, dtype=np.int64),
        "gt_cls": np.asarray(g_cls, dtype=np.int64),
        "gt_missed": np.asarray(g_missed, dtype=bool),
    }


def _counts(flat: dict[str, Any]) -> dict[str, int]:
    counts = {t: int((flat["type"] == t).sum()) for t in ERROR_TYPES if t != "Miss"}
    counts["Miss"] = int(flat["gt_missed"].sum())
    return counts


def _n_gt(gt_cls: np.ndarray, mask: np.ndarray | None = None) -> dict[int, int]:
    sel = gt_cls if mask is None else gt_cls[mask]
    unique, counts = np.unique(sel, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist(), strict=True)}


def _oracle(
    flat: dict[str, Any], kind: str
) -> tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
    """Return (keep, tp, per-class GT counts, class labels) after fixing one error."""
    keep = np.ones(flat["cls"].shape[0], dtype=bool)
    tp = flat["tp"].copy()
    cls = flat["cls"].copy()
    n_gt = _n_gt(flat["gt_cls"])

    if kind in ("Dupe", "Bkg"):
        keep &= flat["type"] != kind
        return keep, tp, n_gt, cls

    if kind == "Miss":
        alive = ~flat["gt_missed"]
        return keep, tp, _n_gt(flat["gt_cls"], alive), cls

    if kind == "FP":
        keep &= flat["tp"]
        return keep, tp, n_gt, cls

    if kind == "FN":
        alive = ~flat["gt_missed"]
        return keep, tp, _n_gt(flat["gt_cls"], alive), cls

    targets = {"Loc": ("Loc",), "Cls": ("Cls",), "Both": ("Both",)}[kind]
    consumed = set(flat["gt_index"][flat["tp"]].tolist())
    # Promote in descending confidence so a second detection on the same object stays a
    # false positive, exactly as the baseline match would have decided it.
    idx = np.where(np.isin(flat["type"], targets))[0]
    for d in idx[np.argsort(-flat["conf"][idx])]:
        g = int(flat["ogt"][d])
        if g < 0 or g in consumed:
            continue
        consumed.add(g)
        tp[d] = True
        cls[d] = int(flat["alt_cls"][d])
    return keep, tp, n_gt, cls


def compute_tide(
    records: list[dict],
    names: list[str],
    *,
    delta_ap: bool = True,
    sampled_of: int | None = None,
) -> dict[str, Any] | None:
    """Return the TIDE decomposition, or None when there is nothing to decompose.

    `delta_ap=False` (or an empty sample) yields the counts-only variant, which is what
    the report shows when the capture is unavailable; the section title then names the
    reason instead of quietly showing a different thing under the same heading.
    """
    if not records:
        return None
    flat = _assemble(records)
    if flat["cls"].size == 0 and flat["gt_cls"].size == 0:
        return None

    counts = _counts(flat)
    out: dict[str, Any] = {
        "mode": "counts",
        "counts": counts,
        "types": list(ERROR_TYPES),
        "delta_ap": {},
        "ceilings": {},
        "baseline_ap50": None,
        "n_images": len(records),
        "sampled_of": sampled_of if sampled_of is not None else len(records),
        "n_classes": len(names),
        "recommendations": dict(RECOMMENDATIONS),
    }
    if not delta_ap:
        return out

    try:
        keep0 = np.ones(flat["cls"].shape[0], dtype=bool)
        n_gt0 = _n_gt(flat["gt_cls"])
        base = _ap50(flat["cls"], flat["conf"], flat["tp"], keep0, n_gt0)
        deltas = {}
        for kind in ERROR_TYPES:
            keep, tp, n_gt, cls = _oracle(flat, kind)
            deltas[kind] = round(_ap50(cls, flat["conf"], tp, keep, n_gt) - base, 6)
        ceilings = {}
        for label, kind in (("fp", "FP"), ("fn", "FN")):
            keep, tp, n_gt, cls = _oracle(flat, kind)
            ceilings[label] = round(_ap50(cls, flat["conf"], tp, keep, n_gt) - base, 6)
    except Exception as e:
        logger.warning("TIDE delta-AP oracles failed, falling back to counts: %s", e)
        return out

    out["mode"] = "delta_ap"
    out["baseline_ap50"] = round(base, 6)
    out["delta_ap"] = deltas
    out["ceilings"] = ceilings
    return out


def match_iou_note() -> str:
    """Return the caption every TIDE display has to carry."""
    return (
        f"Matching at IoU {T_F:g} (foreground) and {T_B:g} (background), box IoU only "
        "-- even on a segmentation run. The six ΔAP values are independent oracles and "
        "do not sum to the total headroom."
    )


__all__ = [
    "ERROR_TYPES",
    "MATCH_IOU",
    "RECOMMENDATIONS",
    "compute_tide",
    "match_iou_note",
]
