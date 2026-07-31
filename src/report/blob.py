"""Assembly of the report's single JSON blob, and the only place caps are applied.

The page carries all of its data in one `<script type="application/json">` element and
builds its charts and galleries from it client-side. That is the pytest-html pattern
rather than the ydata-profiling one, and it is what keeps the markup small enough to
stay inside a DOM-node budget while the data stays inspectable.

Every cap in `CAPS` is enforced here and asserted by `tests/report/test_report_size.py`.
Nothing in the blob is a per-image or per-detection array: every distribution arrives
pre-binned, every gallery is a fixed item count, and the only epoch-dependent payload
(the training curves) is downsampled. That is what makes the file size independent of
both dataset size and epoch count -- the project's report-volume law.
"""

from __future__ import annotations
import math

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.report import figures as figs
from src.report import theme
from src.report.dataset_scan import DatasetScan, scan_dataset
from src.report.thumbs import ThumbnailStore
from src.report.tide import compute_tide
from src.utils.logging import get_logger


logger = get_logger(__name__)

SCHEMA = 1

MAX_FIGURES = 24
MAX_TABLE_ROWS = 120
MAX_CM_CLASSES = 60
MAX_HIST_BINS = 50
MAX_BOXES_PER_ITEM = 20
MAX_CLASS_NAMES = 1000
MAX_WARNINGS = 30
MAX_DEGRADATIONS = 20
MAX_CONFIG_FULL = 200
MAX_CONFIG_DIFF = 60
MAX_KPIS = 12
MAX_GRIDS = 6

CAPS = {
    "figures": MAX_FIGURES,
    "table_rows": MAX_TABLE_ROWS,
    "cm_classes": MAX_CM_CLASSES,
    "hist_bins": MAX_HIST_BINS,
    "boxes_per_item": MAX_BOXES_PER_ITEM,
    "class_names": MAX_CLASS_NAMES,
    "warnings": MAX_WARNINGS,
    "degradations": MAX_DEGRADATIONS,
    "kpis": MAX_KPIS,
    "grids": MAX_GRIDS,
}

GRID_SPECS = (
    (
        "g_worst",
        "Worst images",
        "Ranked by false positives plus false negatives",
        "whole image, conf >= {conf}",
    ),
    (
        "g_hiconf_fp",
        "High-confidence false positives",
        "The missing-annotation detector: the model is sure and the labels disagree",
        "instance crop, score >= {hi}",
    ),
    (
        "g_missed",
        "Missed ground truth",
        "Largest first -- a big miss is rarely a hard example",
        "instance crop, unmatched at IoU 0.50",
    ),
    (
        "g_lowconf_tp",
        "Low-confidence true positives",
        "Correct detections a deploy threshold would throw away",
        "instance crop, lowest scoring matches",
    ),
    (
        "g_random",
        "Random control",
        "A fixed-seed sample, so the other four grids can be read against something",
        "whole image, seeded sample",
    ),
)

AR_BUCKETS = ("tall", "portrait", "square", "landscape", "wide")


@dataclass
class ReportContext:
    """Everything `build_blob` reads. All of it optional; the report degrades."""

    task: Any = None
    validator: Any = None
    final_metrics: Any = None
    trainer: Any = None
    val_stats: Any = None
    capture: Any = None
    dataset_dir: str | None = None
    datadotyaml: dict | None = None
    class_2_idx: dict | None = None
    task_yolo: str = "detect"
    split_label: str = ""
    args_train: dict = field(default_factory=dict)
    args_val: dict = field(default_factory=dict)
    args_visualization: dict = field(default_factory=dict)
    dataset_scan: DatasetScan | None = None


class _Builder:
    """One report build. Collects degradations and warnings as it goes."""

    def __init__(self, ctx: ReportContext) -> None:
        self.ctx = ctx
        self.cfg = ctx.args_visualization or {}
        self.degradations: list[str] = []
        self.warnings: list[dict[str, str]] = []
        self.figures: dict[str, Any] = {}
        self.tables: dict[str, Any] = {}
        self.grids: dict[str, Any] = {}
        self.metrics = getattr(ctx.validator, "metrics", None) or ctx.final_metrics
        self.names = self._names()
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.is_seg = _is_segment(self.metrics, ctx.task_yolo)
        self.summary = self._summary()
        self.optimal = self._optimal()
        self.thumbs = ThumbnailStore(
            size=int(self.cfg.get("report_thumbnail_px", 192)),
            max_thumbs=int(self.cfg.get("report_max_thumbnails", 200)),
        )

    # --- inputs ---------------------------------------------------------------------
    def _names(self) -> list[str]:
        source = self.ctx.class_2_idx
        if source:
            names = [n for n, _ in sorted(source.items(), key=lambda kv: kv[1])]
        else:
            raw = getattr(self.metrics, "names", None) or {}
            if isinstance(raw, dict):
                names = [str(raw[k]) for k in sorted(raw)]
            else:
                names = [str(n) for n in raw]
        return names[:MAX_CLASS_NAMES]

    def _summary(self) -> list[dict[str, Any]]:
        if self.metrics is None:
            return []
        try:
            return list(self.metrics.summary())
        except Exception as e:
            logger.warning("metrics.summary() failed, no per-class table: %s", e)
            self.degradations.append(
                "Per-class metrics were unavailable (`metrics.summary()` failed)."
            )
            return []

    def _optimal(self) -> dict[str, Any] | None:
        from src.yolov8.metrics_utils import extract_optimal_confidence

        if self.ctx.validator is None and self.metrics is None:
            return None
        try:
            return extract_optimal_confidence(
                self.ctx.validator
                if self.ctx.validator is not None
                else _Wrap(self.metrics)
            )
        except Exception as e:
            logger.debug("optimal confidence unavailable: %s", e)
            return None

    # --- top level ------------------------------------------------------------------
    def build(self) -> dict[str, Any]:
        self._scan_dataset()
        meta = self._meta()
        kpis = self._kpis()
        self._section_dataset()
        self._section_per_class()
        self._section_confusion()
        self._section_threshold()
        tide = self._section_tide()
        self._section_strata()
        if self.is_seg:
            self._section_boxmask()
        self._section_galleries()
        self._section_training()
        self._collect_capture_degradations()
        self._build_warnings(meta)

        blob = {
            "schema": SCHEMA,
            "meta": meta,
            "kpis": kpis[:MAX_KPIS],
            "model_card": self._model_card(),
            "warnings": self.warnings[:MAX_WARNINGS],
            "classes": self.names,
            "figures": dict(list(self.figures.items())[:MAX_FIGURES]),
            "plotly_template": {
                "light": theme.plotly_template(False),
                "dark": theme.plotly_template(True),
            },
            "tables": self.tables,
            "grids": dict(list(self.grids.items())[:MAX_GRIDS]),
            "thumbs": self.thumbs.thumbs,
            "tide_mode": (tide or {}).get("mode", "none"),
            "caveats": self._caveats(),
            "degradations": self.degradations[:MAX_DEGRADATIONS],
        }
        self.thumbs.log_summary()
        return blob

    # --- section 1 / 2 --------------------------------------------------------------
    def _meta(self) -> dict[str, Any]:
        import ultralytics

        from src.params import DOCKER_IMAGE, VERSION

        task = self.ctx.task
        val_args = getattr(self.ctx.validator, "args", None)
        # Never `args_val`: train.py overwrites batch, split and visualize *after* the
        # ClearML connect, so the connected group is not what the pass ran with.
        keys = ("conf", "iou", "max_det", "batch", "imgsz", "half", "rect", "split")
        actual = {k: _plain(getattr(val_args, k, None)) for k in keys}

        scan = self.ctx.dataset_scan
        split_counts = scan.split_counts() if scan else {}
        tau = (self.optimal or {}).get("global_conf")
        per_tau = {
            str(e["class_name"]): round(float(e["conf"]), 4)
            for e in (self.optimal or {}).get("per_class", [])
        }
        return {
            "run_name": str(getattr(task, "name", "") or "training run")[:200],
            "task_id": str(getattr(task, "id", "") or ""),
            "task_url": _task_url(task),
            "project": str(_task_project(task)),
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(task),
            "image_tag": DOCKER_IMAGE,
            "code_version": VERSION,
            "ultralytics": ultralytics.__version__,
            "task_yolo": "segment" if self.is_seg else str(self.ctx.task_yolo),
            "is_seg": self.is_seg,
            "model_name": str(self.ctx.args_train.get("model_name") or _model(self.ctx)),
            "imgsz": int(actual.get("imgsz") or self.ctx.args_train.get("imgsz") or 640),
            "split_name": str(
                actual.get("split") or self.ctx.split_label.strip() or "val"
            ),
            "split_counts": split_counts,
            "split_label": self.ctx.split_label.strip() or "Final",
            "val_images": int(getattr(self.ctx.validator, "seen", 0) or 0),
            "val_args": actual,
            "thresholds": {
                "nms_floor": float(actual.get("conf") or 0.001),
                "matrix_display": float(self.cfg.get("confusion_matrix_conf", 0.25)),
                "f1_optimal": None if tau is None else round(float(tau), 4),
                "per_class_tau": dict(list(per_tau.items())[:MAX_CLASS_NAMES]),
            },
        }

    def _kpis(self) -> list[dict[str, Any]]:
        results = dict(getattr(self.metrics, "results_dict", None) or {})
        tau = (self.optimal or {}).get("global_conf")
        tau_txt = "not computed" if tau is None else f"tau* = {tau:.3f}"
        nms = float(getattr(getattr(self.ctx.validator, "args", None), "conf", 0.001))
        map_basis = f"mAP, conf={nms:g}, IoU 0.50:0.95"
        map50_basis = f"mAP, conf={nms:g}, IoU 0.50"
        pt_basis = f"per-class max-F1 operating point, global {tau_txt}"

        tiles: list[dict[str, Any]] = [
            _kpi("Box mAP50-95", results.get("metrics/mAP50-95(B)"), "num2", map_basis),
            _kpi("Box mAP50", results.get("metrics/mAP50(B)"), "num2", map50_basis),
        ]
        if self.is_seg:
            tiles += [
                _kpi(
                    "Mask mAP50-95", results.get("metrics/mAP50-95(M)"), "num2", map_basis
                ),
                _kpi("Mask mAP50", results.get("metrics/mAP50(M)"), "num2", map50_basis),
            ]
        tiles += [
            _kpi("Precision", results.get("metrics/precision(B)"), "num2", pt_basis),
            _kpi("Recall", results.get("metrics/recall(B)"), "num2", pt_basis),
            _kpi("F1", _f1(results), "num2", pt_basis),
        ]
        fitness = results.get("fitness")
        tiles.append(
            {
                "label": "Fitness",
                "value": None if fitness is None else round(float(fitness), 4),
                "fmt": "num2",
                # scale is None on a seg run: SegmentMetrics.fitness is box + mask
                # mAP50-95, i.e. 0-2. A meter or a percentage would be a lie.
                "scale": None if self.is_seg else [0.0, 1.0],
                "basis": (
                    "0-2 (box + mask mAP50-95)"
                    if self.is_seg
                    else "= box mAP50-95 on a detect run"
                ),
                "tone": "neutral",
            }
        )
        return tiles

    def _model_card(self) -> dict[str, Any]:
        run_args = _args_dict(getattr(self.ctx.trainer, "args", None))
        diff, full = [], []
        try:
            from ultralytics.utils import DEFAULT_CFG_DICT

            for key in sorted(run_args):
                value = run_args[key]
                full.append([key, _short(value)])
                default = DEFAULT_CFG_DICT.get(key, "<absent>")
                if _short(default) != _short(value):
                    diff.append([key, _short(value), _short(default)])
        except Exception as e:
            logger.debug("config diff unavailable: %s", e)

        thr = self.ctx.args_visualization
        nms = float(getattr(getattr(self.ctx.validator, "args", None), "conf", 0.001))
        tau = (self.optimal or {}).get("global_conf")
        note = (
            f"Three confidence thresholds coexist in this task and they are not "
            f"alternatives. mAP and every curve are computed at the NMS floor "
            f"({nms:g}) -- anything below it never reaches the metrics at all. The "
            f"confusion matrix and every gallery here are drawn at the display "
            f"threshold ({float(thr.get('confusion_matrix_conf', 0.25)):g}), because a "
            f"matrix at the floor is all background. The threshold you would deploy at "
            f"is the F1-optimal one "
            + ("(not computed)." if tau is None else f"({float(tau):.3f}).")
        )
        return {
            "intended_use": str(self.cfg.get("report_intended_use") or "")[:600]
            or (
                "Not stated. Set `8_Visualization/report_intended_use` to record what "
                "this model is for."
            ),
            "out_of_scope": str(self.cfg.get("report_out_of_scope") or "")[:600]
            or (
                "Not stated. Set `8_Visualization/report_out_of_scope` to record what "
                "this model must not be used for."
            ),
            "config_diff": diff[:MAX_CONFIG_DIFF],
            "config_full": full[:MAX_CONFIG_FULL],
            "thresholds_note": note,
        }

    # --- section 3 ------------------------------------------------------------------
    def _scan_dataset(self) -> None:
        if self.ctx.dataset_scan is not None:
            return
        if not self.cfg.get("report_scan_labels", True):
            self.degradations.append(
                "Label scanning was disabled (`report_scan_labels=False`), so the "
                "split composition section is unavailable."
            )
            return
        imgsz = int(self.ctx.args_train.get("imgsz") or 640)
        try:
            self.ctx.dataset_scan = scan_dataset(self.ctx.dataset_dir, self.names, imgsz)
        except Exception as e:
            logger.warning("dataset scan failed: %s", e)
            self.ctx.dataset_scan = None
        if self.ctx.dataset_scan is None:
            self.degradations.append(
                "No YOLO label directories were readable under the dataset directory, "
                "so the dataset section shows pre-split figures only."
            )

    def _section_dataset(self) -> None:
        scan = self.ctx.dataset_scan
        if scan is None:
            return
        imgsz = scan.imgsz
        totals = {i: 0 for i in range(len(self.names))}
        per_split: dict[str, list[int]] = {}
        for split in ("train", "valid", "test"):
            s = scan.splits.get(split)
            counts = [0] * len(self.names)
            if s:
                for cls, n in s.instances.items():
                    if 0 <= cls < len(counts):
                        counts[cls] += n
                        totals[cls] += n
            if s:
                per_split[split] = counts
        order = sorted(range(len(self.names)), key=lambda i: totals[i])

        self._fig(
            "f_class_bars",
            figs.class_bars,
            self.names,
            [totals[i] for i in range(len(self.names))],
        )
        if len(per_split) > 1:
            self._fig("f_split_stack", figs.split_stack, self.names, per_split, order)
        p995 = figs.percentile_from_counter(scan.objects_per_image, 0.995)
        self._fig(
            "f_objects_per_image",
            figs.objects_per_image,
            dict(scan.objects_per_image),
            p995,
        )
        terciles = scan.terciles()
        self._fig("f_box_size", figs.box_size, scan.area_hist, terciles, imgsz)
        self._fig("f_aspect", figs.aspect_ratio, scan.ar_hist)
        self._fig("f_center_heat", figs.centre_heatmap, scan.heat)
        if self.is_seg and scan.polygon_instances:
            frac = scan.rectangle_polygons / max(scan.polygon_instances, 1)
            self._fig("f_mask_fill", figs.mask_fill, scan.fill_hist, frac)
            self._fig("f_poly_vertices", figs.polygon_vertices, dict(scan.vertex_hist))
            if frac > 0.2:
                self.warnings.append(
                    {
                        "severity": "serious",
                        "text": (
                            f"{frac:.0%} of polygons fill more than 95% of their "
                            "bounding box -- these are rectangles drawn as polygons, "
                            "and mask mAP measured against them is close to box mAP."
                        ),
                    }
                )

        # Valid ascending, so a class with no ground truth in the validation split is
        # the first row -- that class is silently absent from mAP.
        valid = per_split.get("valid") or [0] * len(self.names)
        rows = [
            [
                self.names[i],
                (per_split.get("train") or [0] * len(self.names))[i],
                valid[i],
                (per_split.get("test") or [0] * len(self.names))[i],
                totals[i],
            ]
            for i in sorted(range(len(self.names)), key=lambda i: (valid[i], -totals[i]))
        ]
        self.tables["t_split_composition"] = {
            "columns": ["Class", "train", "valid", "test", "total"],
            "rows": rows[:MAX_TABLE_ROWS],
            "sort": "valid:asc",
            "note": (
                f"Counted from the label files on disk. Object sizes elsewhere use this "
                f"dataset's own terciles at {terciles[0] * imgsz:.0f}px and "
                f"{terciles[1] * imgsz:.0f}px (imgsz={imgsz}), not COCO's 32/96 "
                f"absolutes."
            ),
        }
        self.tables["t_quality_flags"] = {
            "columns": ["Flag", "Count", "Example"],
            "rows": [
                [flag, count, ", ".join(scan.flags.examples.get(flag, [])) or "--"]
                for flag, count in scan.flags.counts.most_common(MAX_TABLE_ROWS)
            ],
            "sort": "Count:desc",
            "note": "Heuristics over the label files; each is a smell, not a verdict.",
        }

    # --- section 4 ------------------------------------------------------------------
    def _section_per_class(self) -> None:
        if not self.summary:
            return
        cap = self.capture_or_none()
        tau_by_class = {
            str(e["class_name"]): float(e["conf"])
            for e in (self.optimal or {}).get("per_class", [])
        }
        seg = getattr(self.metrics, "seg", None) if self.is_seg else None
        # No `or []`: these are numpy arrays, whose truth value raises.
        mask_ap = np.asarray(getattr(seg, "ap", []), dtype=float)
        mask_ap50 = np.asarray(getattr(seg, "ap50", []), dtype=float)

        columns = ["Class", "Support", "AP50-95", "AP50"]
        if self.is_seg and mask_ap.size == len(self.summary):
            columns += ["Mask AP50-95", "Mask AP50"]
        columns += ["P", "R", "F1", "TP", "FP", "FN", "tau*"]

        rows, tones = [], []
        low = int(self.cfg.get("report_low_support_threshold", 30))
        for i, entry in enumerate(self.summary):
            name = str(entry.get("Class", f"class_{i}"))
            idx = self.name_to_idx.get(name, -1)
            support = int(entry.get("Instances", 0) or 0)
            row = [
                name,
                support,
                _r(entry.get("mAP50-95")),
                _r(entry.get("mAP50")),
            ]
            if "Mask AP50-95" in columns:
                row += [_r(mask_ap[i]), _r(mask_ap50[i])]
            row += [
                _r(entry.get("Box-P")),
                _r(entry.get("Box-R")),
                _r(entry.get("Box-F1")),
                int(cap.tp_by_class.get(idx, 0)) if cap else None,
                int(cap.fp_by_class.get(idx, 0)) if cap else None,
                int(cap.fn_by_class.get(idx, 0)) if cap else None,
                _r(tau_by_class.get(name)),
            ]
            rows.append(row)
            tones.append("dim" if 0 < support < low else None)

        order = sorted(range(len(rows)), key=lambda i: rows[i][1])
        self.tables["t_per_class"] = {
            "columns": columns,
            "rows": [rows[i] for i in order][:MAX_TABLE_ROWS],
            "row_tone": [tones[i] for i in order][:MAX_TABLE_ROWS],
            "sort": "Support:asc",
            "class_col": 0,
            "note": (
                "Built from `metrics.summary()`, which resolves names through "
                "`ap_class_index` -- the only index-safe source. P/R/F1 are at each "
                "class's own max-F1 point; TP/FP/FN are counted at IoU 0.50 and the "
                f"display threshold {self.meta_display_conf():g}. Classes with fewer "
                f"than {low} instances are dimmed: their AP is not reliable."
            ),
        }
        if len(rows) > MAX_TABLE_ROWS:
            self.degradations.append(
                f"The per-class table shows {MAX_TABLE_ROWS} of {len(rows)} classes; "
                "the full set is in the ClearML per-class table."
            )

    # --- section 5 ------------------------------------------------------------------
    def _section_confusion(self) -> None:
        from src.yolov8.metrics_utils import extract_confusion_matrix_data

        try:
            _, labels, counts = extract_confusion_matrix_data(
                self.ctx.validator, self.names or None
            )
        except Exception as e:
            logger.debug("confusion matrix unavailable: %s", e)
            labels, counts = [], None
        if counts is None or not labels:
            self.degradations.append(
                "The confusion matrix was not available; it requires "
                "`5_Testing/plots=True`."
            )
        elif float(np.asarray(counts, dtype=float).sum()) == 0.0:
            self.degradations.append(
                "The confusion matrix was empty -- it requires `5_Testing/plots=True`."
            )
        else:
            counts = np.asarray(counts, dtype=float)
            cap = int(self.cfg.get("report_cm_max_classes", MAX_CM_CLASSES))
            cap = min(cap, MAX_CM_CLASSES)
            labels, counts, truncated = _cap_matrix(labels, counts, cap)
            payload = figs.safe(figs.confusion, labels, counts)
            if payload is not None:
                self.figures["f_confusion"] = payload
            if truncated:
                self.degradations.append(
                    f"The confusion matrix shows the {cap} classes with the most "
                    f"ground truth, of {truncated}."
                )

        cap_obj = self.capture_or_none()
        if cap_obj and cap_obj.confused:
            self.tables["t_confused_pairs"] = self._confused_table(cap_obj)

    def _confused_table(self, cap: Any) -> dict[str, Any]:
        totals = cap.gt_by_class
        seen: set[tuple[int, int]] = set()
        rows = []
        for (i, j), n_ij in cap.confused.items():
            key = (min(i, j), max(i, j))
            if key in seen or i == j:
                continue
            seen.add(key)
            n_ji = cap.confused.get((j, i), 0)
            denom = totals.get(i, 0) + totals.get(j, 0)
            rate = (n_ij + n_ji) / denom if denom else 0.0
            rows.append(
                [
                    f"{cap._name(i)} <-> {cap._name(j)}",
                    int(n_ij),
                    int(n_ji),
                    round(float(rate), 4),
                    f"{cap._name(i)}>{cap._name(j)}",
                ]
            )
        rows.sort(key=lambda r: -r[3])
        return {
            "columns": ["Pair", "n(i->j)", "n(j->i)", "rate", "pair"],
            "rows": rows[:5],
            "sort": "rate:desc",
            "pair_col": 4,
            "hidden": [4],
            "note": (
                "Rate is (n_ij + n_ji) / (n_i + n_j) over ground-truth instances. "
                "Click a row to filter the galleries to that pair."
            ),
        }

    # --- section 6 ------------------------------------------------------------------
    def _section_threshold(self) -> None:
        from src.yolov8.clearml_logger import binned_counts
        from src.yolov8.metrics_utils import extract_calibration_bins, extract_curve_data

        tau = (self.optimal or {}).get("global_conf")
        curves = None
        try:
            curves = extract_curve_data(self.ctx.validator)
        except Exception as e:
            logger.debug("curve data unavailable: %s", e)
        if curves:
            self._fig("f_pr_f1_conf", figs.pr_f1_vs_conf, curves, tau)
        else:
            self.degradations.append(
                "Precision/recall/F1 curves were not available from this validation pass."
            )

        split = None
        if self.ctx.val_stats is not None:
            try:
                split = self.ctx.val_stats.confidence_split()
            except Exception as e:
                logger.debug("confidence split unavailable: %s", e)
        if not split:
            self.degradations.append(
                "The TP-vs-FP confidence split and the reliability diagram need the "
                "`8_Visualization/log_calibration` capture, which did not run."
            )
            return
        centres, tp_counts, _ = binned_counts(split["tp"], bins=MAX_HIST_BINS, lo=0, hi=1)
        _, fp_counts, _ = binned_counts(split["fp"], bins=MAX_HIST_BINS, lo=0, hi=1)
        if centres:
            self._fig("f_tp_fp_hist", figs.tp_fp_hist, centres, tp_counts, fp_counts)
        calibration = extract_calibration_bins(split)
        if calibration:
            calibration = dict(calibration)
            calibration["precision"] = figs.nan_to_none(calibration["precision"])
            calibration["mean_confidence"] = figs.nan_to_none(
                calibration["mean_confidence"]
            )
            self._fig("f_reliability", figs.reliability, calibration)

    # --- section 7 ------------------------------------------------------------------
    def _section_tide(self) -> dict[str, Any] | None:
        cap = self.capture_or_none()
        if cap is None or not cap.records:
            self.degradations.append(
                "The error decomposition was not computed: no per-image capture was "
                "available for this validation pass."
            )
            return None
        want_delta = bool(self.cfg.get("report_tide", True))
        tide = compute_tide(
            cap.records,
            self.names,
            delta_ap=want_delta,
            sampled_of=cap._seen_images,
        )
        if tide is None:
            return None
        self._fig("f_tide", figs.tide_bar, tide)
        deltas = tide.get("delta_ap") or {}
        counts = tide.get("counts") or {}
        rows = [
            [
                kind,
                int(counts.get(kind, 0)),
                _r(deltas.get(kind)) if tide["mode"] == "delta_ap" else None,
                tide["recommendations"][kind],
            ]
            for kind in tide["types"]
        ]
        rows.sort(key=lambda r: -(r[2] or 0))
        self.tables["t_tide"] = {
            "columns": ["Type", "Count", "dAP50", "What to do"],
            "rows": rows,
            "sort": "dAP50:desc",
            "note": self._tide_note(tide),
        }
        if tide["mode"] == "counts" and want_delta:
            self.degradations.append(
                "The TIDE dAP oracles could not be computed; only error counts are shown."
            )
        return tide

    def _tide_note(self, tide: dict[str, Any]) -> str:
        base = tide.get("baseline_ap50")
        sampled = tide.get("sampled_of") or tide["n_images"]
        scope = (
            f"all {tide['n_images']:,} images"
            if sampled <= tide["n_images"]
            else f"{tide['n_images']:,} images sampled of {sampled:,}"
        )
        head = (
            f"Baseline AP50 over {scope}: {base:.4f}. It differs from the headline "
            "mAP50 because it is computed over this subsample with a single greedy "
            "match at IoU 0.50. "
            if base is not None
            else f"Error counts over {scope}. "
        )
        from src.report.tide import match_iou_note

        return head + match_iou_note()

    # --- section 8 ------------------------------------------------------------------
    def _section_strata(self) -> None:
        cap = self.capture_or_none()
        if cap is None or int(cap.area_total.sum()) == 0:
            return
        scan = self.ctx.dataset_scan
        imgsz = int(self.ctx.args_train.get("imgsz") or 640)
        terciles = scan.terciles() if scan else _terciles(cap.area_total)
        bins = len(cap.area_total)
        edges = [int(terciles[0] * bins), int(terciles[1] * bins)]
        groups = [(0, edges[0]), (edges[0], edges[1]), (edges[1], bins)]
        recalls, supports = [], []
        for lo, hi in groups:
            total = int(cap.area_total[lo:hi].sum())
            hit = int(cap.area_matched[lo:hi].sum())
            supports.append(total)
            recalls.append(round(hit / total, 4) if total else 0.0)
        labels = [
            f"small (<{terciles[0] * imgsz:.0f}px)",
            f"medium ({terciles[0] * imgsz:.0f}-{terciles[1] * imgsz:.0f}px)",
            f"large (>{terciles[1] * imgsz:.0f}px)",
        ]
        self._fig("f_recall_by_size", figs.recall_by_size, labels, recalls, supports)

        if int(cap.iou_hist.sum()):
            self._fig("f_iou_hist_tp", figs.iou_hist, cap.iou_hist)

        deciles_x, deciles_y = _deciles(cap.area_total, cap.area_matched, imgsz)
        if deciles_x:
            self._fig(
                "f_recall_vs_area", figs.recall_vs_area, deciles_x, deciles_y, imgsz
            )

        ar_labels, ar_recalls = _ar_buckets(cap.ar_total, cap.ar_matched)
        if ar_labels:
            self._fig("f_recall_by_ar", figs.recall_by_ar, ar_labels, ar_recalls)

    # --- section 9 ------------------------------------------------------------------
    def _section_boxmask(self) -> None:
        seg = getattr(self.metrics, "seg", None)
        if seg is None or not self.summary:
            return
        box_ap = [float(e.get("mAP50-95") or 0.0) for e in self.summary]
        mask_ap = np.asarray(getattr(seg, "ap", []), dtype=float).tolist()
        names = [str(e.get("Class")) for e in self.summary]
        if len(mask_ap) == len(box_ap) and box_ap:
            self._fig("f_boxmask_class", figs.boxmask_class, names, box_ap, mask_ap)
        cap = self.capture_or_none()
        if cap is None or int(cap.box_mask_hist.sum()) == 0:
            return
        self._fig("f_boxmask_iou", figs.boxmask_iou, cap.box_mask_hist)
        levels = cap.box_mask_hist.sum(axis=0)
        self._fig("f_iou_overlay", figs.iou_overlay, cap.iou_hist, levels)
        # Mask level is quantised to the 10 IoU thresholds, so the "IoU" it implies is
        # level/10; the delta against exact box IoU is what the diverging bar shows.
        buckets, deltas = _boxmask_size_delta(cap)
        if buckets:
            self._fig("f_boxmask_size", figs.boxmask_by_size, buckets, deltas)

    # --- section 10 -----------------------------------------------------------------
    def _section_galleries(self) -> None:
        cap = self.capture_or_none()
        per_grid = int(self.cfg.get("report_gallery_per_grid", 24))
        display = self.meta_display_conf()
        hi = float(self.cfg.get("report_high_conf_fp_threshold", 0.7))
        for gid, title, subtitle, basis in GRID_SPECS:
            items_in = cap.grid_items(gid)[:per_grid] if cap else []
            items_out = []
            for item in items_in:
                key = self.thumbs.add(item)
                if key is None:
                    continue
                items_out.append(
                    {
                        "thumb": key,
                        "label": str(item.get("label", ""))[:120],
                        "classes": list(item.get("classes") or [])[:12],
                        "pairs": list(item.get("pairs") or [])[:6],
                        "outcome": str(item.get("outcome", "mixed")),
                        "boxes": _norm_boxes(item)[:MAX_BOXES_PER_ITEM],
                    }
                )
            self.grids[gid] = {
                "title": title,
                "subtitle": subtitle,
                "basis": basis.format(conf=f"{display:g}", hi=f"{hi:g}"),
                "items": items_out,
                "empty_reason": None if items_out else _empty_reason(gid, cap, hi),
            }

    # --- section 11 -----------------------------------------------------------------
    def _section_training(self) -> None:
        save_dir = getattr(self.ctx.trainer, "save_dir", None)
        path = Path(save_dir) / "results.csv" if save_dir else None
        if path is None or not path.is_file():
            self.degradations.append(
                "Training history was not found (`results.csv`), so the training-curve "
                "appendix is empty."
            )
            return
        try:
            import pandas as pd

            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("could not read %s: %s", path, e)
            self.degradations.append(f"Training history could not be parsed: {e}.")
            return

        from src.yolov8.metrics_utils import MAX_CURVE_POINTS

        step = max(1, math.ceil(len(df) / MAX_CURVE_POINTS))
        df = df.iloc[::step]
        epochs = [int(v) for v in df.get("epoch", range(len(df)))]
        maps = {
            c: [float(v) for v in df[c]]
            for c in df.columns
            if c.startswith("metrics/mAP50-95")
        }
        losses = {
            c.split("/")[-1]: [float(v) for v in df[c]]
            for c in df.columns
            if c.startswith("train/") and c.endswith("loss")
        }
        if maps:
            self._fig("f_val_map", figs.val_map_curve, epochs, maps)
        if losses:
            self._fig("f_losses", figs.loss_curves, epochs, losses)

    # --- warnings and caveats -------------------------------------------------------
    def _collect_capture_degradations(self) -> None:
        cap = self.ctx.capture
        if cap is not None:
            self.degradations.extend(cap.degradations())
            cap.log_summary()

    def _build_warnings(self, meta: dict[str, Any]) -> None:
        low = int(self.cfg.get("report_low_support_threshold", 30))
        present = {str(e.get("Class")) for e in self.summary}
        absent = [n for n in self.names if n not in present] if self.summary else []
        if absent:
            self.warnings.append(
                {
                    "severity": "serious",
                    "text": (
                        f"{len(absent)} class(es) had no ground truth in the "
                        f"'{meta['split_name']}' split and are silently absent from "
                        f"mAP: {', '.join(absent[:12])}"
                        + (" ..." if len(absent) > 12 else "")
                    ),
                }
            )
        weak = [
            str(e.get("Class"))
            for e in self.summary
            if 0 < int(e.get("Instances", 0) or 0) < low
        ]
        if weak:
            self.warnings.append(
                {
                    "severity": "warning",
                    "text": (
                        f"{len(weak)} class(es) have fewer than {low} instances, so "
                        f"their AP is noise: {', '.join(weak[:12])}"
                        + (" ..." if len(weak) > 12 else "")
                    ),
                }
            )
        val_args = meta["val_args"]
        try:
            from ultralytics.utils import DEFAULT_CFG_DICT

            drift = [
                f"{k}={val_args[k]} (default {DEFAULT_CFG_DICT.get(k)})"
                for k in ("conf", "max_det", "batch", "split")
                if val_args.get(k) is not None
                and val_args.get(k) != DEFAULT_CFG_DICT.get(k)
            ]
        except Exception:
            drift = []
        if drift:
            self.warnings.append(
                {
                    "severity": "warning",
                    "text": (
                        "The final validation pass did not use the ultralytics "
                        "defaults, and the per-epoch passes during training did: "
                        + "; ".join(drift)
                    ),
                }
            )
        if self.ctx.args_train.get("resume"):
            self.warnings.append(
                {
                    "severity": "warning",
                    "text": (
                        "`resume=True` was set. `model_latest_id` does not actually "
                        "resume in this template -- ultralytics' `check_resume` only "
                        "honours a path, so it globs for `./**/last*.pt` instead."
                    ),
                }
            )
        if not getattr(getattr(self.ctx.validator, "args", None), "plots", True):
            self.warnings.append(
                {
                    "severity": "serious",
                    "text": (
                        "`5_Testing/plots` was off, so there is no confusion matrix "
                        "and no per-image diagnostic panels for this pass."
                    ),
                }
            )
        for line in self.degradations[:MAX_WARNINGS]:
            self.warnings.append({"severity": "warning", "text": line})

    def _caveats(self) -> list[str]:
        display = self.meta_display_conf()
        nms = float(getattr(getattr(self.ctx.validator, "args", None), "conf", 0.001))
        out = [
            f"mAP and every curve are computed at conf={nms:g}; the confusion matrix "
            f"and every gallery are drawn at conf={display:g}. The two disagree by "
            "construction, not by mistake.",
        ]
        if self.is_seg:
            out.append(
                "The confusion matrix and the error decomposition are box-IoU based "
                "even on this segmentation run -- `SegmentationValidator` inherits "
                "`update_metrics`, so neither ever sees a mask."
            )
        broken = []
        if self.ctx.args_train.get("label_smoothing"):
            broken.append("`label_smoothing`")
        if self.ctx.args_val.get("save_hybrid"):
            broken.append("`save_hybrid`")
        if self.ctx.args_val.get("save_crop"):
            broken.append("`save_crop`")
        if broken:
            out.append(
                f"{', '.join(broken)} were set but do nothing in ultralytics "
                f"{_ul_version()}."
            )
        out.extend(self.degradations[:MAX_DEGRADATIONS])
        out.append(
            f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}, report "
            f"schema {SCHEMA}."
        )
        return out

    # --- helpers --------------------------------------------------------------------
    def capture_or_none(self) -> Any:
        cap = self.ctx.capture
        if cap is None or not getattr(cap, "n_images", 0):
            return None
        return cap

    def meta_display_conf(self) -> float:
        return float(self.cfg.get("confusion_matrix_conf", 0.25))

    def _fig(self, fig_id: str, builder, *args, **kwargs) -> None:
        if len(self.figures) >= MAX_FIGURES:
            return
        payload = figs.safe(builder, *args, **kwargs)
        if payload is not None:
            self.figures[fig_id] = payload


class _Wrap:
    """Adapt a bare metrics object to the validator shape the extractors expect."""

    def __init__(self, metrics: Any) -> None:
        self.metrics = metrics


# --- module-level helpers -----------------------------------------------------------


def build_blob(ctx: ReportContext) -> dict[str, Any]:
    """Assemble the whole blob. Never raises for a missing input; degrades instead."""
    return _Builder(ctx).build()


def _kpi(label: str, value: Any, fmt: str, basis: str) -> dict[str, Any]:
    return {
        "label": label,
        "value": None if value is None else round(float(value), 4),
        "fmt": fmt,
        "scale": [0.0, 1.0],
        "basis": basis,
        "tone": "neutral",
    }


def _f1(results: dict[str, Any]) -> float | None:
    p = results.get("metrics/precision(B)")
    r = results.get("metrics/recall(B)")
    if p is None or r is None:
        return None
    p, r = float(p), float(r)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _r(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except TypeError, ValueError:
        return None
    return None if math.isnan(f) else round(f, digits)


def _plain(value: Any) -> Any:
    """Coerce a config value into something json can hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _short(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _args_dict(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    for attr in ("__dict__", "_asdict"):
        got = getattr(args, attr, None)
        if isinstance(got, dict):
            return dict(got)
        if callable(got):
            try:
                return dict(got())
            except Exception:
                pass
    return {}


def _is_segment(metrics: Any, task_yolo: str) -> bool:
    """Decide by whether the mask `Metric` ran, not by what the task was called."""
    seg = getattr(metrics, "seg", None)
    if seg is None:
        return False
    # `or []` is wrong here: `p` is a numpy array, whose truth value raises.
    return len(getattr(seg, "p", [])) > 0 or str(task_yolo).lower() == "segment"


def _model(ctx: ReportContext) -> str:
    args = getattr(ctx.trainer, "args", None)
    return str(getattr(args, "model", "") or "unknown")


def _task_url(task: Any) -> str | None:
    for attr in ("get_output_log_web_page",):
        fn = getattr(task, attr, None)
        if callable(fn):
            try:
                url = fn()
                if isinstance(url, str) and url.startswith("http"):
                    return url
            except Exception:
                return None
    return None


def _task_project(task: Any) -> str:
    data = getattr(task, "data", None)
    return str(getattr(data, "project", "") or getattr(task, "project", "") or "")


def _git_commit(task: Any) -> str:
    try:
        return str(task.data.script.version_num or "unknown")
    except Exception:
        return "unknown"


def _ul_version() -> str:
    import ultralytics

    return ultralytics.__version__


def _cap_matrix(
    labels: list[str], counts: np.ndarray, cap: int
) -> tuple[list[str], np.ndarray, int]:
    """Keep the `cap` classes with the most ground truth, background always last."""
    n = counts.shape[0]
    if n <= cap:
        return labels, counts, 0
    has_bg = bool(labels) and labels[-1] == "background"
    body = n - 1 if has_bg else n
    support = counts[:body].sum(axis=1)
    keep = sorted(np.argsort(-support)[: cap - (1 if has_bg else 0)].tolist())
    if has_bg:
        keep.append(n - 1)
    idx = np.asarray(keep, dtype=int)
    return [labels[i] for i in keep], counts[np.ix_(idx, idx)], body


def _terciles(hist: np.ndarray) -> tuple[float, float]:
    total = int(hist.sum())
    if total == 0:
        return (1 / 3, 2 / 3)
    cum = np.cumsum(hist)
    n = len(hist)
    return tuple(
        min(int(np.searchsorted(cum, q * total)) + 1, n) / n for q in (1 / 3, 2 / 3)
    )


def _deciles(
    total: np.ndarray, matched: np.ndarray, imgsz: int
) -> tuple[list[float], list[float]]:
    """Recall against sqrt(area), grouped into equal-count deciles of the histogram."""
    n_total = int(total.sum())
    if n_total == 0:
        return [], []
    bins = len(total)
    cum = np.cumsum(total)
    xs, ys = [], []
    prev = 0
    for k in range(1, 11):
        edge = int(np.searchsorted(cum, k / 10 * n_total)) + 1
        edge = min(edge, bins)
        if edge <= prev:
            continue
        seg_total = int(total[prev:edge].sum())
        if seg_total == 0:
            prev = edge
            continue
        xs.append(round((prev + edge) / 2 / bins * imgsz, 1))
        ys.append(round(int(matched[prev:edge].sum()) / seg_total, 4))
        prev = edge
    return xs, ys


def _ar_buckets(total: np.ndarray, matched: np.ndarray) -> tuple[list[str], list[float]]:
    """Five fixed log(w/h) buckets over the [-3, 3] histogram range."""
    if int(total.sum()) == 0:
        return [], []
    bins = len(total)
    edges = [0, int(bins * 0.2), int(bins * 0.4), int(bins * 0.6), int(bins * 0.8), bins]
    labels, recalls = [], []
    for i, name in enumerate(AR_BUCKETS):
        lo, hi = edges[i], edges[i + 1]
        n = int(total[lo:hi].sum())
        labels.append(name)
        recalls.append(round(int(matched[lo:hi].sum()) / n, 4) if n else 0.0)
    return labels, recalls


def _boxmask_size_delta(cap: Any) -> tuple[list[str], list[float]]:
    """Mask minus box IoU by box-IoU tercile, from the quantised 2-D histogram."""
    hist = cap.box_mask_hist
    if int(hist.sum()) == 0:
        return [], []
    n_box, n_lvl = hist.shape
    thirds = [(0, n_box // 3), (n_box // 3, 2 * n_box // 3), (2 * n_box // 3, n_box)]
    labels, deltas = [], []
    for name, (lo, hi) in zip(("low IoU", "mid IoU", "high IoU"), thirds, strict=True):
        block = hist[lo:hi]
        n = int(block.sum())
        if n == 0:
            continue
        box_mean = float(
            (block.sum(axis=1) * (np.arange(lo, hi) + 0.5) / n_box).sum() / n
        )
        mask_mean = float((block.sum(axis=0) * np.arange(n_lvl) / (n_lvl - 1)).sum() / n)
        labels.append(name)
        deltas.append(round(mask_mean - box_mean, 4))
    return labels, deltas


def _norm_boxes(item: dict) -> list[list]:
    """Convert native-pixel boxes to thumbnail-normalised 0..1 floats.

    The lightbox draws them as percentage-positioned divs over the same base64 the grid
    shows, so the geometry has to be relative to the *crop*, not the original image.
    """
    h, w = (float(v) for v in item.get("ori_shape") or (0, 0))
    focus = item.get("focus")
    if focus:
        x0, y0, x1, y1 = (float(v) for v in focus[:4])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(max(x1 - x0, y1 - y0) * 1.6, 64.0)
        side = min(side, max(w, h) or side)
        half = side / 2
        cx = min(max(cx, half), max(w - half, half))
        cy = min(max(cy, half), max(h - half, half))
        ox0, oy0 = max(cx - half, 0.0), max(cy - half, 0.0)
        cw, ch = min(cx + half, w) - ox0, min(cy + half, h) - oy0
    else:
        ox0, oy0, cw, ch = 0.0, 0.0, w, h
    if cw <= 0 or ch <= 0:
        return []
    # The thumbnail letterboxes the crop into a square, so both axes share one scale
    # and the shorter one is centred -- the same arithmetic the encoder used.
    span = max(cw, ch)
    pad_x = (span - cw) / 2
    pad_y = (span - ch) / 2
    out = []
    for box in (item.get("boxes") or [])[:MAX_BOXES_PER_ITEM]:
        bx0 = (float(box[0]) - ox0 + pad_x) / span
        by0 = (float(box[1]) - oy0 + pad_y) / span
        bx1 = (float(box[2]) - ox0 + pad_x) / span
        by1 = (float(box[3]) - oy0 + pad_y) / span
        out.append(
            [
                round(min(max(bx0, 0.0), 1.0), 4),
                round(min(max(by0, 0.0), 1.0), 4),
                round(min(max(bx1, 0.0), 1.0), 4),
                round(min(max(by1, 0.0), 1.0), 4),
                box[4],
                box[5],
                box[6],
                box[7],
            ]
        )
    return out


def _empty_reason(gid: str, cap: Any, hi: float) -> str:
    if cap is None:
        return "Per-image capture was not available for this validation pass."
    return {
        "g_hiconf_fp": f"No false positive scored above {hi:g}.",
        "g_missed": "Nothing was missed at IoU 0.50.",
        "g_lowconf_tp": "No matched detection to show.",
        "g_worst": "No image had a false positive or a false negative.",
        "g_random": "No images were validated.",
    }.get(gid, "Nothing to show.")
