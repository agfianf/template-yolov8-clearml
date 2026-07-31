"""The single entry point `src/train.py` calls, and the upload flow behind it.

Called once, after the final `val()` and before the export stage, inside its own
try/except so a failed report can never cost an exported model at the end of a GPU run.
Everything below it degrades rather than raises: a missing capture, an unreadable label
directory or a broken figure builder each cost their own card in the page.

The artifact is always a **file path**, never a directory: ClearML zips a folder
artifact, and a zip is not something the fileserver will serve as `text/html`.
"""

from __future__ import annotations
import tempfile

from pathlib import Path
from typing import Any

from src.report.blob import ReportContext, build_blob
from src.report.render import drop_galleries, render_report, split_blob
from src.utils.logging import get_logger


logger = get_logger(__name__)

ARTIFACT_NAME = "evaluation_report"
GALLERY_ARTIFACT_NAME = "evaluation_report_galleries"
PREVIEW_LIMIT = 60_000


def build_evaluation_report(
    *,
    task: Any = None,
    validator: Any = None,
    final_metrics: Any = None,
    trainer: Any = None,
    val_stats: Any = None,
    capture: Any = None,
    dataset_dir: str | None = None,
    datadotyaml: dict | None = None,
    class_2_idx: dict | None = None,
    task_yolo: str = "detect",
    split_label: str = "",
    args_train: dict | None = None,
    args_val: dict | None = None,
    args_visualization: dict | None = None,
    output_dir: str | Path | None = None,
) -> str | None:
    """Build the report and upload it. Returns the artifact URL, or None.

    Returns None rather than raising for every failure mode, including "the report was
    switched off in `8_Visualization`".
    """
    cfg = args_visualization or {}
    if not cfg.get("html_report", True):
        logger.info("html_report is off; skipping the evaluation report")
        return None

    ctx = ReportContext(
        task=task,
        validator=validator,
        final_metrics=final_metrics,
        trainer=trainer,
        val_stats=val_stats,
        capture=capture,
        dataset_dir=dataset_dir,
        datadotyaml=datadotyaml,
        class_2_idx=class_2_idx,
        task_yolo=task_yolo,
        split_label=split_label,
        args_train=args_train or {},
        args_val=args_val or {},
        args_visualization=cfg,
    )
    blob = build_blob(ctx)
    return publish_report(blob, task=task, cfg=cfg, output_dir=output_dir)


def publish_report(
    blob: dict,
    *,
    task: Any = None,
    cfg: dict | None = None,
    output_dir: str | Path | None = None,
) -> str | None:
    """Render, size-check and upload an already-assembled blob.

    Split apart from `build_evaluation_report` so the tests can drive the rendering and
    the upload without standing up a validator.
    """
    cfg = cfg or {}
    split_bytes = int(cfg.get("report_split_bytes", 5_000_000))
    max_bytes = int(cfg.get("report_max_bytes", 15_000_000))
    directory = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
    directory.mkdir(parents=True, exist_ok=True)

    document = render_report(blob)
    size = len(document.encode("utf-8"))
    logger.info(
        "evaluation report: %.2f MB, %d figures, %d thumbnails",
        size / 1e6,
        len(blob.get("figures") or {}),
        len(blob.get("thumbs") or {}),
    )

    from src.yolov8.clearml_logger import YOLOClearMLLogger

    uploader = YOLOClearMLLogger(task) if task is not None else None
    gallery_url = None

    if size > split_bytes and blob.get("thumbs"):
        # Exactly one split point, measured rather than guessed, and the child goes
        # first so the index can link to its absolute URL.
        index_blob, child_blob = split_blob(blob)
        child_path = directory / f"{GALLERY_ARTIFACT_NAME}.html"
        child_path.write_text(
            render_report(child_blob, only_galleries=True), encoding="utf-8"
        )
        if uploader is not None:
            gallery_url = uploader.upload_html_report(
                child_path,
                name=GALLERY_ARTIFACT_NAME,
                preview="Gallery half of the evaluation report.",
                metadata=_metadata(child_blob, child_path, parts=2, part="galleries"),
                link_media=False,
            )
        blob = index_blob
        document = render_report(blob, gallery_link=gallery_url)
        size = len(document.encode("utf-8"))

    if size > max_bytes:
        blob = drop_galleries(blob)
        document = render_report(blob, gallery_link=gallery_url)
        size = len(document.encode("utf-8"))
        logger.warning(
            "evaluation report exceeded the hard ceiling; galleries dropped (%.2f MB)",
            size / 1e6,
        )

    path = directory / f"{ARTIFACT_NAME}.html"
    path.write_text(document, encoding="utf-8")
    if uploader is None:
        logger.info("no ClearML task; evaluation report written to %s", path)
        return str(path)

    return uploader.upload_html_report(
        path,
        name=ARTIFACT_NAME,
        preview=build_preview(blob),
        metadata=_metadata(
            blob, path, parts=2 if gallery_url else 1, part="index", size=size
        ),
        link_media=True,
    )


def build_preview(blob: dict) -> str:
    """Plain-text summary shown in the ClearML artifact panel without opening the file."""
    meta = blob["meta"]
    lines = [
        f"Evaluation report - {meta['run_name']}",
        f"task {meta['task_id']}  split {meta['split_name']}  "
        f"{meta['val_images']:,} images  ultralytics {meta['ultralytics']}",
        "",
        "Headline metrics (each with the basis it was computed at):",
    ]
    for kpi in blob.get("kpis") or []:
        value = kpi.get("value")
        shown = "not computed" if value is None else f"{float(value):.4f}"
        lines.append(f"  {kpi['label']:<16} {shown:<12} [{kpi.get('basis', '')}]")

    thresholds = meta.get("thresholds") or {}
    tau = thresholds.get("f1_optimal")
    lines += [
        "",
        "Thresholds:",
        f"  NMS floor (mAP and curves)   {thresholds.get('nms_floor')}",
        f"  Display (matrix, galleries)  {thresholds.get('matrix_display')}",
        f"  F1-optimal (deploy at this)  {tau if tau is not None else 'not computed'}",
    ]

    per_class = (blob.get("tables") or {}).get("t_per_class")
    if per_class and per_class.get("rows"):
        lines += ["", "Weakest classes by AP50-95 (class, support, AP50-95):"]
        rows = sorted(per_class["rows"], key=lambda r: (r[2] is None, r[2] or 0))
        for row in rows[:5]:
            lines.append(f"  {row[0]:<24} n={row[1]:<8} {row[2]}")

    tide = (blob.get("tables") or {}).get("t_tide")
    if tide and tide.get("rows"):
        top = tide["rows"][0]
        lines += [
            "",
            f"Largest error type: {top[0]} (count {top[1]}, dAP50 {top[2]})",
            f"  {top[3]}",
        ]

    warnings = blob.get("warnings") or []
    if warnings:
        lines += ["", "Warnings:"]
        lines += [f"  [{w['severity']}] {w['text']}" for w in warnings]
    degradations = blob.get("degradations") or []
    if degradations:
        lines += ["", "Degradations:"] + [f"  {d}" for d in degradations]

    lines += ["", "Open the artifact URL in a new tab for the interactive report."]
    return "\n".join(lines)[:PREVIEW_LIMIT]


def _metadata(
    blob: dict,
    path: Path,
    *,
    parts: int,
    part: str,
    size: int | None = None,
) -> dict[str, str]:
    meta = blob["meta"]
    return {
        key: str(value)
        for key, value in {
            "schema": blob.get("schema"),
            "generated_utc": meta.get("generated_utc"),
            "task_yolo": meta.get("task_yolo"),
            "split": meta.get("split_name"),
            "val_images": meta.get("val_images"),
            "classes": len(blob.get("classes") or []),
            "is_segment": meta.get("is_seg"),
            "bytes": size if size is not None else path.stat().st_size,
            "thumbnails": len(blob.get("thumbs") or {}),
            "figures": len(blob.get("figures") or {}),
            "parts": parts,
            "part": part,
            "ultralytics": meta.get("ultralytics"),
            "code_version": meta.get("code_version"),
            "image_tag": meta.get("image_tag"),
            "git_commit": meta.get("git_commit"),
            "tide_mode": blob.get("tide_mode", "none"),
            "degradations": "; ".join(blob.get("degradations") or [])[:900],
        }.items()
    }
