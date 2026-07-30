"""Dataset composition reporting for ClearML.

The data stage already knows everything worth knowing about a dataset -- how many
instances each class has, how big the boxes are, what was filtered out and why -- and
until now none of it reached ClearML. All of it went to the console, and the
`FilterReport` describing exactly which annotations were dropped was received into a
throwaway variable and discarded (`coco2yolo.py`, `img_to_anns, _report = ...`).

Class imbalance and box-size distribution are baked into every metric reported later, so
they belong next to those metrics rather than in a log nobody re-reads. A class with 12
instances and a mAP of 0 is not a modelling problem.

Accumulation is deliberately tally-only: `note_annotation` is called once per annotation
across the whole dataset and must never log, per the project rule that log volume cannot
grow with dataset size. One summary is reported at the end of the data stage.
"""

from __future__ import annotations
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.logging import get_logger


if TYPE_CHECKING:
    from src.schema.coco import FilterReport


logger = get_logger(__name__)

# Above this many classes a bar chart stops being readable; the table still carries
# every row, so nothing is lost -- only the chart is capped.
MAX_CHART_CLASSES = 60


class DatasetStats:
    """Accumulate dataset composition across every converted source.

    A run can convert several CVAT tasks or S3 prefixes, so this accumulates across all of
    them and is reported once, rather than emitting one plot set per source.
    """

    def __init__(self) -> None:
        self.instances: Counter[str] = Counter()
        self.images_with_class: Counter[str] = Counter()
        self.areas: list[float] = []
        self.aspects: list[float] = []
        self.image_count = 0
        self.filter_total = 0
        self.filter_kept = 0
        self.dropped_area = 0
        self.dropped_class: Counter[str] = Counter()
        self.dropped_attr: Counter[str] = Counter()

    def reset(self) -> None:
        """Clear everything, so a second data stage does not stack onto the first."""
        self.__init__()

    def note_annotation(self, class_name: str, norm_w: float, norm_h: float) -> None:
        """Record one kept annotation. Called per annotation; must not log."""
        self.instances[class_name] += 1
        if norm_w > 0 and norm_h > 0:
            self.areas.append(float(norm_w * norm_h))
            self.aspects.append(float(norm_w / norm_h))

    def note_image(self, class_names: set[str]) -> None:
        """Record one image and which classes appeared in it."""
        self.image_count += 1
        for name in class_names:
            self.images_with_class[name] += 1

    def note_filter_report(self, report: FilterReport) -> None:
        """Fold one source's filtering outcome into the totals."""
        self.filter_total += report.total
        self.filter_kept += report.kept
        self.dropped_area += report.dropped_area
        # Tally is dict-like; fall back quietly if that ever stops being true.
        for tally, target in (
            (report.dropped_class, self.dropped_class),
            (report.dropped_attr, self.dropped_attr),
        ):
            try:
                target.update(dict(tally))
            except Exception as e:
                logger.debug("could not fold a filter tally: %s", e)

    @property
    def is_empty(self) -> bool:
        """True when nothing was accumulated and there is nothing to report."""
        return not self.instances and not self.filter_total

    def class_table(self) -> pd.DataFrame | None:
        """Per-class instance and image counts, rarest class first."""
        if not self.instances:
            return None
        rows = [
            {
                "Class": name,
                "Instances": count,
                "Images": self.images_with_class.get(name, 0),
                "Instances per image": round(
                    count / self.images_with_class.get(name, 1), 2
                ),
                "Share of instances": round(count / sum(self.instances.values()), 4),
            }
            for name, count in self.instances.items()
        ]
        df = pd.DataFrame(rows).sort_values("Instances", ascending=True, kind="stable")
        return df.reset_index(drop=True)

    def filter_table(self) -> pd.DataFrame | None:
        """Return what was dropped and why, as counts."""
        if not self.filter_total:
            return None
        rows = [
            {"Reason": "kept", "Annotations": self.filter_kept},
            {"Reason": "dropped: below min area", "Annotations": self.dropped_area},
        ]
        rows += [
            {"Reason": f"dropped: excluded class '{name}'", "Annotations": count}
            for name, count in self.dropped_class.items()
        ]
        rows += [
            {"Reason": f"dropped: excluded attribute '{name}'", "Annotations": count}
            for name, count in self.dropped_attr.items()
        ]
        return pd.DataFrame(rows)


# Module-level: the converter is a set of static methods called from several places, and
# the report is emitted once at the end of the data stage. Only ever touched in the main
# process -- the data stage runs before any dataloader exists.
dataset_stats = DatasetStats()


def _imbalance_figure(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar of instances per class on a log axis.

    Log scale because the interesting datasets are the skewed ones: on a linear axis a
    class with 20 instances next to one with 20,000 is an invisible sliver, which is
    exactly the class whose metrics will be meaningless.
    """
    shown = df.head(MAX_CHART_CLASSES)
    fig = go.Figure(
        go.Bar(
            x=shown["Instances"].tolist(),
            y=shown["Class"].astype(str).tolist(),
            orientation="h",
            text=shown["Instances"].tolist(),
            textposition="outside",
            hovertemplate="%{y}<br>%{x:,} instances<extra></extra>",
        )
    )
    title = "Instances per class (rarest first)"
    if len(df) > MAX_CHART_CLASSES:
        title += f" - showing {MAX_CHART_CLASSES} of {len(df)}"
    fig.update_layout(
        title=title,
        xaxis_title="Instances (log scale)",
        yaxis_title="Class",
        xaxis_type="log",
        height=max(320, 18 * len(shown)),
    )
    return fig


def _geometry_figure(areas: list[float], aspects: list[float]) -> go.Figure | None:
    """Box area and aspect-ratio distributions.

    Area is plotted as sqrt(area) relative to image size, i.e. the fraction of the image
    edge an object spans, because that is what compares against `imgsz` and the model's
    stride: an object spanning 1% of the edge is ~6px at imgsz=640 and cannot survive
    downsampling regardless of how the model is trained.
    """
    if not areas:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=np.sqrt(np.asarray(areas, dtype=float)).tolist(),
            name="sqrt(area) / image edge",
            nbinsx=50,
        )
    )
    if aspects:
        fig.add_trace(
            go.Histogram(
                x=np.asarray(aspects, dtype=float).tolist(),
                name="aspect ratio (w/h)",
                nbinsx=50,
                visible="legendonly",
            )
        )
    fig.update_layout(
        title="Object geometry (toggle series in the legend)",
        xaxis_title="Value",
        yaxis_title="Instances",
        barmode="overlay",
    )
    return fig


def report_dataset_composition(stats: DatasetStats | None = None) -> bool:
    """Report the accumulated dataset composition to ClearML.

    Called once, at the end of the data stage. Returns False when there is no ClearML task
    or nothing was accumulated, so it is safe to call unconditionally.
    """
    stats = stats if stats is not None else dataset_stats
    if stats.is_empty:
        logger.debug("no dataset statistics accumulated, nothing to report")
        return False

    try:
        from clearml import Task

        task = Task.current_task()
    except Exception as e:
        logger.debug("no ClearML task for the dataset report: %s", e)
        return False

    if task is None:
        return False

    from src.yolov8.clearml_logger import YOLOClearMLLogger

    clearml_logger = YOLOClearMLLogger(task)
    reported: list[str] = []

    class_df = stats.class_table()
    if class_df is not None:
        if clearml_logger.log_dataframe_table(
            class_df, title="Dataset", series="Class Distribution"
        ):
            reported.append("class table")
        if clearml_logger.log_plotly_figure(
            _imbalance_figure(class_df), title="Dataset", series="Class Imbalance"
        ):
            reported.append("imbalance chart")

    geometry = _geometry_figure(stats.areas, stats.aspects)
    if geometry is not None and clearml_logger.log_plotly_figure(
        geometry, title="Dataset", series="Object Geometry"
    ):
        reported.append("geometry")

    filter_df = stats.filter_table()
    if filter_df is not None and clearml_logger.log_dataframe_table(
        filter_df, title="Dataset", series="Annotation Filtering"
    ):
        reported.append("filter table")

    summary: dict[str, Any] = {
        "images": stats.image_count,
        "classes": len(stats.instances),
        "instances_kept": stats.filter_kept or sum(stats.instances.values()),
        "annotations_dropped": max(stats.filter_total - stats.filter_kept, 0),
    }
    if stats.instances:
        counts = list(stats.instances.values())
        # A single number for how lopsided the dataset is, which is what actually drives
        # whether per-class metrics are trustworthy.
        summary["imbalance_ratio_max_over_min"] = round(max(counts) / min(counts), 2)
    clearml_logger.log_scalar_group("Dataset", summary, 0, "dataset summary")

    logger.info("dataset report: %s", ", ".join(reported) if reported else "nothing sent")
    return bool(reported)
