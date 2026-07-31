"""ClearML logging wrapper for YOLO training visualization.

This module provides a unified interface for logging various metrics and visualizations
to ClearML, including interactive confusion matrices, per-class tables, PR curves,
histograms, and scalar metrics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.logging import get_logger


if TYPE_CHECKING:
    from pathlib import Path

    from clearml import Task


logger = get_logger(__name__)

# Ultralytics suffixes curve names with the metric domain: "(B)" for box, "(M)" for
# mask. See Metric.curves / SegmentMetrics.curves in ultralytics/utils/metrics.py.
_CURVE_DOMAINS = {"(B)": "Box", "(M)": "Mask", "(P)": "Pose"}


def binned_counts(
    values: np.ndarray,
    bins: int = 50,
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[list[float], list[int], float]:
    """Bin values into counts, returning (bin_centers, counts, bin_width).

    Plotly's ``go.Histogram`` bins in the browser, which means the *raw samples* are
    serialised into the plot payload. On a real validation set that is tens of thousands
    of floats: a single TP-vs-FP histogram measured 1.0 MB, and the ClearML UI does not
    render plots that size -- the panel simply comes up blank, with no error to explain
    it. Binning here and shipping a bar chart of counts takes the same plot to a few KB.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [], [], 0.0

    lo = float(values.min()) if lo is None else lo
    hi = float(values.max()) if hi is None else hi
    if hi <= lo:
        hi = lo + 1e-9

    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return centers, counts.astype(int).tolist(), float(edges[1] - edges[0])


def _split_curve_name(name: str) -> tuple[str, str]:
    """Split "Precision-Recall(B)" into ("Precision-Recall", "Box").

    Falls back to ("<name>", "Box") for an unsuffixed name so a future ultralytics
    rename degrades to a mislabelled series rather than a crash inside a callback.
    """
    for suffix, domain in _CURVE_DOMAINS.items():
        if name.endswith(suffix):
            return name[: -len(suffix)].strip(), domain
    return name, "Box"


class YOLOClearMLLogger:
    """Wrapper class for consistent ClearML logging in YOLO training."""

    def __init__(self, task: Task | None = None):
        """Initialize the logger with an optional ClearML task.

        Args:
            task: ClearML Task instance. If None, will try to get current task.

        """
        if task is None:
            from clearml import Task as ClearMLTask

            task = ClearMLTask.current_task()
        self._task = task

    @property
    def task(self) -> Task | None:
        """Return the current ClearML task."""
        return self._task

    def _get_logger(self):
        """Get the ClearML logger from the task."""
        if self._task is None:
            return None
        return self._task.get_logger()

    def log_interactive_confusion_matrix(
        self,
        matrix: np.ndarray,
        labels: list[str],
        iteration: int = 0,
        title: str = "Confusion Matrix",
        series: str = "Normalized",
        xlabels: list[str] | None = None,
        ylabels: list[str] | None = None,
    ) -> bool:
        """Log an interactive confusion matrix to ClearML.

        Args:
            matrix: 2D numpy array with confusion matrix values.
            labels: List of class labels for default x and y labels.
            iteration: Iteration/epoch number.
            title: Title for the plot.
            series: Series name (e.g., "Normalized", "Counts").
            xlabels: Optional custom labels for x-axis (predicted).
            ylabels: Optional custom labels for y-axis (actual).

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            logger.warning("No ClearML task available for logging confusion matrix")
            return False

        try:
            task_logger.report_confusion_matrix(
                title=title,
                series=series,
                matrix=matrix,
                iteration=iteration,
                xaxis="Predicted",
                yaxis="Actual",
                xlabels=xlabels or labels,
                ylabels=ylabels or labels,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log confusion matrix: %s", e)
            return False

    def log_per_class_table(
        self,
        metrics_df: pd.DataFrame,
        iteration: int = 0,
        title: str = "Per-Class Metrics",
        series: str = "Detailed",
    ) -> bool:
        """Log a per-class metrics table to ClearML.

        Args:
            metrics_df: DataFrame with per-class metrics.
            iteration: Iteration/epoch number.
            title: Title for the table.
            series: Series name.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            logger.warning("No ClearML task available for logging table")
            return False

        try:
            task_logger.report_table(
                title=title,
                series=series,
                iteration=iteration,
                table_plot=metrics_df,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log per-class table: %s", e)
            return False

    def log_confidence_histogram(
        self,
        confidences: list[float],
        iteration: int = 0,
        title: str = "Distributions",
        series: str = "All Classes - Confidence",
        xaxis: str = "Confidence",
        yaxis: str = "Count",
    ) -> bool:
        """Log a confidence score histogram to ClearML.

        Args:
            confidences: List of confidence scores.
            iteration: Iteration/epoch number.
            title: Title for the histogram.
            series: Series name.
            xaxis: X-axis label.
            yaxis: Y-axis label.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            logger.warning("No ClearML task available for logging histogram")
            return False

        if not confidences:
            logger.warning("No confidence scores to log")
            return False

        try:
            task_logger.report_histogram(
                title=title,
                series=series,
                iteration=iteration,
                values=confidences,
                xaxis=xaxis,
                yaxis=yaxis,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log confidence histogram: %s", e)
            return False

    def log_curve_family(
        self,
        curve: dict,
        iteration: int = 0,
        title_prefix: str = "Curves",
    ) -> bool:
        """Log one per-class curve family (PR, F1-conf, P-conf or R-conf) as Plotly.

        Takes a dict from ``extract_curve_data``. Ultralytics names the family with a
        ``(B)`` / ``(M)`` suffix; we split that into the ClearML series so box and mask
        land as two selectable variants of the same plot instead of eight unrelated
        plots, or -- worse -- overlaid without saying which is which.

        Args:
            curve: Dict with keys name, xlabel, ylabel, x, series.
            iteration: Iteration/epoch number.
            title_prefix: Group prefix for the plot title.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            logger.warning("No ClearML task available for logging curves")
            return False

        if not curve or not curve.get("series"):
            return False

        raw_name = str(curve.get("name", "Curve"))
        base, variant = _split_curve_name(raw_name)

        try:
            fig = go.Figure()
            x = curve["x"]
            for s in curve["series"]:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=s["y"],
                        mode="lines",
                        name=str(s["class_name"]),
                        hovertemplate=(
                            f"Class: {s['class_name']}<br>"
                            f"{curve['xlabel']}: %{{x:.3f}}<br>"
                            f"{curve['ylabel']}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title=f"{base} ({variant})",
                xaxis_title=curve["xlabel"],
                yaxis_title=curve["ylabel"],
                legend_title="Class",
                hovermode="closest",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05]),
            )

            task_logger.report_plotly(
                title=f"{title_prefix}/{base}",
                series=variant,
                iteration=iteration,
                figure=fig,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log curve %s: %s", raw_name, e)
            return False

    def log_per_class_bar(
        self,
        metrics_df: pd.DataFrame,
        iteration: int = 0,
        sort_by: str = "Box-mAP50-95",
        title: str = "Per-Class Performance",
        series: str = "mAP50-95 (sorted)",
    ) -> bool:
        """Log a per-class mAP bar chart, sorted worst-first, box beside mask.

        The sort is the point: it puts the classes you have to fix on the left, which an
        alphabetical or class-index ordering buries. Mask bars are drawn alongside box
        bars whenever the run is a segmentation run, so a class whose masks lag its boxes
        is visible at a glance.

        Args:
            metrics_df: DataFrame from ``extract_per_class_metrics``.
            iteration: Iteration/epoch number.
            sort_by: Column to sort ascending by.
            title: Title for the plot.
            series: Series name.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            logger.warning("No ClearML task available for logging per-class bar")
            return False

        if metrics_df is None or metrics_df.empty or sort_by not in metrics_df.columns:
            return False

        try:
            df = metrics_df.sort_values(by=sort_by, ascending=True, kind="stable")
            fig = go.Figure()
            for col, label in (
                ("Box-mAP50-95", "Box"),
                ("Mask-mAP50-95", "Mask"),
            ):
                if col not in df.columns:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=df["Class"].astype(str).tolist(),
                        y=df[col].tolist(),
                        name=label,
                        text=[f"{v:.3f}" for v in df[col]],
                        textposition="outside",
                        hovertemplate=(
                            f"%{{x}}<br>{label} mAP50-95: %{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title="mAP50-95 per Class (worst first)",
                xaxis_title="Class",
                yaxis_title="mAP50-95",
                barmode="group",
                yaxis=dict(range=[0, 1.05]),
            )

            task_logger.report_plotly(
                title=title, series=series, iteration=iteration, figure=fig
            )
            return True
        except Exception as e:
            logger.warning("Failed to log per-class bar: %s", e)
            return False

    def log_scalar_grouped(
        self,
        title: str,
        series: str,
        value: float,
        iteration: int,
    ) -> bool:
        """Log a scalar value with grouping support.

        Args:
            title: Group/category title (e.g., "Losses/Train").
            series: Series name within the group (e.g., "box_loss").
            value: Scalar value to log.
            iteration: Iteration/epoch number.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            return False

        try:
            task_logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=iteration,
            )
            return True
        except Exception as e:
            # DEBUG, not WARNING: this is called once per scalar, so a sick
            # ClearML connection produced eleven warnings per epoch. The batch
            # helpers below warn once for the whole group instead.
            logger.debug("Failed to log scalar %s/%s: %s", title, series, e)
            return False

    def _log_scalar_batch(
        self,
        title: str,
        values: dict[str, float],
        iteration: int,
        what: str,
    ) -> bool:
        """Report a group of scalars and warn at most once for the group."""
        task_logger = self._get_logger()
        if task_logger is None:
            return False

        failed = [
            name
            for name, value in values.items()
            if not self.log_scalar_grouped(title, name, value, iteration)
        ]
        if failed:
            logger.warning(
                "Failed to log %d/%d %s at iteration %s: %s",
                len(failed),
                len(values),
                what,
                iteration,
                ", ".join(failed),
            )
            return False
        return True

    def log_scalar_group(
        self,
        title: str,
        values: dict[str, float],
        iteration: int,
        what: str = "scalars",
    ) -> bool:
        """Report a named group of scalars, warning at most once for the whole group.

        Args:
            title: Group/category title.
            values: Mapping of series name to value.
            iteration: Iteration/epoch number.
            what: Human-readable name of the group, used in the warning.

        Returns:
            True if every scalar was logged, False otherwise.

        """
        if not values:
            return False
        return self._log_scalar_batch(title, values, iteration, what)

    def log_per_class_scatter(
        self,
        class_names: list[str],
        metric_values: list[float],
        metric_name: str,
        iteration: int = 0,
        title: str = "Per-Class Performance",
    ) -> bool:
        """Log a per-class scatter plot to ClearML.

        Args:
            class_names: List of class names.
            metric_values: List of metric values corresponding to each class.
            metric_name: Name of the metric (e.g., "mAP50", "Precision").
            iteration: Iteration/epoch number.
            title: Title for the plot.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            logger.warning("No ClearML task available for logging scatter plot")
            return False

        if not class_names or not metric_values:
            logger.warning("No data for scatter plot")
            return False

        try:
            fig = go.Figure()

            hover_tpl = f"Class: %{{x}}<br>{metric_name}: %{{y:.3f}}<extra></extra>"
            fig.add_trace(
                go.Bar(
                    x=class_names,
                    y=metric_values,
                    text=[f"{v:.3f}" for v in metric_values],
                    textposition="outside",
                    hovertemplate=hover_tpl,
                )
            )

            y_max = max(metric_values) * 1.15 if metric_values else 1
            fig.update_layout(
                title=f"{metric_name} per Class",
                xaxis_title="Class",
                yaxis_title=metric_name,
                yaxis=dict(range=[0, y_max]),
            )

            task_logger.report_plotly(
                title=title,
                series=metric_name,
                iteration=iteration,
                figure=fig,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log scatter plot: %s", e)
            return False

    def log_speed_metrics(
        self,
        speeds: dict[str, float],
        iteration: int = 0,
        title: str = "Speed/Inference",
    ) -> bool:
        """Log speed metrics to ClearML.

        Args:
            speeds: Dictionary with speed metrics (preprocess_ms, inference_ms, etc.).
            iteration: Iteration/epoch number.
            title: Title group for the metrics.

        Returns:
            True if logged successfully, False otherwise.

        """
        return self._log_scalar_batch(title, speeds, iteration, "speed metrics")

    def log_learning_rates(
        self,
        learning_rates: dict[str, float],
        iteration: int,
        title: str = "Learning Rate",
    ) -> bool:
        """Log learning rate values to ClearML.

        Args:
            learning_rates: Dictionary with param group names and their LR values.
            iteration: Iteration/epoch number.
            title: Title group for the metrics.

        Returns:
            True if logged successfully, False otherwise.

        """
        return self._log_scalar_batch(title, learning_rates, iteration, "learning rates")

    def log_loss_components(
        self,
        losses: dict[str, float],
        iteration: int,
        title: str = "Losses/Train",
    ) -> bool:
        """Log individual loss components to ClearML.

        Args:
            losses: Dictionary with loss component names and values.
            iteration: Iteration/epoch number.
            title: Title group for the losses.

        Returns:
            True if logged successfully, False otherwise.

        """
        return self._log_scalar_batch(title, losses, iteration, "loss components")

    def log_mask_box_gap(
        self,
        gap: dict[str, float],
        iteration: int,
        title: str = "Metrics/Mask-vs-Box",
    ) -> bool:
        """Log the mask-minus-box mAP gap for a segmentation run.

        Args:
            gap: Output of ``extract_mask_box_gap``.
            iteration: Iteration/epoch number.
            title: Title group for the scalars.

        Returns:
            True if logged successfully, False otherwise.

        """
        if not gap:
            return False
        return self._log_scalar_batch(title, gap, iteration, "mask/box gap")

    def log_optimal_confidence(
        self,
        optimal: dict[str, Any],
        iteration: int = 0,
        title: str = "Operating Point",
    ) -> bool:
        """Log the F1-optimal confidence threshold, globally and per class.

        This is the threshold to deploy at, and it is none of the three thresholds
        already visible elsewhere in the task: mAP is computed at conf=0.001, the
        confusion matrix at conf=0.25/IoU=0.45, and ``args_val["conf"]`` is only what
        validation filtered at. Reported as its own group so nobody has to guess which
        number to ship.

        Args:
            optimal: Output of ``extract_optimal_confidence``.
            iteration: Iteration/epoch number.
            title: Title group.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None or not optimal:
            return False

        ok = self._log_scalar_batch(
            title,
            {
                "f1_optimal_confidence": float(optimal["global_conf"]),
                "f1_at_optimal_confidence": float(optimal["global_f1"]),
            },
            iteration,
            "optimal confidence",
        )

        per_class = optimal.get("per_class") or []
        if not per_class:
            return ok

        try:
            df = pd.DataFrame(per_class).rename(
                columns={
                    "class_name": "Class",
                    "conf": "F1-optimal conf",
                    "f1": "F1 at that conf",
                }
            )
            task_logger.report_table(
                title=title,
                series="Per-Class Threshold",
                iteration=iteration,
                table_plot=df,
            )
        except Exception as e:
            logger.warning("Failed to log per-class thresholds: %s", e)
            return False
        return ok

    def log_worst_images_table(
        self,
        worst_df: pd.DataFrame,
        iteration: int = 0,
        title: str = "Error Analysis",
    ) -> bool:
        """Log the worst-scoring validation images as a table.

        Args:
            worst_df: Output of ``extract_worst_images``.
            iteration: Iteration/epoch number.
            title: Title group.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None or worst_df is None or worst_df.empty:
            return False

        basis = worst_df.attrs.get("basis", "box")
        try:
            task_logger.report_table(
                title=title,
                series=f"Worst Images ({basis} F1)",
                iteration=iteration,
                table_plot=worst_df,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log worst-images table: %s", e)
            return False

    def log_image_files(
        self,
        files: list[Path],
        title: str = "Error Analysis",
        series_prefix: str = "",
        iteration: int = 0,
    ) -> int:
        """Report a bounded list of image files, returning how many were sent.

        The series name is the *rank* only, deliberately excluding the filename.
        ClearML's image retention (``sdk.metrics.file_history_size``) is applied per
        title/series, so a stable series per rank gives each slot its own history that
        can be scrubbed across iterations. Folding the filename in would mint a new
        series every time a different image became the worst, defeating that.

        Args:
            files: Image paths to upload.
            title: Title group.
            series_prefix: Prefix for each image's series name.
            iteration: Iteration number.

        Returns:
            Number of images successfully reported.

        """
        task_logger = self._get_logger()
        if task_logger is None or not files:
            return 0

        sent = 0
        failed = 0
        for rank, path in enumerate(files):
            try:
                task_logger.report_image(
                    title=title,
                    series=f"{series_prefix}{rank:02d}",
                    local_path=str(path),
                    iteration=iteration,
                )
                sent += 1
            except Exception:
                failed += 1
        # One summary line, not one per image: this loops over dataset items.
        if failed:
            logger.warning("failed to report %d/%d images", failed, len(files))
        return sent

    def log_confidence_split(
        self,
        split: dict[str, Any],
        iteration: int = 0,
        title: str = "Distributions",
    ) -> bool:
        """Log matched vs unmatched detection confidence as an overlaid histogram.

        Answers a question the existing undifferentiated confidence histogram cannot: is
        there any threshold that separates true from false detections? Two overlapping
        humps mean no threshold will fix precision -- the model, or the labels, need work.

        Args:
            split: Output of ``ValStatsAccumulator.confidence_split``.
            iteration: Iteration/epoch number.
            title: Title group.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None or not split:
            return False

        tp = np.asarray(split.get("tp", []), dtype=float)
        fp = np.asarray(split.get("fp", []), dtype=float)
        if tp.size == 0 and fp.size == 0:
            return False

        basis = split.get("basis", "box")
        try:
            fig = go.Figure()
            for values, label in ((tp, "Matched (TP)"), (fp, "Unmatched (FP)")):
                if values.size == 0:
                    continue
                # Binned here rather than by go.Histogram, which would serialise every
                # raw confidence into the payload -- 1 MB for a real validation set, which
                # the ClearML UI silently declines to render.
                centers, counts, width = binned_counts(values, bins=50, lo=0.0, hi=1.0)
                fig.add_trace(
                    go.Bar(
                        x=centers,
                        y=counts,
                        name=label,
                        opacity=0.65,
                        width=width,
                        hovertemplate=(
                            f"{label}<br>confidence ~%{{x:.3f}}"
                            "<br>%{y} detections<extra></extra>"
                        ),
                    )
                )
            fig.update_layout(
                title=f"Detection confidence, matched vs unmatched ({basis} IoU@0.50)",
                xaxis_title="Confidence",
                yaxis_title="Detections",
                barmode="overlay",
            )
            task_logger.report_plotly(
                title=title,
                series="TP vs FP Confidence",
                iteration=iteration,
                figure=fig,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log TP/FP confidence split: %s", e)
            return False

    def log_reliability_diagram(
        self,
        calibration: dict[str, Any],
        iteration: int = 0,
        title: str = "Calibration",
    ) -> bool:
        """Log a reliability diagram plus the calibration error scalar.

        Args:
            calibration: Output of ``extract_calibration_bins``.
            iteration: Iteration/epoch number.
            title: Title group.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None or not calibration:
            return False

        basis = calibration.get("basis", "box")
        try:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect calibration",
                    line=dict(dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=calibration["mean_confidence"],
                    y=calibration["precision"],
                    mode="lines+markers",
                    name="Observed",
                    text=[f"n={c}" for c in calibration["count"]],
                    hovertemplate=(
                        "Confidence: %{x:.3f}<br>Precision: %{y:.3f}"
                        "<br>%{text}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title=(
                    f"Reliability ({basis} IoU@0.50), ECE={calibration['ece']:.4f}"
                    " - ignores false negatives, read with recall"
                ),
                xaxis_title="Mean predicted confidence",
                yaxis_title="Observed precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05]),
            )
            task_logger.report_plotly(
                title=title,
                series="Reliability Diagram",
                iteration=iteration,
                figure=fig,
            )
        except Exception as e:
            logger.warning("Failed to log reliability diagram: %s", e)
            return False

        return self._log_scalar_batch(
            title,
            {"expected_calibration_error": float(calibration["ece"])},
            iteration,
            "calibration error",
        )

    def upload_html_report(
        self,
        path: Path | str,
        name: str = "evaluation_report",
        preview: str = "",
        metadata: dict[str, str] | None = None,
        link_media: bool = True,
    ) -> str | None:
        """Upload one self-contained HTML file as an artifact, returning its URL.

        `artifact_object` is always a **file path string**, never a directory: ClearML
        turns a folder artifact into a zip, which the fileserver will not serve as
        `text/html` and which is therefore unopenable in a browser. The `.html`
        extension is what makes the fileserver hand it back with scripts enabled.

        With `link_media`, the URL is additionally reported as media so the report shows
        up in **Debug Samples** as well as **Artifacts** -- the artifact panel is not
        where anyone looks first. Do not rely on that iframe executing the page's
        JavaScript; the report carries its own "open in a new tab" banner for exactly
        that reason.

        Args:
            path: The rendered HTML file.
            name: Artifact name; also its directory on the fileserver.
            preview: Plain-text summary shown without opening the file.
            metadata: String map recorded beside the artifact.
            link_media: Also report the URL under Debug Samples.

        Returns:
            The artifact URL, or None when the upload failed.

        """
        if self._task is None:
            logger.warning("No ClearML task available to upload %s", name)
            return None
        try:
            self._task.upload_artifact(
                name,
                artifact_object=str(path),
                metadata=metadata or {},
                preview=preview,
                wait_on_upload=True,
            )
        except Exception as e:
            logger.warning("Failed to upload %s: %s", name, e)
            return None

        url = None
        try:
            url = self._task.artifacts[name].url
        except Exception as e:
            logger.warning("Uploaded %s but could not read its URL: %s", name, e)
        if url and link_media:
            self.link_report_media(url)
        logger.info("uploaded %s (%s)", name, url or "no URL")
        return url

    def link_report_media(
        self,
        url: str,
        title: str = "Evaluation",
        series: str = "report",
        iteration: int = 0,
    ) -> bool:
        """Surface an already-uploaded artifact URL under Debug Samples.

        Args:
            url: Absolute artifact URL.
            title: Debug-sample title group.
            series: Series name within the group.
            iteration: Iteration to file it under.

        Returns:
            True if reported successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None or not url:
            return False
        try:
            task_logger.report_media(
                title=title, series=series, iteration=iteration, url=url
            )
            return True
        except Exception as e:
            logger.warning("Failed to link the report under Debug Samples: %s", e)
            return False

    def log_plotly_figure(
        self,
        figure: go.Figure,
        title: str,
        series: str,
        iteration: int = 0,
    ) -> bool:
        """Report an already-built Plotly figure.

        Args:
            figure: The figure to report.
            title: Title group.
            series: Series name.
            iteration: Iteration number.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None or figure is None:
            return False
        try:
            task_logger.report_plotly(
                title=title, series=series, iteration=iteration, figure=figure
            )
            return True
        except Exception as e:
            logger.warning("Failed to log figure %s/%s: %s", title, series, e)
            return False

    def log_dataframe_table(
        self,
        df: pd.DataFrame,
        title: str,
        series: str,
        iteration: int = 0,
    ) -> bool:
        """Report any DataFrame as a ClearML table.

        Args:
            df: The table to report.
            title: Title group.
            series: Series name.
            iteration: Iteration number.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None or df is None or df.empty:
            return False
        try:
            task_logger.report_table(
                title=title, series=series, iteration=iteration, table_plot=df
            )
            return True
        except Exception as e:
            logger.warning("Failed to log table %s/%s: %s", title, series, e)
            return False
