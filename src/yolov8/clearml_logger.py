"""ClearML logging wrapper for YOLO training visualization.

This module provides a unified interface for logging various metrics and visualizations
to ClearML, including interactive confusion matrices, per-class tables, PR curves,
histograms, and scalar metrics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.logging import get_logger


if TYPE_CHECKING:
    from clearml import Task


logger = get_logger(__name__)


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

    def log_pr_curve_plotly(
        self,
        pr_curves: list[dict],
        iteration: int = 0,
        title: str = "PR Curves",
        series: str = "All Classes",
    ) -> bool:
        """Log interactive PR curves using Plotly to ClearML.

        Args:
            pr_curves: List of dicts with keys: class_name, precision, recall, ap50.
            iteration: Iteration/epoch number.
            title: Title for the plot.
            series: Series name.

        Returns:
            True if logged successfully, False otherwise.

        """
        task_logger = self._get_logger()
        if task_logger is None:
            logger.warning("No ClearML task available for logging PR curves")
            return False

        if not pr_curves:
            logger.warning("No PR curve data to log")
            return False

        try:
            fig = go.Figure()

            for curve in pr_curves:
                fig.add_trace(
                    go.Scatter(
                        x=curve["recall"],
                        y=curve["precision"],
                        mode="lines",
                        name=f"{curve['class_name']} (AP50={curve.get('ap50', 0):.3f})",
                        hovertemplate=(
                            f"Class: {curve['class_name']}<br>"
                            "Recall: %{x:.3f}<br>"
                            "Precision: %{y:.3f}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title="Precision-Recall Curves",
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend_title="Class (AP50)",
                hovermode="closest",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05]),
            )

            task_logger.report_plotly(
                title=title,
                series=series,
                iteration=iteration,
                figure=fig,
            )
            return True
        except Exception as e:
            logger.warning("Failed to log PR curves: %s", e)
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
            logger.warning("Failed to log scalar %s/%s: %s", title, series, e)
            return False

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
        task_logger = self._get_logger()
        if task_logger is None:
            return False

        success = True
        for metric_name, value in speeds.items():
            if not self.log_scalar_grouped(title, metric_name, value, iteration):
                success = False

        return success

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
        task_logger = self._get_logger()
        if task_logger is None:
            return False

        success = True
        for group_name, lr_value in learning_rates.items():
            if not self.log_scalar_grouped(title, group_name, lr_value, iteration):
                success = False

        return success

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
        task_logger = self._get_logger()
        if task_logger is None:
            return False

        success = True
        for loss_name, value in losses.items():
            if not self.log_scalar_grouped(title, loss_name, value, iteration):
                success = False

        return success
