"""Unit tests for src/yolov8/clearml_logger.py module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.yolov8.clearml_logger import YOLOClearMLLogger, _split_curve_name


class TestYOLOClearMLLoggerInit:
    """Tests for YOLOClearMLLogger initialization."""

    def test_init_with_provided_task(self):
        """Test initialization with provided task."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        assert logger.task == mock_task

    def test_init_gets_current_task_when_none_provided(self):
        """Test that logger gets current task when none provided."""
        # When task is None, the logger tries to get the current task
        # We test this by checking that the task property returns the mocked task
        with patch("clearml.Task.current_task") as mock_current:
            mock_current.return_value = MagicMock()
            _ = YOLOClearMLLogger(task=None)

        # The task should be fetched during init
        mock_current.assert_called_once()


class TestLogInteractiveConfusionMatrix:
    """Tests for log_interactive_confusion_matrix method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_interactive_confusion_matrix(
            matrix=np.array([[1, 0], [0, 1]]),
            labels=["a", "b"],
        )

        assert result is False

    def test_logs_confusion_matrix(self):
        """Test that method logs confusion matrix successfully."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)
        matrix = np.array([[10, 2], [3, 15]])
        labels = ["cat", "dog"]

        result = logger.log_interactive_confusion_matrix(
            matrix=matrix,
            labels=labels,
            iteration=5,
            title="Test CM",
            series="Normalized",
        )

        assert result is True
        mock_logger.report_confusion_matrix.assert_called_once()
        call_kwargs = mock_logger.report_confusion_matrix.call_args[1]
        assert call_kwargs["title"] == "Test CM"
        assert call_kwargs["series"] == "Normalized"
        assert call_kwargs["iteration"] == 5

    def test_handles_exception(self):
        """Test that method handles exceptions gracefully."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        mock_logger.report_confusion_matrix.side_effect = Exception("Test error")

        logger = YOLOClearMLLogger(task=mock_task)

        result = logger.log_interactive_confusion_matrix(
            matrix=np.array([[1]]),
            labels=["a"],
        )

        assert result is False


class TestLogPerClassTable:
    """Tests for log_per_class_table method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_per_class_table(
            metrics_df=pd.DataFrame({"Class": ["a"]}),
        )

        assert result is False

    def test_logs_table(self):
        """Test that method logs table successfully."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)
        df = pd.DataFrame(
            {
                "Class": ["cat", "dog"],
                "Precision": [0.8, 0.9],
            }
        )

        result = logger.log_per_class_table(
            metrics_df=df,
            iteration=10,
            title="Test Table",
        )

        assert result is True
        mock_logger.report_table.assert_called_once()


class TestLogConfidenceHistogram:
    """Tests for log_confidence_histogram method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_confidence_histogram(confidences=[0.8, 0.9])

        assert result is False

    def test_returns_false_when_empty_confidences(self):
        """Test that method returns False when confidences are empty."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        result = logger.log_confidence_histogram(confidences=[])

        assert result is False

    def test_logs_histogram(self):
        """Test that method logs histogram successfully."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)
        confidences = [0.8, 0.85, 0.9, 0.95]

        result = logger.log_confidence_histogram(
            confidences=confidences,
            iteration=1,
            title="Test Hist",
            series="Confidence",
        )

        assert result is True
        mock_logger.report_histogram.assert_called_once()
        call_kwargs = mock_logger.report_histogram.call_args[1]
        assert call_kwargs["title"] == "Test Hist"
        assert call_kwargs["series"] == "Confidence"


def _curve(name="Precision-Recall(B)"):
    """Build a curve family in the shape extract_curve_data returns."""
    return {
        "name": name,
        "xlabel": "Recall",
        "ylabel": "Precision",
        "x": [0.0, 0.5, 1.0],
        "series": [
            {"class_name": "cat", "y": [1.0, 0.9, 0.8]},
            {"class_name": "dog", "y": [1.0, 0.85, 0.7]},
        ],
    }


class TestSplitCurveName:
    """Tests for the (B)/(M) suffix parsing."""

    def test_box_and_mask_suffixes(self):
        """Ultralytics' suffixes map to readable domain names."""
        assert _split_curve_name("Precision-Recall(B)") == ("Precision-Recall", "Box")
        assert _split_curve_name("F1-Confidence(M)") == ("F1-Confidence", "Mask")

    def test_unsuffixed_name_degrades_gracefully(self):
        """An unrecognised name must not raise -- these run inside callbacks."""
        assert _split_curve_name("Something") == ("Something", "Box")


class TestLogCurveFamily:
    """Tests for log_curve_family method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        assert logger.log_curve_family(_curve()) is False

    def test_returns_false_when_no_series(self):
        """Test that method returns False when the curve has no series."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        curve = _curve()
        curve["series"] = []
        assert logger.log_curve_family(curve) is False

    def test_logs_curve_with_domain_in_series(self):
        """Box and mask land as two series of one titled plot, not two plots."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        assert logger.log_curve_family(_curve(), iteration=1) is True

        kwargs = mock_logger.report_plotly.call_args[1]
        assert kwargs["title"] == "Curves/Precision-Recall"
        assert kwargs["series"] == "Box"

    def test_mask_curve_gets_mask_series(self):
        """A mask family must be distinguishable from the box one."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        logger.log_curve_family(_curve("Precision-Recall(M)"))

        kwargs = mock_logger.report_plotly.call_args[1]
        assert kwargs["title"] == "Curves/Precision-Recall"
        assert kwargs["series"] == "Mask"


class TestLogPerClassBar:
    """Tests for log_per_class_bar method."""

    DF = pd.DataFrame(
        {
            "Class": ["good", "bad", "mid"],
            "Box-mAP50-95": [0.9, 0.1, 0.5],
            "Mask-mAP50-95": [0.7, 0.05, 0.4],
        }
    )

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        assert logger.log_per_class_bar(self.DF) is False

    def test_returns_false_when_sort_column_missing(self):
        """Without the sort column there is nothing meaningful to draw."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        assert logger.log_per_class_bar(pd.DataFrame({"Class": ["a"]})) is False

    def test_sorts_worst_class_first(self):
        """The whole point of the chart: the class you must fix is leftmost."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        assert logger.log_per_class_bar(self.DF) is True

        fig = mock_logger.report_plotly.call_args[1]["figure"]
        assert list(fig.data[0].x) == ["bad", "mid", "good"]

    def test_includes_mask_trace_when_present(self):
        """Box and mask bars are grouped so a lagging mask head is visible."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        logger.log_per_class_bar(self.DF)

        fig = mock_logger.report_plotly.call_args[1]["figure"]
        assert [t.name for t in fig.data] == ["Box", "Mask"]

    def test_omits_mask_trace_for_detection(self):
        """A detect run has no mask column, so only one trace is drawn."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        logger.log_per_class_bar(self.DF.drop(columns=["Mask-mAP50-95"]))

        fig = mock_logger.report_plotly.call_args[1]["figure"]
        assert [t.name for t in fig.data] == ["Box"]


class TestLogImageFiles:
    """Tests for log_image_files method."""

    def test_returns_zero_when_no_task(self):
        """Test that method reports nothing when no task available."""
        logger = YOLOClearMLLogger(task=None)

        assert logger.log_image_files([Path("a.jpg")]) == 0

    def test_series_is_rank_only_not_filename(self):
        """ClearML retention is per title/series, so the slot must be stable.

        Including the filename would mint a new series each time a different image became
        the worst, so no rank slot would accumulate history.
        """
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        sent = logger.log_image_files(
            [Path("/tmp/zebra.jpg"), Path("/tmp/apple.jpg")],
            title="Error Analysis",
            series_prefix="worst-box-",
        )

        assert sent == 2
        series = [c[1]["series"] for c in mock_logger.report_image.call_args_list]
        assert series == ["worst-box-00", "worst-box-01"]
        assert not any("zebra" in s for s in series)

    def test_counts_only_successful_reports(self):
        """A failing upload must not be counted, nor abort the rest."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_logger.report_image.side_effect = [None, Exception("boom"), None]
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        sent = logger.log_image_files([Path("a.jpg"), Path("b.jpg"), Path("c.jpg")])

        assert sent == 2


class TestLogConfidenceSplit:
    """Tests for log_confidence_split method."""

    def test_returns_false_when_empty(self):
        """No detections means no plot."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        split = {"tp": np.array([]), "fp": np.array([]), "basis": "box"}
        assert logger.log_confidence_split(split) is False

    def test_logs_two_overlaid_histograms(self):
        """Matched and unmatched are drawn together so overlap is visible."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        split = {
            "tp": np.array([0.9, 0.8]),
            "fp": np.array([0.2, 0.3]),
            "basis": "mask",
        }
        assert logger.log_confidence_split(split) is True

        fig = mock_logger.report_plotly.call_args[1]["figure"]
        assert [t.name for t in fig.data] == ["Matched (TP)", "Unmatched (FP)"]
        assert fig.layout.barmode == "overlay"


class TestLogReliabilityDiagram:
    """Tests for log_reliability_diagram method."""

    CALIBRATION = {
        "bin_center": [0.25, 0.75],
        "precision": [0.2, 0.9],
        "mean_confidence": [0.25, 0.75],
        "count": [10, 20],
        "ece": 0.075,
        "basis": "box",
    }

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        assert logger.log_reliability_diagram(self.CALIBRATION) is False

    def test_logs_diagonal_and_observed_plus_scalar(self):
        """The plot carries a reference diagonal; the ECE is also a scalar."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        assert logger.log_reliability_diagram(self.CALIBRATION) is True

        fig = mock_logger.report_plotly.call_args[1]["figure"]
        assert [t.name for t in fig.data] == ["Perfect calibration", "Observed"]
        mock_logger.report_scalar.assert_called_once_with(
            title="Calibration",
            series="expected_calibration_error",
            value=0.075,
            iteration=0,
        )

    def test_title_warns_about_false_negatives(self):
        """The metric ignores FNs; the plot has to say so where it is read."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        logger.log_reliability_diagram(self.CALIBRATION)

        fig = mock_logger.report_plotly.call_args[1]["figure"]
        assert "false negative" in fig.layout.title.text


class TestLogMaskBoxGap:
    """Tests for log_mask_box_gap method."""

    def test_returns_false_for_empty_gap(self):
        """A detect run yields no gap, which is not a failure to report."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        assert logger.log_mask_box_gap({}, iteration=0) is False

    def test_reports_each_gap_scalar(self):
        """Both the mAP50-95 and mAP50 gaps are reported."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        gap = {"mask_minus_box_mAP50-95": -0.2, "mask_minus_box_mAP50": -0.3}
        assert logger.log_mask_box_gap(gap, iteration=3) is True
        assert mock_logger.report_scalar.call_count == 2


class TestLogOptimalConfidence:
    """Tests for log_optimal_confidence method."""

    OPTIMAL = {
        "global_conf": 0.32,
        "global_f1": 0.61,
        "per_class": [
            {"class_name": "cat", "conf": 0.31, "f1": 0.6},
            {"class_name": "dog", "conf": 0.41, "f1": 0.64},
        ],
    }

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        assert logger.log_optimal_confidence(self.OPTIMAL) is False

    def test_reports_scalars_and_per_class_table(self):
        """The global threshold is a scalar; per-class thresholds are a table."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        logger = YOLOClearMLLogger(task=mock_task)

        assert logger.log_optimal_confidence(self.OPTIMAL) is True

        assert mock_logger.report_scalar.call_count == 2
        table = mock_logger.report_table.call_args[1]["table_plot"]
        assert list(table.columns) == ["Class", "F1-optimal conf", "F1 at that conf"]


class TestLogScalarGrouped:
    """Tests for log_scalar_grouped method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_scalar_grouped(
            title="Test", series="value", value=1.0, iteration=1
        )

        assert result is False

    def test_logs_scalar(self):
        """Test that method logs scalar successfully."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)

        result = logger.log_scalar_grouped(
            title="Losses/Train",
            series="box_loss",
            value=0.5,
            iteration=10,
        )

        assert result is True
        mock_logger.report_scalar.assert_called_once_with(
            title="Losses/Train",
            series="box_loss",
            value=0.5,
            iteration=10,
        )


class TestLogPerClassScatter:
    """Tests for log_per_class_scatter method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_per_class_scatter(
            class_names=["a", "b"],
            metric_values=[0.8, 0.9],
            metric_name="mAP50",
        )

        assert result is False

    def test_returns_false_when_empty_data(self):
        """Test that method returns False when data is empty."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        result = logger.log_per_class_scatter(
            class_names=[],
            metric_values=[],
            metric_name="mAP50",
        )

        assert result is False

    def test_logs_scatter_plot(self):
        """Test that method logs scatter plot successfully."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)

        result = logger.log_per_class_scatter(
            class_names=["cat", "dog", "bird"],
            metric_values=[0.8, 0.85, 0.9],
            metric_name="mAP50",
            iteration=5,
        )

        assert result is True
        mock_logger.report_plotly.assert_called_once()


class TestLogSpeedMetrics:
    """Tests for log_speed_metrics method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_speed_metrics(
            speeds={"preprocess_ms": 1.0},
            iteration=1,
        )

        assert result is False

    def test_logs_all_speed_metrics(self):
        """Test that method logs all speed metrics."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)
        speeds = {
            "preprocess_ms": 1.5,
            "inference_ms": 10.0,
            "postprocess_ms": 2.5,
        }

        result = logger.log_speed_metrics(speeds=speeds, iteration=5)

        assert result is True
        assert mock_logger.report_scalar.call_count == 3


class TestLogLearningRates:
    """Tests for log_learning_rates method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_learning_rates(
            learning_rates={"param_group_0": 0.001},
            iteration=1,
        )

        assert result is False

    def test_logs_all_learning_rates(self):
        """Test that method logs all learning rates."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)
        learning_rates = {
            "param_group_0": 0.001,
            "param_group_1": 0.0001,
        }

        result = logger.log_learning_rates(learning_rates=learning_rates, iteration=10)

        assert result is True
        assert mock_logger.report_scalar.call_count == 2


class TestLogLossComponents:
    """Tests for log_loss_components method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_loss_components(
            losses={"box_loss": 0.5},
            iteration=1,
        )

        assert result is False

    def test_logs_all_loss_components(self):
        """Test that method logs all loss components."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)
        losses = {
            "box_loss": 0.5,
            "cls_loss": 0.3,
            "dfl_loss": 0.2,
        }

        result = logger.log_loss_components(losses=losses, iteration=5)

        assert result is True
        assert mock_logger.report_scalar.call_count == 3
