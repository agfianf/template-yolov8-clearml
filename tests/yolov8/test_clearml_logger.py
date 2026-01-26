"""Unit tests for src/yolov8/clearml_logger.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.yolov8.clearml_logger import YOLOClearMLLogger


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
        df = pd.DataFrame({
            "Class": ["cat", "dog"],
            "Precision": [0.8, 0.9],
        })

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


class TestLogPrCurvePlotly:
    """Tests for log_pr_curve_plotly method."""

    def test_returns_false_when_no_task(self):
        """Test that method returns False when no task available."""
        logger = YOLOClearMLLogger(task=None)

        result = logger.log_pr_curve_plotly(pr_curves=[])

        assert result is False

    def test_returns_false_when_empty_curves(self):
        """Test that method returns False when curves are empty."""
        mock_task = MagicMock()
        logger = YOLOClearMLLogger(task=mock_task)

        result = logger.log_pr_curve_plotly(pr_curves=[])

        assert result is False

    def test_logs_pr_curves(self):
        """Test that method logs PR curves successfully."""
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        logger = YOLOClearMLLogger(task=mock_task)
        pr_curves = [
            {
                "class_name": "cat",
                "precision": [1.0, 0.9, 0.8],
                "recall": [0.0, 0.5, 1.0],
                "ap50": 0.85,
            },
            {
                "class_name": "dog",
                "precision": [1.0, 0.85, 0.7],
                "recall": [0.0, 0.5, 1.0],
                "ap50": 0.78,
            },
        ]

        result = logger.log_pr_curve_plotly(
            pr_curves=pr_curves,
            iteration=1,
            title="PR Curves",
        )

        assert result is True
        mock_logger.report_plotly.assert_called_once()


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
