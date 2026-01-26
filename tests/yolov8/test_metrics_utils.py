"""Unit tests for src/yolov8/metrics_utils.py module."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.yolov8.metrics_utils import (
    collect_prediction_confidences,
    extract_confusion_matrix_data,
    extract_learning_rate,
    extract_loss_components,
    extract_per_class_metrics,
    extract_pr_curve_data,
    extract_speed_metrics,
)


class TestExtractPerClassMetrics:
    """Tests for extract_per_class_metrics function."""

    def test_returns_none_when_no_box_metrics(self):
        """Test that function returns None when validator has no box metrics."""
        validator = MagicMock()
        validator.metrics.box = None

        result = extract_per_class_metrics(validator)
        assert result is None

    def test_returns_none_when_no_metrics(self):
        """Test that function returns None when metrics attribute is missing."""
        validator = MagicMock()
        del validator.metrics.box

        result = extract_per_class_metrics(validator)
        assert result is None

    def test_returns_dataframe_with_metrics(self):
        """Test that function returns DataFrame with correct columns."""
        validator = MagicMock()
        box_metrics = MagicMock()
        box_metrics.p = np.array([0.8, 0.9])
        box_metrics.r = np.array([0.7, 0.85])
        box_metrics.ap50 = np.array([0.75, 0.88])
        box_metrics.ap = np.array([0.6, 0.7])
        validator.metrics.box = box_metrics

        result = extract_per_class_metrics(validator, class_names=["cat", "dog"])

        assert isinstance(result, pd.DataFrame)
        expected_cols = ["Class", "Precision", "Recall", "mAP50", "mAP50-95"]
        assert list(result.columns) == expected_cols
        assert result["Class"].tolist() == ["cat", "dog"]
        assert result["Precision"].tolist() == [0.8, 0.9]
        assert result["Recall"].tolist() == [0.7, 0.85]

    def test_uses_default_class_names_when_not_provided(self):
        """Test that function uses default class names when none provided."""
        validator = MagicMock()
        box_metrics = MagicMock()
        box_metrics.p = np.array([0.8, 0.9])
        box_metrics.r = np.array([0.7, 0.85])
        box_metrics.ap50 = np.array([0.75, 0.88])
        box_metrics.ap = np.array([0.6, 0.7])
        validator.metrics.box = box_metrics

        result = extract_per_class_metrics(validator)

        assert result["Class"].tolist() == ["class_0", "class_1"]


class TestExtractConfusionMatrixData:
    """Tests for extract_confusion_matrix_data function."""

    def test_returns_none_when_no_confusion_matrix(self):
        """Test that function returns None tuple when no confusion matrix."""
        validator = MagicMock()
        validator.confusion_matrix = None

        normalized, labels, counts = extract_confusion_matrix_data(validator)

        assert normalized is None
        assert labels == []
        assert counts is None

    def test_returns_matrices_and_labels(self):
        """Test that function returns normalized matrix, labels, and counts."""
        validator = MagicMock()
        cm = MagicMock()
        cm.matrix = np.array([[10, 2], [3, 15]])
        cm.names = {0: "cat", 1: "dog"}
        validator.confusion_matrix = cm

        normalized, labels, counts = extract_confusion_matrix_data(validator)

        assert counts is not None
        np.testing.assert_array_equal(counts, np.array([[10, 2], [3, 15]]))
        assert labels == ["cat", "dog"]
        assert normalized is not None
        # Check normalization: row sums should be ~1
        assert np.allclose(normalized.sum(axis=1), [1.0, 1.0], atol=0.01)

    def test_uses_provided_class_names(self):
        """Test that function uses provided class names."""
        validator = MagicMock()
        cm = MagicMock()
        cm.matrix = np.array([[10, 2], [3, 15]])
        cm.names = None
        validator.confusion_matrix = cm

        _, labels, _ = extract_confusion_matrix_data(
            validator, class_names=["person", "car"]
        )

        assert labels == ["person", "car"]


class TestExtractLossComponents:
    """Tests for extract_loss_components function."""

    def test_returns_empty_dict_when_no_loss_items(self):
        """Test that function returns empty dict when no loss items."""
        trainer = MagicMock()
        trainer.loss_items = None
        trainer.tloss = None

        result = extract_loss_components(trainer)
        assert result == {}

    def test_returns_loss_components(self):
        """Test that function returns individual loss components."""
        trainer = MagicMock()
        trainer.loss_items = [0.5, 0.3, 0.2]
        trainer.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        trainer.tloss = 1.0

        result = extract_loss_components(trainer)

        assert "box_loss" in result
        assert result["box_loss"] == 0.5
        assert result["cls_loss"] == 0.3
        assert result["dfl_loss"] == 0.2
        assert result["total_loss"] == 1.0


class TestExtractLearningRate:
    """Tests for extract_learning_rate function."""

    def test_returns_empty_dict_when_no_optimizer(self):
        """Test that function returns empty dict when no optimizer."""
        trainer = MagicMock()
        trainer.optimizer = None

        result = extract_learning_rate(trainer)
        assert result == {}

    def test_returns_learning_rates(self):
        """Test that function returns learning rates for each param group."""
        trainer = MagicMock()
        trainer.optimizer.param_groups = [
            {"lr": 0.001},
            {"lr": 0.0001},
        ]

        result = extract_learning_rate(trainer)

        assert result["param_group_0"] == 0.001
        assert result["param_group_1"] == 0.0001


class TestExtractSpeedMetrics:
    """Tests for extract_speed_metrics function."""

    def test_returns_empty_dict_when_no_speed(self):
        """Test that function returns empty dict when no speed data."""
        validator = MagicMock()
        validator.speed = None

        result = extract_speed_metrics(validator)
        assert result == {}

    def test_returns_speed_metrics(self):
        """Test that function returns speed metrics."""
        validator = MagicMock()
        validator.speed = {
            "preprocess": 1.5,
            "inference": 10.0,
            "postprocess": 2.5,
        }

        result = extract_speed_metrics(validator)

        assert result["preprocess_ms"] == 1.5
        assert result["inference_ms"] == 10.0
        assert result["postprocess_ms"] == 2.5
        assert result["total_ms"] == 14.0


class TestExtractPrCurveData:
    """Tests for extract_pr_curve_data function."""

    def test_returns_none_when_no_box_metrics(self):
        """Test that function returns None when no box metrics."""
        validator = MagicMock()
        validator.metrics.box = None

        result = extract_pr_curve_data(validator)
        assert result is None

    def test_returns_none_when_no_pr_data(self):
        """Test that function returns None when no PR curve data."""
        validator = MagicMock()
        box_metrics = MagicMock()
        box_metrics.px = None
        box_metrics.py = None
        validator.metrics.box = box_metrics

        result = extract_pr_curve_data(validator)
        assert result is None

    def test_returns_pr_curve_data(self):
        """Test that function returns PR curve data."""
        validator = MagicMock()
        box_metrics = MagicMock()
        box_metrics.px = np.linspace(0, 1, 10)
        box_metrics.py = np.array([np.linspace(1, 0, 10), np.linspace(0.9, 0.1, 10)])
        box_metrics.ap50 = np.array([0.8, 0.75])
        validator.metrics.box = box_metrics

        result = extract_pr_curve_data(validator, class_names=["cat", "dog"])

        assert result is not None
        assert len(result) == 2
        assert result[0]["class_name"] == "cat"
        assert result[0]["ap50"] == 0.8
        assert result[1]["class_name"] == "dog"
        assert result[1]["ap50"] == 0.75


class TestCollectPredictionConfidences:
    """Tests for collect_prediction_confidences function."""

    def test_returns_empty_when_no_results(self):
        """Test that function returns empty lists when no results."""
        results = []

        all_confs, per_class = collect_prediction_confidences(results)

        assert all_confs == []
        assert per_class == {}

    def test_collects_confidences(self):
        """Test that function collects confidences from results."""
        import torch

        r1 = MagicMock()
        r1.boxes.conf = torch.tensor([0.9, 0.8])
        r1.boxes.cls = torch.tensor([0, 1])

        r2 = MagicMock()
        r2.boxes.conf = torch.tensor([0.7])
        r2.boxes.cls = torch.tensor([0])

        results = [r1, r2]

        all_confs, per_class = collect_prediction_confidences(
            results, class_names=["cat", "dog"]
        )

        assert len(all_confs) == 3
        # Use approximate comparison for floating point values
        assert any(abs(c - 0.9) < 0.01 for c in all_confs)
        assert any(abs(c - 0.8) < 0.01 for c in all_confs)
        assert any(abs(c - 0.7) < 0.01 for c in all_confs)

        assert "cat" in per_class
        assert "dog" in per_class
        assert len(per_class["cat"]) == 2
        assert len(per_class["dog"]) == 1

    def test_handles_missing_boxes(self):
        """Test that function handles results without boxes."""
        r1 = MagicMock()
        r1.boxes = None

        results = [r1]

        all_confs, per_class = collect_prediction_confidences(results)

        assert all_confs == []
        assert per_class == {}
