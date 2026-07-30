"""Unit tests for src/yolov8/metrics_utils.py module."""

import tempfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ultralytics.utils.metrics import DetMetrics, Metric, SegmentMetrics

from src.yolov8.metrics_utils import (
    ValStatsAccumulator,
    collect_prediction_confidences,
    extract_calibration_bins,
    extract_confusion_matrix_data,
    extract_curve_data,
    extract_learning_rate,
    extract_loss_components,
    extract_mask_box_gap,
    extract_optimal_confidence,
    extract_per_class_metrics,
    extract_speed_metrics,
    extract_worst_images,
)


# Class 2 deliberately has no ground truth. Ultralytics drops absent classes from every
# per-class array (they are indexed by ap_class_index), so a naive class_names[:n] slice
# mislabels every row after the gap. These fixtures exist to catch exactly that.
NAMES = {0: "ripe", 1: "unripe", 2: "absent_class"}


def _im(precision, recall, f1, tp, fp, fn):
    """One entry of Metric.image_metrics."""
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _build_metrics(cls, seed: int = 0):
    """Build a real, processed metrics object -- not a mock.

    Mocks are why the PR-curve bug survived: a MagicMock answers to any attribute, so a
    test could assert against `box_metrics.py` (which has never existed in this
    ultralytics) and pass. Anything that reads the ultralytics API is tested against the
    real classes.
    """
    rng = np.random.default_rng(seed)
    n_pred = 400
    conf = rng.random(n_pred)
    stats = {
        "tp": [rng.random((n_pred, 10)) < conf[:, None] * 0.9],
        "conf": [conf],
        "pred_cls": [rng.choice([0, 1], size=n_pred)],
        "target_cls": [rng.choice([0, 1], size=300)],
        "target_img": [np.array([0, 1])],
    }
    if cls is SegmentMetrics:
        # Masks deliberately worse than boxes, as they almost always are in practice.
        stats["tp_m"] = [rng.random((n_pred, 10)) < conf[:, None] * 0.6]

    metrics = cls()
    metrics.names = NAMES
    metrics.stats = stats
    with tempfile.TemporaryDirectory() as td:
        metrics.process(save_dir=Path(td), plot=False)
    return metrics


def _validator(metrics):
    return SimpleNamespace(metrics=metrics)


class TestUltralyticsApiSurface:
    """Pin the ultralytics attributes this module depends on.

    Every assertion here is the shape of a bug we actually shipped. `Metric.py` was read
    for PR curves and does not exist, so the plot was silently never reported for the
    whole life of that code. A mock cannot catch that; only the real class can.
    """

    def test_metric_exposes_prec_values_not_py(self):
        """`prec_values` is the real attribute; `py` never existed.

        Read off a *processed* Metric: `prec_values` is assigned by `update()`, so a bare
        Metric() has neither name and would prove nothing.
        """
        metric = _build_metrics(DetMetrics).box

        assert isinstance(metric, Metric)
        assert hasattr(metric, "prec_values")
        assert not hasattr(metric, "py")
        # The curve accessor the extractor actually calls.
        assert hasattr(metric, "curves_results")

    def test_curves_results_pairs_with_curve_names(self):
        """curves_results entries are [x, y, xlabel, ylabel], aligned with `curves`."""
        metrics = _build_metrics(DetMetrics)
        assert len(metrics.curves_results) == len(metrics.curves)
        for entry in metrics.curves_results:
            assert len(entry) == 4

    def test_segment_summary_carries_mask_columns(self):
        """SegmentMetrics.summary() is the source of per-class mask metrics."""
        metrics = _build_metrics(SegmentMetrics)
        assert "Mask-P" in metrics.summary()[0]

    def test_segment_curves_cover_box_and_mask(self):
        """A seg run must expose both (B) and (M) curve families."""
        metrics = _build_metrics(SegmentMetrics)
        names = metrics.curves
        assert any(n.endswith("(B)") for n in names)
        assert any(n.endswith("(M)") for n in names)

    def test_absent_classes_are_dropped_from_per_class_arrays(self):
        """The premise of the name-alignment bug: absent classes vanish."""
        metrics = _build_metrics(DetMetrics)
        assert list(metrics.ap_class_index) == [0, 1]
        assert len(metrics.box.p) == 2  # not 3, despite 3 names


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

    def test_detection_columns(self):
        """A detect run gets box columns and no mask columns."""
        result = extract_per_class_metrics(_validator(_build_metrics(DetMetrics)))

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "Class",
            "Images",
            "Instances",
            "Box-P",
            "Box-R",
            "Box-F1",
            "Box-mAP50",
            "Box-mAP50-95",
        ]
        assert not [c for c in result.columns if c.startswith("Mask")]

    def test_segmentation_adds_mask_columns_and_delta(self):
        """A seg run reports mask metrics alongside box, plus the gap between them."""
        result = extract_per_class_metrics(_validator(_build_metrics(SegmentMetrics)))

        for col in ("Mask-P", "Mask-R", "Mask-F1", "Mask-mAP50", "Mask-mAP50-95"):
            assert col in result.columns

        delta = result["Mask-Box-mAP50-95-delta"]
        expected = result["Mask-mAP50-95"] - result["Box-mAP50-95"]
        assert np.allclose(delta, expected)
        # Masks were generated worse than boxes, so every delta must be negative.
        assert (delta < 0).all()

    def test_names_skip_absent_classes(self):
        """Only classes with ground truth appear, correctly named.

        Regression guard for the mislabelling bug: 'absent_class' has no instances, so it
        must be missing entirely rather than shifting the other rows' names.
        """
        result = extract_per_class_metrics(_validator(_build_metrics(SegmentMetrics)))

        assert result["Class"].tolist() == ["ripe", "unripe"]
        assert "absent_class" not in result["Class"].tolist()
        assert (result["Instances"] > 0).all()


class TestExtractConfusionMatrixData:
    """Tests for extract_confusion_matrix_data function."""

    # Ultralytics stores matrix[predicted, ground_truth]. Rows = predictions.
    #   GT ripe   = 10  -> 8 called ripe, 1 called unripe, 1 missed
    #   GT unripe = 20  -> 2 called ripe, 15 called unripe, 3 missed
    #   background      -> 4 spurious ripe, 5 spurious unripe
    RAW = np.array(
        [
            [8, 2, 4],
            [1, 15, 5],
            [1, 3, 0],
        ],
        dtype=float,
    )

    def _extract(self):
        cm = SimpleNamespace(matrix=self.RAW.copy(), names={0: "ripe", 1: "unripe"}, nc=2)
        return extract_confusion_matrix_data(SimpleNamespace(confusion_matrix=cm))

    def test_returns_none_when_no_confusion_matrix(self):
        """Test that function returns None tuple when no confusion matrix."""
        validator = MagicMock()
        validator.confusion_matrix = None

        normalized, labels, counts = extract_confusion_matrix_data(validator)

        assert normalized is None
        assert labels == []
        assert counts is None

    def test_background_label_appended(self):
        """The matrix is one wider than the class list; that column is background."""
        _, labels, _ = self._extract()
        assert labels == ["ripe", "unripe", "background"]

    def test_counts_are_transposed_to_actual_major(self):
        """Rows of the reported matrix must be ground truth, matching yaxis='Actual'.

        Ultralytics fills matrix[pred, gt]; we report yaxis='Actual', so the matrix has to
        be transposed or the plot is a transposed lie.
        """
        _, _, counts = self._extract()

        np.testing.assert_array_equal(counts, self.RAW.T)
        # Row sums are now per-actual-class totals: 10 ripe, 20 unripe, 9 background FPs.
        np.testing.assert_array_equal(counts.sum(axis=1), [10, 20, 9])

    def test_normalized_matches_ultralytics_column_normalization(self):
        """Normalization must be by ground-truth class, as ultralytics does it.

        Ultralytics divides by `matrix.sum(0)` (utils/metrics.py:549). Dividing by rows
        instead yields precision-like numbers, which disagreed with the
        confusion_matrix_normalized.png uploaded under the same ClearML title.
        """
        normalized, _, _ = self._extract()

        np.testing.assert_allclose(normalized, (self.RAW / self.RAW.sum(0)).T)
        # Each actual class's row sums to 1 ...
        np.testing.assert_allclose(normalized.sum(axis=1), [1.0, 1.0, 1.0])
        # ... and the diagonal therefore reads as per-class recall.
        assert normalized[0, 0] == pytest.approx(8 / 10)
        assert normalized[1, 1] == pytest.approx(15 / 20)

    def test_uses_provided_class_names(self):
        """Test that function uses provided class names."""
        cm = SimpleNamespace(matrix=np.array([[10, 2], [3, 15]]), names=None, nc=2)

        _, labels, _ = extract_confusion_matrix_data(
            SimpleNamespace(confusion_matrix=cm), class_names=["person", "car"]
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

    def test_returns_loss_components_legacy_sequence(self):
        """Test the pre-8.4 shape: loss_items a sequence, tloss a scalar."""
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

    def test_returns_loss_components_dict_shape(self):
        """Test the ultralytics >=8.4 shape: loss_items and tloss are dicts.

        Regression guard. Iterating a dict yields its keys, so the old
        zip(loss_names, loss_items) paired names with names and float() raised
        ValueError: could not convert string to float: 'box_loss' -- which killed
        training at the first on_train_epoch_end callback.
        """
        trainer = MagicMock()
        trainer.loss_items = {"box_loss": 0.5, "cls_loss": 0.3, "dfl_loss": 0.2}
        trainer.loss_names = tuple(trainer.loss_items)
        trainer.tloss = {"box_loss": 0.4, "cls_loss": 0.2, "dfl_loss": 0.1}

        result = extract_loss_components(trainer)

        assert result["box_loss"] == 0.5
        assert result["cls_loss"] == 0.3
        assert result["dfl_loss"] == 0.2
        assert result["total_loss"] == pytest.approx(0.7)


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


class TestExtractCurveData:
    """Tests for extract_curve_data function.

    Replaces the tests for extract_pr_curve_data, which asserted against a mocked `py`
    attribute that does not exist on the real Metric -- so the tests passed while the
    function returned None on every real run.
    """

    def test_returns_none_when_no_box_metrics(self):
        """Test that function returns None when no box metrics."""
        validator = MagicMock()
        validator.metrics.box = None

        result = extract_curve_data(validator)
        assert result is None

    def test_detection_returns_four_curve_families(self):
        """PR, F1-conf, P-conf and R-conf, all box."""
        result = extract_curve_data(_validator(_build_metrics(DetMetrics)))

        assert result is not None
        names = [c["name"] for c in result]
        assert names == [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
        ]

    def test_segmentation_returns_box_and_mask_families(self):
        """A seg run doubles the curve count, four box plus four mask."""
        result = extract_curve_data(_validator(_build_metrics(SegmentMetrics)))

        names = [c["name"] for c in result]
        assert len([n for n in names if n.endswith("(B)")]) == 4
        assert len([n for n in names if n.endswith("(M)")]) == 4

    def test_axis_labels_come_from_ultralytics(self):
        """The labels are carried in curves_results, not guessed by us."""
        result = extract_curve_data(_validator(_build_metrics(DetMetrics)))
        by_name = {c["name"]: c for c in result}

        pr = by_name["Precision-Recall(B)"]
        assert (pr["xlabel"], pr["ylabel"]) == ("Recall", "Precision")

        f1 = by_name["F1-Confidence(B)"]
        assert (f1["xlabel"], f1["ylabel"]) == ("Confidence", "F1")

    def test_series_are_named_and_aligned_with_x(self):
        """One series per present class, each the same length as x."""
        result = extract_curve_data(_validator(_build_metrics(DetMetrics)))
        curve = result[0]

        assert [s["class_name"] for s in curve["series"]] == ["ripe", "unripe"]
        for s in curve["series"]:
            assert len(s["y"]) == len(curve["x"])


class TestExtractOptimalConfidence:
    """Tests for extract_optimal_confidence function."""

    def test_returns_none_without_curves(self):
        """No curve data means no operating point."""
        validator = MagicMock()
        validator.metrics.box = None
        assert extract_optimal_confidence(validator) is None

    def test_reports_global_and_per_class_thresholds(self):
        """Thresholds are confidences in [0, 1], one per present class."""
        result = extract_optimal_confidence(_validator(_build_metrics(DetMetrics)))

        assert 0.0 <= result["global_conf"] <= 1.0
        assert 0.0 <= result["global_f1"] <= 1.0
        assert [p["class_name"] for p in result["per_class"]] == ["ripe", "unripe"]
        for entry in result["per_class"]:
            assert 0.0 <= entry["conf"] <= 1.0

    def test_per_class_conf_maximises_that_class_f1(self):
        """Each per-class threshold is the argmax of that class's own F1 curve."""
        metrics = _build_metrics(DetMetrics)
        result = extract_optimal_confidence(_validator(metrics))

        for i, entry in enumerate(result["per_class"]):
            assert entry["f1"] == pytest.approx(metrics.box.f1_curve[i].max())


class TestExtractMaskBoxGap:
    """Tests for extract_mask_box_gap function."""

    def test_empty_for_detection(self):
        """A detect model has no mask metrics, so there is no gap to report."""
        assert extract_mask_box_gap(_validator(_build_metrics(DetMetrics))) == {}

    def test_negative_gap_for_segmentation(self):
        """Masks were built worse than boxes, so the gap must be negative."""
        metrics = _build_metrics(SegmentMetrics)
        gap = extract_mask_box_gap(_validator(metrics))

        assert gap["mask_minus_box_mAP50-95"] == pytest.approx(
            metrics.seg.map - metrics.box.map
        )
        assert gap["mask_minus_box_mAP50-95"] < 0


class TestExtractWorstImages:
    """Tests for extract_worst_images function."""

    IMAGE_METRICS = {
        "empty.jpg": _im(1.0, 1.0, 1.0, 0, 0, 0),
        "mid.jpg": _im(0.5, 0.4, 0.44, 4, 4, 6),
        "worst.jpg": _im(0.0, 0.0, 0.0, 0, 9, 7),
    }

    def test_returns_none_without_per_image_data(self):
        """image_metrics is only populated by a full validation pass."""
        metrics = _build_metrics(DetMetrics)
        metrics.box.image_metrics = {}
        assert extract_worst_images(_validator(metrics)) is None

    def test_sorted_worst_first_with_rank(self):
        """Worst F1 first, and a Rank column tying rows to the gallery series."""
        metrics = _build_metrics(DetMetrics)
        metrics.box.image_metrics = dict(self.IMAGE_METRICS)

        result = extract_worst_images(_validator(metrics), limit=3)

        assert result["Image"].tolist() == ["worst.jpg", "mid.jpg", "empty.jpg"]
        assert result["Rank"].tolist() == ["00", "01", "02"]
        assert result.attrs["basis"] == "box"

    def test_limit_is_respected(self):
        """Only the requested number of rows come back."""
        metrics = _build_metrics(DetMetrics)
        metrics.box.image_metrics = dict(self.IMAGE_METRICS)

        assert len(extract_worst_images(_validator(metrics), limit=2)) == 2

    def test_segmentation_ranks_on_mask_f1(self):
        """Rank by mask F1 on a seg run, since that is what mask AP measures.

        metrics.seg.image_metrics is fed by mask IoU (segment/val.py:164), so preferring
        it is what makes the ranking mask-aware rather than a box ranking mislabelled.
        """
        metrics = _build_metrics(SegmentMetrics)
        metrics.box.image_metrics = {
            "a.jpg": _im(0.1, 0.1, 0.1, 1, 9, 9),
            "b.jpg": _im(0.9, 0.9, 0.9, 9, 1, 1),
        }
        # Masks disagree with boxes: b is the bad one by mask F1.
        metrics.seg.image_metrics = {
            "a.jpg": _im(0.8, 0.8, 0.8, 8, 2, 2),
            "b.jpg": _im(0.0, 0.0, 0.0, 0, 9, 9),
        }

        result = extract_worst_images(_validator(metrics), limit=2)

        assert result.attrs["basis"] == "mask"
        assert result["Image"].tolist() == ["b.jpg", "a.jpg"]

    def test_empty_images_never_rank_worst(self):
        """Ultralytics scores a no-GT/no-pred image 1.0, so it must sort last."""
        metrics = _build_metrics(DetMetrics)
        metrics.box.image_metrics = dict(self.IMAGE_METRICS)

        result = extract_worst_images(_validator(metrics), limit=3)
        assert result["Image"].tolist()[0] != "empty.jpg"


class TestValStatsAccumulator:
    """Tests for the on_val_batch_end capture and the calibration it feeds."""

    @staticmethod
    def _append(metrics, n, conf_scale, rng, with_mask=False):
        conf = rng.random(n) * conf_scale
        metrics.stats.setdefault("conf", []).append(conf)
        metrics.stats.setdefault("tp", []).append(rng.random((n, 10)) < conf[:, None])
        if with_mask:
            metrics.stats.setdefault("tp_m", []).append(
                rng.random((n, 10)) < conf[:, None] * 0.5
            )
        return conf

    def test_captures_across_batches_without_double_counting(self):
        """Repeated update() calls consume only what is new."""
        rng = np.random.default_rng(1)
        metrics = DetMetrics()
        metrics.stats = {"conf": [], "tp": []}
        acc = ValStatsAccumulator()
        validator = _validator(metrics)

        for _ in range(3):
            self._append(metrics, 5, 1.0, rng)
            acc.update(validator)
        assert len(acc.conf) == 3

        acc.update(validator)  # nothing new appended
        assert len(acc.conf) == 3

    def test_resets_when_stats_are_cleared(self):
        """A fresh validation pass must not be appended to the previous one."""
        rng = np.random.default_rng(2)
        metrics = DetMetrics()
        metrics.stats = {"conf": [], "tp": []}
        acc = ValStatsAccumulator()
        validator = _validator(metrics)

        self._append(metrics, 5, 1.0, rng)
        acc.update(validator)
        assert len(acc.conf) == 1

        metrics.stats = {"conf": [], "tp": []}  # what clear_stats() effectively does
        acc.update(validator)
        assert acc.conf == []

    def test_confidence_split_separates_matched_from_unmatched(self):
        """TP confidences should sit higher than FP ones for a sane model."""
        rng = np.random.default_rng(3)
        metrics = DetMetrics()
        metrics.stats = {"conf": [], "tp": []}
        acc = ValStatsAccumulator()
        for _ in range(20):
            self._append(metrics, 20, 1.0, rng)
        acc.update(_validator(metrics))

        split = acc.confidence_split()
        assert split["basis"] == "box"
        assert split["tp"].size
        assert split["fp"].size
        assert split["tp"].mean() > split["fp"].mean()

    def test_prefers_mask_when_available(self):
        """On a seg run the split is mask-based unless asked otherwise."""
        rng = np.random.default_rng(4)
        metrics = SegmentMetrics()
        metrics.stats = {"conf": [], "tp": [], "tp_m": []}
        acc = ValStatsAccumulator()
        for _ in range(5):
            self._append(metrics, 10, 1.0, rng, with_mask=True)
        acc.update(_validator(metrics))

        assert acc.confidence_split(prefer_mask=True)["basis"] == "mask"
        assert acc.confidence_split(prefer_mask=False)["basis"] == "box"


class TestExtractCalibrationBins:
    """Tests for extract_calibration_bins function."""

    def test_returns_none_for_empty_split(self):
        """Nothing to bin means nothing to report."""
        assert extract_calibration_bins(None) is None
        assert extract_calibration_bins({"tp": np.array([]), "fp": np.array([])}) is None

    def test_perfect_calibration_has_zero_error(self):
        """A bin whose precision equals its confidence contributes no error.

        All detections at conf 0.95 and all matched: precision 1.0 vs mean conf 0.95, so
        ECE is the 0.05 residual and nothing more.
        """
        split = {"tp": np.full(100, 0.95), "fp": np.array([]), "basis": "box"}
        result = extract_calibration_bins(split, n_bins=10)

        assert result["ece"] == pytest.approx(0.05, abs=1e-6)

    def test_overconfident_model_scores_badly(self):
        """High confidence with half the detections wrong is poor calibration."""
        split = {"tp": np.full(50, 0.9), "fp": np.full(50, 0.9), "basis": "box"}
        result = extract_calibration_bins(split, n_bins=10)

        # precision 0.5 against mean confidence 0.9
        assert result["ece"] == pytest.approx(0.4, abs=1e-6)

    def test_bin_structure_and_counts(self):
        """Every bin is reported, empty ones as NaN, and counts sum to the input."""
        split = {"tp": np.full(10, 0.25), "fp": np.full(10, 0.75), "basis": "mask"}
        result = extract_calibration_bins(split, n_bins=10)

        assert len(result["bin_center"]) == 10
        assert sum(result["count"]) == 20
        assert result["basis"] == "mask"
        empty_precisions = [
            p for p, c in zip(result["precision"], result["count"], strict=True) if c == 0
        ]
        assert np.isnan(empty_precisions).all()


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


class TestCurveDownsampling:
    """Curve payloads must not scale with class count.

    Ultralytics computes 1000 x-points per class, per curve family, and a segmentation run
    has eight families. At 80 classes that is megabytes of JSON, which the ClearML UI
    renders as a blank panel rather than an error.
    """

    def test_curves_are_capped(self):
        """Every family is thinned to at most MAX_CURVE_POINTS."""
        from src.yolov8.metrics_utils import MAX_CURVE_POINTS

        curves = extract_curve_data(_validator(_build_metrics(SegmentMetrics)))

        for c in curves:
            assert len(c["x"]) <= MAX_CURVE_POINTS
            for s in c["series"]:
                assert len(s["y"]) == len(c["x"])

    def test_endpoints_are_preserved(self):
        """Thinning must not clip the ends of the curve."""
        metrics = _build_metrics(DetMetrics)
        curves = extract_curve_data(_validator(metrics))
        px = np.asarray(metrics.box.px, dtype=float)

        pr = next(c for c in curves if c["name"] == "Precision-Recall(B)")
        assert pr["x"][0] == pytest.approx(px[0])
        assert pr["x"][-1] == pytest.approx(px[-1])

    def test_shape_is_preserved(self):
        """A thinned curve still tracks the original within tolerance."""
        metrics = _build_metrics(DetMetrics)
        curves = extract_curve_data(_validator(metrics))
        f1 = next(c for c in curves if c["name"] == "F1-Confidence(B)")

        full = np.asarray(metrics.box.f1_curve[0], dtype=float)
        thinned = np.asarray(f1["series"][0]["y"], dtype=float)
        assert thinned.max() == pytest.approx(full.max(), abs=0.02)

    def test_short_curves_are_untouched(self):
        """Below the cap nothing is dropped."""
        from src.yolov8.metrics_utils import _downsample

        x = np.linspace(0, 1, 10)
        y = np.vstack([x, x])
        dx, dy = _downsample(x, y)
        assert dx.shape[0] == 10
        assert dy.shape == (2, 10)
