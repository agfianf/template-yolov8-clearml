"""Unit tests for src/yolov8/callbacks.py.

This module previously had no tests at all, which is how two reporting bugs shipped and
survived: PR curves that never rendered, and a confusion matrix normalized on the wrong
axis. Nothing here asserts that ClearML works -- it asserts *which* hook reports *what*,
because that routing is the part that was wrong.

Note on imports: `callbacks.py` sets its module-level `clearml` to None under pytest (it
asserts `not TESTS_RUNNING`), so the `callbacks` dict is empty during tests. The hook
functions themselves are still defined and are what these tests drive directly.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.yolov8.callbacks as cb


@pytest.fixture
def task():
    """Return a mock ClearML task whose logger records every report_* call."""
    mock_task = MagicMock()
    mock_task.get_logger.return_value = MagicMock()
    with patch.object(cb, "Task") as task_cls:
        task_cls.current_task.return_value = mock_task
        yield mock_task


@pytest.fixture
def trainer(tmp_path):
    """Return a trainer shaped like ultralytics', with current validation metrics."""
    validator = SimpleNamespace(
        metrics=SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.7,
                "metrics/recall(B)": 0.6,
                "metrics/mAP50(B)": 0.65,
                "metrics/mAP50-95(B)": 0.5,
                "fitness": 0.5,
            },
            box=SimpleNamespace(map=0.5, map50=0.65),
        ),
        speed={"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0},
        save_dir=tmp_path,
    )
    return SimpleNamespace(
        epoch=3,
        epoch_time=42.0,
        save_dir=tmp_path,
        validator=validator,
        metrics={"val/box_loss": 1.2, "val/cls_loss": 0.8},
        loss_items={"box_loss": 1.0, "cls_loss": 0.5, "dfl_loss": 0.25},
        loss_names=("box_loss", "cls_loss", "dfl_loss"),
        tloss={"box_loss": 1.0, "cls_loss": 0.5, "dfl_loss": 0.25},
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
        args=SimpleNamespace(
            data="data.yaml", model="yolo11n-seg.pt", imgsz=640, task="segment"
        ),
    )


def _scalar_titles(task):
    return {
        c[0][0] if c[0] else c[1]["title"]
        for c in task.get_logger.return_value.report_scalar.call_args_list
    }


class TestEpochLagFix:
    """The validation metrics must be reported from the hook where they are current.

    `on_train_epoch_end` fires at engine/trainer.py:569; `validate()` does not run until
    :577. Reading `trainer.validator.metrics` in that hook therefore yields the previous
    epoch's numbers -- zeros on epoch 0. `on_fit_epoch_end` (:605) is after validation.
    """

    def test_train_epoch_end_reports_no_validation_metrics(self, task, trainer):
        """Regression guard: no Metrics/* or Losses/Validation from the early hook."""
        cb.on_train_epoch_end(trainer)

        titles = _scalar_titles(task)
        assert not any(t.startswith("Metrics/") for t in titles)
        assert "Losses/Validation" not in titles

    def test_train_epoch_end_still_reports_train_side_values(self, task, trainer):
        """Train losses and LR are read from the trainer and are valid in this hook."""
        cb.on_train_epoch_end(trainer)

        titles = _scalar_titles(task)
        assert "Losses/Train" in titles
        assert "Learning Rate" in titles

    def test_fit_epoch_end_reports_validation_metrics(self, task, trainer):
        """The metrics moved here, where trainer.validator.metrics is current."""
        with patch.object(cb, "model_info_for_loggers", return_value={}):
            cb.on_fit_epoch_end(trainer)

        titles = _scalar_titles(task)
        assert {
            "Metrics/Precision",
            "Metrics/Recall",
            "Metrics/mAP",
            "Metrics/Summary",
        } <= titles
        assert "Losses/Validation" in titles

    def test_fit_epoch_end_uses_current_epoch_number(self, task, trainer):
        """Reports are filed under the epoch that was just validated."""
        with patch.object(cb, "model_info_for_loggers", return_value={}):
            cb.on_fit_epoch_end(trainer)

        iterations = {
            c[1]["iteration"]
            for c in task.get_logger.return_value.report_scalar.call_args_list
            if "iteration" in c[1]
        }
        assert iterations == {3}


class TestMetricScalarRouting:
    """Each results_dict key must land in the right titled group."""

    def test_mask_metrics_route_alongside_box(self, task, trainer):
        """A seg run's (M) keys must be grouped with their (B) counterparts."""
        trainer.validator.metrics.results_dict.update(
            {
                "metrics/precision(M)": 0.4,
                "metrics/mAP50-95(M)": 0.3,
            }
        )
        with patch.object(cb, "model_info_for_loggers", return_value={}):
            cb.on_fit_epoch_end(trainer)

        calls = task.get_logger.return_value.report_scalar.call_args_list
        routed = {c[0][1]: c[0][0] for c in calls if len(c[0]) >= 2}
        assert routed["metrics/precision(M)"] == "Metrics/Precision"
        assert routed["metrics/mAP50-95(M)"] == "Metrics/mAP"


class TestDebugSamples:
    """_log_debug_samples must not raise on an unexpected filename."""

    def test_filename_without_batch_suffix_does_not_raise(self, task, tmp_path):
        """Regression guard for an unguarded `it.group()` on a non-matching name.

        The upstream version guards the iteration but not the series, so a file lacking a
        `_batchN` suffix raised inside a callback -- which aborts training, not just the
        report.
        """
        odd = tmp_path / "val_summary.jpg"
        odd.write_bytes(b"x")

        cb._log_debug_samples([odd], "Validation")

        kwargs = task.get_logger.return_value.report_image.call_args[1]
        assert kwargs["series"] == "val_summary.jpg"
        assert kwargs["iteration"] == 0

    def test_batch_suffix_is_stripped_into_iteration(self, task, tmp_path):
        """A normal name still yields a stable series and a batch iteration."""
        f = tmp_path / "val_batch2_pred.jpg"
        f.write_bytes(b"x")

        cb._log_debug_samples([f], "Validation")

        kwargs = task.get_logger.return_value.report_image.call_args[1]
        assert kwargs["series"] == "val_pred.jpg"
        assert kwargs["iteration"] == 2

    def test_missing_file_is_skipped(self, task, tmp_path):
        """Nonexistent paths are ignored rather than raising."""
        cb._log_debug_samples([tmp_path / "nope.jpg"], "Validation")

        task.get_logger.return_value.report_image.assert_not_called()

    @pytest.mark.parametrize("epoch", [0, 1])
    def test_mosaic_logged_in_first_two_epochs(self, task, trainer, epoch):
        """Guarding on epoch == 1 alone means a 1-epoch run logs no mosaic at all."""
        (trainer.save_dir / "train_batch0.jpg").write_bytes(b"x")
        trainer.epoch = epoch

        cb.on_train_epoch_end(trainer)

        titles = {
            c[1]["title"]
            for c in task.get_logger.return_value.report_image.call_args_list
        }
        assert "Mosaic" in titles

    def test_mosaic_not_relogged_every_epoch(self, task, trainer):
        """Report volume must not grow with epoch count."""
        (trainer.save_dir / "train_batch0.jpg").write_bytes(b"x")
        trainer.epoch = 7

        cb.on_train_epoch_end(trainer)

        task.get_logger.return_value.report_image.assert_not_called()


class TestVisualizationFlags:
    """Every new report has to be switchable from the ClearML UI."""

    def test_mask_box_gap_flag_gates_the_scalar(self, task, trainer):
        """log_mask_box_gap=False suppresses the gap group."""
        trainer.validator.metrics.seg = SimpleNamespace(
            map=0.3, map50=0.4, p=np.array([0.5, 0.5])
        )

        with (
            patch.dict(cb.args_visualization, {"log_mask_box_gap": False}),
            patch.object(cb, "model_info_for_loggers", return_value={}),
        ):
            cb.on_fit_epoch_end(trainer)
        assert "Metrics/Mask-vs-Box" not in _scalar_titles(task)

        task.get_logger.return_value.report_scalar.reset_mock()
        with (
            patch.dict(cb.args_visualization, {"log_mask_box_gap": True}),
            patch.object(cb, "model_info_for_loggers", return_value={}),
        ):
            cb.on_fit_epoch_end(trainer)
        assert "Metrics/Mask-vs-Box" in _scalar_titles(task)

    def test_loss_component_flag_gates_train_losses(self, task, trainer):
        """log_loss_components=False suppresses Losses/Train."""
        with patch.dict(cb.args_visualization, {"log_loss_components": False}):
            cb.on_train_epoch_end(trainer)

        assert "Losses/Train" not in _scalar_titles(task)

    def test_learning_rate_flag_gates_lr(self, task, trainer):
        """log_learning_rate=False suppresses the LR group."""
        with patch.dict(cb.args_visualization, {"log_learning_rate": False}):
            cb.on_train_epoch_end(trainer)

        assert "Learning Rate" not in _scalar_titles(task)

    def test_calibration_flag_gates_the_val_batch_capture(self):
        """With calibration off, on_val_batch_end must not accumulate anything."""
        validator = SimpleNamespace(
            metrics=SimpleNamespace(
                stats={"conf": [np.array([0.5])], "tp": [np.ones((1, 10))]}
            )
        )
        cb.val_stats.reset()

        with patch.dict(cb.args_visualization, {"log_calibration": False}):
            cb.on_val_batch_end(validator)
        assert cb.val_stats.conf == []

        with patch.dict(cb.args_visualization, {"log_calibration": True}):
            cb.on_val_batch_end(validator)
        assert len(cb.val_stats.conf) == 1
        cb.val_stats.reset()


class TestReportValidationAnalysis:
    """The heavy report set runs once per validation, not once per epoch."""

    def test_no_task_is_a_noop(self):
        """Without a ClearML task nothing is attempted."""
        with patch.object(cb, "Task") as task_cls:
            task_cls.current_task.return_value = None
            cb.report_validation_analysis(MagicMock())  # must not raise

    def test_title_prefix_separates_splits(self, task):
        """The test split must not overwrite the validation split's plots."""
        validator = MagicMock()
        with (
            patch.object(cb, "extract_per_class_metrics") as per_class,
            patch.object(cb, "extract_curve_data", return_value=None),
            patch.object(cb, "extract_optimal_confidence", return_value=None),
            patch.object(cb, "extract_worst_images", return_value=None),
        ):
            import pandas as pd

            per_class.return_value = pd.DataFrame({"Class": ["a"], "Box-mAP50-95": [0.5]})
            cb.report_validation_analysis(validator, title_prefix="Test ")

        titles = {
            c[1]["title"]
            for c in task.get_logger.return_value.report_table.call_args_list
        }
        assert "Test Per-Class Metrics" in titles

    def test_gallery_orders_panels_by_worst_rank(self, task, tmp_path):
        """Uploaded panels follow the worst-image ranking, not directory order."""
        import pandas as pd

        vis = tmp_path / "visualizations"
        vis.mkdir()
        for name in ("aaa.jpg", "zzz.jpg"):
            (vis / name).write_bytes(b"x")

        # zzz is the worst, so it must be uploaded first despite sorting last by name.
        worst = pd.DataFrame({"Rank": ["00", "01"], "Image": ["zzz.jpg", "aaa.jpg"]})
        worst.attrs["basis"] = "mask"

        logger = cb.YOLOClearMLLogger(task)
        cb._log_error_gallery(logger, worst, tmp_path, iteration=0, title_prefix="")

        paths = [
            Path(c[1]["local_path"]).name
            for c in task.get_logger.return_value.report_image.call_args_list
        ]
        assert paths == ["zzz.jpg", "aaa.jpg"]

    def test_gallery_skipped_when_visualizations_absent(self, task, tmp_path):
        """Without visualize=True there is no directory, and that is not an error."""
        import pandas as pd

        worst = pd.DataFrame({"Rank": ["00"], "Image": ["a.jpg"]})
        logger = cb.YOLOClearMLLogger(task)

        cb._log_error_gallery(logger, worst, tmp_path, iteration=0, title_prefix="")

        task.get_logger.return_value.report_image.assert_not_called()


class TestConfusionMatrixConfPinning:
    """The matrix's threshold must be independent of the NMS threshold.

    Ultralytics passes one `args.conf` to both (detect/val.py:195), but metrics need a
    floor near zero to keep the PR curve's high-recall tail while a readable matrix needs
    ~0.25. args_val["conf"] serves the metrics; this pinning serves the matrix.
    """

    @staticmethod
    def _validator():
        matrix = MagicMock()
        matrix._conf_pinned = False
        return SimpleNamespace(confusion_matrix=matrix)

    def test_forces_display_threshold_over_caller_value(self):
        """Whatever conf update_metrics passes, the matrix uses the display value."""
        validator = self._validator()
        original = validator.confusion_matrix.process_batch

        cb.on_val_batch_start(validator)
        # This is what detect/val.py does: passes args.conf, which is now 0.001.
        validator.confusion_matrix.process_batch({}, {}, conf=0.001)

        assert original.call_args[1]["conf"] == 0.25

    def test_threshold_is_configurable(self):
        """The pinned value comes from args_visualization, not a constant."""
        validator = self._validator()
        original = validator.confusion_matrix.process_batch

        with patch.dict(cb.args_visualization, {"confusion_matrix_conf": 0.4}):
            cb.on_val_batch_start(validator)
        validator.confusion_matrix.process_batch({}, {}, conf=0.001)

        assert original.call_args[1]["conf"] == 0.4

    def test_idempotent_across_batches(self):
        """Fires once per batch, so it must not wrap the wrapper repeatedly."""
        validator = self._validator()

        cb.on_val_batch_start(validator)
        wrapped_once = validator.confusion_matrix.process_batch
        for _ in range(5):
            cb.on_val_batch_start(validator)

        assert validator.confusion_matrix.process_batch is wrapped_once

    def test_no_matrix_is_a_noop(self):
        """With plots=False there is no matrix, and that is not an error."""
        cb.on_val_batch_start(SimpleNamespace(confusion_matrix=None))
        cb.on_val_batch_start(SimpleNamespace())  # must not raise

    def test_positional_args_are_preserved(self):
        """The wrapper must pass detections and batch through untouched."""
        validator = self._validator()
        original = validator.confusion_matrix.process_batch

        cb.on_val_batch_start(validator)
        detections, batch = {"cls": [1]}, {"cls": [1]}
        validator.confusion_matrix.process_batch(detections, batch)

        assert original.call_args[0] == (detections, batch)


class TestReportVolume:
    """Report volume must not grow with epoch count.

    The sibling of the project's log-volume rule. Scalars are series and are *meant* to
    grow one point per epoch, but plots, tables and images are not: a 300-epoch run must
    not upload 300 confusion matrices. This runs the per-epoch hooks at two epoch counts
    and asserts the heavy report count is identical, exactly as
    tests/data/test_data_stage_smoke.py does for log lines at two dataset sizes.
    """

    @staticmethod
    def _heavy_report_count(task):
        log = task.get_logger.return_value
        return (
            log.report_plotly.call_count
            + log.report_table.call_count
            + log.report_confusion_matrix.call_count
            + log.report_matplotlib_figure.call_count
        )

    def _run_epochs(self, task, trainer, n_epochs):
        with patch.object(cb, "model_info_for_loggers", return_value={}):
            for epoch in range(n_epochs):
                trainer.epoch = epoch
                cb.on_train_epoch_end(trainer)
                cb.on_fit_epoch_end(trainer)
        return self._heavy_report_count(task)

    def test_heavy_reports_do_not_scale_with_epochs(self, task, trainer):
        """Three epochs and thirty epochs must emit the same number of plots/tables."""
        short = self._run_epochs(task, trainer, 3)

        task.get_logger.return_value.reset_mock()
        long = self._run_epochs(task, trainer, 30)

        assert short == long == 0

    def test_scalars_do_grow_with_epochs(self, task, trainer):
        """Confirm scalars do scale, unlike the heavy reports.

        The control for the test above, which would otherwise pass simply because nothing
        was reported at all.
        """
        with patch.object(cb, "model_info_for_loggers", return_value={}):
            trainer.epoch = 0
            cb.on_fit_epoch_end(trainer)
            after_one = task.get_logger.return_value.report_scalar.call_count

            trainer.epoch = 1
            cb.on_fit_epoch_end(trainer)
            after_two = task.get_logger.return_value.report_scalar.call_count

        assert after_two > after_one

    def test_image_gallery_is_bounded_by_the_limit(self, task, tmp_path):
        """The worst-image gallery is capped, not "every image with a panel"."""
        import pandas as pd

        vis = tmp_path / "visualizations"
        vis.mkdir()
        names = [f"img{i:03d}.jpg" for i in range(50)]
        for name in names:
            (vis / name).write_bytes(b"x")

        # extract_worst_images already applied the limit; the gallery must not re-scan
        # the directory and upload everything it finds there.
        worst = pd.DataFrame({"Rank": [f"{i:02d}" for i in range(4)], "Image": names[:4]})
        logger = cb.YOLOClearMLLogger(task)
        cb._log_error_gallery(logger, worst, tmp_path, iteration=0, title_prefix="")

        assert task.get_logger.return_value.report_image.call_count == 4


class TestRegistrationMetrics:
    """Model registration must carry the quality numbers it always accepted."""

    def test_detection_metrics(self):
        """A detect run tags box mAP only."""
        validator = SimpleNamespace(
            metrics=SimpleNamespace(box=SimpleNamespace(map=0.5, map50=0.65))
        )
        assert cb._registration_metrics(validator) == {"map50": 0.65, "map": 0.5}

    def test_segmentation_adds_mask_metrics(self):
        """A seg run additionally tags mask mAP, so models can be ranked on it."""
        validator = SimpleNamespace(
            metrics=SimpleNamespace(
                box=SimpleNamespace(map=0.5, map50=0.65),
                seg=SimpleNamespace(map=0.3, map50=0.4, p=np.array([0.5, 0.5])),
            )
        )
        result = cb._registration_metrics(validator)

        assert result == {
            "map50": 0.65,
            "map": 0.5,
            "mask_map50": 0.4,
            "mask_map": 0.3,
        }

    def test_missing_metrics_yields_empty_dict(self):
        """No metrics is survivable; registration proceeds without tags."""
        assert cb._registration_metrics(SimpleNamespace(metrics=None)) == {}


class TestLossRatios:
    """Loss ratios make runs with different loss gains comparable."""

    def test_ratios_are_relative_to_box_loss(self, task):
        """Each non-box component is reported as a multiple of box loss."""
        logger = cb.YOLOClearMLLogger(task)
        cb._report_loss_ratios(
            logger, {"box_loss": 2.0, "cls_loss": 1.0, "seg_loss": 4.0}, iteration=1
        )

        calls = task.get_logger.return_value.report_scalar.call_args_list
        reported = {c[1]["series"]: c[1]["value"] for c in calls}
        assert reported["cls_loss_over_box"] == pytest.approx(0.5)
        assert reported["seg_loss_over_box"] == pytest.approx(2.0)

    def test_zero_box_loss_is_skipped(self, task):
        """No division by zero when box loss has not been populated yet."""
        logger = cb.YOLOClearMLLogger(task)
        cb._report_loss_ratios(logger, {"box_loss": 0.0, "cls_loss": 1.0}, iteration=1)

        task.get_logger.return_value.report_scalar.assert_not_called()

    def test_total_loss_is_not_ratioed(self, task):
        """total_loss against box_loss is not a meaningful balance figure."""
        logger = cb.YOLOClearMLLogger(task)
        cb._report_loss_ratios(logger, {"box_loss": 2.0, "total_loss": 6.0}, iteration=1)

        task.get_logger.return_value.report_scalar.assert_not_called()


class TestValidatorRef:
    """A standalone `val()` needs the validator, and ultralytics does not expose it.

    `Model.val()` constructs its validator locally and returns only `validator.metrics`
    (engine/model.py:625-627). There is no `model.validator`; that attribute resolves
    through `Model.__getattr__` to the underlying nn.Module and raises AttributeError.
    src/train.py briefly relied on it, which -- inside its try/except -- would have
    silently skipped all final-split reporting in production.
    """

    @pytest.mark.usefixtures("task")
    def test_on_val_end_captures_the_validator(self, tmp_path):
        """The hook is handed the validator, so it is the reliable capture point."""
        cb.validator_ref.current = None
        validator = SimpleNamespace(save_dir=tmp_path)

        cb.on_val_end(validator)

        assert cb.validator_ref.current is validator

    def test_captured_even_without_a_clearml_task(self, tmp_path):
        """Capture must not depend on reporting being active."""
        cb.validator_ref.current = None
        validator = SimpleNamespace(save_dir=tmp_path)

        with patch.object(cb, "Task") as task_cls:
            task_cls.current_task.return_value = None
            cb.on_val_end(validator)

        assert cb.validator_ref.current is validator

    @pytest.mark.usefixtures("task")
    def test_latest_validator_wins(self, tmp_path):
        """Training validates every epoch; the standalone val() must be the one kept."""
        first = SimpleNamespace(save_dir=tmp_path)
        last = SimpleNamespace(save_dir=tmp_path)

        cb.on_val_end(first)
        cb.on_val_end(last)

        assert cb.validator_ref.current is last

    def test_ultralytics_model_really_has_no_validator_attribute(self):
        """Pin the upstream fact this works around, against the real class."""
        from ultralytics.engine.model import Model

        assert not hasattr(Model, "validator")
        # ...and it is not assigned during val() either.
        import inspect

        assert "self.validator" not in inspect.getsource(Model.val)


class TestStaticPlotUploads:
    """Static PNGs are only worth uploading when nothing interactive replaces them."""

    ALL_ULTRALYTICS_PLOTS = [
        "results.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "MaskPR_curve.png",
    ]

    def _run(self, trainer):
        for name in self.ALL_ULTRALYTICS_PLOTS:
            (trainer.save_dir / name).write_bytes(b"x")
        with (
            patch.object(cb, "_log_plot") as log_plot,
            patch.object(cb, "register_model_to_clearml"),
            patch.object(cb, "report_validation_analysis"),
            patch.object(cb, "yaml_loader", return_value={"names": ["a"]}),
            patch.object(
                cb, "extract_confusion_matrix_data", return_value=(None, [], None)
            ),
        ):
            trainer.best = trainer.save_dir / "best.pt"
            trainer.last = trainer.save_dir / "last.pt"
            cb.on_train_end(trainer)
        return [c[1]["title"] for c in log_plot.call_args_list]

    @pytest.mark.usefixtures("task")
    def test_duplicated_plots_are_not_uploaded(self, trainer):
        """Curves and confusion matrices already exist as interactive plots.

        Uploading them again produced a second, non-interactive copy of the same data --
        and those copies are exactly the panels that render blank when the ClearML
        fileserver cannot serve images, because report_matplotlib_figure uploads a PNG and
        references it by URL.
        """
        titles = self._run(trainer)

        assert not [t for t in titles if "curve" in t]
        assert not [t for t in titles if "confusion_matrix" in t]

    @pytest.mark.usefixtures("task")
    def test_unduplicated_plots_are_still_uploaded(self, trainer):
        """results/labels have no interactive equivalent, so they stay."""
        titles = self._run(trainer)

        assert set(titles) == {"results", "labels", "labels_correlogram"}

    @pytest.mark.usefixtures("task")
    def test_flag_disables_static_uploads_entirely(self, trainer):
        """One switch for deployments whose fileserver cannot serve images."""
        with patch.dict(cb.args_visualization, {"log_static_plots": False}):
            titles = self._run(trainer)

        assert titles == []
