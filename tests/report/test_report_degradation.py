"""Every input the report reads is allowed to be missing.

This runs at the end of a GPU run, after the model is already trained. A report that
raises because `results.csv` was not written, or because the capture wrapper hit an
ultralytics rename, costs the run its export stage for nothing. So each missing input
degrades to a card that names what is absent and why, the artifact still uploads, and
the degradation is listed in the caveats footer.
"""

import logging

from types import SimpleNamespace

import numpy as np
import pytest

from src.report.blob import build_blob
from src.report.build import build_evaluation_report, publish_report
from src.report.capture import MAX_CAPTURE_ERRORS, ReportCapture
from src.report.render import render_report

from .conftest import DEFAULT_CFG, make_capture, make_context, markup_only


def _document(ctx, tmp_path, name="out"):
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    blob = build_blob(ctx)
    publish_report(blob, cfg=DEFAULT_CFG, output_dir=out)
    return blob, (out / "evaluation_report.html").read_text()


class TestMissingInputs:
    """One missing input at a time; the page still builds and says what is gone."""

    def test_builds_with_no_capture(self, tmp_path, task):
        """No per-image wrapper: no TIDE, no strata, no galleries, still a report."""
        ctx = make_context(tmp_path, capture=None)
        blob = build_blob(ctx)
        publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=tmp_path / "o")

        assert task.upload_artifact.call_count == 1
        assert blob["tide_mode"] == "none"
        assert not any(g["items"] for g in blob["grids"].values())
        document = render_report(blob)
        assert "Not captured" in document
        assert blob["degradations"]

    def test_builds_with_empty_val_stats(self, tmp_path):
        """`log_calibration=False` removes the reliability pair, nothing else."""
        ctx = make_context(tmp_path, with_val_stats=False)
        blob, document = _document(ctx, tmp_path, "cal")

        assert "f_reliability" not in blob["figures"]
        assert "f_tp_fp_hist" not in blob["figures"]
        assert "log_calibration" in document
        assert any("log_calibration" in d for d in blob["degradations"])
        # ...and the rest of the report is intact.
        assert "t_per_class" in blob["tables"]

    def test_builds_with_no_confusion_matrix(self, tmp_path):
        """`plots=False` means no matrix, which the section says out loud."""
        ctx = make_context(tmp_path, with_matrix=False)
        blob, document = _document(ctx, tmp_path, "cm")

        assert "f_confusion" not in blob["figures"]
        assert "plots=True" in document
        assert any("plots" in d for d in blob["degradations"])

    def test_builds_with_missing_results_csv(self, tmp_path):
        """No trainer, no training history; the appendix says so."""
        ctx = make_context(tmp_path, with_trainer=False)
        blob, document = _document(ctx, tmp_path, "csv")

        assert "f_val_map" not in blob["figures"]
        assert "results.csv" in document

    def test_builds_with_no_dataset_directory(self, tmp_path):
        """A cleaned-up dataset directory costs the split section and nothing else."""
        ctx = make_context(tmp_path, with_dataset=False)
        blob, _document_html = _document(ctx, tmp_path, "ds")

        assert "t_split_composition" not in blob["tables"]
        assert "f_split_stack" not in blob["figures"]
        assert any("label directories" in d for d in blob["degradations"])
        assert "t_per_class" in blob["tables"]

    def test_builds_with_unreadable_images(self, tmp_path):
        """Deleted images cost thumbnails, and each grid says why it is empty."""
        ctx = make_context(tmp_path)
        for path in (tmp_path / "images").glob("*.jpg"):
            path.unlink()
        blob, document = _document(ctx, tmp_path, "img")

        assert blob["thumbs"] == {}
        for grid in blob["grids"].values():
            assert grid["items"] == []
            assert grid["empty_reason"]
        assert "<html" not in document.split("<body>")[1]  # still one valid document

    def test_builds_with_nothing_at_all(self, tmp_path, task):
        """The floor: no validator, no metrics, no capture, no dataset."""
        ctx = make_context(
            tmp_path,
            capture=None,
            with_val_stats=False,
            with_dataset=False,
            with_trainer=False,
        )
        ctx.validator = None
        ctx.final_metrics = None
        blob = build_blob(ctx)
        publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=tmp_path / "bare")

        assert task.upload_artifact.call_count == 1
        assert render_report(blob).startswith("<!doctype html>")

    def test_report_can_be_switched_off(self, tmp_path, task):
        """`html_report=False` uploads nothing and raises nothing."""
        result = build_evaluation_report(
            task=task,
            args_visualization={**DEFAULT_CFG, "html_report": False},
            output_dir=tmp_path,
        )

        assert result is None
        assert task.upload_artifact.call_count == 0

    def test_caveats_are_never_empty(self, tmp_path):
        """The footer always ends with the timestamp and the schema version."""
        blob = build_blob(make_context(tmp_path, capture=None))

        assert blob["caveats"]
        assert "schema 1" in blob["caveats"][-1]


class TestCaptureNeverBreaksValidation:
    """The wrapper is on the hot path of every validated image. It cannot raise."""

    def test_capture_wrapper_never_raises_into_validation(self, caplog):
        """A `note()` that raises on every call must not change what ultralytics sees."""
        capture = ReportCapture()
        sentinel = {"tp": np.ones((2, 10), dtype=bool)}
        validator = SimpleNamespace(_process_batch=lambda _preds, _batch: sentinel)
        capture.install(validator)

        def boom(*_args, **_kwargs):
            raise RuntimeError("capture is broken")

        capture._note = boom
        with caplog.at_level(logging.WARNING):
            for _ in range(MAX_CAPTURE_ERRORS + 5):
                assert validator._process_batch({}, {}) is sentinel

        # It self-disables rather than paying for a doomed try/except per image ...
        assert capture.available is False
        assert capture.installed is False
        # ... and it never logged from inside the per-image loop.
        assert caplog.records == []
        assert capture.errors.total() >= MAX_CAPTURE_ERRORS

    def test_install_is_idempotent(self):
        """`on_val_batch_start` fires once per batch, so this runs hundreds of times."""
        capture = ReportCapture()
        validator = SimpleNamespace(_process_batch=lambda _preds, _batch: {"tp": None})
        capture.install(validator)
        wrapped = validator._process_batch
        for _ in range(5):
            capture.install(validator)

        assert validator._process_batch is wrapped

    def test_install_on_a_validator_without_the_method_is_a_noop(self):
        """An ultralytics rename degrades to "not captured", not to a crash."""
        capture = ReportCapture()

        assert capture.install(SimpleNamespace()) is False
        assert capture.available is False

    def test_note_survives_ragged_input(self):
        """Shapes that do not line up are tallied and dropped, never raised."""
        capture = ReportCapture()
        capture.configure(DEFAULT_CFG)
        capture.note(
            {"bboxes": np.zeros((3, 4)), "conf": np.zeros(2), "cls": np.zeros(5)},
            {"bboxes": None, "cls": None, "ori_shape": (10, 10)},
            {"tp": np.zeros((0, 10))},
        )

        assert capture.n_images >= 0  # the point is that it returned at all

    def test_new_pass_clears_data_but_keeps_the_wrapper(self, tmp_path):
        """Ultralytics reuses `trainer.validator`, so epoch 2 must not pool with 1."""
        capture = make_capture(tmp_path / "im", n_images=20)
        validator = SimpleNamespace(_process_batch=lambda _preds, _batch: {"tp": None})
        capture.install(validator)
        wrapped = validator._process_batch
        assert capture.n_images == 20

        capture.new_pass()

        assert capture.n_images == 0
        assert capture.installed is True
        assert validator._process_batch is wrapped

    def test_reset_drops_everything(self, tmp_path):
        capture = make_capture(tmp_path / "im2", n_images=20)
        capture.reset()

        assert capture.n_images == 0
        assert capture.records == []
        assert all(len(v) == 0 for v in capture.grids.values())


class TestTideDegradation:
    """The decomposition has two honest modes and never a silent third."""

    def test_counts_mode_when_oracles_are_disabled(self, tmp_path):
        blob = build_blob(make_context(tmp_path, cfg={"report_tide": False}))

        assert blob["tide_mode"] == "counts"
        rows = blob["tables"]["t_tide"]["rows"]
        assert all(row[2] is None for row in rows)
        assert "not computed" in render_report(blob)

    def test_counts_table_is_shown_in_both_modes(self, tmp_path):
        """The section keeps its shape, so a reader always sees raw counts."""
        with_oracles = build_blob(make_context(tmp_path / "a", cfg={"report_tide": True}))
        without = build_blob(make_context(tmp_path / "b", cfg={"report_tide": False}))

        assert (
            with_oracles["tables"]["t_tide"]["columns"]
            == (without["tables"]["t_tide"]["columns"])
        )

    def test_tide_note_states_the_subsample_and_the_box_iou_caveat(self, tmp_path):
        note = build_blob(make_context(tmp_path))["tables"]["t_tide"]["note"]

        assert "box IoU only" in note
        assert "independent oracles" in note
        assert "images" in note

    def test_reservoir_bounds_memory(self, tmp_path):
        """The sample is capped, and the report says how many of how many it saw."""
        capture = make_capture(
            tmp_path / "res", n_images=400, cfg={"report_tide_max_images": 50}
        )

        assert len(capture.records) == 50
        assert any("sample of" in d for d in capture.degradations())


class TestUploadFlow:
    """One artifact, one media link, and never a directory."""

    def test_artifact_is_a_file_path_not_a_directory(self, tmp_path, task):
        """A folder artifact becomes a zip, which no browser will render."""
        blob = build_blob(make_context(tmp_path))
        publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=tmp_path / "up")

        path = task.upload_artifact.call_args[1]["artifact_object"]
        assert path.endswith(".html")
        assert isinstance(path, str)

    def test_upload_waits_so_the_url_is_readable(self, tmp_path, task):
        blob = build_blob(make_context(tmp_path))
        publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=tmp_path / "up2")

        assert task.upload_artifact.call_args[1]["wait_on_upload"] is True

    def test_media_link_uses_the_artifact_url(self, tmp_path, task):
        blob = build_blob(make_context(tmp_path))
        publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=tmp_path / "up3")

        kwargs = task.get_logger.return_value.report_media.call_args[1]
        assert kwargs["url"] == "https://files.example/report.html"
        assert kwargs["iteration"] == 0

    def test_split_produces_two_artifacts_with_the_child_first(self, tmp_path, task):
        """Above the split threshold: galleries out, uploaded first, linked absolutely."""
        blob = build_blob(make_context(tmp_path, n_images=200))
        publish_report(
            blob,
            task=task,
            cfg={**DEFAULT_CFG, "report_split_bytes": 1},
            output_dir=tmp_path / "sp",
        )

        names = [c[0][0] for c in task.upload_artifact.call_args_list]
        assert names == ["evaluation_report_galleries", "evaluation_report"]
        # Only the index is linked under Debug Samples.
        assert task.get_logger.return_value.report_media.call_count == 1
        index = (tmp_path / "sp" / "evaluation_report.html").read_text()
        assert "https://files.example/galleries.html" in index

    def test_hard_ceiling_drops_galleries_rather_than_the_report(self, tmp_path, task):
        """A degraded report at the end of a GPU run beats no report."""
        blob = build_blob(make_context(tmp_path, n_images=200))
        publish_report(
            blob,
            task=task,
            cfg={**DEFAULT_CFG, "report_split_bytes": 10**9, "report_max_bytes": 1},
            output_dir=tmp_path / "hc",
        )
        document = markup_only((tmp_path / "hc" / "evaluation_report.html").read_text())

        assert task.upload_artifact.call_count == 1
        assert "data-thumb" not in document

    def test_upload_failure_returns_none_and_does_not_raise(self, tmp_path, task):
        task.upload_artifact.side_effect = RuntimeError("fileserver is down")
        blob = build_blob(make_context(tmp_path))

        assert (
            publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=tmp_path / "f")
            is None
        )

    def test_metadata_values_are_all_strings(self, tmp_path, task):
        """ClearML artifact metadata is a string map; an int silently drops the key."""
        blob = build_blob(make_context(tmp_path))
        publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=tmp_path / "md")

        metadata = task.upload_artifact.call_args[1]["metadata"]
        assert metadata
        assert all(isinstance(v, str) for v in metadata.values())
        assert metadata["parts"] == "1"


@pytest.mark.parametrize("seg", [False, True])
def test_end_to_end_through_the_public_entry_point(tmp_path, task, seg):
    """`build_evaluation_report` is what train.py calls; drive that, not the internals."""
    ctx = make_context(tmp_path, seg=seg, task=task)
    url = build_evaluation_report(
        task=task,
        validator=ctx.validator,
        final_metrics=ctx.final_metrics,
        trainer=ctx.trainer,
        val_stats=ctx.val_stats,
        capture=ctx.capture,
        dataset_dir=ctx.dataset_dir,
        class_2_idx=ctx.class_2_idx,
        task_yolo=ctx.task_yolo,
        split_label="Test ",
        args_train=ctx.args_train,
        args_val=ctx.args_val,
        args_visualization=ctx.args_visualization,
        output_dir=tmp_path / "e2e",
    )

    assert url == "https://files.example/report.html"
    assert task.upload_artifact.call_count == 1
