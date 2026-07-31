"""The report has to build from a genuine ultralytics metrics object.

`_build_metrics` is imported from `tests/yolov8/test_metrics_utils.py` rather than
copied, so this suite is pinned to the same real `DetMetrics` / `SegmentMetrics`
behaviour the metrics tests are -- including its `NAMES` fixture, whose class 2 has no
ground truth. That absent class is the `ap_class_index` trap: ultralytics drops it from
every per-class array, so anything that slices a plain names list against `box.p`
mislabels every row after it.
"""

import pytest

from ultralytics.utils.metrics import DetMetrics, SegmentMetrics

from src.report.blob import build_blob
from src.report.build import build_preview, publish_report
from src.report.render import render_report

from .conftest import DEFAULT_CFG, extract_blob, make_context
from .conftest import SUITE_NAMES as NAMES
from .conftest import suite_metrics as _build_metrics


@pytest.fixture
def real_det(tmp_path):
    """Build a context whose metrics are the metrics suite's own, absent class and all."""
    ctx = make_context(tmp_path, n_classes=3)
    metrics = _build_metrics(DetMetrics)
    ctx.final_metrics = metrics
    ctx.validator.metrics = metrics
    ctx.class_2_idx = {name: idx for idx, name in NAMES.items()}
    return ctx


@pytest.fixture
def real_seg(tmp_path):
    ctx = make_context(tmp_path, n_classes=3, seg=True)
    metrics = _build_metrics(SegmentMetrics)
    ctx.final_metrics = metrics
    ctx.validator.metrics = metrics
    ctx.class_2_idx = {name: idx for idx, name in NAMES.items()}
    return ctx


class TestBuildsFromRealMetrics:
    """A full page, from a processed metrics object, on both task types."""

    def test_full_html_builds_from_det_metrics(self, real_det, tmp_path):
        path = publish_report(
            build_blob(real_det), cfg=DEFAULT_CFG, output_dir=tmp_path / "det"
        )
        document = (tmp_path / "det" / "evaluation_report.html").read_text()

        assert path.endswith("evaluation_report.html")
        assert extract_blob(document)["schema"] == 1
        assert 'id="s-per-class"' in document
        # No mask sections on a detect run; the whole section is omitted, not emptied.
        assert 'id="s-boxmask"' not in document

    def test_full_html_builds_from_segment_metrics(self, real_seg, tmp_path):
        publish_report(build_blob(real_seg), cfg=DEFAULT_CFG, output_dir=tmp_path / "seg")
        document = (tmp_path / "seg" / "evaluation_report.html").read_text()

        assert 'id="s-boxmask"' in document
        blob = extract_blob(document)
        assert blob["meta"]["is_seg"] is True
        assert "Mask AP50-95" in blob["tables"]["t_per_class"]["columns"]

    def test_absent_class_is_not_mislabelled(self, real_det):
        """The `ap_class_index` trap: an absent class must vanish, not shift names."""
        blob = build_blob(real_det)
        rows = blob["tables"]["t_per_class"]["rows"]

        assert [r[0] for r in rows] == ["ripe", "unripe"]
        assert "absent_class" not in [r[0] for r in rows]
        # ...and it is reported as absent rather than silently dropped.
        zero = [w for w in blob["warnings"] if "no ground truth" in w["text"]]
        assert zero
        assert "absent_class" in zero[0]["text"]
        assert zero[0]["severity"] == "serious"

    def test_fitness_tile_has_no_meter_on_segment(self, real_seg):
        """`SegmentMetrics.fitness` is box + mask mAP50-95, i.e. 0-2, so no meter."""
        blob = build_blob(real_seg)
        fitness = next(k for k in blob["kpis"] if k["label"] == "Fitness")

        assert fitness["scale"] is None
        assert "0-2" in fitness["basis"]

    def test_fitness_tile_keeps_its_meter_on_detect(self, real_det):
        """The control: on a detect run fitness *is* mAP50-95 and 0-1 is honest."""
        blob = build_blob(real_det)
        fitness = next(k for k in blob["kpis"] if k["label"] == "Fitness")

        assert fitness["scale"] == [0.0, 1.0]

    def test_every_kpi_states_its_basis(self, real_seg):
        """Non-negotiable: a metric with no stated confidence and IoU is a rumour."""
        for kpi in build_blob(real_seg)["kpis"]:
            assert kpi["basis"], f"{kpi['label']} has no basis"

    def test_map_tiles_state_the_nms_floor(self, real_det):
        """MAP is computed at conf=0.001 and the tile has to say so."""
        blob = build_blob(real_det)
        tile = next(k for k in blob["kpis"] if k["label"] == "Box mAP50-95")

        assert "0.001" in tile["basis"]

    def test_preview_is_plain_text_and_bounded(self, real_seg):
        """The artifact preview has to be readable without opening the file."""
        preview = build_preview(build_blob(real_seg))

        assert "<" not in preview.split("Open the artifact")[0][:200]
        assert len(preview.encode()) < 60_000
        assert "Open the artifact URL in a new tab" in preview

    def test_thresholds_are_all_three_and_distinct(self, real_det):
        """The report must never present one confidence as if it were the only one."""
        thresholds = build_blob(real_det)["meta"]["thresholds"]

        assert thresholds["nms_floor"] == pytest.approx(0.001)
        assert thresholds["matrix_display"] == pytest.approx(0.25)
        assert thresholds["f1_optimal"] is not None

    def test_split_settings_come_from_the_validator_not_args_val(self, real_det):
        """train.py overwrites batch and split after the connect, so `args_val` lies."""
        real_det.args_val = {"batch": 16, "split": "val", "conf": 0.25}
        blob = build_blob(real_det)

        assert blob["meta"]["val_args"]["batch"] == 32
        assert blob["meta"]["split_name"] == "test"
        assert blob["meta"]["val_args"]["conf"] == pytest.approx(0.001)

    def test_report_renders_without_a_clearml_task(self, real_det):
        """Nothing here depends on ClearML; the page is built either way."""
        document = render_report(build_blob(real_det))

        assert document.startswith("<!doctype html>")
        assert "report-data" in document
