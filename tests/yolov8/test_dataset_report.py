"""Unit tests for src/yolov8/dataset_report.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.yolov8 import dataset_report as dr


@pytest.fixture
def stats():
    """Return a fresh accumulator, so tests never inherit module-level state."""
    return dr.DatasetStats()


def _filter_report(total, kept, dropped_area=0, dropped_class=None, dropped_attr=None):
    """A FilterReport-shaped object; the real one is a dataclass with Tally fields."""
    return SimpleNamespace(
        total=total,
        kept=kept,
        dropped_area=dropped_area,
        dropped_class=dropped_class or {},
        dropped_attr=dropped_attr or {},
    )


class TestDatasetStats:
    """Accumulation must be tally-only and stable across many sources."""

    def test_starts_empty(self, stats):
        """Nothing accumulated means nothing to report."""
        assert stats.is_empty

    def test_note_annotation_tallies_instances_and_geometry(self, stats):
        """Instances count up; area and aspect ratio are derived from the box."""
        stats.note_annotation("ripe", 0.5, 0.25)
        stats.note_annotation("ripe", 0.2, 0.2)

        assert stats.instances["ripe"] == 2
        assert stats.areas == [pytest.approx(0.125), pytest.approx(0.04)]
        assert stats.aspects == [pytest.approx(2.0), pytest.approx(1.0)]

    def test_degenerate_boxes_are_not_measured(self, stats):
        """A zero-width box still counts as an instance but has no geometry."""
        stats.note_annotation("ripe", 0.0, 0.5)

        assert stats.instances["ripe"] == 1
        assert stats.areas == []

    def test_note_image_counts_images_per_class(self, stats):
        """Images-per-class is what distinguishes 'rare' from 'clustered'."""
        stats.note_image({"ripe", "unripe"})
        stats.note_image({"ripe"})

        assert stats.image_count == 2
        assert stats.images_with_class["ripe"] == 2
        assert stats.images_with_class["unripe"] == 1

    def test_filter_reports_accumulate_across_sources(self, stats):
        """A run converts several CVAT tasks; their filtering totals must add up."""
        stats.note_filter_report(
            _filter_report(100, 80, dropped_area=5, dropped_class={"stalk": 15})
        )
        stats.note_filter_report(
            _filter_report(50, 45, dropped_area=2, dropped_class={"stalk": 3})
        )

        assert stats.filter_total == 150
        assert stats.filter_kept == 125
        assert stats.dropped_area == 7
        assert stats.dropped_class["stalk"] == 18

    def test_reset_clears_everything(self, stats):
        """A second data stage must not stack onto the first."""
        stats.note_annotation("ripe", 0.5, 0.5)
        stats.note_filter_report(_filter_report(10, 10))
        stats.reset()

        assert stats.is_empty
        assert stats.areas == []
        assert stats.image_count == 0

    def test_class_table_is_rarest_first(self, stats):
        """Sorted ascending, because the rare classes are the ones to look at."""
        for _ in range(50):
            stats.note_annotation("common", 0.2, 0.2)
        for _ in range(3):
            stats.note_annotation("rare", 0.2, 0.2)
        stats.note_image({"common", "rare"})

        df = stats.class_table()

        assert df["Class"].tolist() == ["rare", "common"]
        assert df["Instances"].tolist() == [3, 50]
        assert df["Share of instances"].tolist() == [
            pytest.approx(3 / 53, abs=1e-4),
            pytest.approx(50 / 53, abs=1e-4),
        ]

    def test_class_table_none_when_no_instances(self, stats):
        """No annotations means no table rather than an empty frame."""
        assert stats.class_table() is None

    def test_filter_table_lists_every_reason(self, stats):
        """Kept and each drop reason appear as their own row."""
        stats.note_filter_report(
            _filter_report(
                100,
                70,
                dropped_area=10,
                dropped_class={"stalk": 15},
                dropped_attr={"maturity": 5},
            )
        )

        reasons = stats.filter_table()["Reason"].tolist()

        assert "kept" in reasons
        assert any("below min area" in r for r in reasons)
        assert any("stalk" in r for r in reasons)
        assert any("maturity" in r for r in reasons)


class TestReportDatasetComposition:
    """The report is emitted once, and only when there is something to say."""

    def test_no_task_returns_false(self, stats):
        """Without a ClearML task the call is a safe no-op."""
        stats.note_annotation("ripe", 0.5, 0.5)

        with patch("clearml.Task.current_task", return_value=None):
            assert dr.report_dataset_composition(stats) is False

    def test_empty_stats_returns_false_without_touching_clearml(self, stats):
        """Nothing accumulated short-circuits before any ClearML call."""
        with patch("clearml.Task.current_task") as current:
            assert dr.report_dataset_composition(stats) is False
            current.assert_not_called()

    def test_reports_tables_charts_and_summary(self, stats):
        """One table per topic, two figures, and the summary scalars."""
        for _ in range(20):
            stats.note_annotation("common", 0.3, 0.3)
        stats.note_annotation("rare", 0.1, 0.2)
        stats.note_image({"common", "rare"})
        stats.note_filter_report(_filter_report(30, 21, dropped_class={"stalk": 9}))

        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        with patch("clearml.Task.current_task", return_value=mock_task):
            assert dr.report_dataset_composition(stats) is True

        table_series = {c[1]["series"] for c in mock_logger.report_table.call_args_list}
        assert table_series == {"Class Distribution", "Annotation Filtering"}

        plot_series = {c[1]["series"] for c in mock_logger.report_plotly.call_args_list}
        assert plot_series == {"Class Imbalance", "Object Geometry"}

        scalars = {c[1]["series"] for c in mock_logger.report_scalar.call_args_list}
        assert "imbalance_ratio_max_over_min" in scalars
        assert "annotations_dropped" in scalars

    def test_imbalance_ratio_is_max_over_min(self, stats):
        """A single number for how lopsided the dataset is."""
        for _ in range(40):
            stats.note_annotation("common", 0.3, 0.3)
        for _ in range(4):
            stats.note_annotation("rare", 0.3, 0.3)
        stats.note_image({"common", "rare"})

        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        with patch("clearml.Task.current_task", return_value=mock_task):
            dr.report_dataset_composition(stats)

        reported = {
            c[1]["series"]: c[1]["value"]
            for c in mock_logger.report_scalar.call_args_list
        }
        assert reported["imbalance_ratio_max_over_min"] == pytest.approx(10.0)

    def test_chart_is_capped_but_table_is_not(self, stats):
        """Above the cap the chart truncates and says so; the table keeps every row."""
        n_classes = dr.MAX_CHART_CLASSES + 15
        for i in range(n_classes):
            stats.note_annotation(f"class_{i}", 0.2, 0.2)
        stats.note_image({f"class_{i}" for i in range(n_classes)})

        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        with patch("clearml.Task.current_task", return_value=mock_task):
            dr.report_dataset_composition(stats)

        table = mock_logger.report_table.call_args[1]["table_plot"]
        assert len(table) == n_classes

        fig = mock_logger.report_plotly.call_args_list[0][1]["figure"]
        assert len(fig.data[0].x) == dr.MAX_CHART_CLASSES
        assert f"of {n_classes}" in fig.layout.title.text

    def test_geometry_omitted_when_no_boxes_measured(self, stats):
        """Filter-only data still reports, just without the geometry figure."""
        stats.note_filter_report(_filter_report(10, 0, dropped_class={"stalk": 10}))

        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        with patch("clearml.Task.current_task", return_value=mock_task):
            dr.report_dataset_composition(stats)

        plot_series = {c[1]["series"] for c in mock_logger.report_plotly.call_args_list}
        assert "Object Geometry" not in plot_series
