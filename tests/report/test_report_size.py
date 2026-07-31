"""The rendered page has to stay inside its byte and DOM budgets.

Two axes, and they fail differently. Bytes are what a browser downloads over whatever
link the ClearML fileserver is on; DOM nodes are what it then has to lay out, and a page
with 20,000 elements scrolls badly whatever it weighs.

Both numbers moved when the figures became hand-authored SVG inlined into the document.
Bytes fell hard -- the 1.42 MB plotly bundle is gone and there is no chart library at
all. Nodes rose, because every mark is now an element: the ceiling below is set from
what the two capped matrices cost (60x60 cells and 16x16) plus headroom, and it is still
an order of magnitude under the number that makes a page scroll badly.
"""

import re

from src.report.blob import (
    MAX_BOXES_PER_ITEM,
    MAX_CLASS_NAMES,
    MAX_CM_CLASSES,
    MAX_DEGRADATIONS,
    MAX_FIGURES,
    MAX_KPIS,
    MAX_TABLE_ROWS,
    MAX_WARNINGS,
)
from src.report.render import dom_estimate, render_report

from .conftest import DEFAULT_CFG, make_blob


BUDGET = 5_000_000
DOM_BUDGET = 4_500
STRESS_DOM_BUDGET = 12_000


def _render(tmp_path, **kwargs):
    return render_report(make_blob(tmp_path, **kwargs))


class TestSizeBudget:
    """Target 5 MB, hard ceiling 15 MB, and the bundle is not our doing."""

    def test_generated_html_under_budget(self, tmp_path):
        """A realistic run: 30 classes, 2,000 images, detect."""
        document = _render(tmp_path, n_classes=30, n_images=2000)
        total = len(document.encode("utf-8"))

        assert total < BUDGET
        # Well under, and the headroom is the point: this was ~2.3 MB when two thirds of
        # it was a chart library that drew the same figures at runtime.
        assert total < 1_500_000

    def test_stress_run_under_hard_ceiling(self, tmp_path):
        """100 classes and a large split still fit under the hard ceiling."""
        document = _render(tmp_path, n_classes=100, n_images=5000, seg=True)

        assert len(document.encode("utf-8")) < DEFAULT_CFG["report_max_bytes"]

    def test_dom_node_estimate(self, tmp_path):
        """Every mark is an element now, so this is the number that has to be watched."""
        document = _render(tmp_path, n_classes=30, n_images=2000)

        assert dom_estimate(document) < DOM_BUDGET

    def test_dom_node_estimate_under_stress(self, tmp_path):
        """100 classes is a 60x60 matrix, which is the largest figure the report draws.

        The gallery is deliberately absent from this count: its overlay layers are built
        by the runtime a tile at a time, so tripling the grids does not triple the DOM.
        """
        document = _render(tmp_path, n_classes=100, n_images=1000, seg=True)

        assert dom_estimate(document) < STRESS_DOM_BUDGET

    def test_thumbnails_are_the_largest_controllable_cost(self, tmp_path):
        """Sanity on where the bytes go, so a future change notices when it shifts."""
        blob = make_blob(tmp_path, n_classes=30, n_images=500)
        thumb_bytes = sum(len(v) for v in blob["thumbs"].values())

        assert thumb_bytes < 2_500_000
        assert len(blob["thumbs"]) <= DEFAULT_CFG["report_max_thumbnails"]


class TestCapsAreEnforced:
    """Every cap in the schema, asserted against deliberately over-sized input."""

    def test_caps_are_enforced_in_blob(self, tmp_path):
        blob = make_blob(tmp_path, n_classes=100, n_images=2000, seg=True)

        assert len(blob["figures"]) <= MAX_FIGURES
        assert len(blob["kpis"]) <= MAX_KPIS
        assert len(blob["classes"]) <= MAX_CLASS_NAMES
        assert len(blob["warnings"]) <= MAX_WARNINGS
        assert len(blob["degradations"]) <= MAX_DEGRADATIONS
        for table in blob["tables"].values():
            assert len(table["rows"]) <= MAX_TABLE_ROWS
        for grid in blob["grids"].values():
            for item in grid["items"]:
                assert len(item["boxes"]) <= MAX_BOXES_PER_ITEM

    def test_confusion_matrix_is_capped_by_class_count(self, tmp_path):
        """The matrix is O(n^2), so it is the one payload that has to be truncated."""
        blob = make_blob(tmp_path, n_classes=100, n_images=300)
        figure = blob["figures"]["f_confusion"]
        cells = figure["svg"].count("data-tip")

        assert figure["classes"] <= MAX_CM_CLASSES
        assert cells <= MAX_CM_CLASSES**2
        assert any("confusion matrix shows" in d for d in blob["degradations"])

    def test_curve_payloads_are_binned_not_raw(self, tmp_path):
        """No figure may carry one mark per detection.

        The SVG form of the old claim: a line is a `points` list and a distribution is a
        set of marks, and both are a function of the bin count. The ceiling is the
        largest figure the report can draw -- the capped matrix -- so anything shipping a
        sample of a 2,000-image split fails it by orders of magnitude.
        """
        blob = make_blob(tmp_path, n_images=2000)
        for fig_id, fig in blob["figures"].items():
            svg = fig["svg"]
            for points in re.findall(r'points="([^"]+)"', svg):
                assert len(points.split()) <= 260, f"{fig_id} ships a raw sample"
            assert svg.count("data-tip") <= MAX_CM_CLASSES**2, f"{fig_id} is unbounded"

    def test_per_grid_cap_is_respected_when_lowered(self, tmp_path):
        """The cap is a parameter, and lowering it actually lowers the payload."""
        blob = make_blob(tmp_path, n_images=300, cfg={"report_gallery_per_grid": 4})

        for grid in blob["grids"].values():
            assert len(grid["items"]) <= 4

    def test_thumbnail_cap_stops_encoding(self, tmp_path):
        """Once the cap is hit, no further thumbnails are produced at any cost."""
        blob = make_blob(tmp_path, n_images=300, cfg={"report_max_thumbnails": 10})

        assert len(blob["thumbs"]) <= 10
