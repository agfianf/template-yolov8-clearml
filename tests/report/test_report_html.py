"""The page has to be valid, self-contained and internally consistent.

Self-containment is the load-bearing property: the report is served from the ClearML
fileserver and viewed inside a Debug Samples iframe, so a single CDN reference turns
half the page blank on any deployment without outbound network access, with no error
anywhere. Every asset is inlined and this suite is what keeps it that way.
"""

import json
import re

from src.report.render import render_report

from .conftest import extract_blob, inline_css, make_blob, markup_only


NETWORK_REF = re.compile(r'(?:src|href)\s*=\s*"(?!#|data:)([^"]*)"', re.IGNORECASE)
CSS_URL = re.compile(r"url\(\s*['\"]?(?!data:)([^)'\"]+)", re.IGNORECASE)


class TestJsonBlob:
    """The blob is the whole data model; if it does not parse, nothing renders."""

    def test_json_blob_parses(self, det_blob):
        blob = extract_blob(render_report(det_blob))

        assert blob["schema"] == 1
        assert blob["meta"]["run_name"]

    def test_blob_contains_no_nan(self, seg_blob):
        """`NaN` is valid Python and invalid JSON; `JSON.parse` rejects the whole page."""
        payload = json.dumps(seg_blob, allow_nan=False)  # raises if any NaN survived

        assert "NaN" not in payload

    def test_script_close_tag_cannot_escape_the_blob(self, tmp_path):
        """A class named `</script>` must not be able to end the element early."""
        blob = make_blob(tmp_path)
        blob["classes"] = ["</script><img>"]
        document = render_report(blob)

        assert "</script><img>" not in document.split("report-data")[1][:4000]
        assert extract_blob(document)["classes"] == ["</script><img>"]


class TestReferentialIntegrity:
    """Every id the markup points at has to exist in the blob, and the converse."""

    def test_every_data_fig_id_exists_in_blob(self, seg_blob):
        document = markup_only(render_report(seg_blob))
        referenced = set(re.findall(r'data-fig="([^"]+)"', document))

        assert referenced <= set(seg_blob["figures"])
        assert referenced, "no figures were referenced at all"

    def test_every_figure_in_blob_is_referenced(self, seg_blob):
        """A figure nobody draws is dead payload."""
        document = markup_only(render_report(seg_blob))
        referenced = set(re.findall(r'data-fig="([^"]+)"', document))

        assert set(seg_blob["figures"]) - referenced == set()

    def test_every_data_thumb_key_exists_in_blob(self, det_blob):
        document = markup_only(render_report(det_blob))
        referenced = set(re.findall(r'data-thumb="([^"]+)"', document))

        assert referenced <= set(det_blob["thumbs"])
        assert referenced

    def test_grid_items_reference_real_thumbs(self, det_blob):
        for grid in det_blob["grids"].values():
            for item in grid["items"]:
                assert item["thumb"] in det_blob["thumbs"]


class TestSelfContained:
    """Not one byte may be fetched from anywhere."""

    def test_no_external_network_references(self, seg_blob):
        rendered = render_report(seg_blob)
        document = markup_only(rendered)
        task_url = seg_blob["meta"].get("task_url") or "\0"

        for ref in NETWORK_REF.findall(document):
            if ref == "" or ref == task_url:
                continue  # the empty self-link banner, and the deliberate task link
            raise AssertionError(f"external reference: {ref}")
        # The stylesheet is ours, so it is held to the same rule: no fonts, no images.
        assert not CSS_URL.findall(inline_css(rendered))

    def test_assets_are_inlined_not_linked(self, det_blob):
        document = render_report(det_blob)

        assert "<link" not in document
        assert "Plotly" in document  # the bundle itself, not a script src
        assert 'class="sortable"' in document

    def test_banner_href_is_empty(self, det_blob):
        """`href=""` resolves to this document's own URL -- the sandboxed-iframe escape.

        It works with scripts blocked and without the generator knowing its own address.
        Filling it in with a guessed URL breaks both properties.
        """
        document = render_report(det_blob)

        assert re.search(r'<a class="banner" href=""', document)


class TestPlotlyContainerRules:
    """A Plotly div in a hidden container renders zero-width and never recovers."""

    def test_no_plotly_div_inside_details_without_the_lazy_marker(self, seg_blob):
        document = markup_only(render_report(seg_blob))
        for block in re.findall(
            r"<details\b([^>]*)>(.*?)</details>", document, re.DOTALL
        ):
            attrs, body = block
            if "data-fig" in body:
                assert "lazy" in attrs, "a figure sits in a <details> with no lazy marker"

    def test_the_lazy_details_is_the_training_appendix_only(self, seg_blob):
        document = markup_only(render_report(seg_blob))
        lazy = re.findall(r'<details class="lazy">(.*?)</details>', document, re.DOTALL)

        assert len(lazy) == 1
        assert "f_val_map" in lazy[0]

    def test_sections_carry_the_content_visibility_pair(self, det_blob):
        """`content-visibility` without `contain-intrinsic-size` makes the page jump."""
        document = render_report(det_blob)

        assert "content-visibility: auto" in document
        assert "contain-intrinsic-size: auto 900px" in document


class TestNotApplicableIsNotMissing:
    """A card means "not captured"; "cannot exist on this task type" means no card.

    A detect run has no polygons, so a mask-fill histogram is not a thing that failed --
    it is a thing that does not apply, and printing a "not captured" card for it sends
    the reader looking for a switch that does not exist. Those figures are omitted
    exactly as the whole box-vs-mask section is.
    """

    SEG_ONLY = ("f_mask_fill", "f_poly_vertices")

    def test_detect_omits_seg_only_dataset_figures(self, det_blob):
        document = markup_only(render_report(det_blob))
        dataset = re.search(
            r'<section id="s-dataset">(.*?)</section>', document, re.DOTALL
        ).group(1)

        for fig_id in self.SEG_ONLY:
            assert fig_id not in det_blob["figures"]
            assert fig_id not in dataset
        assert "Not captured for this run." not in dataset

    def test_segment_still_shows_them(self, seg_blob):
        """The control: on a run where they do apply, they are drawn."""
        document = markup_only(render_report(seg_blob))

        for fig_id in self.SEG_ONLY:
            assert f'data-fig="{fig_id}"' in document

    def test_missing_cards_are_still_used_where_they_apply(self, tmp_path):
        """Applicable-but-absent keeps its card -- this rule must not silence findings."""
        blob = make_blob(tmp_path, with_matrix=False)
        document = markup_only(render_report(blob))

        assert "f_confusion" not in blob["figures"]
        assert 'class="missing"' in document
        assert "plots=True" in document


class TestTideFigureLayout:
    """The stacked bar's annotations and legend must not overprint each other."""

    def test_legend_order_matches_the_segment_order(self, det_blob):
        """A horizontal stacked bar reverses its legend by default; Cls must lead."""
        fig = det_blob["figures"]["f_tide"]

        assert [t["name"] for t in fig["data"]] == [
            "Cls",
            "Loc",
            "Both",
            "Dupe",
            "Bkg",
            "Miss",
        ]
        assert fig["layout"]["legend"]["traceorder"] == "normal"

    def test_ceiling_annotations_are_staggered(self, det_blob):
        """The two ceilings sit close in x, so they are separated in y and by anchor."""
        layout = det_blob["figures"]["f_tide"]["layout"]
        annotations = [
            a for a in layout.get("annotations", []) if "removed" in a.get("text", "")
        ]

        assert len(annotations) == 2
        assert annotations[0]["yshift"] != annotations[1]["yshift"]
        assert annotations[0]["xanchor"] != annotations[1]["xanchor"]

    def test_legend_clears_the_axis_title(self, det_blob):
        """Bottom margin has to hold the axis title *and* the legend row under it."""
        layout = det_blob["figures"]["f_tide"]["layout"]

        assert layout["margin"]["b"] >= 100
        assert layout["legend"]["y"] < -0.3


class TestMarkupSanity:
    """Enough structure that a browser is not guessing."""

    def test_tables_have_real_head_and_body(self, det_blob):
        """The vendored sorter needs `<thead>`/`<tbody>`; a div grid would not sort."""
        document = markup_only(render_report(det_blob))
        for table in re.findall(r"<table[^>]*>(.*?)</table>", document, re.DOTALL):
            assert "<thead>" in table
            assert "<tbody>" in table

    def test_numeric_cells_carry_data_sort(self, det_blob):
        """A string sort puts 9 above 1,024, so numeric cells need an explicit key."""
        document = render_report(det_blob)

        assert 'data-sort="' in document

    def test_every_section_id_has_a_nav_link(self, seg_blob):
        document = markup_only(render_report(seg_blob))
        sections = set(re.findall(r'<section id="([^"]+)"', document))
        links = set(re.findall(r'<a href="#([^"]+)"', document))

        assert sections == links

    def test_gallery_items_carry_filter_attributes(self, det_blob):
        """The class filter is one CSS rule against these attributes, not per-item JS."""
        document = markup_only(render_report(det_blob))

        assert 'data-classes="' in document
        assert 'data-gid="' in document
        assert 'data-idx="' in document

    def test_document_declares_utf8_and_a_title(self, det_blob):
        document = render_report(det_blob)

        assert '<meta charset="utf-8">' in document
        assert "<title>Evaluation report" in document
