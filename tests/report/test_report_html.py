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
        assert 'class="sortable"' in document
        assert '<svg class="chart"' in document  # the figures, drawn into the document

    def test_no_chart_library_ships_at_all(self, det_blob):
        """The figures are hand-authored SVG, so there is nothing to vendor any more.

        This used to be a 1.42 MB plotly bundle and two thirds of the file. It is gone,
        and the report has no runtime chart dependency to reintroduce by accident.
        """
        document = render_report(det_blob)

        assert "Plotly" not in document
        assert "plotly" not in document

    def test_banner_href_is_empty(self, det_blob):
        """`href=""` resolves to this document's own URL -- the sandboxed-iframe escape.

        It works with scripts blocked and without the generator knowing its own address.
        Filling it in with a guessed URL breaks both properties.
        """
        document = render_report(det_blob)

        assert re.search(r'<a class="banner" href=""', document)


class TestCollapsedAppendix:
    """A collapsed <details> used to mean a figure that never rendered; now it cannot.

    Plotly drew zero-width inside a hidden container and never recovered, which is why
    the training appendix carried a `lazy` marker and observed its own figures on
    `toggle`. Inline SVG has no such failure mode: the figure is in the document whether
    the appendix is open or not, which is what these two assert.
    """

    def test_the_collapsed_appendix_still_contains_its_drawn_figures(self, seg_blob):
        document = markup_only(render_report(seg_blob))
        blocks = re.findall(r"<details\b[^>]*>(.*?)</details>", document, re.DOTALL)
        with_figures = [b for b in blocks if "data-fig" in b]

        assert len(with_figures) == 1, "only the training appendix collapses a figure"
        assert "f_val_map" in with_figures[0]
        assert '<svg class="chart"' in with_figures[0]

    def test_no_lazy_marker_survives(self, seg_blob):
        """The marker existed only for Plotly; leaving it would imply it still matters."""
        document = markup_only(render_report(seg_blob))

        assert 'class="lazy"' not in document
        assert "details.lazy" not in document

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
    """The stacked bar's labels and its two ceilings must not overprint each other."""

    def _svg(self, blob):
        return blob["figures"]["f_tide"]["svg"]

    def test_segments_are_drawn_largest_first(self, det_blob):
        """Left to right by size, so the bar reads as a ranking and not as an order."""
        svg = self._svg(det_blob)
        drawn = re.findall(r'data-tip="([A-Za-z]+) \u00b7', svg)

        assert drawn, "no segment carried a tooltip"
        widths = [
            float(m)
            for m in re.findall(
                r'class="f-[a-z0-9]+" x="[\d.]+" y="48.0" width="([\d.]+)"', svg
            )
        ]
        assert widths == sorted(widths, reverse=True)

    def test_ceiling_labels_are_staggered(self, det_blob):
        """The two ceilings land within a hundredth of each other on the same axis."""
        svg = self._svg(det_blob)
        labels = re.findall(
            r'<text class="dl f-peer" x="[\d.]+" y="([\d.]+)" text-anchor="(\w+)">', svg
        )

        assert len(labels) == 2
        assert labels[0][0] != labels[1][0], "the two labels share a y"
        assert labels[0][1] != labels[1][1], "the two labels run the same way"

    def test_the_thin_tail_gets_one_leader_below_the_tick_row(self):
        """Sub-pixel segments share one leader, and its elbow clears the axis labels.

        Driven straight from the shape that produces a tail -- one error type carrying
        most of the delta and three carrying almost none -- because a synthetic run with
        six comparable segments never draws a leader at all.
        """
        from src.report import figures

        svg = figures.tide_bar(
            {
                "mode": "delta_ap",
                "types": ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"],
                "delta_ap": {
                    "Cls": 0.0001,
                    "Loc": 0.0786,
                    "Both": 0.0061,
                    "Dupe": 0.0005,
                    "Bkg": 0.2527,
                    "Miss": 0.0970,
                },
                "counts": {
                    "Cls": 517,
                    "Loc": 10291,
                    "Both": 1219,
                    "Dupe": 409,
                    "Bkg": 35010,
                    "Miss": 1703,
                },
                "ceilings": {"fp": 0.45, "fn": 0.097},
            }
        )
        leader = re.search(r'<path class="lead" d="M[\d.]+ \d+ L[\d.]+ (\d+)', svg)
        ticks = [float(y) for y in re.findall(r'class="tk" x="[\d.]+" y="([\d.]+)"', svg)]
        tail = re.findall(r'<text class="dl f-ink2" x="[\d.]+" y="[\d.]+">', svg)

        assert leader, "the thin segments were left with no leader at all"
        assert ticks, "the axis printed no ticks to clear"
        assert float(leader.group(1)) > max(ticks)
        assert len(tail) == 3, "one label per unlabelled segment, and no more"

    def test_the_bar_stays_inside_the_viewbox(self):
        """The stack is drawn end to end, so the axis has to cover its sum.

        Scaling to the largest segment instead ran a bar of six comparable errors four
        viewBox widths off the canvas, with every label past the right edge.
        """
        from src.report import figures

        even = dict.fromkeys(["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"], 0.08)
        svg = figures.tide_bar(
            {
                "mode": "delta_ap",
                "types": list(even),
                "delta_ap": even,
                "counts": dict.fromkeys(even, 100),
                "ceilings": {"fp": 0.2, "fn": 0.1},
            }
        )
        width = float(re.search(r'viewBox="0 0 (\d+)', svg).group(1))
        right = max(
            float(m.group(1)) + float(m.group(2))
            for m in re.finditer(
                r'<rect class="f-[a-z0-9]+" x="([\d.]+)" y="48.0" '
                r'width="([\d.]+)"',
                svg,
            )
        )

        assert right <= width


class TestImageFileSection:
    """The header-only pass over `<split>/images`, and what it puts on the page.

    Cheap enough to cover every image rather than a sample -- `Image.open` parses the
    header and stops -- which is what lets these figures be read as facts about the
    dataset instead of about a subsample.
    """

    def test_the_image_figures_and_stat_row_are_rendered(self, det_blob):
        document = markup_only(render_report(det_blob))
        dataset = re.search(
            r'<section id="s-dataset">(.*?)</section>', document, re.DOTALL
        ).group(1)

        for fig_id in ("f_resolutions", "f_img_aspect", "f_megapixels"):
            assert fig_id in det_blob["figures"]
            assert f'data-fig="{fig_id}"' in dataset
        assert "Image files" in dataset
        assert 'class="statrow' in dataset

    def test_the_stats_describe_the_files_on_disk(self, det_blob):
        """Three columns, and the greyscale image in the fixture is counted as one."""
        stats = det_blob["image_stats"]
        columns = {c["heading"]: dict(c["rows"]) for c in stats["columns"]}

        assert stats["scanned"] > 0
        assert columns["Colour mode"]["L"] > 0
        assert columns["File format"]["JPEG"] > 0
        fit = next(c for h, c in columns.items() if h.startswith("Fit at imgsz="))
        assert sum(fit.values()) > 0
        assert any(k.startswith("short side <") for k in fit)

    def test_the_scan_reads_headers_only(self, det_blob):
        """A resolution ranking is only meaningful if every image was looked at."""
        rows = det_blob["figures"]["f_resolutions"]["svg"]

        assert "320 \u00d7 240" in rows
        assert "240 \u00d7 320" in rows


class TestExifOrientation:
    """The scan must report the size the *loader* sees, not the size in the file.

    Ultralytics transposes a JPEG whose EXIF orientation tag is 6 or 8, which is how
    every phone photo is stored. A scan that skipped that would call a portrait split
    landscape and put the resolution ranking, the orientation strip and the fit-at-imgsz
    column all one axis out -- describing a dataset nobody trained on.
    """

    def _write(self, path, size, orientation=None):
        from PIL import Image

        image = Image.new("RGB", size, (90, 110, 130))
        if orientation is None:
            image.save(path)
            return
        exif = image.getexif()
        exif[274] = orientation
        image.save(path, exif=exif)

    def test_a_rotated_jpeg_is_counted_on_the_axis_the_loader_uses(self, tmp_path):
        from src.report.dataset_scan import DatasetScan

        plain, rotated = tmp_path / "a.jpg", tmp_path / "b.jpg"
        self._write(plain, (320, 240))
        self._write(rotated, (320, 240), orientation=6)

        scan = DatasetScan(["x"], imgsz=640)
        scan.note_image_file(plain)
        scan.note_image_file(rotated)

        assert scan.orientation["landscape"] == 1
        assert scan.orientation["portrait"] == 1
        assert scan.resolutions == {"320x240": 1, "240x320": 1}

    def test_it_agrees_with_ultralytics(self, tmp_path):
        """The control: the two implementations must not drift apart."""
        from PIL import Image
        from ultralytics.data.utils import exif_size

        from src.report.dataset_scan import _exif_size

        for orientation in (None, 1, 3, 6, 8):
            path = tmp_path / f"o{orientation}.jpg"
            self._write(path, (320, 240), orientation)
            with Image.open(path) as image:
                assert _exif_size(image) == exif_size(image)


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
