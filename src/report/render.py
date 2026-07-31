"""Markup generation, asset inlining and the multi-part split.

No Jinja2. It is present only as a transitive torch dependency, and declaring it would
rewrite `pyproject.toml` and `uv.lock` -- both of which the Dockerfile bind-mounts into
its `uv sync` layer, so a templating library would cost an ~8 GB dependency rebuild.
The template is one HTML file with `{{TOKEN}}` placeholders and `str.replace` fills them.

The markup here is deliberately thin, but the line has moved: figures are now finished
SVG and are inlined straight into the document, so they are in the DOM before any script
runs and they survive printing and a script-blocked iframe. Everything whose node count
would grow with the gallery -- the three overlay layers over each thumbnail -- is still
built by the runtime from the blob, a tile at a time.

**Figures are stripped from the JSON blob before it is serialised.** They are already in
the document; shipping both would put every chart in the file twice.
"""

from __future__ import annotations
import html
import json
import re

from pathlib import Path
from typing import Any

from src.utils.logging import get_logger


logger = get_logger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"

# (anchor, nav label). A section with nothing to say is dropped from both.
SECTION_ORDER = (
    ("s-header", "Summary"),
    ("s-highlights", "Highlights"),
    ("s-model-card", "Model card"),
    ("s-dataset", "Dataset"),
    ("s-per-class", "Per-class"),
    ("s-confusion", "Confusion"),
    ("s-threshold", "Threshold"),
    ("s-tide", "Errors"),
    ("s-strata", "Size &amp; shape"),
    ("s-boxmask", "Box vs mask"),
    ("s-galleries", "Galleries"),
    ("s-training", "Training"),
    ("s-environment", "Environment"),
    ("s-caveats", "Caveats"),
)

# The three overlay layers, and what each one's legend says. The layers themselves are
# drawn by `report.js` from `grids[].items[].boxes`; this is only the strip of text.
GALLERY_TABS = (
    ("outcome", "Outcome"),
    ("pred", "Prediction"),
    ("gt", "Ground truth"),
)

# A run of digits, thousands separators, one decimal point and an optional unit sign.
# Highlights arrive from the blob as plain sentences; this is what makes their numbers
# scan down the column without the builder having to emit markup. The lookbehind is what
# keeps it from weighting the "50" inside dAP50 and the "1" inside F1 -- those are parts
# of a name, not figures.
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?x?")


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def e(value: Any) -> str:
    """Escape for text content and attribute values alike."""
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(value: Any, kind: str = "num2") -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        f = float(value)
    except TypeError, ValueError:
        return e(value)
    if kind == "pct":
        return f"{f * 100:.1f}%"
    return f"{f:.4f}" if kind == "num2" else f"{f:g}"


class _Figures:
    """Continuous `Fig N.` numbering across sections, skipping the ones not built."""

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return self.n


# --- building blocks ----------------------------------------------------------------


def fig_block(
    blob: dict,
    fig_id: str,
    caption: str = "",
    missing: str = "",
    applicable: bool = True,
    *,
    title: str = "",
    num: _Figures | None = None,
    wide: bool = True,
) -> str:
    """One inlined figure, or a card explaining why it is not there.

    `applicable` separates two things a reader must not have to tell apart. A "not
    captured" card means the figure *could* have existed on this run and something
    stopped it -- that is a finding, and it belongs on the page. A figure that cannot
    exist on this task type at all, like the mask-fill histogram on a detect run, is
    omitted entirely, exactly as the whole box-vs-mask section is. Printing "not
    captured" for it would invite somebody to go looking for a switch that does not
    exist.
    """
    if not applicable:
        return ""
    payload = (blob.get("figures") or {}).get(fig_id)
    if not payload or not payload.get("svg"):
        text = missing or "Not captured for this run."
        return f'<p class="missing">{e(text)}</p>'
    head = f'<p class="fig-t">{e(title)}</p>' if title else ""
    label = f"<b>Fig {num.next()}.</b> " if num is not None else ""
    foot = f"<figcaption>{label}{caption}</figcaption>" if (caption or label) else ""
    cls = "wide" if wide else ""
    return (
        f'<figure class="{cls}" data-fig="{e(fig_id)}">{head}{payload["svg"]}{foot}'
        "</figure>"
    )


def table_block(blob: dict, table_id: str, missing: str = "") -> str:
    """Render a sortable table. Numeric cells carry `data-sort`, so "1,024" sorts."""
    table = (blob.get("tables") or {}).get(table_id)
    if not table or not table.get("rows"):
        return f'<p class="missing">{e(missing or "No rows to show.")}</p>'
    columns = table["columns"]
    hidden = set(table.get("hidden") or [])
    class_col = table.get("class_col")
    pair_col = table.get("pair_col")
    tones = table.get("row_tone") or []

    head = "".join(
        f'<th class="{"hidecol" if i in hidden else ""}">{e(c)}</th>'
        for i, c in enumerate(columns)
    )
    body = []
    for r, row in enumerate(table["rows"]):
        attrs = []
        tone = tones[r] if r < len(tones) else None
        classes = [tone] if tone else []
        if class_col is not None:
            attrs.append(f'data-class="{e(row[class_col])}"')
            classes.append("clickable")
        if pair_col is not None and pair_col < len(row):
            attrs.append(f'data-pair="{e(row[pair_col])}"')
            classes.append("clickable")
        if classes:
            attrs.append(f'class="{" ".join(classes)}"')
        cells = []
        for i, value in enumerate(row):
            css = "hidecol" if i in hidden else ("num" if _numeric(value) else "")
            sort = f' data-sort="{value}"' if _numeric(value) else ""
            cells.append(f'<td class="{css}"{sort}>{_cell(value)}</td>')
        body.append(f"<tr {' '.join(attrs)}>{''.join(cells)}</tr>")
    note = f'<p class="caption">{table.get("note", "")}</p>' if table.get("note") else ""
    return (
        f'<div class="scrollx wide"><table class="sortable"><thead><tr>{head}</tr>'
        f"</thead><tbody>{''.join(body)}</tbody></table></div>{note}"
    )


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cell(value: Any) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return e(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return e(value)


def _section(anchor: str, title: str, sub: str, body: str) -> str:
    heading = f"<h2>{e(title)}</h2>" if anchor != "s-header" else ""
    subtitle = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<section id="{anchor}">{heading}{subtitle}{body}</section>'


def _stat_column(heading: str, rows) -> str:
    """One flat column of `label  value` lines. No box, no accent edge, no glass."""
    lines = []
    for label, value in rows:
        zero = " zero" if value in (0, "0", "", None) else ""
        # Escaped even though every value here is ours: the image-stat rows are keyed by
        # PIL's `mode` and `format`, which come off a file header, and a value read from
        # a file is never markup.
        shown = e(value) if isinstance(value, str) else _cell(value)
        lines.append(f'<p class="stat-l{zero}"><span>{e(label)}</span><b>{shown}</b></p>')
    return (
        f'<div class="statcol"><p class="stat-h">{e(heading)}</p>{"".join(lines)}</div>'
    )


def _bold_numbers(text: str) -> str:
    """Escape one highlight line and weight its figures so they scan down the column."""
    # quote=False on purpose: `&#x27;` carries digits, and bolding the guts of an entity
    # would print the escape itself.
    return NUMBER.sub(lambda m: f"<b>{m.group(0)}</b>", html.escape(text, quote=False))


# --- sections -----------------------------------------------------------------------


def _header(blob: dict) -> str:
    meta = blob["meta"]
    link = (
        f'<a href="{e(meta["task_url"])}" target="_blank" rel="noopener">'
        f"{e(meta['task_id'][:12] or 'task')}</a>"
        if meta.get("task_url")
        else e(meta.get("task_id") or "-")
    )
    tags = "".join(f"<li>{e(t)}</li>" for t in (meta.get("tags") or []))
    # Omitted entirely when there are none: an empty rail reads as a missing feature.
    tag_rail = (
        f'<ul class="tags" aria-label="Experiment tags">{tags}</ul>' if tags else ""
    )
    bits = [
        f"<b>{e(meta['model_name'])}</b>",
        f"imgsz {e(meta['imgsz'])}",
        f"split <b>{e(meta['split_name'])}</b> ({meta['val_images']:,} images)",
        f"generated {e(meta['generated_utc'])}",
        f'commit <span class="mono">{e((meta.get("git_commit") or "?")[:12])}</span>',
        f'<span class="mono">{e(meta["image_tag"])}</span>',
        f"ultralytics {e(meta['ultralytics'])}",
        f"task {link}",
    ]
    tiles = []
    for kpi in blob.get("kpis") or []:
        value = kpi.get("value")
        scale = kpi.get("scale")
        ring = ""
        if scale and value is not None:
            lo, hi = scale
            pct = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
            ring = _ring(pct, f"{kpi['label']} {_fmt(value, kpi.get('fmt', 'num2'))}")
        tiles.append(
            f'<div class="kpi">{ring}'
            f'<span class="kpi-v">{_fmt(value, kpi.get("fmt", "num2"))}</span>'
            f'<span class="kpi-l">{e(kpi["label"])}</span>'
            f'<span class="kpi-b">{e(kpi.get("basis") or "not computed")}</span></div>'
        )
    return _section(
        "s-header",
        "",
        "",
        f"<h1>{e(meta['run_name'])}</h1>{tag_rail}"
        f'<p class="meta">{" &middot; ".join(bits)}</p>'
        f'<div class="kpis glass wide">{"".join(tiles)}</div>',
    )


def _ring(fraction: float, tip: str) -> str:
    """Draw a 30px progress ring. Its value is printed beside it, never inside it."""
    r = 12.0
    circumference = 2 * 3.14159265 * r
    filled = circumference * max(0.0, min(1.0, fraction))
    return (
        f'<svg class="ring" viewBox="0 0 30 30" role="img" data-tip="{e(tip)}" '
        f'aria-label="{e(tip)}"><circle class="rk" cx="15" cy="15" r="{r:g}"/>'
        f'<circle class="ra" cx="15" cy="15" r="{r:g}" '
        f'stroke-dasharray="{filled:.2f} {circumference - filled:.2f}" '
        f'transform="rotate(-90 15 15)"/></svg>'
    )


def _highlights(blob: dict) -> str:
    """Measured facts, then the warnings, in the same flat register.

    Deliberately siblings rather than competing banners: warnings say something is
    wrong, highlights say what the run is, and neither is a box with a coloured edge.
    """
    lines = "".join(f"<li>{_bold_numbers(h)}</li>" for h in blob.get("highlights") or [])
    body = (
        f'<ul class="hl">{lines}</ul>'
        if lines
        else '<p class="missing">Nothing measurable was available to summarise.</p>'
    )
    flags = "".join(
        f'<li><span class="sd {e(w["severity"])}" aria-hidden="true"></span>'
        f"<span>{e(w['text'])}</span></li>"
        for w in blob.get("warnings") or []
    )
    if flags:
        body += f'<h2>Warnings</h2><ul class="flags">{flags}</ul>'
    return _section(
        "s-highlights",
        "Highlights",
        "Each line states what was measured, and stops there. Every one of them is "
        "checkable against a figure further down.",
        body,
    )


def _model_card(blob: dict) -> str:
    card = blob["model_card"]
    diff_rows = "".join(
        f"<tr><td>{e(k)}</td><td>{e(v)}</td><td>{e(d)}</td></tr>"
        for k, v, d in card.get("config_diff") or []
    )
    diff = (
        '<div class="scrollx"><table class="sortable"><thead><tr><th>Parameter</th>'
        "<th>This run</th><th>Ultralytics default</th></tr></thead><tbody>"
        f"{diff_rows}</tbody></table></div>"
        if diff_rows
        else '<p class="missing">The trainer configuration was not available.</p>'
    )
    full = "\n".join(f"{k} = {v}" for k, v in card.get("config_full") or [])
    body = (
        f"<h3>Intended use</h3><p>{e(card['intended_use'])}</p>"
        f"<h3>Out of scope</h3><p>{e(card['out_of_scope'])}</p>"
        f"<h3>Three thresholds</h3><p>{e(card['thresholds_note'])}</p>"
        f"<details open><summary>Configuration that differs from the ultralytics "
        f"defaults</summary>{diff}</details>"
        f"<details><summary>Full configuration</summary>"
        f'<pre class="cfg">{e(full)}</pre></details>'
    )
    return _section(
        "s-model-card",
        "Model card",
        "What this run was, what it is for, and where it departs from the defaults.",
        body,
    )


def _dataset(blob: dict, num: _Figures) -> str:
    # Mask fill ratio and polygon vertex counts do not exist on a detect run -- the
    # label files carry four numbers per line and there is no polygon to measure.
    seg = bool(blob["meta"].get("is_seg"))
    body = (
        fig_block(
            blob,
            "f_class_bars",
            "Log axis, because the class you have to worry about is the sliver.",
            "The dataset label files were not readable, so composition is unavailable.",
            title="Instances per class, all splits",
            num=num,
        )
        + fig_block(
            blob,
            "f_split_stack",
            "Sorted so a zero-validation class is first.",
            title="Instances per class by split",
            num=num,
        )
        + table_block(blob, "t_split_composition", "No split composition was recounted.")
        + fig_block(
            blob,
            "f_objects_per_image",
            "Only the modal bin and the p99.5 cut are labelled; thirty printed numbers "
            "would bury the shape they describe.",
            title="Objects per image",
            num=num,
        )
        + fig_block(
            blob,
            "f_box_size",
            "The dashed lines are this dataset's own terciles, in pixels at this run's "
            "imgsz.",
            title="Object size",
            num=num,
        )
        + fig_block(
            blob,
            "f_aspect",
            "log(w/h), so a tall box and a wide one of the same ratio sit the same "
            "distance either side of zero.",
            title="Object aspect ratio",
            num=num,
        )
        + fig_block(
            blob,
            "f_center_heat",
            "One cell is a sixteenth of the frame across by a sixteenth down. The ramp "
            "is a single hue, so the reading is magnitude and never category.",
            title="Box centres over the frame",
            num=num,
        )
        + fig_block(
            blob,
            "f_mask_fill",
            "A mass above 0.95 means the polygons are rectangles, and mask mAP measured "
            "against them is close to box mAP.",
            "No polygons were found in the label files, so the mask fill ratio could "
            "not be measured.",
            applicable=seg,
            title="Mask fill ratio",
            num=num,
        )
        + fig_block(
            blob,
            "f_poly_vertices",
            "Four vertices is a rectangle drawn with the polygon tool, and the mask "
            "metric cannot tell it from a box.",
            applicable=seg,
            title="Polygon vertex count",
            num=num,
        )
        + "<h3>Annotation quality flags</h3>"
        + table_block(blob, "t_quality_flags", "No quality flags were raised.")
        + _image_files(blob, num)
    )
    return _section(
        "s-dataset",
        "Dataset and split composition",
        "Read from the label files on disk, after filtering and after the split.",
        body,
    )


def _image_files(blob: dict, num: _Figures) -> str:
    """Render the image-header pass: the files, before anything was annotated."""
    stats = blob.get("image_stats")
    if not stats:
        return ""
    columns = "".join(
        _stat_column(col["heading"], col["rows"]) for col in stats.get("columns") or []
    )
    return (
        "<h3>Image files</h3>"
        f"<p>Read from the header of every image under the split directories -- "
        f"{stats['scanned']:,} of them, no pixel decoded.</p>"
        + fig_block(
            blob,
            "f_resolutions",
            "Top six distinct sizes by image count; everything rarer is one row.",
            title="Image resolutions",
            num=num,
        )
        + fig_block(blob, "f_img_aspect", wide=False)
        + fig_block(
            blob,
            "f_megapixels",
            "The axis stops one bin past the largest image, so a 44 MP scale does not "
            "squash a 12 MP dataset into the first tenth of it.",
            title="Pixels per image",
            num=num,
        )
        + f'<div class="statrow wide">{columns}</div>'
        + f'<p class="note">{e(stats.get("note") or "")}</p>'
    )


def _per_class(blob: dict) -> str:
    return _section(
        "s-per-class",
        "Per-class evaluation",
        "Sorted by support ascending: the unreliable rows are at the top, where they "
        "belong. Click a row to filter the galleries to that class.",
        table_block(
            blob,
            "t_per_class",
            "Per-class metrics were unavailable (`metrics.summary()` failed).",
        ),
    )


def _confusion(blob: dict, num: _Figures) -> str:
    meta = blob["meta"]
    caption = (
        f"conf = {meta['thresholds']['matrix_display']:g} (display), IoU 0.45 "
        "(ultralytics' fixed matrix threshold). Rows are ground truth, columns are "
        "predictions; the last row and column are background, i.e. false positives and "
        "false negatives. The background column stays as raw counts in both modes "
        "&mdash; "
        "normalising it would print a count as if it were a rate."
    )
    if meta.get("is_seg"):
        caption += " The matrix is box-IoU based even in segment mode."
    has_figure = "f_confusion" in (blob.get("figures") or {})
    toggle = (
        '<button id="cm-toggle" type="button">Show row-normalised</button>'
        if has_figure
        else ""
    )
    figure = fig_block(
        blob,
        "f_confusion",
        caption,
        "The confusion matrix requires `5_Testing/plots=True`.",
        title="Confusion matrix",
        num=num,
    )
    # Both ramps ship inside the one SVG; this attribute is what selects between them,
    # so the toggle is a class swap and never a redraw.
    body = (
        toggle
        + (f'<div id="cm-wrap" data-cm="counts">{figure}</div>' if has_figure else figure)
        + "<h3>Most confused pairs</h3>"
        + table_block(blob, "t_confused_pairs", "No cross-class confusion was captured.")
    )
    return _section(
        "s-confusion",
        "Confusion matrix",
        "Which classes get mistaken for which, at the display threshold.",
        body,
    )


def _threshold(blob: dict, num: _Figures) -> str:
    body = (
        fig_block(
            blob,
            "f_pr_f1_conf",
            "tau* is the F1-optimal threshold, and it is none of the other three "
            "confidences in this task.",
            "Curve data was not available from this validation pass.",
            title="Precision, recall and F1 against confidence",
            num=num,
        )
        + fig_block(
            blob,
            "f_tp_fp_hist",
            "Two overlapping humps mean no threshold fixes precision &mdash; the model "
            "or the labels need work, not the tuning.",
            "The TP-vs-FP split needs `8_Visualization/log_calibration`.",
            title="Detection confidence, matched against unmatched",
            num=num,
        )
        + fig_block(
            blob,
            "f_reliability",
            "Computed over predictions only, so it is blind to false negatives. Read it "
            "next to recall.",
            "The reliability diagram needs `8_Visualization/log_calibration`.",
            title="Reliability",
            num=num,
        )
    )
    return _section(
        "s-threshold",
        "Operating threshold",
        "Where to set the deploy threshold, and whether the scores mean anything.",
        body,
    )


def _tide(blob: dict, num: _Figures) -> str:
    mode = blob.get("tide_mode", "none")
    title = {
        "delta_ap": "Error decomposition (dAP oracles)",
        "counts": "TIDE error counts (dAP oracles not computed)",
    }.get(mode, "Error decomposition (not captured)")
    body = fig_block(
        blob,
        "f_tide",
        "Matched at IoU 0.5 for foreground and 0.1 for background, box IoU only "
        "&mdash; even on a segmentation run. The two dashed ceilings are the "
        "false-positive and false-negative oracles, which are not segments of the bar.",
        "Not captured: the per-image wrapper did not run for this validation pass.",
        title="dAP50 by error type",
        num=num,
    ) + table_block(blob, "t_tide", "Not captured for this run.")
    return _section(
        "s-tide",
        title,
        "Six error types, each with the AP an oracle that fixed only that error would "
        "recover. They are independent oracles and do not sum to the total headroom.",
        body,
    )


def _strata(blob: dict, num: _Figures) -> str:
    body = (
        fig_block(
            blob,
            "f_recall_by_size",
            "These are this dataset's own terciles, not COCO's 32x32 / 96x96 absolutes.",
            "Not captured: the per-image wrapper did not run.",
            title="Recall by object size",
            num=num,
        )
        + fig_block(
            blob,
            "f_iou_hist_tp",
            "Mass piled just above 0.50 means localisation is the bottleneck &mdash; it "
            "should agree with the Loc segment of the error decomposition.",
            title="IoU of matched detections",
            num=num,
        )
        + fig_block(
            blob,
            "f_recall_vs_area",
            "Deciles of this split's own size distribution, in pixels at this run's "
            "imgsz.",
            title="Recall against object size",
            num=num,
        )
        + fig_block(
            blob,
            "f_recall_by_ar",
            "A bucket the model never recalls is usually a bucket the training split "
            "barely had.",
            title="Recall by aspect ratio",
            num=num,
        )
    )
    return _section("s-strata", "Size and shape", "Where the misses actually are.", body)


def _boxmask(blob: dict, num: _Figures) -> str:
    body = (
        fig_block(
            blob,
            "f_boxmask_class",
            "Upper bar is box, lower is mask. A class whose two bars are the same "
            "length was annotated as rectangles.",
            title="Box against mask, per class",
            num=num,
        )
        + fig_block(
            blob,
            "f_boxmask_iou",
            "The mask axis is quantised to the ten IoU thresholds, because ultralytics "
            "discards the pairwise mask IoU matrix. Mass on the diagonal at high IoU is "
            "the rectangle-annotation signature.",
            title="Box IoU against mask IoU level",
            num=num,
        )
        + fig_block(
            blob,
            "f_iou_overlay",
            "Density rather than share: the box axis has 50 bins and the mask axis 11, "
            "so their raw shares are not comparable.",
            title="Box and mask IoU distributions",
            num=num,
        )
        + fig_block(
            blob,
            "f_boxmask_size",
            "Signed, and every bar carries its sign: a negative bucket is one where the "
            "mask fits worse than the box.",
            title="Mask minus box IoU by size",
            num=num,
        )
    )
    return _section(
        "s-boxmask",
        "Box against mask",
        "A 6-10 point mask-minus-box gap is normal; much less usually means the "
        "polygons are rectangles.",
        body,
    )


def _galleries(blob: dict, gallery_link: str | None = None) -> str:
    if gallery_link:
        body = (
            '<p class="missing">The galleries were split into their own artifact to '
            "keep this file inside the size budget. "
            f'<a href="{e(gallery_link)}" target="_blank" rel="noopener">Open the '
            "gallery report</a>.</p>"
        )
        return _section("s-galleries", "Galleries", "", body)

    parts = [
        (
            '<p class="gal-switch"><button class="anno" id="anno" type="button" '
            'role="switch" aria-checked="true"><span>annotations</span>'
            '<span class="trk" aria-hidden="true"><span class="knb"></span></span>'
            "</button></p>"
        )
    ]
    for gid, grid in (blob.get("grids") or {}).items():
        items = grid.get("items") or []
        if not items:
            parts.append(
                f"<h3>{e(grid['title'])}</h3>"
                f'<p class="missing">{e(grid.get("empty_reason") or "Nothing to show.")}'
                "</p>"
            )
            continue
        tiles = []
        for i, item in enumerate(items):
            tiles.append(
                f'<figure data-gid="{e(gid)}" data-idx="{i}" tabindex="0" '
                f'data-outcome="{e(item["outcome"])}" '
                f'data-classes="{e(" ".join(item.get("classes") or []))}" '
                f'data-pairs="{e(" ".join(item.get("pairs") or []))}">'
                f'<img data-thumb="{e(item["thumb"])}" alt="" loading="lazy">'
                f"<figcaption>{e(item['label'])}</figcaption></figure>"
            )
        parts.append(
            f"<h3>{e(grid['title'])}</h3>"
            f'<div class="galwrap wide" data-gid="{e(gid)}" data-tab="outcome">'
            f'<div class="gal-head">{_tabs(gid)}'
            f'<span class="gal-meta"><span class="shown"></span> &middot; '
            f"{e(grid['subtitle'])} &middot; {e(grid['basis'])}</span></div>"
            f'<div class="gal at-start"><div class="grid">{"".join(tiles)}</div></div>'
            f"{_legend()}</div>"
        )
    return _section(
        "s-galleries",
        "Galleries",
        "Three readings of the same thumbnails: only the overlay layer changes, never "
        "the image. The switch governs the grids and the lightbox together, and the "
        "<b>a</b> key toggles it. Click any tile for the full frame.",
        "".join(parts),
    )


def _tabs(gid: str) -> str:
    buttons = []
    for i, (key, label) in enumerate(GALLERY_TABS):
        first = i == 0
        buttons.append(
            f'<button class="tab" type="button" role="tab" data-tab="{key}" '
            f'aria-selected="{"true" if first else "false"}"'
            f"{'' if first else ' tabindex=-1'}>{label}</button>"
        )
    return (
        f'<div class="tabs" role="tablist" aria-label="Overlay layer for {e(gid)}">'
        f"{''.join(buttons)}</div>"
    )


def _legend() -> str:
    return (
        '<p class="gal-legend">'
        '<span class="lg-outcome">'
        '<i><span class="sw tp" aria-hidden="true"></span>true positive</i>'
        '<i><span class="sw fp" aria-hidden="true"></span>false positive</i>'
        '<i><span class="sw fn" aria-hidden="true"></span>missed ground truth</i></span>'
        '<span class="lg-pred">'
        '<i><span class="sw pd" aria-hidden="true"></span>detection, confidence '
        "printed</i></span>"
        '<span class="lg-gt">'
        '<i><span class="sw gt" aria-hidden="true"></span>annotated instance</i></span>'
        "</p>"
    )


def _training(blob: dict, num: _Figures) -> str:
    body = (
        "<details><summary>Training curves</summary>"
        + fig_block(
            blob,
            "f_val_map",
            "",
            "Training history not found (`results.csv`).",
            title="Validation mAP against epoch",
            num=num,
        )
        + fig_block(
            blob,
            "f_losses",
            "Raw loss values on one axis, so the shapes are comparable and the "
            "magnitudes are not.",
            title="Training losses against epoch",
            num=num,
        )
        + "</details>"
    )
    return _section(
        "s-training",
        "Training history",
        "Downsampled to a fixed point count, which is what keeps report size "
        "independent of epoch count.",
        body,
    )


def _environment(blob: dict) -> str:
    env = blob.get("environment") or {}
    # No `num=`: the epoch chart sits inside a stat column, where a "Fig N." caption
    # would be wider than the column and would number a figure nobody cross-references.
    chart = fig_block(blob, "f_epoch_seconds", wide=False)
    columns = (
        _stat_column("Where", env.get("where") or [])
        + _stat_column("How long", env.get("timing") or [])
        + (
            f'<div class="statcol chart-col"><p class="stat-h">Seconds per epoch</p>'
            f"{chart}</div>"
            if "f_epoch_seconds" in (blob.get("figures") or {})
            else ""
        )
    )
    return _section(
        "s-environment",
        "Environment",
        "Where this run executed and how long it took. The machine facts are read "
        "locally on the agent; the worker id and the start timestamp come from the "
        "ClearML task, and any field that cannot be read prints "
        '<span class="mono">unknown</span> rather than disappearing.',
        f'<div class="statrow env wide">{columns}</div>'
        f'<p class="note">{e(env.get("note") or "")}</p>',
    )


def _caveats(blob: dict) -> str:
    items = "".join(f"<li>{e(line)}</li>" for line in blob.get("caveats") or [])
    return _section(
        "s-caveats",
        "Caveats",
        "Everything this report knows it cannot tell you.",
        f'<ul class="caveats">{items}</ul>',
    )


# --- assembly -----------------------------------------------------------------------


def render_report(
    blob: dict,
    *,
    gallery_link: str | None = None,
    only_galleries: bool = False,
) -> str:
    """Return the whole self-contained HTML document.

    `only_galleries` renders the child of a two-part split; `gallery_link` renders the
    index of one, whose gallery section becomes a link to that child's **absolute** URL
    (relative links between artifacts break -- each artifact name is its own directory).
    """
    num = _Figures()
    if only_galleries:
        sections = [_galleries(blob), _caveats(blob)]
        anchors = {"s-galleries", "s-caveats"}
    else:
        builders = [
            ("s-header", _header),
            ("s-highlights", _highlights),
            ("s-model-card", _model_card),
            ("s-dataset", lambda b: _dataset(b, num)),
            ("s-per-class", _per_class),
            ("s-confusion", lambda b: _confusion(b, num)),
            ("s-threshold", lambda b: _threshold(b, num)),
            ("s-tide", lambda b: _tide(b, num)),
            ("s-strata", lambda b: _strata(b, num)),
        ]
        if blob["meta"].get("is_seg"):
            builders.append(("s-boxmask", lambda b: _boxmask(b, num)))
        builders += [
            ("s-galleries", lambda b: _galleries(b, gallery_link)),
            ("s-training", lambda b: _training(b, num)),
            ("s-environment", _environment),
            ("s-caveats", _caveats),
        ]
        sections = [fn(blob) for _, fn in builders]
        anchors = {a for a, _ in builders}

    nav = "".join(
        f'<a href="#{a}">{label}</a>' for a, label in SECTION_ORDER if a in anchors
    )
    title = f"Evaluation report - {blob['meta'].get('run_name', 'run')}"

    # The figures are already in the document as SVG. Shipping them in the blob as well
    # would put every chart in the file twice.
    payload = dict(blob)
    payload["figures"] = {}
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False, default=str)
    # `</script>` inside the JSON would close the element early; nothing else can.
    body = body.replace("</", "<\\/")

    document = _asset("template.html")
    for token, value in (
        ("{{TITLE}}", e(title)),
        ("{{RUN_NAME}}", e(blob["meta"].get("run_name", "run"))),
        ("{{CSS}}", _asset("report.css")),
        ("{{NAV}}", nav),
        ("{{TOC}}", nav),
        ("{{SECTIONS}}", "".join(sections)),
        ("{{BLOB}}", body),
        ("{{SORTABLE_JS}}", _asset("sortable.min.js")),
        ("{{REPORT_JS}}", _asset("report.js")),
    ):
        document = document.replace(token, value)
    return document


def split_blob(blob: dict) -> tuple[dict, dict]:
    """Split one blob into (index without galleries, galleries-only child)."""
    child = {
        "schema": blob["schema"],
        "meta": blob["meta"],
        "kpis": [],
        "highlights": [],
        "model_card": {
            "intended_use": "",
            "out_of_scope": "",
            "config_diff": [],
            "config_full": [],
            "thresholds_note": "",
        },
        "warnings": [],
        "classes": blob["classes"],
        "figures": {},
        "image_stats": None,
        "environment": {},
        "tables": {},
        "grids": blob["grids"],
        "thumbs": blob["thumbs"],
        "tide_mode": blob.get("tide_mode", "none"),
        "caveats": blob.get("caveats", []),
        "degradations": blob.get("degradations", []),
    }
    index = dict(blob)
    index["grids"] = {}
    index["thumbs"] = {}
    return index, child


def drop_galleries(blob: dict) -> dict:
    """Last resort above the hard ceiling: a report without galleries, never no report."""
    out = dict(blob)
    out["grids"] = {}
    out["thumbs"] = {}
    out["degradations"] = list(out.get("degradations") or []) + [
        "The galleries were dropped: the rendered report exceeded the hard size ceiling."
    ]
    return out


def dom_estimate(document: str) -> int:
    """Rough element count, used by the size tests. Each element opens and closes.

    The inlined `<script>` and `<style>` payloads are stripped first: they are three
    elements between them however many angle brackets they happen to contain.
    """
    stripped = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", "<script></script>", document, flags=re.DOTALL
    )
    return stripped.count("<") // 2
