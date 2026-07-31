"""Markup generation, asset inlining and the multi-part split.

No Jinja2. It is present only as a transitive torch dependency, and declaring it would
rewrite `pyproject.toml` and `uv.lock` -- both of which the Dockerfile bind-mounts into
its `uv sync` layer, so a templating library would cost an ~8 GB dependency rebuild.
The template is one HTML file with `{{TOKEN}}` placeholders and `str.replace` fills them.

The markup here is deliberately thin: tables (which need real `<thead>`/`<tbody>` for the
vendored sorter) and gallery shells (which need `data-classes` / `data-pairs` for the CSS
filter) are server-rendered; everything else is one `<div data-fig>` that the runtime
fills. That is what keeps the DOM inside its node budget while every number in the page
stays inspectable in the JSON blob.
"""

from __future__ import annotations
import html
import json
import re

from pathlib import Path
from typing import Any

from src.report import theme
from src.utils.logging import get_logger


logger = get_logger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"

# (anchor, nav label). A section with nothing to say is dropped from both.
SECTION_ORDER = (
    ("s-header", "Summary"),
    ("s-model-card", "Model card"),
    ("s-dataset", "Dataset & splits"),
    ("s-per-class", "Per-class"),
    ("s-confusion", "Confusion"),
    ("s-threshold", "Thresholds"),
    ("s-tide", "Error types"),
    ("s-strata", "Size & shape"),
    ("s-boxmask", "Box vs mask"),
    ("s-galleries", "Galleries"),
    ("s-training", "Training curves"),
    ("s-caveats", "Caveats"),
)

SEVERITY_ICON = {"warning": "!", "serious": "!!", "critical": "!!!"}


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


# --- building blocks ----------------------------------------------------------------


def fig_block(
    blob: dict,
    fig_id: str,
    caption: str = "",
    missing: str = "",
    applicable: bool = True,
) -> str:
    """One lazily-rendered figure, or a card explaining why it is not there.

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
    if fig_id not in (blob.get("figures") or {}):
        text = missing or "Not captured for this run."
        return f'<p class="missing">{e(text)}</p>'
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="fig" data-fig="{e(fig_id)}"></div>{cap}'


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
        f'<div class="scrollx"><table class="sortable"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{note}"
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
    subtitle = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<section id="{anchor}"><h2>{e(title)}</h2>{subtitle}{body}</section>'


# --- sections -----------------------------------------------------------------------


def _header(blob: dict) -> str:
    meta = blob["meta"]
    link = (
        f'<a href="{e(meta["task_url"])}" target="_blank" rel="noopener">'
        f"{e(meta['task_id'][:12] or 'task')}</a>"
        if meta.get("task_url")
        else e(meta.get("task_id") or "-")
    )
    splits = ", ".join(f"{k} {v:,}" for k, v in (meta.get("split_counts") or {}).items())
    fields = [
        ("Model", meta["model_name"]),
        ("Task", f"{meta['task_yolo']} @ imgsz {meta['imgsz']}"),
        ("Split", f"{meta['split_name']} ({meta['val_images']:,} images)"),
        ("Split sizes", splits or "not recounted"),
        ("ClearML task", link),
        ("Git commit", (meta.get("git_commit") or "unknown")[:12]),
        ("Image tag", meta["image_tag"]),
        ("Template version", meta["code_version"]),
        ("Ultralytics", meta["ultralytics"]),
        ("Generated", meta["generated_utc"]),
    ]
    cells = "".join(
        f"<div><span>{e(label)}</span>{value if label == 'ClearML task' else e(value)}"
        "</div>"
        for label, value in fields
    )
    tiles = []
    for kpi in blob.get("kpis") or []:
        value = kpi.get("value")
        meter = ""
        scale = kpi.get("scale")
        if scale and value is not None:
            lo, hi = scale
            pct = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
            meter = f'<div class="meter"><i style="width:{pct * 100:.0f}%"></i></div>'
        tiles.append(
            f'<div class="kpi"><div class="lab">{e(kpi["label"])}</div>'
            f'<div class="val">{_fmt(value, kpi.get("fmt", "num2"))}</div>'
            f"{meter}"
            f'<div class="basis">{e(kpi.get("basis") or "not computed")}</div></div>'
        )
    return _section(
        "s-header",
        e(meta["run_name"]),
        "Every tile states the confidence and IoU it was computed at. They differ on "
        "purpose.",
        f'<div class="meta">{cells}</div><div class="kpis">{"".join(tiles)}</div>',
    )


def _model_card(blob: dict) -> str:
    card = blob["model_card"]
    banners = "".join(
        f'<div class="warn {e(w["severity"])}"><b>{SEVERITY_ICON.get(w["severity"], "!")}'
        f"</b><span>{e(w['text'])}</span></div>"
        for w in blob.get("warnings") or []
    )
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
        f"{banners}"
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


def _dataset(blob: dict) -> str:
    # Mask fill ratio and polygon vertex counts do not exist on a detect run -- the
    # label files carry four numbers per line and there is no polygon to measure.
    seg = bool(blob["meta"].get("is_seg"))
    body = (
        fig_block(
            blob,
            "f_class_bars",
            "Log axis, because the class you have to worry about is the sliver.",
            "The dataset label files were not readable, so composition is unavailable.",
        )
        + fig_block(blob, "f_split_stack", "Sorted so a zero-validation class is first.")
        + table_block(blob, "t_split_composition", "No split composition was recounted.")
        + fig_block(blob, "f_objects_per_image")
        + fig_block(blob, "f_box_size")
        + fig_block(blob, "f_aspect")
        + fig_block(blob, "f_center_heat")
        + fig_block(
            blob,
            "f_mask_fill",
            "A mass above 0.95 means the polygons are rectangles, and mask mAP measured "
            "against them is close to box mAP.",
            "No polygons were found in the label files, so the mask fill ratio could "
            "not be measured.",
            applicable=seg,
        )
        + fig_block(blob, "f_poly_vertices", applicable=seg)
        + "<h3>Annotation quality flags</h3>"
        + table_block(blob, "t_quality_flags", "No quality flags were raised.")
    )
    return _section(
        "s-dataset",
        "Dataset and split composition",
        "Read from the label files on disk, after filtering and after the split.",
        body,
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


def _confusion(blob: dict) -> str:
    meta = blob["meta"]
    caption = (
        f"conf = {meta['thresholds']['matrix_display']:g} (display), IoU 0.45 "
        "(ultralytics' fixed matrix threshold). Rows are ground truth, columns are "
        "predictions; the last row and column are background, i.e. false positives and "
        "false negatives. The background column stays as raw counts in both modes -- "
        "normalising it would print a count as if it were a rate."
    )
    if meta.get("is_seg"):
        caption += " The matrix is box-IoU based even in segment mode."
    toggle = (
        '<button id="cm-toggle" type="button" data-mode="counts">Show row-normalised'
        "</button>"
        if "f_confusion" in (blob.get("figures") or {})
        else ""
    )
    body = (
        toggle
        + fig_block(
            blob,
            "f_confusion",
            caption,
            "The confusion matrix requires `5_Testing/plots=True`.",
        )
        + "<h3>Most confused pairs</h3>"
        + table_block(blob, "t_confused_pairs", "No cross-class confusion was captured.")
    )
    return _section(
        "s-confusion",
        "Confusion matrix",
        "Which classes get mistaken for which, at the display threshold.",
        body,
    )


def _threshold(blob: dict) -> str:
    body = (
        fig_block(
            blob,
            "f_pr_f1_conf",
            "tau* is the F1-optimal threshold, and it is none of the other three "
            "confidences in this task.",
            "Curve data was not available from this validation pass.",
        )
        + fig_block(
            blob,
            "f_tp_fp_hist",
            "Two overlapping humps mean no threshold fixes precision -- the model or "
            "the labels need work, not the tuning.",
            "The TP-vs-FP split needs `8_Visualization/log_calibration`.",
        )
        + fig_block(
            blob,
            "f_reliability",
            "Computed over predictions only, so it is blind to false negatives. Read it "
            "next to recall.",
            "The reliability diagram needs `8_Visualization/log_calibration`.",
        )
    )
    return _section(
        "s-threshold",
        "Operating threshold",
        "Where to set the deploy threshold, and whether the scores mean anything.",
        body,
    )


def _tide(blob: dict) -> str:
    mode = blob.get("tide_mode", "none")
    title = {
        "delta_ap": "Error decomposition (dAP oracles)",
        "counts": "TIDE error counts (dAP oracles not computed)",
    }.get(mode, "Error decomposition (not captured)")
    body = fig_block(
        blob,
        "f_tide",
        "",
        "Not captured: the per-image wrapper did not run for this validation pass.",
    ) + table_block(blob, "t_tide", "Not captured for this run.")
    return _section(
        "s-tide",
        title,
        "Six error types, each with the AP an oracle that fixed only that error would "
        "recover.",
        body,
    )


def _strata(blob: dict) -> str:
    body = (
        fig_block(
            blob,
            "f_recall_by_size",
            "These are this dataset's own terciles, not COCO's 32x32 / 96x96 absolutes.",
            "Not captured: the per-image wrapper did not run.",
        )
        + fig_block(
            blob,
            "f_iou_hist_tp",
            "Mass piled just above 0.50 means localisation is the bottleneck -- it "
            "should agree with the Loc segment of the error decomposition.",
        )
        + fig_block(blob, "f_recall_vs_area")
        + fig_block(blob, "f_recall_by_ar")
    )
    return _section(
        "s-strata",
        "Size and shape",
        "Where the misses actually are.",
        body,
    )


def _boxmask(blob: dict) -> str:
    body = (
        fig_block(blob, "f_boxmask_class", "Box in slot 1, mask in slot 3.")
        + fig_block(
            blob,
            "f_boxmask_iou",
            "The mask axis is quantised to the ten IoU thresholds, because ultralytics "
            "discards the pairwise mask IoU matrix. Mass on the diagonal at high IoU is "
            "the rectangle-annotation signature.",
        )
        + fig_block(blob, "f_iou_overlay")
        + fig_block(blob, "f_boxmask_size")
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

    parts = []
    for gid, grid in (blob.get("grids") or {}).items():
        items = grid.get("items") or []
        if not items:
            parts.append(
                f"<h3>{e(grid['title'])}</h3>"
                f'<p class="missing">{e(grid.get("empty_reason") or "Nothing to show.")}'
                "</p>"
            )
            continue
        figs = []
        for i, item in enumerate(items):
            figs.append(
                f'<figure data-gid="{e(gid)}" data-idx="{i}" '
                f'data-outcome="{e(item["outcome"])}" '
                f'data-classes="{e(" ".join(item.get("classes") or []))}" '
                f'data-pairs="{e(" ".join(item.get("pairs") or []))}">'
                f'<img data-thumb="{e(item["thumb"])}" alt="" loading="lazy">'
                f"<figcaption>{e(item['label'])}</figcaption></figure>"
            )
        parts.append(
            f"<h3>{e(grid['title'])}</h3>"
            f'<p class="caption">{e(grid["subtitle"])} &mdash; {e(grid["basis"])}. '
            f'<span class="shown"></span></p>'
            f'<div class="grid">{"".join(figs)}</div>'
        )
    return _section(
        "s-galleries",
        "Galleries",
        "Colour is the outcome and nothing else: green solid is a true positive, red "
        "solid a false positive, amber dashed a miss. Click any thumbnail for the "
        "labelled overlay.",
        "".join(parts),
    )


def _training(blob: dict) -> str:
    body = (
        '<details class="lazy"><summary>Training curves</summary>'
        + fig_block(blob, "f_val_map", "", "Training history not found (`results.csv`).")
        + fig_block(blob, "f_losses")
        + "</details>"
    )
    return _section(
        "s-training",
        "Training history",
        "Downsampled to a fixed point count, which is what keeps report size "
        "independent of epoch count.",
        body,
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
    if only_galleries:
        sections = [_galleries(blob), _caveats(blob)]
        anchors = {"s-galleries", "s-caveats"}
    else:
        builders = [
            ("s-header", _header),
            ("s-model-card", _model_card),
            ("s-dataset", _dataset),
            ("s-per-class", _per_class),
            ("s-confusion", _confusion),
            ("s-threshold", _threshold),
            ("s-tide", _tide),
            ("s-strata", _strata),
        ]
        if blob["meta"].get("is_seg"):
            builders.append(("s-boxmask", _boxmask))
        builders += [
            ("s-galleries", lambda b: _galleries(b, gallery_link)),
            ("s-training", _training),
            ("s-caveats", _caveats),
        ]
        sections = [fn(blob) for _, fn in builders]
        anchors = {a for a, _ in builders}

    nav = "".join(
        f'<a href="#{a}">{e(label)}</a>' for a, label in SECTION_ORDER if a in anchors
    )
    title = f"Evaluation report - {blob['meta'].get('run_name', 'run')}"
    payload = json.dumps(blob, separators=(",", ":"), allow_nan=False, default=str)
    # `</script>` inside the JSON would close the element early; nothing else can.
    payload = payload.replace("</", "<\\/")

    document = _asset("template.html")
    for token, value in (
        ("{{TITLE}}", e(title)),
        ("{{CSS}}", theme.css_variables() + _asset("report.css")),
        ("{{NAV}}", nav),
        ("{{SECTIONS}}", "".join(sections)),
        ("{{BLOB}}", payload),
        ("{{PLOTLY_JS}}", _asset("plotly-cartesian.min.js")),
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
        "plotly_template": blob["plotly_template"],
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
