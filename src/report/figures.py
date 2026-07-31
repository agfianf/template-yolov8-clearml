"""Every Plotly figure the report draws, plus the payload trimming.

Two rules bind every builder here.

**Nothing raw ships.** Plotly bins in the browser, which means handing `go.Histogram` a
sample serialises the whole sample into the page. Every distribution below is binned in
Python -- most of them were already binned by the capture -- so a figure's payload is a
function of its bin count and nothing else. That is what makes report size independent
of dataset size.

**The layout template is hoisted.** `figure_payload` drops `layout.template` from every
figure; the JS assigns the shared one (light or dark) at render time. Serialising a
template per figure is roughly two thirds of a naive payload.

Palette slots refer to the fixed categorical order in `theme.py`. Status colours are
never used for a series and series colours are never used for an outcome.
"""

from __future__ import annotations
import json
import math

from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from src.report import theme
from src.utils.logging import get_logger


logger = get_logger(__name__)

MAX_HIST_BINS = 50
MAX_CM_CLASSES_DEFAULT = 60
BAR_RADIUS = 4


def figure_payload(fig: go.Figure) -> dict[str, Any]:
    """Serialise a figure to `{"data": …, "layout": …}` with the template removed."""
    payload = json.loads(pio.to_json(fig, remove_uids=True))
    payload.get("layout", {}).pop("template", None)
    return payload


def _bar(x, y, name=None, slot=1, **kw) -> go.Bar:
    return go.Bar(
        x=x,
        y=y,
        name=name,
        marker=dict(color=theme.categorical(slot), cornerradius=BAR_RADIUS),
        **kw,
    )


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str, **kw) -> go.Figure:
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, **kw)
    return fig


def _centres(hist: np.ndarray, lo: float, hi: float) -> list[float]:
    n = len(hist)
    edges = np.linspace(lo, hi, n + 1)
    return ((edges[:-1] + edges[1:]) / 2).tolist()


# --- section 3: dataset -------------------------------------------------------------


def class_bars(names: list[str], counts: list[int]) -> go.Figure:
    """Instances per class, rarest first, on a log axis.

    Log because the interesting datasets are the skewed ones: on a linear axis the class
    with 20 instances beside one with 20,000 is an invisible sliver -- and that is
    exactly the class whose mAP will be noise.
    """
    order = np.argsort(counts)
    fig = go.Figure(
        _bar(
            x=[counts[i] for i in order],
            y=[names[i] for i in order],
            orientation="h",
            hovertemplate="%{y}<br>%{x:,} instances<extra></extra>",
        )
    )
    _layout(
        fig,
        "Instances per class (rarest first)",
        "Instances (log scale)",
        "Class",
        xaxis_type="log",
        height=max(280, 18 * len(order) + 90),
    )
    return fig


def split_stack(
    names: list[str], per_split: dict[str, list[int]], order: list[int]
) -> go.Figure:
    """Per-class instance counts stacked across train / valid / test.

    Sorted so that a class absent from the validation split is the first row: that class
    is silently dropped from mAP, and this is the chart that shows it.
    """
    fig = go.Figure()
    for slot, split in zip((1, 3, 4), ("train", "valid", "test"), strict=True):
        counts = per_split.get(split)
        if not counts:
            continue
        fig.add_trace(
            _bar(
                x=[counts[i] for i in order],
                y=[names[i] for i in order],
                name=split,
                slot=slot,
                orientation="h",
                hovertemplate=f"%{{y}}<br>{split}: %{{x:,}}<extra></extra>",
            )
        )
    _layout(
        fig,
        "Instances per class by split",
        "Instances",
        "Class",
        barmode="stack",
        height=max(280, 18 * len(order) + 90),
    )
    return fig


def objects_per_image(counter: dict[int, int], p995: float) -> go.Figure:
    keys = sorted(counter)
    fig = go.Figure(
        _bar(
            x=keys,
            y=[counter[k] for k in keys],
            hovertemplate="%{x} objects<br>%{y:,} images<extra></extra>",
        )
    )
    _layout(fig, "Objects per image", "Objects in the image", "Images")
    if p995 > 0:
        fig.add_vline(
            x=p995,
            line_dash="dot",
            line_color=theme.STATUS["warning"],
            annotation_text=f"p99.5 = {p995:g}",
        )
    return fig


def box_size(hist: np.ndarray, terciles: tuple[float, float], imgsz: int) -> go.Figure:
    """sqrt(area) as a fraction of the image edge, with this dataset's terciles shaded."""
    centres = _centres(hist, 0.0, 1.0)
    fig = go.Figure(
        _bar(
            x=centres,
            y=hist.tolist(),
            hovertemplate=(
                "~%{x:.3f} of the image edge<br>%{y:,} instances<extra></extra>"
            ),
        )
    )
    _layout(
        fig,
        f"Object size (sqrt area / image edge; x imgsz={imgsz} for pixels)",
        "sqrt(area) / image edge",
        "Instances",
    )
    for edge in terciles:
        fig.add_vline(
            x=edge,
            line_dash="dot",
            line_color=theme.MUTED_LIGHT,
            annotation_text=f"{edge * imgsz:.0f}px",
        )
    return fig


def aspect_ratio(hist: np.ndarray) -> go.Figure:
    fig = go.Figure(
        _bar(
            x=_centres(hist, -3.0, 3.0),
            y=hist.tolist(),
            hovertemplate="log(w/h) ~%{x:.2f}<br>%{y:,} instances<extra></extra>",
        )
    )
    _layout(fig, "Aspect ratio, log(w/h)", "log(w / h)", "Instances")
    fig.add_vline(x=0, line_dash="dot", line_color=theme.MUTED_LIGHT)
    return fig


def centre_heatmap(heat: np.ndarray) -> go.Figure:
    n = heat.shape[0]
    coords = [(i + 0.5) / n for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=heat.tolist(),
            x=coords,
            y=coords,
            colorscale=theme.SEQUENTIAL_SCALE,
            hovertemplate="x %{x:.2f}, y %{y:.2f}<br>%{z:,} boxes<extra></extra>",
        )
    )
    _layout(
        fig,
        "Where objects sit in the frame",
        "x (fraction of width)",
        "y (fraction of height)",
        yaxis=dict(autorange="reversed"),
        height=380,
    )
    return fig


def mask_fill(hist: np.ndarray, rectangle_fraction: float) -> go.Figure:
    fig = go.Figure(
        _bar(
            x=_centres(hist, 0.0, 1.0),
            y=hist.tolist(),
            slot=3,
            hovertemplate="fill ~%{x:.2f}<br>%{y:,} instances<extra></extra>",
        )
    )
    _layout(
        fig,
        f"Mask fill ratio (mask area / box area) -- {rectangle_fraction:.1%} above 0.95",
        "mask area / bounding-box area",
        "Instances",
    )
    fig.add_vline(
        x=0.95,
        line_dash="dot",
        line_color=theme.STATUS["warning"],
        annotation_text="rectangle-polygon",
    )
    return fig


def polygon_vertices(counter: dict[int, int]) -> go.Figure:
    keys = sorted(counter)
    fig = go.Figure(
        _bar(
            x=keys,
            y=[counter[k] for k in keys],
            slot=3,
            hovertemplate="%{x} vertices<br>%{y:,} polygons<extra></extra>",
        )
    )
    _layout(fig, "Polygon vertex count", "Vertices", "Polygons")
    return fig


# --- section 5: confusion -----------------------------------------------------------


def confusion(labels: list[str], counts: np.ndarray) -> dict[str, Any]:
    """Draw the matrix as counts, and carry a row-normalised `alt_z` for the toggle.

    The background column is left as raw counts in both modes. Normalising it would
    divide the misses of one class by that class's total and print it beside genuine
    per-class rates, which reads as a probability and is not one.
    """
    z = np.asarray(counts, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rows = z.sum(axis=1, keepdims=True)
        alt = np.where(rows > 0, z / rows, 0.0)
    if z.shape[1] == len(labels) and labels and labels[-1] == "background":
        alt[:, -1] = z[:, -1]
    fig = go.Figure(
        go.Heatmap(
            z=z.tolist(),
            x=labels,
            y=labels,
            colorscale=theme.SEQUENTIAL_SCALE,
            hovertemplate="actual %{y}<br>predicted %{x}<br>%{z}<extra></extra>",
        )
    )
    _layout(
        fig,
        "Confusion matrix (counts)",
        "Predicted",
        "Actual",
        height=max(360, 16 * len(labels) + 140),
        yaxis=dict(autorange="reversed"),
    )
    payload = figure_payload(fig)
    payload["alt_z"] = np.round(alt, 4).tolist()
    return payload


# --- section 6: threshold -----------------------------------------------------------


def pr_f1_vs_conf(curves: list[dict], tau: float | None) -> go.Figure | None:
    """Mean precision, recall and F1 against confidence, with tau* marked."""
    wanted = {
        "Precision-Confidence(B)": ("Precision", 1),
        "Recall-Confidence(B)": ("Recall", 2),
        "F1-Confidence(B)": ("F1", 3),
    }
    fig = go.Figure()
    drawn = 0
    for curve in curves or []:
        entry = wanted.get(str(curve.get("name")))
        if entry is None:
            continue
        label, slot = entry
        series = curve.get("series") or []
        if not series:
            continue
        mean = np.mean([s["y"] for s in series], axis=0)
        fig.add_trace(
            go.Scatter(
                x=curve["x"],
                y=mean.tolist(),
                mode="lines",
                name=label,
                line=dict(color=theme.categorical(slot), width=2),
                hovertemplate=f"{label} %{{y:.3f}} at conf %{{x:.3f}}<extra></extra>",
            )
        )
        drawn += 1
    if not drawn:
        return None
    _layout(
        fig,
        "Precision, recall and F1 against confidence (mean over classes)",
        "Confidence",
        "Value",
        yaxis=dict(range=[0, 1.05]),
    )
    if tau is not None:
        fig.add_vline(
            x=tau,
            line_dash="dot",
            line_color=theme.MUTED_LIGHT,
            annotation_text=f"tau* = {tau:.3f}",
        )
    return fig


def tp_fp_hist(centres: list[float], tp: list[int], fp: list[int]) -> go.Figure:
    """Overlay matched and unmatched confidence as densities, so FP mass is visible."""
    fig = go.Figure()
    for values, label, colour in (
        (tp, "Matched (TP)", theme.STATUS["good"]),
        (fp, "Unmatched (FP)", theme.STATUS["critical"]),
    ):
        total = sum(values) or 1
        fig.add_trace(
            go.Bar(
                x=centres,
                y=[v / total for v in values],
                name=label,
                opacity=0.65,
                marker=dict(color=colour),
                hovertemplate=f"{label}<br>conf ~%{{x:.2f}}<br>%{{y:.3f}}<extra></extra>",
            )
        )
    _layout(
        fig,
        "Detection confidence, matched vs unmatched (normalised density)",
        "Confidence",
        "Share of that population",
        barmode="overlay",
    )
    return fig


def reliability(calibration: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(dash="dash", color=theme.MUTED_LIGHT, width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=calibration["mean_confidence"],
            y=calibration["precision"],
            mode="lines+markers",
            name="Observed",
            line=dict(color=theme.categorical(1), width=2),
            marker=dict(size=8),
            text=[f"n={c}" for c in calibration["count"]],
            hovertemplate=(
                "confidence %{x:.3f}<br>precision %{y:.3f}<br>%{text}<extra></extra>"
            ),
        )
    )
    _layout(
        fig,
        (
            f"Reliability ({calibration.get('basis', 'box')} IoU@0.50), "
            f"D-ECE={calibration['ece']:.4f} over conf >= 0.001 "
            "-- ignores false negatives, read with recall"
        ),
        "Mean predicted confidence",
        "Observed precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


# --- section 7: TIDE ----------------------------------------------------------------


def tide_bar(tide: dict[str, Any]) -> go.Figure:
    """One stacked bar, six segments, every segment directly labelled.

    Direct labels are mandatory rather than decorative: slots 4 and 5 fail contrast
    against the light surface, so the numbers -- not the colours -- have to carry the
    reading.
    """
    fig = go.Figure()
    deltas = tide.get("delta_ap") or {}
    counts = tide.get("counts") or {}
    use_delta = tide.get("mode") == "delta_ap"
    for slot, kind in enumerate(tide.get("types") or [], start=1):
        value = abs(float(deltas.get(kind, 0.0))) if use_delta else counts.get(kind, 0)
        fig.add_trace(
            go.Bar(
                x=[value],
                y=["errors"],
                name=kind,
                orientation="h",
                marker=dict(
                    color=theme.categorical(slot),
                    line=dict(color=theme.SURFACE_LIGHT, width=1),
                ),
                text=[f"{kind} {value:.3f}" if use_delta else f"{kind} {value:,}"],
                textposition="inside",
                hovertemplate=(
                    f"{kind}<br>%{{x}}<br>{counts.get(kind, 0):,} detections"
                    "<extra></extra>"
                ),
            )
        )
    _layout(
        fig,
        (
            "Error decomposition by dAP50 (independent oracles)"
            if use_delta
            else "Error decomposition by count (dAP oracles not computed)"
        ),
        "dAP50 recoverable" if use_delta else "Errors",
        "",
        barmode="stack",
        height=320,
        showlegend=True,
        # Room above for the two ceiling annotations and below for the axis title with
        # the legend under it; at the template's default margins the two collide.
        margin=dict(l=70, r=30, t=88, b=104),
        # A horizontal stacked bar reverses its legend by default, which put Miss first
        # and read as if the bar were drawn right to left.
        legend=dict(
            traceorder="normal",
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.42,
            yanchor="top",
        ),
    )
    # The two ceilings routinely land within a few hundredths of each other, so they get
    # two separate defences: a vertical stagger, and an anchor chosen by value so each
    # label runs *away* from the other line rather than across it.
    ceilings = [
        (label, abs(float(value)))
        for label, key in (
            ("dAP if all FP removed", "fp"),
            ("dAP if all FN removed", "fn"),
        )
        if use_delta and (value := (tide.get("ceilings") or {}).get(key))
    ]
    ceilings.sort(key=lambda c: c[1])
    for i, (label, value) in enumerate(ceilings):
        last = i == len(ceilings) - 1
        fig.add_vline(
            x=value,
            line_dash="dot",
            line_color=theme.MUTED_LIGHT,
            annotation_text=f"{label}: {value:.3f}",
            # Smaller value's label extends left, larger value's extends right.
            annotation_position="top right" if last else "top left",
            annotation_yshift=6 if last else 26,
            annotation_font=dict(size=11, color=theme.MUTED_LIGHT),
        )
    return fig


# --- section 8: strata --------------------------------------------------------------


def recall_by_size(
    buckets: list[str], recalls: list[float], supports: list[int]
) -> go.Figure:
    fig = go.Figure(
        _bar(
            x=buckets,
            y=recalls,
            text=[f"{r:.2f} (n={n:,})" for r, n in zip(recalls, supports, strict=True)],
            textposition="outside",
            hovertemplate="%{x}<br>recall %{y:.3f}<extra></extra>",
        )
    )
    _layout(
        fig,
        "Recall by object size (this dataset's terciles)",
        "Size bucket",
        "Recall at IoU 0.50",
        yaxis=dict(range=[0, 1.1]),
    )
    return fig


def iou_hist(hist: np.ndarray) -> go.Figure:
    fig = go.Figure(
        _bar(
            x=_centres(hist, 0.0, 1.0),
            y=hist.tolist(),
            hovertemplate="IoU ~%{x:.2f}<br>%{y:,} matches<extra></extra>",
        )
    )
    _layout(fig, "IoU of matched detections", "Box IoU", "Matches")
    fig.add_vline(
        x=0.5,
        line_dash="dot",
        line_color=theme.STATUS["warning"],
        annotation_text="match threshold",
    )
    return fig


def recall_vs_area(edges: list[float], recalls: list[float], imgsz: int) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=edges,
            y=recalls,
            mode="lines+markers",
            line=dict(color=theme.categorical(1), width=2),
            marker=dict(size=8),
            hovertemplate="~%{x:.0f}px<br>recall %{y:.3f}<extra></extra>",
        )
    )
    _layout(
        fig,
        f"Recall against object size (px at imgsz={imgsz})",
        "sqrt(area) in px at imgsz",
        "Recall at IoU 0.50",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def recall_by_ar(buckets: list[str], recalls: list[float]) -> go.Figure:
    fig = go.Figure(
        _bar(
            x=buckets,
            y=recalls,
            text=[f"{r:.2f}" for r in recalls],
            textposition="outside",
            hovertemplate="%{x}<br>recall %{y:.3f}<extra></extra>",
        )
    )
    _layout(
        fig,
        "Recall by aspect ratio",
        "log(w / h) bucket",
        "Recall at IoU 0.50",
        yaxis=dict(range=[0, 1.1]),
    )
    return fig


# --- section 9: box vs mask ---------------------------------------------------------


def boxmask_class(names: list[str], box: list[float], mask: list[float]) -> go.Figure:
    fig = go.Figure()
    for values, label, slot in ((box, "Box mAP50-95", 1), (mask, "Mask mAP50-95", 3)):
        fig.add_trace(
            _bar(
                x=names,
                y=values,
                name=label,
                slot=slot,
                hovertemplate=f"%{{x}}<br>{label} %{{y:.3f}}<extra></extra>",
            )
        )
    _layout(
        fig,
        "Box against mask mAP, per class",
        "Class",
        "mAP50-95",
        barmode="group",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def boxmask_iou(hist: np.ndarray) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=hist.T.tolist(),
            x=_centres(np.zeros(hist.shape[0]), 0.0, 1.0),
            y=list(range(hist.shape[1])),
            colorscale=theme.SEQUENTIAL_SCALE,
            hovertemplate=(
                "box IoU ~%{x:.2f}<br>mask IoU levels matched %{y}"
                "<br>%{z:,} instances<extra></extra>"
            ),
        )
    )
    _layout(
        fig,
        "Box IoU against mask IoU level (mask axis is quantised to the 10 thresholds)",
        "Box IoU (exact)",
        "Mask IoU thresholds matched (0-10)",
        height=380,
    )
    return fig


def iou_overlay(box_hist: np.ndarray, mask_levels: np.ndarray) -> go.Figure:
    fig = go.Figure()
    total_box = int(box_hist.sum()) or 1
    fig.add_trace(
        go.Bar(
            x=_centres(box_hist, 0.0, 1.0),
            y=(box_hist / total_box).tolist(),
            name="Box IoU (exact)",
            opacity=0.7,
            marker=dict(color=theme.categorical(1)),
            hovertemplate="box IoU ~%{x:.2f}<br>%{y:.3f}<extra></extra>",
        )
    )
    total_mask = int(mask_levels.sum()) or 1
    fig.add_trace(
        go.Bar(
            x=[i / (len(mask_levels) - 1) for i in range(len(mask_levels))],
            y=(mask_levels / total_mask).tolist(),
            name="Mask IoU (quantised)",
            opacity=0.7,
            marker=dict(color=theme.categorical(3)),
            hovertemplate="mask level %{x:.2f}<br>%{y:.3f}<extra></extra>",
        )
    )
    _layout(
        fig,
        "Box and mask IoU distributions, matched instances",
        "IoU",
        "Share of matches",
        barmode="overlay",
    )
    return fig


def boxmask_by_size(buckets: list[str], deltas: list[float]) -> go.Figure:
    colours = [
        theme.DIVERGING_LIGHT[0] if d >= 0 else theme.DIVERGING_LIGHT[-1] for d in deltas
    ]
    fig = go.Figure(
        go.Bar(
            x=buckets,
            y=deltas,
            marker=dict(color=colours, cornerradius=BAR_RADIUS),
            text=[f"{d:+.3f}" for d in deltas],
            textposition="outside",
            hovertemplate="%{x}<br>mask - box %{y:+.3f}<extra></extra>",
        )
    )
    _layout(fig, "Mask minus box IoU by size bucket", "Size bucket", "Mask - box")
    return fig


# --- section 11: training curves ----------------------------------------------------


def val_map_curve(epochs: list[int], series: dict[str, list[float]]) -> go.Figure:
    fig = go.Figure()
    for slot, (name, values) in zip((1, 3, 2, 4), series.items(), strict=False):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=theme.categorical(slot), width=2),
                hovertemplate=f"{name}<br>epoch %{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    _layout(fig, "Validation mAP against epoch", "Epoch", "mAP")
    return fig


def loss_curves(epochs: list[int], series: dict[str, list[float]]) -> go.Figure:
    fig = go.Figure()
    for slot, (name, values) in enumerate(series.items(), start=1):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=theme.categorical(slot), width=2),
                hovertemplate=f"{name}<br>epoch %{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    _layout(fig, "Training losses against epoch", "Epoch", "Loss")
    return fig


def percentile_from_counter(counter: dict[int, int], q: float) -> float:
    """Return the q-quantile of a value->count histogram, without building a sample."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    target = q * total
    seen = 0
    for key in sorted(counter):
        seen += counter[key]
        if seen >= target:
            return float(key)
    return float(max(counter))


def safe(builder, *args, **kwargs) -> dict[str, Any] | None:
    """Run one figure builder, returning None instead of raising.

    A figure that cannot be built costs its own card, never the report: the blob has to
    stay valid JSON no matter which builder ran into a shape it did not expect.
    """
    try:
        fig = builder(*args, **kwargs)
    except Exception as e:
        logger.warning("figure %s failed: %s", getattr(builder, "__name__", "?"), e)
        return None
    if fig is None:
        return None
    if isinstance(fig, dict):
        return fig
    try:
        return figure_payload(fig)
    except Exception as e:
        logger.warning("could not serialise figure %s: %s", builder.__name__, e)
        return None


def nan_to_none(values) -> list:
    """Replace NaN with None so the payload is valid JSON rather than `NaN`."""
    return [
        None if v is None or (isinstance(v, float) and math.isnan(v)) else v
        for v in values
    ]
