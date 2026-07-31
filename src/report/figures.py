"""Every figure the report draws, hand-authored as inline SVG.

Four rules bind every builder here.

**Nothing raw ships.** Every distribution below is binned in Python -- most of them were
already binned by the capture -- so a figure's payload is a function of its bin count and
nothing else. That is what makes report size independent of dataset size.

**Nothing is drawn client-side.** A builder returns a finished `<svg>` string which
`render.py` inlines into the document. There is no chart library in the page, the figures
are in the DOM before any script runs, and they survive printing and a blocked-script
iframe. The cost is DOM nodes, which is why the two matrices are the only figures whose
mark count is quadratic and why both are capped.

**Zero hex inside an SVG.** Colour is always `class="f-cyan"` or `fill="var(--…)"`, never
a literal, so the theme toggle reaches the charts. `report.css` owns the palette; this
module owns geometry.

**Every mark carries its number.** Direct labels, not hover -- with one exception: where
labelling every mark would bury the shape it describes (histograms, the heatmaps), only
the peak and the reference lines are labelled and the rest is left to `data-tip`.

Builder signatures are the seam with `blob.py` and do not change; only the return type
does. `safe()` still swallows every exception and returns `None`, which is what the
degradation tests assert.
"""

from __future__ import annotations
import html
import math

from typing import Any

import numpy as np

from src.utils.logging import get_logger


logger = get_logger(__name__)

MAX_HIST_BINS = 50
MAX_CM_CLASSES_DEFAULT = 60

# One viewBox width for every full-width chart, so a column of figures shares a left
# edge and a tick rhythm. Height is per figure; `width:100%` scales both.
VB_W = 660
PLOT_L, PLOT_R = 52, 616

# The sequential ramp is seven steps of one hue (`.hm-0` .. `.hm-6` in report.css).
RAMP_STEPS = 6
# Above this many rows a matrix cell is too small to hold a legible number, so the
# counts are left to the tooltip rather than printed at 4px.
CM_LABEL_MAX = 14
# Bars thinner than this cannot carry a label between their neighbours.
LABEL_MIN_BAND = 13.0


# --- primitives ---------------------------------------------------------------------


def _e(value: Any) -> str:
    """Escape one value for text content and attribute values alike."""
    return html.escape("" if value is None else str(value), quote=True)


def _n(value: float) -> str:
    return f"{value:,.0f}"


def _compact(value: float) -> str:
    """Axis ticks: 12,000 is four glyphs too many under a 30px-wide bar."""
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= cut:
            return f"{value / cut:g}{suffix}"
    return f"{value:,g}"


def _round_step(span: float, target: int) -> float:
    """Return a 1 / 2 / 2.5 / 5 x 10^k step, about `target` intervals over `span`."""
    raw = max(span, 1e-12) / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    return next((m * mag for m in (1, 2, 2.5, 5) if m * mag >= raw), 10 * mag)


def _nice(hi: float, target: int = 4) -> tuple[float, list[float]]:
    """Return `(top, ticks)` covering `hi` with a round step.

    Chosen rather than `hi` itself so the gridlines land on numbers a reader can add up.
    """
    if not math.isfinite(hi) or hi <= 0:
        return 1.0, [0.0, 1.0]
    step = _round_step(hi, target)
    count = max(1, math.ceil(hi / step))
    return step * count, [step * i for i in range(count + 1)]


def _ramp(value: float, vmax: float) -> int:
    """Map one cell value onto a ramp step, with 0 reserved for an empty cell."""
    if value <= 0 or vmax <= 0:
        return 0
    return min(RAMP_STEPS, 1 + int(value / vmax * (RAMP_STEPS - 0.001)))


def _rect(
    cls: str, x: float, y: float, w: float, h: float, rx: float = 1.5, tip: str = ""
) -> str:
    attr = f' data-tip="{_e(tip)}"' if tip else ""
    return (
        f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" '
        f'height="{max(h, 0):.1f}" rx="{rx:g}"{attr}></rect>'
    )


def _text(cls: str, x: float, y: float, text: Any, anchor: str = "") -> str:
    at = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}"{at}>{_e(text)}</text>'


def _vline(cls: str, x: float, y1: float, y2: float, tip: str = "") -> str:
    attr = f' data-tip="{_e(tip)}"' if tip else ""
    return (
        f'<line class="{cls}" x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" '
        f'y2="{y2:.1f}"{attr}></line>'
    )


def _hline(cls: str, x1: float, x2: float, y: float) -> str:
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}"/>'


class _Chart:
    """One SVG canvas: a plot rectangle, two scales and a list of parts.

    Exists so a builder reads as the chart it draws rather than as arithmetic. Every
    coordinate below goes through `sx`/`sy`, which is also what keeps a log axis from
    leaking into six separate places.
    """

    def __init__(
        self,
        aria: str,
        *,
        height: float,
        width: float = VB_W,
        left: float = PLOT_L,
        right: float = PLOT_R,
        top: float = 26,
        bottom: float | None = None,
    ) -> None:
        self.aria = aria
        self.w, self.h = width, height
        self.x0, self.x1 = left, right
        self.yt = top
        self.yb = height - 42 if bottom is None else bottom
        self.parts: list[str] = []
        self._x: tuple[float, float, bool] = (0.0, 1.0, False)
        self._y: tuple[float, float] = (0.0, 1.0)

    # --- scales ---------------------------------------------------------------------
    def xdomain(self, lo: float, hi: float, log: bool = False) -> _Chart:
        self._x = (lo, hi, log)
        return self

    def ydomain(self, lo: float, hi: float) -> _Chart:
        self._y = (lo, hi)
        return self

    def sx(self, value: float) -> float:
        lo, hi, log = self._x
        if log:
            value, lo, hi = (math.log10(max(v, 1.0)) for v in (value, lo, hi))
        span = (hi - lo) or 1.0
        return self.x0 + (value - lo) / span * (self.x1 - self.x0)

    def sy(self, value: float) -> float:
        lo, hi = self._y
        span = (hi - lo) or 1.0
        return self.yb - (value - lo) / span * (self.yb - self.yt)

    @property
    def band(self) -> float:
        return self.x1 - self.x0

    # --- furniture ------------------------------------------------------------------
    def add(self, *parts: str) -> _Chart:
        self.parts.extend(parts)
        return self

    def grid_y(self, ticks, fmt=_compact) -> _Chart:
        """Horizontal gridlines with their labels. The zero line carries the axis."""
        for tick in ticks:
            y = self.sy(tick)
            self.add(_hline("gl" if tick == 0 else "gl-2", self.x0, self.x1, y))
            self.add(_text("tk", self.x0 - 9, y + 4, fmt(tick), "end"))
        return self

    def grid_x(self, ticks, fmt=_compact, y0=None, y1=None) -> _Chart:
        """Vertical gridlines, for the horizontal-bar charts. Labels sit under them."""
        top = self.yt if y0 is None else y0
        bottom = self.yb if y1 is None else y1
        for tick in ticks:
            x = self.sx(tick)
            self.add(_vline("gl" if tick == ticks[0] else "gl-2", x, top, bottom))
            self.add(_text("tk", x, bottom + 15, fmt(tick), "middle"))
        return self

    def auto_ticks_x(self, target: int = 6, fmt=_compact) -> _Chart:
        """Ticks inside the x domain, whatever it starts at.

        A tick list generated from the maximum alone printed `0` and `150` outside a
        plot that ran from 8 to 128 -- two labels pointing at nothing.
        """
        lo, hi, _ = self._x
        step = _round_step(hi - lo, target)
        first = math.ceil(lo / step - 1e-9)
        last = math.floor(hi / step + 1e-9)
        return self.ticks_x([step * k for k in range(first, last + 1)], fmt)

    def ticks_x(self, values, fmt=_compact) -> _Chart:
        for value in values:
            self.add(_text("tk", self.sx(value), self.yb + 18, fmt(value), "middle"))
        return self

    def axis(self, left: str, right: str = "", dy: float = 36) -> _Chart:
        self.add(_text("axl", self.x0, self.yb + dy, left))
        if right:
            self.add(_text("axl", self.x1, self.yb + dy, right, "end"))
        return self

    def vref(
        self, value: float, label: str, anchor: str = "end", tip: str = ""
    ) -> _Chart:
        """Draw a dashed reference line, labelled 11px away from the plot."""
        x = self.sx(value)
        self.add(_vline("ref", x, self.yt - 4, self.yb, tip or label))
        offset = -7 if anchor == "end" else 7
        self.add(_text("dl f-peer", x + offset, self.yt + 2, label, anchor))
        return self

    def render(self) -> str:
        return (
            f'<svg class="chart" viewBox="0 0 {self.w:g} {self.h:g}" role="img" '
            f'aria-label="{_e(self.aria)}">{"".join(self.parts)}</svg>'
        )


def _bars(chart: _Chart, counts, tip, cls: str = "f-cyan", width: float = 0.66) -> int:
    """Draw `counts` as evenly spaced bars over the plot width; return the peak index.

    Every bar gets a full-height invisible hit target rather than relying on the bar
    itself: a bar two pixels tall is not a hover target, and those are exactly the bins
    a reader wants the count for.
    """
    values = list(counts)
    if not values:
        return -1
    slot = chart.band / len(values)
    bw = slot * width
    for i, count in enumerate(values):
        x = chart.x0 + i * slot + (slot - bw) / 2
        if count:
            y = min(chart.sy(count), chart.yb - 1.5)
            chart.add(_rect(cls, x, y, bw, chart.yb - y, 1))
        chart.add(
            _rect(
                "hit",
                chart.x0 + i * slot,
                chart.yt - 8,
                slot,
                chart.yb - chart.yt + 8,
                0,
                tip(i, count),
            )
        )
    return max(range(len(values)), key=lambda i: values[i])


def _peak_label(chart: _Chart, counts, index: int, cls: str = "f-cyan") -> None:
    """Label the modal bar only. Thirty printed numbers bury the shape they describe."""
    values = list(counts)
    if index < 0 or not values or not values[index]:
        return
    slot = chart.band / len(values)
    x = chart.x0 + index * slot + slot / 2
    # Always above the cap, never inside it: a bin is a few pixels wide, so a label
    # centred in the bar overflows on both sides and prints paper-on-paper. The top
    # gridline is the tallest the bar can be, and the row above it is empty by design.
    chart.add(
        _text(f"dl {cls}", x, chart.sy(values[index]) - 7, _n(values[index]), "middle")
    )


def _rows(count: int, bh: float, gap: float, top: float = 16) -> tuple[float, float]:
    """Height of a horizontal-bar chart and its baseline, for `count` rows."""
    bottom = top + count * bh + max(count - 1, 0) * gap + 4
    return bottom + 40, bottom


def _hbars(
    chart: _Chart, rows, top: float, bh: float, gap: float, label_cls: str = "b-lbl"
) -> None:
    """Rows of `(name, value, class, label, tip)`, each labelled at its bar end."""
    for i, (name, value, cls, label, tip) in enumerate(rows):
        y = top + i * (bh + gap)
        w = max(chart.sx(value) - chart.x0, 1.5 if value > 0 else 0.0)
        if w > 0:
            chart.add(_rect(cls, chart.x0, y, w, bh, 2, tip))
        chart.add(_text(label_cls, chart.x0 - 10, y + bh / 2 + 4, name, "end"))
        chart.add(_text(f"dl {cls}", chart.x0 + w + 8, y + bh / 2 + 4, label))


def _centres(hist, lo: float, hi: float) -> list[float]:
    n = len(hist)
    edges = np.linspace(lo, hi, n + 1)
    return ((edges[:-1] + edges[1:]) / 2).tolist()


def _bin_edges(i: int, n: int, lo: float, hi: float) -> tuple[float, float]:
    step = (hi - lo) / max(n, 1)
    return lo + i * step, lo + (i + 1) * step


def figure_payload(svg: str) -> dict[str, Any]:
    """Wrap one finished SVG in the payload `render.py` inlines."""
    if not isinstance(svg, str):
        raise TypeError(f"builder returned {type(svg).__name__}, not an SVG string")
    return {"svg": svg}


# --- section 3: dataset -------------------------------------------------------------


def class_bars(names: list[str], counts: list[int]) -> str:
    """Instances per class, rarest first, on a log axis.

    Log because the interesting datasets are the skewed ones: on a linear axis the class
    with 20 instances beside one with 20,000 is an invisible sliver -- and that is
    exactly the class whose mAP will be noise.
    """
    order = sorted(range(len(names)), key=lambda i: counts[i])
    bh, gap = (24, 18) if len(order) <= 12 else (13, 7)
    height, bottom = _rows(len(order), bh, gap)
    top = float(counts and max(counts) or 1)
    decades = max(1, math.ceil(math.log10(max(top, 10))))
    chart = _Chart(
        f"Instances per class, {len(order)} classes, log scale",
        height=height + 12,
        left=118,
        bottom=bottom,
    ).xdomain(1, 10**decades, log=True)
    ticks = [10**d for d in range(decades + 1)]
    chart.grid_x(ticks, _compact, y0=8, y1=bottom)
    rows = []
    for i in order:
        value = int(counts[i])
        rows.append(
            (
                names[i],
                value,
                "f-cyan" if value else "f-peer",
                _n(value),
                f"{names[i]} - {_n(value)} instances",
            )
        )
    _hbars(chart, rows, 16, bh, gap)
    chart.add(_text("axl", 118, bottom + 33, "instances (log scale)"))
    return chart.render()


def split_stack(
    names: list[str], per_split: dict[str, list[int]], order: list[int]
) -> str:
    """Per-class instance counts stacked across train / valid / test.

    Sorted so that a class absent from the validation split is the first row: that class
    is silently dropped from mAP, and this is the chart that shows it.
    """
    splits = [s for s in ("train", "valid", "test") if per_split.get(s)]
    styles = {"train": "f-cyan", "valid": "f-teal", "test": "f-g1"}
    totals = [sum(per_split[s][i] for s in splits) for i in order]
    bh, gap = (24, 18) if len(order) <= 12 else (13, 7)
    height, bottom = _rows(len(order), bh, gap)
    chart = _Chart(
        "Instances per class by split",
        height=height + 12,
        left=118,
        bottom=bottom,
    ).xdomain(0, max(totals) or 1)
    top, ticks = _nice(max(totals) or 1)
    chart.xdomain(0, top).grid_x(ticks, _compact, y0=8, y1=bottom)
    for row, i in enumerate(order):
        y = 16 + row * (bh + gap)
        x = chart.x0
        for split in splits:
            value = int(per_split[split][i])
            if value <= 0:
                continue
            w = max(chart.sx(value) - chart.x0, 1.0)
            tip = f"{names[i]} - {split} {_n(value)}"
            chart.add(_rect(styles[split], x, y, max(w - 1.5, 0.8), bh, 1.5, tip))
            if w > LABEL_MIN_BAND * 2.4 and bh >= 16:
                chart.add(
                    _text("seg-t f-paper", x + 6, y + bh / 2 + 4, f"{split} {_n(value)}")
                )
            x += w
        chart.add(_text("b-lbl", chart.x0 - 10, y + bh / 2 + 4, names[i], "end"))
        chart.add(_text("dl f-ink2", x + 8, y + bh / 2 + 4, _n(totals[row])))
    legend = " · ".join(splits)
    chart.add(_text("axl", 118, bottom + 33, f"instances · stacked {legend}"))
    return chart.render()


def objects_per_image(counter: dict[int, int], p995: float) -> str:
    hi = max(counter) if counter else 0
    bins = min(30, max(hi + 1, 1))
    width = max(1.0, (hi + 1) / bins)
    counts = [0] * bins
    for key, value in counter.items():
        counts[min(int(key / width), bins - 1)] += int(value)
    top, ticks = _nice(max(counts) or 1)
    chart = _Chart(
        f"Objects per image, {_n(sum(counter.values()))} images",
        height=198,
    )
    chart.xdomain(0, bins * width).ydomain(0, top).grid_y(ticks)
    peak = _bars(
        chart,
        counts,
        lambda i, c: (
            f"{_bin_edges(i, bins, 0, bins * width)[0]:.0f}-"
            f"{_bin_edges(i, bins, 0, bins * width)[1]:.0f} objects · {_n(c)} images"
        ),
    )
    _peak_label(chart, counts, peak)
    if p995 > 0:
        chart.vref(min(p995, bins * width), f"p99.5 = {p995:g}")
    chart.auto_ticks_x(6)
    chart.axis(
        f"objects per image · bin width {width:g}",
        f"{_n(sum(counter.values()))} images",
    )
    return chart.render()


def box_size(hist: np.ndarray, terciles: tuple[float, float], imgsz: int) -> str:
    """sqrt(area) as a fraction of the image edge, with this dataset's terciles shaded."""
    counts = [int(v) for v in hist]
    top, ticks = _nice(max(counts) or 1)
    chart = _Chart("Object size as a fraction of the image edge", height=198)
    chart.xdomain(0, 1).ydomain(0, top).grid_y(ticks)
    peak = _bars(
        chart,
        counts,
        lambda i, c: (
            f"{_bin_edges(i, len(counts), 0, 1)[0]:.02f}-"
            f"{_bin_edges(i, len(counts), 0, 1)[1]:.02f} of the edge "
            f"· {_n(c)} instances"
        ),
    )
    _peak_label(chart, counts, peak)
    for edge, anchor in zip(terciles, ("end", "start"), strict=False):
        chart.vref(
            edge,
            f"{edge * imgsz:.0f}px",
            anchor,
            f"tercile at {edge:.3f} of the edge · {edge * imgsz:.0f}px at imgsz {imgsz}",
        )
    chart.ticks_x([0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:g}")
    chart.axis(
        f"sqrt(area) / image edge · x imgsz {imgsz} for pixels",
        "instances",
    )
    return chart.render()


def aspect_ratio(hist: np.ndarray) -> str:
    counts = [int(v) for v in hist]
    top, ticks = _nice(max(counts) or 1)
    chart = _Chart("Object aspect ratio, log width over height", height=198)
    chart.xdomain(-3, 3).ydomain(0, top).grid_y(ticks)
    peak = _bars(
        chart,
        counts,
        lambda i, c: (
            f"log(w/h) {_bin_edges(i, len(counts), -3, 3)[0]:.2f} to "
            f"{_bin_edges(i, len(counts), -3, 3)[1]:.2f} · {_n(c)} instances"
        ),
    )
    _peak_label(chart, counts, peak)
    chart.vref(0.0, "square", "start", "log(w/h) = 0 · width equals height")
    chart.ticks_x([-3, -2, -1, 0, 1, 2, 3], lambda v: f"{v:g}")
    chart.axis("log(w / h) · negative is tall, positive is wide", "instances")
    return chart.render()


def centre_heatmap(heat: np.ndarray) -> str:
    """Where box centres sit in the frame. One hue, so the reading is never category."""
    grid = _pool(np.asarray(heat), 16)
    ny, nx = grid.shape
    cell = 25.0
    gw, gh = nx * cell - 1, ny * cell - 1
    # Centred in the shared viewBox rather than sized to fit it: a square grid stretched
    # to a 660-wide box would draw a landscape frame, which is a claim about the images.
    x0, yt = max(34.0, (VB_W - gw) / 2), 10.0
    vmax = float(grid.max()) if grid.size else 0.0
    chart = _Chart(
        "Density of box centres over the frame",
        height=yt + gh + 58,
        bottom=yt + gh,
    )
    for r in range(ny):
        for c in range(nx):
            value = int(grid[r, c])
            tip = f"column {c + 1} of {nx}, row {r + 1} of {ny} · {_n(value)} box centres"
            chart.add(
                _rect(
                    f"hm-{_ramp(value, vmax)}",
                    x0 + c * cell,
                    yt + r * cell,
                    cell - 1,
                    cell - 1,
                    1,
                    tip,
                )
            )
    for c, r, value in _hot_cells(grid, 3):
        chart.add(
            _text(
                "hm-lbl",
                x0 + c * cell + (cell - 1) / 2,
                yt + r * cell + cell / 2 + 3,
                _compact(value),
                "middle",
            )
        )
    chart.add(_text("axl", x0, yt + gh + 18, "left edge"))
    chart.add(_text("axl", x0 + gw, yt + gh + 18, "right edge", "end"))
    mid = yt + gh / 2
    chart.add(
        f'<text class="axl" x="14" y="{mid:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {mid:.0f})">top to bottom of frame</text>'
    )
    chart.add(*_ramp_legend(x0, yt + gh + 36, f"{_n(vmax)} box centres in one cell"))
    return chart.render()


def _pool(grid: np.ndarray, limit: int) -> np.ndarray:
    """Halve a histogram grid until it is at most `limit` cells across.

    A 32x32 grid is 1,024 rects in the document and its cells are too small to hold the
    three counts that make the densest cells readable. Pooling is lossless as a sum and
    a sixteenth of the frame is already finer than any decision taken from this chart.
    """
    out = grid
    while out.shape[1] > limit and out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0:
        out = out.reshape(out.shape[0] // 2, 2, out.shape[1] // 2, 2).sum(axis=(1, 3))
    return out


def _hot_cells(grid: np.ndarray, want: int) -> list[tuple[int, int, int]]:
    """Return the `want` densest cells, none adjacent -- labels must not collide."""
    if grid.size == 0:
        return []
    flat = np.argsort(grid, axis=None)[::-1][: want * 40]
    hot: list[tuple[int, int, int]] = []
    for index in flat:
        r, c = divmod(int(index), grid.shape[1])
        value = int(grid[r, c])
        if value <= 0:
            break
        if all(abs(c - hc) > 1 or abs(r - hr) > 1 for hc, hr, _ in hot):
            hot.append((c, r, value))
        if len(hot) == want:
            break
    return hot


def _ramp_legend(x: float, y: float, caption: str) -> list[str]:
    parts = [_text("axl", x, y + 9, "0")]
    for i in range(RAMP_STEPS + 1):
        parts.append(_rect(f"hm-{i}", x + 14 + i * 21, y, 20, 9, 1))
    parts.append(_text("axl", x + 14 + (RAMP_STEPS + 1) * 21 + 5, y + 9, caption))
    return parts


def mask_fill(hist: np.ndarray, rectangle_fraction: float) -> str:
    counts = [int(v) for v in hist]
    top, ticks = _nice(max(counts) or 1)
    chart = _Chart("Mask fill ratio, mask area over box area", height=198)
    chart.xdomain(0, 1).ydomain(0, top).grid_y(ticks)
    peak = _bars(
        chart,
        counts,
        lambda i, c: (
            f"fill {_bin_edges(i, len(counts), 0, 1)[0]:.02f}-"
            f"{_bin_edges(i, len(counts), 0, 1)[1]:.02f} · {_n(c)} instances"
        ),
        cls="f-teal",
    )
    _peak_label(chart, counts, peak, "f-teal")
    chart.vref(
        0.95,
        "rectangle-polygon",
        "end",
        f"{rectangle_fraction:.1%} of polygons fill more than 95% of their box",
    )
    chart.ticks_x([0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:g}")
    chart.axis(
        "mask area / bounding-box area",
        f"{rectangle_fraction:.1%} above 0.95",
    )
    return chart.render()


def polygon_vertices(counter: dict[int, int]) -> str:
    keys = sorted(counter)[:MAX_HIST_BINS]
    counts = [int(counter[k]) for k in keys]
    top, ticks = _nice(max(counts) or 1)
    chart = _Chart("Polygon vertex count", height=198)
    chart.xdomain(0, max(len(keys), 1)).ydomain(0, top).grid_y(ticks)
    peak = _bars(
        chart,
        counts,
        lambda i, c: f"{keys[i]} vertices · {_n(c)} polygons",
        cls="f-teal",
    )
    _peak_label(chart, counts, peak, "f-teal")
    step = max(1, len(keys) // 8)
    for i in range(0, len(keys), step):
        slot = chart.band / max(len(keys), 1)
        x = chart.x0 + i * slot + slot / 2
        chart.add(_text("tk", x, chart.yb + 18, keys[i], "middle"))
    chart.axis("vertices per polygon", "polygons")
    return chart.render()


# --- section 3b: image files --------------------------------------------------------


def image_resolutions(
    rows: list[tuple[str, int]], other: int, sizes: int, total: int
) -> str:
    """Top `W x H` sizes by image count, with everything else folded into one row."""
    entries = [(str(k).lower().replace("x", " \u00d7 "), int(v)) for k, v in rows]
    if other > 0:
        label = f"other ({sizes} sizes)" if sizes else "other"
        entries.append((label, other))
    bh, gap = 18, 9
    height, bottom = _rows(len(entries), bh, gap, top=14)
    top, ticks = _nice(max((v for _, v in entries), default=1))
    chart = _Chart(
        f"Image resolutions by count over {_n(total)} images",
        height=height,
        left=118,
        right=560,
        bottom=bottom,
    ).xdomain(0, top)
    chart.grid_x(ticks, _compact, y0=6, y1=bottom)
    bars = []
    for name, value in entries:
        cyan = not name.startswith("other")
        pixels = _pixels_note(name)
        bars.append(
            (
                name,
                value,
                "f-cyan" if cyan else "f-peer",
                _n(value),
                f"{name} · {_n(value)} images{pixels}",
            )
        )
    _hbars(chart, bars, 14, bh, gap)
    chart.add(_text("axl", 118, bottom + 33, f"images · {_n(total)} scanned"))
    return chart.render()


def _pixels_note(key: str) -> str:
    try:
        w, h = (int(v) for v in key.lower().split("x"))
    except ValueError:
        return ""
    return f" · {w * h / 1e6:.1f} MP"


def image_orientation(counts: dict[str, int], total: int) -> str:
    """One thin stacked bar: a sentence made of pixels, so it carries no caption."""
    order = [k for k in ("landscape", "portrait", "square") if counts.get(k)]
    styles = {"landscape": "f-cyan", "portrait": "f-teal", "square": "f-g1"}
    width, bh, y, gap = 636.0, 14, 4, 2.0
    parts = []
    x = 0.0
    for name in order:
        share = counts[name] / max(total, 1)
        w = max(share * width - gap, 2.0)
        tip = f"{name} · {_n(counts[name])} images · {share:.0%}"
        parts.append(_rect(styles[name], x, y, w, bh, 1.5, tip))
        label = f"{name} {share:.0%}"
        if w > len(label) * 5.6:
            parts.append(_text("seg-t f-paper", x + 8, y + bh - 3.5, label))
        else:
            parts.append(_text("seg-t f-peer", x + w + 7, y + bh - 3.5, label))
        x += w + gap
    aria = ", ".join(f"{k} {counts[k] / max(total, 1):.0%}" for k in order)
    return (
        f'<svg class="chart" viewBox="0 0 720 22" role="img" '
        f'aria-label="Image orientation: {_e(aria)}">{"".join(parts)}</svg>'
    )


def megapixels(hist: np.ndarray, median: float | None, mp_max: float, total: int) -> str:
    """Pixels per image. The median is a stub under the bars, never a scar across them."""
    binw = mp_max / max(len(hist), 1)
    # The histogram is sized for a 44 MP sensor; most datasets occupy the first tenth of
    # it. Drawing the empty tail would put every bar in a strip at the far left, so the
    # axis stops one bin past the last image (and one past the median, which lands in
    # the tail only when a single frame is enormous).
    used = max((i for i, v in enumerate(hist) if v), default=len(hist) - 1) + 1
    if median is not None:
        used = max(used, min(len(hist), int(median / binw) + 1))
    counts = [int(v) for v in hist[:used]]
    axis_max = used * binw
    top, ticks = _nice(max(counts) or 1)
    chart = _Chart(f"Megapixels per image over {_n(total)} images", height=214, top=42)
    chart.xdomain(0, axis_max).ydomain(0, top).grid_y(ticks)
    if median is not None:
        x = chart.sx(min(median, axis_max))
        chart.add(_vline("ref", x, chart.yt - 30, chart.yb, f"median {median:g} MP"))
        note = f"median {median:g} MP"
        chart.add(_text("dl f-peer", x - 7, chart.yt - 32, note, "end"))
    peak = _bars(
        chart,
        counts,
        lambda i, c: (
            f"{i * binw:.1f}-{(i + 1) * binw:.1f} MP · {_n(c)} "
            f"image{'' if c == 1 else 's'}"
        ),
    )
    _peak_label(chart, counts, peak)
    chart.auto_ticks_x(6, lambda v: f"{v:g}")
    chart.axis(
        f"megapixels per image · bin width {binw:.1f} MP",
        f"{_n(total)} images",
    )
    return chart.render()


# --- section 5: confusion -----------------------------------------------------------


def confusion(labels: list[str], counts: np.ndarray) -> dict[str, Any]:
    """Draw the matrix once, carrying both ramps, and let CSS pick which one applies.

    Counts and row-normalised rates differ only in the shade of a cell and in the number
    printed in it. Emitting two complete matrices would double an already quadratic
    figure, so every cell carries *both* ramp classes (`hm-3 nm-5`) and the wrapper's
    `data-cm` attribute decides which rule matches -- one of the two selectors can be
    live at a time, so no redraw and no second set of rects.

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

    n = len(labels)
    gutter = min(120.0, max(46.0, 7.0 * max((len(str(x)) for x in labels), default=4)))
    # The viewBox stays the shared width whatever `n` is. Hugging the content instead
    # would let `width:100%` scale a two-class matrix to 660px and print 90px digits.
    #
    # The ceiling on a cell used to be 33px, which is what a 30-class matrix gets anyway.
    # A five-class one therefore drew 165px of matrix into 660px of canvas and, scaled to
    # the sheet, came out a quarter the width of every chart around it while three
    # quarters of its box stood empty. 64px is the largest cell that keeps the printed
    # count looking like a label rather than a headline, and the matrix is centred in the
    # leftover width instead of being pinned to the label gutter.
    cell = min(64.0, (VB_W - gutter - 24) / max(n, 1))
    top = 66.0 if n <= 12 else 84.0
    size = n * cell
    left = max(gutter, (VB_W - size) / 2)
    chart = _Chart(
        f"Confusion matrix, {n} classes",
        height=top + size + 58,
        left=left,
        right=left + size,
        top=top,
        bottom=top + size,
    )
    for j, name in enumerate(labels):
        cx = left + j * cell + cell / 2 - 6
        chart.add(
            f'<text class="mx-lbl" x="{cx:.1f}" y="{top - 7:.0f}" '
            f'transform="rotate(-45 {cx:.1f} {top - 7:.0f})">{_e(name)}</text>'
        )
        chart.add(_text("mx-lbl", left - 8, top + j * cell + cell / 2 + 3, name, "end"))
    vmax = float(z.max()) if z.size else 0.0
    label_cells = n <= CM_LABEL_MAX
    for i in range(n):
        for j in range(min(n, z.shape[1])):
            chart.add(*_cm_cell(i, j, z, alt, vmax, cell, left, top, label_cells))
    chart.add(*_ramp_legend(left, top + size + 38, f"{_n(vmax)} detections in one cell"))
    chart.add(_text("axl", left, top + size + 22, "predicted across, actual down"))
    payload = figure_payload(chart.render())
    payload["classes"] = n
    return payload


def _cm_cell(
    i: int,
    j: int,
    z: np.ndarray,
    alt: np.ndarray,
    vmax: float,
    cell: float,
    left: float,
    top: float,
    label: bool,
) -> list[str]:
    """One matrix cell: one rect carrying both ramps, and the two printed numbers."""
    count = float(z[i, j])
    rate = float(alt[i, j])
    step, alt_step = _ramp(count, vmax), _ramp(rate, 1.0)
    x, y = left + j * cell, top + i * cell
    tip = f"actual row {i + 1}, predicted column {j + 1} · {_n(count)}"
    parts = [
        (
            f'<rect class="hm-{step} nm-{alt_step}" x="{x:.1f}" y="{y:.1f}" '
            f'width="{cell - 1.5:.1f}" height="{cell - 1.5:.1f}" '
            f'data-tip="{_e(tip)}"></rect>'
        )
    ]
    if not label:
        return parts
    cx, cy = x + (cell - 1.5) / 2, y + cell / 2 + 2
    # A cell that has room for it prints its number at a size that matches the cell, not
    # at the 11px every other chart's tick uses.
    big = " cm-big" if cell >= 44 else ""
    for cls, text, step_now in (
        ("cm-c", _compact(count), step),
        ("cm-n", f"{rate:.2f}", alt_step),
    ):
        shade = "hm-lbl" if step_now >= 5 else "hm-lbl-d"
        parts.append(_text(f"{cls} {shade}{big}", cx, cy, text, "middle"))
    return parts


# --- section 6: threshold -----------------------------------------------------------


def _line(chart: _Chart, xs, ys, stroke: str) -> None:
    points = " ".join(
        f"{chart.sx(x):.1f},{chart.sy(y):.1f}"
        for x, y in zip(xs, ys, strict=False)
        if y is not None
    )
    chart.add(f'<polyline class="ln {stroke}" points="{points}"/>')


MAX_HOVER_MARKS = 24


def _marks(chart: _Chart, xs, ys, name: str, fmt="{:.3f}") -> None:
    """Invisible hover targets along a line, evenly thinned to a fixed count.

    Capped rather than one per point: a 3,000-epoch run downsamples to 250 plotted
    points, and 250 circles carrying a tooltip string each is 30 KB of document that
    grows with epoch count -- which is the one thing this report may not do. The values
    themselves are printed at the end of every line, so nothing is lost but hover
    density.
    """
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None]
    step = max(1, math.ceil(len(pairs) / MAX_HOVER_MARKS))
    for x, y in pairs[::step]:
        chart.add(
            f'<circle class="hit" cx="{chart.sx(x):.1f}" cy="{chart.sy(y):.1f}" r="7" '
            f'data-tip="{_e(f"{name} {fmt.format(y)} at {x:g}")}"></circle>'
        )


LABEL_GAP = 13.0


def _end_labels(chart: _Chart, items) -> None:
    """Name and value at the end of each line, which is where the eye leaves it.

    Laid out together rather than one at a time because lines that converge -- three
    losses at the end of training, box and mask mAP on a run where the polygons are
    rectangles -- end within a couple of pixels of each other, and three labels printed
    at their own y are one illegible stack.
    """
    placed = []
    for xs, ys, fill, text in items:
        pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None]
        if not pairs:
            continue
        x, y = pairs[-1]
        placed.append([chart.sy(y), chart.sy(y), chart.sx(x), fill, text])
    placed.sort(key=lambda row: row[0])
    for i in range(1, len(placed)):
        placed[i][0] = max(placed[i][0], placed[i - 1][0] + LABEL_GAP)
    for ly, py, px_point, fill, text in placed:
        # 11px/500 runs about 5.7px a glyph. Where the label would not fit to the right
        # of the last point it is anchored back onto the line instead of overflowing the
        # viewBox, which is where "observed 0.830" ended up on a curve reaching x=1.
        px, anchor = px_point + 9, ""
        if px + len(text) * 5.7 > chart.w - 4:
            px, anchor = px_point - 9, "end"
        chart.add(_text(f"dl {fill}", px, ly - 4, text, anchor))
        chart.add(
            f'<circle class="mk {fill}" cx="{px_point:.1f}" cy="{py:.1f}" r="4.2"/>'
        )


def pr_f1_vs_conf(curves: list[dict], tau: float | None) -> str | None:
    """Mean precision, recall and F1 against confidence, with tau* marked."""
    wanted = {
        "Precision-Confidence(B)": ("Precision", "peer"),
        "Recall-Confidence(B)": ("Recall", "teal"),
        "F1-Confidence(B)": ("F1", "cyan"),
    }
    series = []
    for curve in curves or []:
        entry = wanted.get(str(curve.get("name")))
        if entry is None or not (curve.get("series") or []):
            continue
        name, colour = entry
        mean = np.mean([s["y"] for s in curve["series"]], axis=0)
        series.append((name, colour, list(curve["x"]), [float(v) for v in mean]))
    if not series:
        return None
    hi = max((max(ys) for _, _, _, ys in series), default=1.0)
    chart = _Chart(
        "Precision, recall and F1 against the confidence threshold",
        height=244,
        top=28,
        bottom=202,
    )
    # Precision, recall and F1 are all bounded by 1, so the axis stops there rather than
    # rounding a curve that reaches 1.000 up to a gridline at 1.25 nothing can occupy.
    top, ticks = (1.0, [0, 0.25, 0.5, 0.75, 1.0]) if hi <= 1.0 else _nice(hi * 1.05, 5)
    chart.xdomain(0, max((max(xs) for _, _, xs, _ in series), default=1.0))
    chart.ydomain(0, top).grid_y(ticks, lambda v: f"{v:.2f}")
    for _, colour, xs, ys in series:
        _line(chart, xs, ys, f"s-{colour}")
    _end_labels(
        chart,
        [
            (xs, ys, f"f-{colour}", f"{name} {ys[-1]:.3f}")
            for name, colour, xs, ys in series
        ],
    )
    for name, _colour, xs, ys in series:
        _marks(chart, xs, ys, name)
    if tau is not None:
        chart.vref(tau, f"τ* = {tau:.3f}", "start", f"F1-optimal threshold {tau:.4f}")
    chart.auto_ticks_x(6, lambda v: f"{v:g}")
    chart.axis("confidence threshold", "mean over classes")
    return chart.render()


def tp_fp_hist(centres: list[float], tp: list[int], fp: list[int]) -> str:
    """Draw matched against unmatched confidence, as two densities on one bin grid.

    Grouped rather than overlaid: an alpha blend of two colours makes a third one, and
    the reading here is which hump sits where, not what the overlap is.
    """
    tp_total, fp_total = sum(tp) or 1, sum(fp) or 1
    tp_share = [v / tp_total for v in tp]
    fp_share = [v / fp_total for v in fp]
    top, ticks = _nice(max(max(tp_share, default=0), max(fp_share, default=0)) or 1, 4)
    chart = _Chart("Detection confidence, matched against unmatched", height=210)
    chart.xdomain(0, 1).ydomain(0, top).grid_y(ticks, lambda v: f"{v:.2f}")
    slot = chart.band / max(len(centres), 1)
    for i, centre in enumerate(centres):
        for offset, values, counts, cls, name in (
            (0.06, tp_share, tp, "f-good", "matched"),
            (0.52, fp_share, fp, "f-serious", "unmatched"),
        ):
            y = chart.sy(values[i]) if i < len(values) else chart.yb
            x = chart.x0 + i * slot + slot * offset
            tip = (
                f"{name} · conf ~{centre:.2f} · {_n(counts[i])} "
                f"({values[i]:.1%} of that population)"
            )
            if values[i] > 0:
                chart.add(_rect(cls, x, y, slot * 0.42, chart.yb - y, 1, tip))
    for values, cls, name in (
        (tp_share, "f-good", f"matched · {_n(tp_total)}"),
        (fp_share, "f-serious", f"unmatched · {_n(fp_total)}"),
    ):
        if not values:
            continue
        i = max(range(len(values)), key=lambda k: values[k])
        chart.add(
            _text(
                f"dl {cls}",
                chart.x0 + i * slot + slot / 2,
                chart.sy(values[i]) - 7,
                name,
                "middle",
            )
        )
    chart.ticks_x([0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:g}")
    chart.axis("confidence", "share of that population")
    return chart.render()


def reliability(calibration: dict[str, Any]) -> str:
    xs = [float(v) for v in calibration["mean_confidence"]]
    ys = [float(v) for v in calibration["precision"]]
    chart = _Chart(
        f"Reliability diagram, D-ECE {float(calibration['ece']):.4f}",
        height=244,
        top=28,
        bottom=202,
    )
    chart.xdomain(0, 1).ydomain(0, 1.0)
    chart.grid_y([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}")
    chart.add(
        f'<line class="ref" x1="{chart.sx(0):.1f}" y1="{chart.sy(0):.1f}" '
        f'x2="{chart.sx(1):.1f}" y2="{chart.sy(1):.1f}"/>'
    )
    chart.add(_text("dl-q", chart.sx(0.72), chart.sy(0.78), "perfect calibration"))
    _line(chart, xs, ys, "s-cyan")
    for x, y, count in zip(xs, ys, calibration["count"], strict=False):
        chart.add(
            f'<circle class="mk f-cyan" cx="{chart.sx(x):.1f}" cy="{chart.sy(y):.1f}" '
            f'r="4.2"/>'
        )
        chart.add(
            f'<circle class="hit" cx="{chart.sx(x):.1f}" cy="{chart.sy(y):.1f}" r="8" '
            f'data-tip="{_e(f"confidence {x:.3f} - precision {y:.3f} - n={count}")}">'
            f"</circle>"
        )
    _end_labels(
        chart, [(xs, ys, "f-cyan", f"observed {ys[-1]:.3f}" if ys else "observed")]
    )
    chart.ticks_x([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}")
    chart.axis(
        f"mean predicted confidence · {calibration.get('basis', 'box')} IoU 0.50",
        "observed precision",
    )
    return chart.render()


# --- section 7: TIDE ----------------------------------------------------------------

TIDE_STYLES = ("f-cyan", "f-teal", "f-g1", "f-g2", "f-g3", "f-g3")


def tide_bar(tide: dict[str, Any]) -> str:
    """One stacked bar, six segments, every segment directly labelled.

    Direct labels are mandatory rather than decorative: three of the six segments are
    greys, so the numbers -- not the colours -- have to carry the reading. Segments too
    thin to hold a number share one leader out to a stacked list, because to scale they
    are a few pixels apart and three separate stubs would imply a precision the axis
    cannot carry.
    """
    counts = tide.get("counts") or {}
    deltas = tide.get("delta_ap") or {}
    use_delta = tide.get("mode") == "delta_ap"
    kinds = list(tide.get("types") or [])
    values = [
        (abs(float(deltas.get(k, 0.0))) if use_delta else float(counts.get(k, 0)))
        for k in kinds
    ]
    ceilings = sorted(
        (
            (label, abs(float(value)))
            for label, key in (("ΔAP_FP", "fp"), ("ΔAP_FN", "fn"))
            if use_delta and (value := (tide.get("ceilings") or {}).get(key))
        ),
        key=lambda c: c[1],
    )
    # The axis has to cover the *stack*, not its largest segment: the six oracles are
    # independent and their sum is not the headroom, but they are still drawn end to
    # end, and scaling to the maximum ran the bar four viewBox widths off the canvas on
    # any run whose errors are evenly spread.
    total = max(sum(values), *(v for _, v in ceilings), 1e-9) * 1.02
    order = sorted(range(len(kinds)), key=lambda i: -values[i])

    chart = _Chart(
        "Error decomposition by error type",
        height=190,
        left=8,
        top=48,
        bottom=80,
    ).xdomain(0, total)
    for i, (label, value) in enumerate(ceilings):
        last = i == len(ceilings) - 1
        x = chart.sx(value)
        ly = 30 if last else 15
        text = f"{label} {value:.4f}"
        chart.add(_vline("ref", x, ly + 5, 86, f"{label} ceiling {value:.4f}"))
        anchor = "start" if not last else "end"
        chart.add(
            _text("dl f-peer", x + (6 if anchor == "start" else -6), ly, text, anchor)
        )
    gap, hair = 2.0, 0.75
    x = float(chart.x0)
    outside: list[tuple[str, float]] = []
    tiny_x0 = tiny_x1 = None
    for slot, i in enumerate(order):
        kind, value = kinds[i], values[i]
        span = chart.sx(value) - chart.x0
        cls = TIDE_STYLES[min(slot, len(TIDE_STYLES) - 1)]
        text = f"{value:.4f}" if use_delta else _n(value)
        tip = f"{kind} · {text} · {_n(counts.get(kind, 0))} boxes"
        chart.add(_rect(cls, x, 48, max(span - gap, hair), 32, 1.5, tip))
        if span > 62:
            shade = "f-paper" if slot < 2 else "f-ink1"
            chart.add(_text(f"seg-t {shade}", x + 9, 62, kind))
            chart.add(_text(f"seg-t {shade}", x + 9, 74, text))
        else:
            outside.append((kind, value))
            tiny_x0 = x if tiny_x0 is None else tiny_x0
            tiny_x1 = x + span
        x += span
    chart.add(_vline("gl", chart.x0, 38, 88))
    for tick in _nice(total, 5)[1]:
        if tick <= total:
            label = f"{tick:g}" if use_delta else _compact(tick)
            chart.add(_text("tk", chart.sx(tick), 102, label, "middle"))
    if outside and tiny_x0 is not None:
        cx = (tiny_x0 + tiny_x1) / 2
        chart.add(f'<path class="lead" d="M{cx:.1f} 82 L{cx:.1f} 118 L536 118"/>')
    for (kind, value), ly in zip(outside, (132, 146, 160, 174), strict=False):
        text = f"{value:.4f}" if use_delta else _n(value)
        chart.add(
            f'<text class="dl f-ink2" x="542" y="{ly}">{_e(kind)} '
            f'<tspan class="f-peer">{_e(text)}</tspan></text>'
        )
    chart.add(
        _text(
            "axl",
            chart.x0,
            184,
            "ΔAP50 an oracle would recover by fixing only that error"
            if use_delta
            else "errors of that type (dAP oracles not computed)",
        )
    )
    return chart.render()


# --- section 8: strata --------------------------------------------------------------


def _vbars(
    chart: _Chart, labels: list[str], values: list[float], fmt, cls: str = "f-cyan"
) -> None:
    """Named vertical bars, every one labelled above its cap and under its foot."""
    slot = chart.band / max(len(values), 1)
    bw = min(slot * 0.62, 74.0)
    for i, value in enumerate(values):
        x = chart.x0 + i * slot + (slot - bw) / 2
        y = chart.sy(max(value, 0))
        base = chart.sy(0)
        tip = f"{labels[i]} · {fmt(value)}"
        chart.add(_rect(cls, x, min(y, base), bw, abs(base - y), 2, tip))
        chart.add(_text(f"dl {cls}", x + bw / 2, min(y, base) - 7, fmt(value), "middle"))
        chart.add(_text("tk", x + bw / 2, chart.yb + 18, labels[i], "middle"))


def recall_by_size(buckets: list[str], recalls: list[float], supports: list[int]) -> str:
    chart = _Chart("Recall by object size bucket", height=210)
    chart.xdomain(0, max(len(buckets), 1)).ydomain(0, 1.0)
    chart.grid_y([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}")
    _vbars(chart, buckets, recalls, lambda v: f"{v:.2f}")
    slot = chart.band / max(len(buckets), 1)
    for i, support in enumerate(supports):
        chart.add(
            _text(
                "tk",
                chart.x0 + i * slot + slot / 2,
                chart.yb + 33,
                f"n={_n(support)}",
                "middle",
            )
        )
    chart.axis("this dataset's own terciles", "recall at IoU 0.50", dy=50)
    return chart.render()


def iou_hist(hist: np.ndarray) -> str:
    counts = [int(v) for v in hist]
    top, ticks = _nice(max(counts) or 1)
    chart = _Chart("IoU of matched detections", height=198)
    chart.xdomain(0, 1).ydomain(0, top).grid_y(ticks)
    peak = _bars(
        chart,
        counts,
        lambda i, c: (
            f"IoU {_bin_edges(i, len(counts), 0, 1)[0]:.02f}-"
            f"{_bin_edges(i, len(counts), 0, 1)[1]:.02f} · {_n(c)} matches"
        ),
    )
    _peak_label(chart, counts, peak)
    chart.vref(0.5, "match threshold", "start", "matches are counted at IoU 0.50")
    chart.ticks_x([0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:g}")
    chart.axis("box IoU", "matches")
    return chart.render()


def recall_vs_area(edges: list[float], recalls: list[float], imgsz: int) -> str:
    chart = _Chart(
        f"Recall against object size in pixels at imgsz {imgsz}",
        height=210,
        top=28,
        bottom=168,
    )
    chart.xdomain(min(edges or [0]), max(edges or [1])).ydomain(0, 1.0)
    chart.grid_y([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}")
    _line(chart, edges, recalls, "s-cyan")
    for x, y in zip(edges, recalls, strict=False):
        chart.add(
            f'<circle class="mk f-cyan" cx="{chart.sx(x):.1f}" cy="{chart.sy(y):.1f}" '
            f'r="4.2"/>'
        )
        chart.add(
            f'<circle class="hit" cx="{chart.sx(x):.1f}" cy="{chart.sy(y):.1f}" r="8" '
            f'data-tip="{_e(f"~{x:.0f}px - recall {y:.3f}")}"></circle>'
        )
    label = f"{recalls[-1]:.2f}" if recalls else ""
    _end_labels(chart, [(edges, recalls, "f-cyan", label)])
    chart.auto_ticks_x(6, lambda v: f"{v:g}")
    chart.axis(f"sqrt(area) in px at imgsz {imgsz}", "recall at IoU 0.50")
    return chart.render()


def recall_by_ar(buckets: list[str], recalls: list[float]) -> str:
    chart = _Chart("Recall by aspect-ratio bucket", height=198)
    chart.xdomain(0, max(len(buckets), 1)).ydomain(0, 1.0)
    chart.grid_y([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}")
    _vbars(chart, buckets, recalls, lambda v: f"{v:.2f}", "f-teal")
    chart.axis("log(w / h) bucket", "recall at IoU 0.50")
    return chart.render()


# --- section 9: box vs mask ---------------------------------------------------------


def boxmask_class(names: list[str], box: list[float], mask: list[float]) -> str:
    """Two bars per class, horizontal: a class name is a word, not a tick."""
    bh, gap, inner = 9, 12, 3
    rows = len(names)
    height, bottom = _rows(rows, bh * 2 + inner, gap)
    chart = _Chart(
        "Box against mask mAP50-95 per class",
        height=height,
        left=118,
        bottom=bottom,
    ).xdomain(0, 1.0)
    chart.grid_x([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}", y0=8, y1=bottom)
    for i, name in enumerate(names):
        y = 16 + i * (bh * 2 + inner + gap)
        for k, (values, cls, label) in enumerate(
            ((box, "f-cyan", "box"), (mask, "f-teal", "mask"))
        ):
            value = float(values[i]) if i < len(values) else 0.0
            w = max(chart.sx(value) - chart.x0, 1.0 if value > 0 else 0.0)
            yy = y + k * (bh + inner)
            chart.add(_rect(cls, chart.x0, yy, w, bh, 2, f"{name} · {label} {value:.3f}"))
            chart.add(_text(f"dl {cls}", chart.x0 + w + 7, yy + bh - 1, f"{value:.3f}"))
        chart.add(_text("b-lbl", chart.x0 - 10, y + bh + inner, name, "end"))
    chart.add(_text("axl", 118, bottom + 33, "mAP50-95 · upper bar box, lower mask"))
    return chart.render()


def boxmask_iou(hist: np.ndarray) -> str:
    grid = np.asarray(hist)
    nx, ny = grid.shape[0], grid.shape[1]
    cell_w = (PLOT_R - PLOT_L) / max(nx, 1)
    cell_h = 18.0
    x0, yt = PLOT_L, 20.0
    vmax = float(grid.max()) if grid.size else 0.0
    chart = _Chart(
        "Box IoU against the number of mask IoU thresholds matched",
        height=yt + ny * cell_h + 74,
        bottom=yt + ny * cell_h,
    )
    for i in range(nx):
        for j in range(ny):
            value = int(grid[i, j])
            # y is inverted: level 10 (a perfect mask match) belongs at the top.
            row = ny - 1 - j
            tip = (
                f"box IoU ~{(i + 0.5) / nx:.2f} · {j} of {ny - 1} mask thresholds "
                f"· {_n(value)} instances"
            )
            chart.add(
                _rect(
                    f"hm-{_ramp(value, vmax)}",
                    x0 + i * cell_w,
                    yt + row * cell_h,
                    cell_w - 0.8,
                    cell_h - 0.8,
                    1,
                    tip,
                )
            )
    for j in (0, ny // 2, ny - 1):
        chart.add(
            _text("tk", x0 - 9, yt + (ny - 1 - j) * cell_h + cell_h / 2 + 4, j, "end")
        )
    chart.ticks_x([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}")
    chart.xdomain(0, 1.0)
    chart.axis("box IoU (exact)", "mask IoU thresholds matched")
    chart.add(*_ramp_legend(x0, chart.yb + 44, f"{_n(vmax)} instances in one cell"))
    return chart.render()


def iou_overlay(box_hist: np.ndarray, mask_levels: np.ndarray) -> str:
    """Box IoU as bars, mask IoU as a step line: two bin grids, one axis, no blending."""
    box = [int(v) for v in box_hist]
    levels = [int(v) for v in mask_levels]
    box_total, mask_total = sum(box) or 1, sum(levels) or 1
    # Density, not share: box IoU arrives in 50 bins and the mask axis in 11 levels, so
    # plotting the two shares against each other would make the coarser grid look ten
    # times as populated. Dividing by bin width puts both on one comparable scale.
    box_bins, mask_bins = max(len(box), 1), max(len(levels), 1)
    box_share = [v / box_total * box_bins for v in box]
    mask_share = [v / mask_total * mask_bins for v in levels]
    top, ticks = _nice(max(max(box_share, default=0), max(mask_share, default=0)) or 1, 4)
    chart = _Chart("Box and mask IoU distributions over matched instances", height=210)
    chart.xdomain(0, 1).ydomain(0, top).grid_y(ticks, lambda v: f"{v:.2f}")
    _bars(
        chart,
        box_share,
        lambda i, _c: (
            f"box IoU ~{(i + 0.5) / box_bins:.02f} · {box[i] / box_total:.1%} of matches"
        ),
    )
    xs = [i / max(len(levels) - 1, 1) for i in range(len(levels))]
    _line(chart, xs, mask_share, "s-teal")
    for x, y, count in zip(xs, mask_share, levels, strict=False):
        chart.add(
            f'<circle class="mk f-teal" cx="{chart.sx(x):.1f}" cy="{chart.sy(y):.1f}" '
            f'r="4.2"/>'
        )
        chart.add(
            f'<circle class="hit" cx="{chart.sx(x):.1f}" cy="{chart.sy(y):.1f}" r="8" '
            f'data-tip="{_e(f"mask level {x:.2f} - {_n(count)} instances")}"></circle>'
        )
    _end_labels(chart, [(xs, mask_share, "f-teal", "mask (quantised)")])
    if box_share:
        i = max(range(len(box_share)), key=lambda k: box_share[k])
        slot = chart.band / len(box_share)
        chart.add(
            _text(
                "dl f-cyan",
                chart.x0 + i * slot + slot / 2,
                chart.sy(box_share[i]) - 7,
                "box (exact)",
                "middle",
            )
        )
    chart.ticks_x([0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:g}")
    chart.axis("IoU", "share of matches per unit IoU")
    return chart.render()


def boxmask_by_size(buckets: list[str], deltas: list[float]) -> str:
    """Mask minus box IoU, signed around a zero line, every bar carrying its sign."""
    span = max((abs(d) for d in deltas), default=0.01) * 1.25 or 0.01
    chart = _Chart("Mask minus box IoU by size bucket", height=210, top=30, bottom=160)
    chart.xdomain(0, max(len(buckets), 1)).ydomain(-span, span)
    ticks = _nice(span, 2)[1]
    chart.grid_y([-t for t in reversed(ticks[1:])] + ticks, lambda v: f"{v:+.2f}")
    slot = chart.band / max(len(deltas), 1)
    bw = min(slot * 0.62, 74.0)
    zero = chart.sy(0)
    for i, delta in enumerate(deltas):
        x = chart.x0 + i * slot + (slot - bw) / 2
        y = chart.sy(delta)
        cls = "f-cyan" if delta >= 0 else "f-teal"
        chart.add(
            _rect(
                cls, x, min(y, zero), bw, abs(zero - y), 2, f"{buckets[i]} {delta:+.3f}"
            )
        )
        label_y = (min(y, zero) - 7) if delta >= 0 else (max(y, zero) + 15)
        chart.add(_text(f"dl {cls}", x + bw / 2, label_y, f"{delta:+.3f}", "middle"))
        chart.add(_text("tk", x + bw / 2, chart.yb + 18, buckets[i], "middle"))
    chart.axis("size bucket", "mask - box IoU")
    return chart.render()


# --- section 11: training curves ----------------------------------------------------

CURVE_STYLES = ("cyan", "teal", "peer", "g1")


def _curves(
    aria: str, epochs: list[int], series: dict[str, list[float]], axis: str
) -> str:
    """Shared line chart for the two training figures.

    Four series exceed the two-identity rule, so only the first two carry an identity
    colour; the rest are greys and every line is named at its own end. Colour is never
    the only thing telling two lines apart here.
    """
    values = [v for row in series.values() for v in row if v is not None]
    hi = max(values, default=1.0)
    lo = min([*values, 0.0], default=0.0)
    top, ticks = _nice(hi or 1.0, 4)
    chart = _Chart(aria, height=244, top=28, bottom=196)
    chart.xdomain(min(epochs or [0]), max(epochs or [1]))
    chart.ydomain(min(lo, 0.0), top).grid_y(ticks, lambda v: f"{v:g}")
    for slot, (name, row) in enumerate(series.items()):
        style = CURVE_STYLES[min(slot, len(CURVE_STYLES) - 1)]
        _line(chart, epochs, row, f"s-{style}")
        _marks(chart, epochs, row, name, "{:.4f}")
    labels = []
    for slot, (name, row) in enumerate(series.items()):
        style = CURVE_STYLES[min(slot, len(CURVE_STYLES) - 1)]
        last = next((v for v in reversed(row) if v is not None), None)
        text = name if last is None else f"{name} {last:.4f}"
        labels.append((epochs, row, f"f-{style}", text))
    _end_labels(chart, labels)
    chart.auto_ticks_x(6, lambda v: f"{v:,.0f}")
    chart.axis("epoch", axis)
    return chart.render()


def val_map_curve(epochs: list[int], series: dict[str, list[float]]) -> str:
    return _curves("Validation mAP against epoch", epochs, series, "mAP")


def loss_curves(epochs: list[int], series: dict[str, list[float]]) -> str:
    return _curves("Training losses against epoch", epochs, series, "loss")


# --- section 12: environment --------------------------------------------------------


EPOCH_BARS_MAX = 12


def epoch_seconds(seconds: list[float]) -> str:
    """Wall-clock per epoch. Three epochs is a thin chart; it is not padded.

    Past a dozen epochs it stops being a bar per epoch and becomes a line, because one
    row per epoch is a figure that grows with epoch count -- the thing the whole report
    is built not to do. The line is downsampled on top of that, so a 3,000-epoch run and
    a 30-epoch one cost the same bytes.
    """
    rows = [(i + 1, float(v)) for i, v in enumerate(seconds) if v and v > 0]
    if not rows:
        return ""
    if len(rows) > EPOCH_BARS_MAX:
        return _epoch_line(rows)
    bh, gap = 17, 13
    height, bottom = _rows(len(rows), bh, gap, top=12)
    top, ticks = _nice(max(v for _, v in rows), 4)
    chart = _Chart(
        f"Seconds per epoch over {len(rows)} epochs",
        height=height - 4,
        width=275,
        left=52,
        right=218,
        bottom=bottom,
    ).xdomain(0, top)
    chart.grid_x(ticks, lambda v: f"{v / 60:g}", y0=5, y1=bottom)
    bars = [
        (
            f"epoch {n}",
            value,
            "f-cyan",
            _mmss(value),
            f"epoch {n} · {value:.0f} s · {_mmss(value)}",
        )
        for n, value in rows
    ]
    _hbars(chart, bars, 12, bh, gap)
    chart.add(_text("axl", 52, bottom + 32, "minutes · labels are m:ss"))
    return chart.render()


EPOCH_LINE_POINTS = 60


def _epoch_line(rows: list[tuple[int, float]]) -> str:
    """Seconds per epoch as one downsampled line, for runs too long to draw as bars."""
    step = max(1, math.ceil(len(rows) / EPOCH_LINE_POINTS))
    kept = rows[::step]
    xs = [n for n, _ in kept]
    ys = [v for _, v in kept]
    top, ticks = _nice(max(v for _, v in rows), 3)
    chart = _Chart(
        f"Seconds per epoch over {len(rows)} epochs",
        height=150,
        width=275,
        left=44,
        right=262,
        top=16,
        bottom=112,
    )
    chart.xdomain(min(xs), max(xs)).ydomain(0, top).grid_y(ticks, lambda v: f"{v / 60:g}")
    _line(chart, xs, ys, "s-cyan")
    _marks(chart, xs, ys, "epoch", "{:.0f} s")
    slowest = max(rows, key=lambda r: r[1])
    chart.add(
        _text(
            "dl f-cyan",
            chart.sx(min(max(slowest[0], min(xs)), max(xs))),
            chart.sy(slowest[1]) - 7,
            _mmss(slowest[1]),
            "middle",
        )
    )
    chart.auto_ticks_x(4, lambda v: f"{v:,.0f}")
    chart.axis("epoch", dy=34)
    chart.add(_text("axl", 44, chart.yb + 48, "minutes · label is the slowest epoch"))
    return chart.render()


def _mmss(seconds: float) -> str:
    total = round(seconds)
    return f"{total // 60}:{total % 60:02d}"


# --- shared helpers -----------------------------------------------------------------


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
    if not fig:
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
