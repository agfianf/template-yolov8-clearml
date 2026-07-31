"""Palette constants for the HTML evaluation report.

One source of truth for every colour the report uses, in Python, so the CSS custom
properties and the two Plotly layout templates cannot drift apart. The values were
validated with the dataviz contrast rules and are not to be re-derived here.

Four families, and mixing them is the mistake this module exists to prevent:

* **Brand** -- the six logo gradient stops and one darkened link blue. Decorative only:
  the header bar, a KPI card's top border, the nav's active state, links. Brand colour
  never encodes a data series.
* **Categorical** -- eight fixed slots, light and dark. Slot order is stable so the same
  series is the same colour in every figure of every report.
* **Sequential / diverging** -- the confusion matrix and the centre heatmap use the
  single-hue blue ramp; deltas use blue-to-red across a grey midpoint.
* **Status** -- TP / FN / FP and the warning banners. Never themed, never a series, and
  never carried by colour alone: every use also carries an icon, a dash pattern or a
  label.
"""

from __future__ import annotations
from typing import Any


# --- brand (decorative only) --------------------------------------------------------
BRAND_GRADIENT = (
    "#02B1F0",
    "#3BC2F2",
    "#5CCBF3",
    "#74D5C2",
    "#8CDF8D",
    "#A3E85A",
)
# Darkened brand blue, >=4.5:1 against the light surface below. The gradient stops are
# all far too light to carry text.
BRAND_LINK_LIGHT = "#0369A8"
BRAND_LINK_DARK = "#5CCBF3"

# --- categorical, fixed slot order --------------------------------------------------
# Only the first three slots are safe in every pairing; a figure needing more than three
# simultaneously-compared series must fold or facet rather than add a fourth colour.
CATEGORICAL_LIGHT = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
CATEGORICAL_DARK = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)

# --- sequential (confusion matrix, centre heatmap) ----------------------------------
SEQUENTIAL_BLUE = ("#cde2fb", "#9dc4f4", "#5b93df", "#2a5fa8", "#0d366b")

# --- diverging (deltas) -------------------------------------------------------------
DIVERGING_LIGHT = ("#0d366b", "#5b93df", "#f0efec", "#e08a72", "#a3231f")
DIVERGING_DARK = ("#5b93df", "#9dc4f4", "#383835", "#e08a72", "#d03b3b")

# --- status (never themed, never a series) ------------------------------------------
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# --- surfaces and text --------------------------------------------------------------
SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
PANEL_LIGHT = "#ffffff"
PANEL_DARK = "#232322"
TEXT_LIGHT = "#111111"
TEXT_DARK = "#ffffff"
MUTED_LIGHT = "#5c5b55"
MUTED_DARK = "#c3c2b7"
BORDER_LIGHT = "#e2e1db"
BORDER_DARK = "#3a3a37"
GRID_LIGHT = "#ebeae4"
GRID_DARK = "#33332f"

# Outcome colours for the gallery overlays. The dash pattern is half the signal: FN is
# the only dashed outcome, so the three read apart in greyscale too.
OUTCOME_STYLE = {
    "tp": (STATUS["good"], "solid"),
    "fp": (STATUS["critical"], "solid"),
    "fn": (STATUS["warning"], "dashed"),
    "mixed": (MUTED_LIGHT, "solid"),
}


def _vars(dark: bool) -> dict[str, str]:
    """Return the CSS custom-property map for one mode."""
    cat = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    div = DIVERGING_DARK if dark else DIVERGING_LIGHT
    out = {
        "--surface": SURFACE_DARK if dark else SURFACE_LIGHT,
        "--panel": PANEL_DARK if dark else PANEL_LIGHT,
        "--text": TEXT_DARK if dark else TEXT_LIGHT,
        "--muted": MUTED_DARK if dark else MUTED_LIGHT,
        "--border": BORDER_DARK if dark else BORDER_LIGHT,
        "--grid": GRID_DARK if dark else GRID_LIGHT,
        "--link": BRAND_LINK_DARK if dark else BRAND_LINK_LIGHT,
    }
    for i, c in enumerate(cat, start=1):
        out[f"--cat{i}"] = c
    for i, c in enumerate(div, start=1):
        out[f"--div{i}"] = c
    for name, c in STATUS.items():
        out[f"--{name}"] = c
    out["--brand-gradient"] = "linear-gradient(90deg," + ",".join(BRAND_GRADIENT) + ")"
    return out


def _block(selector: str, values: dict[str, str]) -> str:
    body = "".join(f"{k}:{v};" for k, v in values.items())
    return f"{selector}{{{body}}}"


def css_variables() -> str:
    """Emit the theme block: light default, dark by preference, attribute overrides.

    The attribute selectors are declared last and repeat both modes so the toggle wins
    in *both* directions -- a `data-theme="light"` on a machine preferring dark has to
    beat the media query, and a media query alone cannot be overridden by a later
    preference-scoped rule.
    """
    light, dark = _vars(False), _vars(True)
    return "".join(
        [
            ":root{color-scheme:light dark;}",
            _block(":root", light),
            f"@media (prefers-color-scheme:dark){{{_block(':root', dark)}}}",
            _block(':root[data-theme="light"]', light),
            _block(':root[data-theme="dark"]', dark),
        ]
    )


def plotly_template(dark: bool) -> dict[str, Any]:
    """Return one Plotly layout template as a plain dict.

    A plain dict rather than a `go.layout.Template`: it is hoisted out of every figure
    and assigned once by the JS, so it never passes through `pio.to_json` and gains
    nothing from the class. Serialising it per figure is what the hoisting avoids -- it
    is roughly two thirds of a naive figure payload.
    """
    text = TEXT_DARK if dark else TEXT_LIGHT
    muted = MUTED_DARK if dark else MUTED_LIGHT
    grid = GRID_DARK if dark else GRID_LIGHT
    panel = PANEL_DARK if dark else PANEL_LIGHT
    axis = {
        "gridcolor": grid,
        "zerolinecolor": grid,
        "linecolor": grid,
        "tickfont": {"color": muted, "size": 11},
        "titlefont": {"color": muted, "size": 12},
        "automargin": True,
    }
    return {
        "layout": {
            "colorway": list(CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT),
            "paper_bgcolor": panel,
            "plot_bgcolor": panel,
            "font": {
                "color": text,
                "size": 12,
                "family": "system-ui,-apple-system,Segoe UI,Roboto,sans-serif",
            },
            "title": {"font": {"color": text, "size": 14}, "x": 0.0, "xanchor": "left"},
            "margin": {"l": 56, "r": 16, "t": 44, "b": 44},
            "xaxis": dict(axis),
            "yaxis": dict(axis),
            "legend": {
                "font": {"color": muted, "size": 11},
                "orientation": "h",
                "y": -0.2,
            },
            "colorscale": {"sequential": _ramp(SEQUENTIAL_BLUE)},
            "hoverlabel": {"font": {"size": 11}},
        },
        "data": {},
    }


def _ramp(colors: tuple[str, ...]) -> list[list[Any]]:
    """Turn an ordered colour list into a Plotly colorscale."""
    n = len(colors) - 1
    return [[i / n, c] for i, c in enumerate(colors)]


SEQUENTIAL_SCALE = _ramp(SEQUENTIAL_BLUE)
DIVERGING_SCALE_LIGHT = _ramp(DIVERGING_LIGHT)


def categorical(slot: int, dark: bool = False) -> str:
    """Return the colour for a 1-based categorical slot, wrapping past slot 8."""
    palette = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    return palette[(slot - 1) % len(palette)]
