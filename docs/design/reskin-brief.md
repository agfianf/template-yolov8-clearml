# Production re-skin brief

Move `src/report/` onto the design the wireframe settled. Read `wireframe-brief.md` (the system), `wireframe-v3-brief.md` and `wireframe-v4-brief.md` (the two deltas) first — they are binding, including the "Do NOT" ban list. `docs/design/gen_wireframe.py` is the reference implementation for every chart's geometry and every CSS token; copy from it rather than reinventing.

Target: `docs/design/wireframe.html` is what a finished run should look like.

## The seam that makes this safe

Every figure is built through one call site — `blob.py::_fig` → `figures.safe(builder, *args)` → stored at `blob["figures"][fig_id]`. **Builder signatures do not change.** Only the return type does: `go.Figure` becomes an SVG string. Nothing outside `figures.py` needs to know how a chart is drawn.

New payload contract:

```python
blob["figures"][fig_id] = {"svg": "<svg class=\"chart\" …></svg>"}
```

`render.py::fig_block` inlines `svg` straight into the document instead of emitting `<div class="fig" data-fig="…">`. The client no longer draws anything, so **figures must be stripped from the JSON blob before it is serialised into `{{BLOB}}`** — they are already in the DOM, and shipping both doubles the file. Keep `fig_block`'s `missing` / `applicable` behaviour exactly as it is; that distinction is load-bearing and tested.

## Work split

### Stream A — `figures.py`

Rewrite all ~22 builders to emit hand-authored SVG. Drop `plotly` from the module entirely; `figure_payload` and `safe` adapt to the new return type (`safe` keeps swallowing exceptions and returning `None` — that is what the degradation tests assert).

Conventions, all verified in the wireframe: transparent background; horizontal-only gridlines 1px `--grid` with `vector-effect:non-scaling-stroke`; no tick marks; tick text 11px `--ink-3`; axis line omitted; **max two identity colours plus `--peer` grey per chart** (hero `--brand-cyan-ink`, secondary `--brand-teal-ink`; lime only for good/TP status); lines 1.7px round cap and join; markers r4.2 stroked with `--paper`; reference lines `stroke-dasharray:5 5`; **direct value labels on every mark**, 11px/500, `paint-order:stroke;stroke:var(--paper);stroke-width:3.5px`. Width via `viewBox` + `width:100%`.

**Zero hex inside any SVG** — `fill="var(--…)"` or a CSS class, always, so the theme toggle reaches the charts. `svg.chart text{pointer-events:none}` or hover dies on the labels. Put `data-tip` on marks; do **not** add SVG `<title>` children (they double the tooltip).

Selective labelling where labelling everything would bury the shape: histograms label the peak bar and the reference line only. Follow `chart_hist`, `chart_tide`, `chart_prf`, `chart_instances`, `chart_heat`, `chart_matrix` in `gen_wireframe.py`.

The confusion matrix keeps its counts/row-normalised toggle: emit both as two `<g>` layers and let CSS switch them, so no redraw is needed.

### Stream B — capture, thumbs, dataset scan, blob

**`capture.py`** — `_overlay` currently emits `tp` (det box), `fp` (det box), `fn` (unmatched gt). Add `gt` for **matched** ground truth, so the Ground truth tab is reconstructible: `gt` + `fn` is the annotation layer, `tp` + `fp` is the prediction layer. Raise `MAX_BOXES_PER_ITEM` proportionally.

**`thumbs.py`** — delete the `ImageDraw.rectangle` loop in `_encode`. Thumbnails become clean pixels; the overlay is drawn as an SVG layer over the `<img>` from `grids[].items[].boxes`, which `blob.py::_norm_boxes` already emits as thumbnail-normalised 0..1 floats. Update the module docstring, which currently documents the baked outline as deliberate.

**`dataset_scan.py`** — add a pass over `<split>/images/*` reading **headers only** (`PIL.Image.open` → `.size`, `.mode`, `.format`; no pixel decode). Accumulate into fixed-size structures only: histograms for megapixels and log aspect ratio, small `Counter`s for mode and format, a resolution `Counter` that stops admitting new keys past ~4,000 distinct sizes and tallies the rest as `other`, and a count of images whose short side is under `imgsz`. **No logging inside the loop** — one summary line at the end, as the existing scan does. Report size must not grow with dataset size; there are tests that assert exactly this.

**`blob.py`** — new payloads:
- image-file stats from the scan above, feeding the new figures;
- `blob["highlights"]`: a list of plain strings, **descriptive only**. State a measured fact with its number and stop. No "usually means", no recommendation, no judgemental adjective. This is the agreed answer to the report's generic prose and its whole value is that it never diagnoses. Derive 5–7 from what the run actually has, skipping any whose inputs are missing.
- `blob["meta"]["tags"]`: ClearML task tags, read inside `try/except`.
- `blob["environment"]`: two groups. Machine facts read locally (`platform.node()`, Python version, `torch.__version__`, CUDA version, GPU name/count/memory via `torch.cuda.get_device_properties`). Timing from ultralytics' `trainer.train_time_start` and the per-epoch `trainer.epoch_time` the callbacks already log, plus the ClearML task start for wall clock. Every field degrades to `"unknown"` on its own line — never drop the row, never fail the report.

### Stream C — assets and `render.py`

`template.html`, `report.css`, `report.js` rewritten against the wireframe. **Delete `src/report/assets/plotly-cartesian.min.js`** and its `{{PLOTLY_JS}}` placeholder, and drop the `fetch-plotlyjs` Makefile target. `sortable.min.js` stays.

`render.py` gains: the Highlights section under the KPI strip (above Warnings, same flat register), tag chips in the header, the Environment section before Caveats, the floating TOC, and the gallery rebuilt as tabbed grids with the global annotation switch. Keep every existing section, its `missing=` string and its `applicable=` gate.

`report.js` keeps: theme toggle, tooltip, lightbox, table sort, class filter, active-section tracking (bind it to **both** `.toc a` and the nav links — the TOC is hidden on narrow viewports). It loses all Plotly instantiation.

Carry over the ten review fixes already made in the wireframe — they are real defects, not mock artefacts: print palette needs `:root,:root[data-theme="dark"],:root[data-theme="light"]`; no unconditional overlay rule in `@media print`; the annotation hotkey must ignore Ctrl/Cmd/Alt; deep links must range-check rather than wrap; `svg.chart text{pointer-events:none}`; and the lightbox scrim and hover strip need the same `@supports not` / `prefers-reduced-transparency` / `forced-colors` fallbacks as `.glass`.

## Constraints

Python 3.14, line length 90, double quotes, ruff `E,F,I,UP,B,W,C90,N,D,PYI,PT,RET,SIM,ARG,ERA,T20`. **`print()` is banned** — everything through `src/utils/logging.py`. No INFO line inside a loop over dataset items. Report volume must not grow with epoch count: scalars may gain a point per epoch, plots and tables and galleries may not.

Only lint paths you touched — `uv run ruff check --fix src/report` — never a bare `ruff` run.

Do not bump `./VERSION` or build an image; that is the final step and it is mine.

## Verification that must pass

```
PYTHONPATH=. uv run pytest tests -q --ignore=tests/data/test_integration.py \
  --ignore=tests/data/downloader/test_method_cvat_http.py \
  --ignore=tests/data/downloader/test_method_cvat_sdk.py
```

Tests under `tests/report/` encode real invariants — self-containment, volume caps, size budget, degradation paths. Where a test asserts something Plotly-specific, update the assertion to the SVG equivalent; **do not weaken what it is checking**. If a test fails because behaviour genuinely changed, say so rather than editing it into passing.
