# Wireframe v3 — delta brief

`wireframe-brief.md` still holds. This file is the **delta only**: three revision points the user raised against v2, plus the answers they gave when asked. Everything not mentioned here stays exactly as v2 built it — same tokens, same glass budget (3 surfaces), same ban list, same hand-authored SVG charts with direct value labels.

Edit `gen_wireframe.py` and regenerate `wireframe.html`. Do not hand-edit the HTML.

## 1. Galleries — minimalist scroll, and a tab strip

The v2 gallery cuts the fourth row in half against a hard edge, which is the thing the user reacted to: the scroll is both obvious and unresolved. Fix by making the block smaller, quieter and continuous.

**Grid.** `repeat(7,1fr)` at full width, `gap:6px` (was 5 columns / 10px). Breakpoints 7 → 5 below 640px → 4 below 440px. Tile stays `aspect-ratio:1`, radius 5px, hairline `box-shadow:0 0 0 1px var(--rule)`.

**No caption under the tile.** The per-tile `<figcaption>` is deleted — three rows of grey filenames is most of the noise. The caption becomes a hover strip inside the tile: absolutely positioned bottom bar, 10.5px, `color-mix(in srgb,var(--paper) 82%,transparent)` + `backdrop-filter:blur(6px)`, `opacity:0` → `1` on hover/focus-visible, 100ms. This is the only new blur surface and it is transient, so the 3-surface glass budget is unaffected — do not give it a border or a shadow.

**Scroll block.** `max-height:352px`, `overflow-y:auto`, `overscroll-behavior:contain`, scrollbar hidden (`scrollbar-width:none`, `-ms-overflow-style:none`, `::-webkit-scrollbar{display:none}`). Soften both cuts with a mask rather than an edge:

```css
.gal{-webkit-mask-image:var(--gal-mask);mask-image:var(--gal-mask);
  --gal-mask:linear-gradient(to bottom,transparent 0,#000 40px,#000 calc(100% - 48px),transparent 100%);}
.gal.at-start{--gal-mask:linear-gradient(to bottom,#000 0,#000 calc(100% - 48px),transparent 100%)}
.gal.at-end{--gal-mask:linear-gradient(to bottom,transparent 0,#000 40px,#000 100%)}
.gal.at-start.at-end{--gal-mask:none}
```

A ~12-line scroll listener toggles `at-start`/`at-end`. Deliberately size the content so the fade is visible on load (the partial row now reads as "more below" instead of as a mistake).

**Header row** above the grid, replacing both the legend paragraph and the trailing note. One flex row, no box, no background, 12.5px:

```
[ Outcome | Prediction | Ground truth ]          18 of 166 · worst FP+FN     [◍ annotations]
```

Left: the tab strip. Right: a `--ink-3` meta string, then the annotation toggle. The legend (TP / FP / FN swatches) moves **below** the grid as one 11px `--ink-3` line, and its content changes with the active tab.

**Tabs.** Plain text buttons — 12.5px, `--ink-3`, no pill, no border, no background. Active: `--ink`, weight 500, and a 2px brand-gradient underline via `::after` (the same signature the nav uses). `role="tablist"`, arrow-key navigation, `aria-selected`.

Three views of the *same* thumbnails; only the overlay layer changes:

| Tab | Draws | Colours |
|---|---|---|
| Outcome (default) | tp, fp, fn | green solid / red solid / amber dashed |
| Prediction | tp + fp as one population | `--brand-cyan-ink` solid, conf printed |
| Ground truth | matched gt + fn as one population | `--brand-teal-ink` solid, class printed |

Legend line under the grid follows: outcome → three swatches; prediction → "detection · conf ≥ 0.25"; ground truth → "annotated instance".

## 2. Annotations show/hide — global

One toggle governs thumbnails **and** the lightbox. Keyboard `a` toggles it; the tabs stay usable while off (turning a tab back on re-enables the overlay).

Control: a 30×16 track switch — `--rule` when off, `var(--brand-grad)` when on, 12px `--ink-2` label "annotations" to its left. `role="switch"`, `aria-checked`. No box around it.

**This forces the overlay to be a separate layer, not baked pixels.** In the wireframe, each tile is `<div class="ph">` holding the placeholder gradient plus an `<svg class="ov" viewBox="0 0 100 100" preserveAspectRatio="none">` sibling with the boxes; `.gal[data-anno="off"] .ov{opacity:0}` and a 120ms transition. Lightbox canvas checks the same flag before drawing boxes.

Record this in the deliverable notes, because production has to follow: `src/report/thumbs.py::_encode` currently bakes a 2px outline into the JPEG for whole-image thumbs, and that must go — the grid will draw from `grids[].items[].boxes`, which `blob.py::_norm_boxes` already emits as thumbnail-normalised 0..1 floats. Production also needs `capture.py::_overlay` to emit matched ground-truth boxes (a new `gt` outcome alongside `tp`/`fp`/`fn`), because the Ground truth tab cannot be reconstructed from the current three.

## 3. Dataset section — image file statistics

The scan reads only label `.txt` files today, so the report can say nothing about the images themselves. Add a subsection **"Image files"** to `s-dataset`, after the annotation-flag table. Source is a PIL header read per image — `.size`, `.mode`, `.format`, no pixel decode — so every figure below is cheap and covers all images, not a sample.

Four blocks, in this order:

**a. Resolutions.** Horizontal bars, top 6 distinct `W×H` by image count plus an "other" row, sorted descending, direct count labels at the bar ends. Mock data: 4032×3024 · 389 · 3024×4032 · 121 · 1920×1440 · 58 · 1280×960 · 31 · 640×480 · 12 · 800×600 · 5 · other (3 sizes) · 2.

**b. Aspect ratio**, inline in the prose that introduces the block: one thin stacked horizontal bar, three segments with in-segment labels — landscape 79% · portrait 20% · square 1%. No axis, no gridlines, 14px tall. This is a sentence made of pixels, not a figure; give it no `fig-t` and no caption.

**c. Pixels per image.** Histogram, ~22 thin bars over megapixels, dashed vline at the median labelled `median 12.2 MP`, peak bar labelled. Same conventions as the existing objects-per-image histogram.

**d. Colour mode, format and fit** — a compact three-column stat row. Flat on the sheet, `border-left:1px solid var(--rule)` between columns, **no glass, no box, no accent edge**. Each column: an 11px uppercase `--ink-3` heading, then 13px tabular lines.

```
COLOUR MODE            FILE FORMAT           FIT AT imgsz=640
RGB          616       JPEG        614       downscaled     612
L (grey)       2       PNG           4       upscaled         6
RGBA           0                             short side <640   6
```

Follow it with one `--ink-3` note line: two greyscale images in an otherwise RGB set is the kind of thing that survives all the way to a confusing per-class number, and a short side under `imgsz` means the loader is inventing detail.

Renumber the existing figures after the insertion point; keep `Fig N.` continuous.

**Production note for later:** `src/report/dataset_scan.py` gains an image pass over `<split>/images/*`. Accumulate into fixed-size structures only — histograms for megapixels and log aspect ratio, small `Counter`s for mode and format, and a resolution `Counter` that stops admitting new keys past ~4,000 distinct sizes and tallies the rest as "other". No logging inside the loop; one summary line at the end. Report size must not grow with dataset size.

## Deliverables

1. Updated `gen_wireframe.py` and regenerated `wireframe.html` (same path, still self-contained).
2. Screenshots: `wireframe_light.png`, `wireframe_dark.png`, plus three crops proving the new work — `gal_tabs.png` (gallery with tabs + fade), `gal_anno_off.png` (annotations toggled off), `ds_imagestats.png` (the new image-files subsection).
3. Return file paths and any deviation from this brief with a one-line reason.
