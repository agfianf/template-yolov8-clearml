# Wireframe v4 — delta brief

`wireframe-brief.md` (the design system) and `wireframe-v3-brief.md` both still hold. This file is the delta only. Edit `gen_wireframe.py`, regenerate `wireframe.html`, never hand-edit the HTML.

Four additions. Nothing already built changes except where stated.

## 1. Highlights — descriptive, never prescriptive

A new block at the very top of the report, directly under the KPI strip, before the model card.

**The rule that governs every line: state what was measured, never what to do about it.** The report describes; the reader concludes. This is a deliberate reversal of how the v2/v3 mock prose reads, and it is the whole point of the block.

Allowed: *"Background is the largest error type: 35,010 detections, ΔAP50 0.2527."*
Banned: *"…which usually means missing annotations before it means a weak model."* — no "usually means", no "you should", no "consider raising", no diagnosis, no recommendation, no adjective of judgement ("poor", "healthy", "concerning").

Shape: heading `Highlights`, then 5–7 lines. Each line is one fact with its number inline, 13.5px, hairline `--rule-2` between lines, no bullets, no box, no background, no icons, no severity colour. Numbers `tabular-nums`; the number itself may take `--ink` weight 500 while the rest of the sentence sits at `--ink-1`, so the figures scan down the left. Deliberately the same flat register as the existing warnings list, so the two read as siblings rather than competing banners.

This is **not** the warnings list and does not replace it. Warnings say something is wrong; highlights say what the run is. Keep both, keep them adjacent, keep them visually identical in weight.

Mock these lines from the real E2E run:

```
Two classes, 21,305 instances. stack carries 58 of them and none landed in the test split.
Background is the largest error type: 35,010 detections, ΔAP50 0.2527, against a baseline AP50 of 0.1354.
F1 peaks at conf 0.000 and falls monotonically from there.
618 images across 6 sensor sizes; the median frame is 12.2 MP, about 30x the pixel budget at imgsz 640.
Objects per image runs from 1 to 146, median 20.
3 epochs on a 90% fraction, 166 images in the test split.
```

Each is checkable against a figure further down. Nothing in the list interprets.

## 2. Table of contents — floating, left

The v2 brief specified a side TOC and it was never built. Build it now, and make it float rather than sit in flow.

`position:fixed`, left gutter, vertically centred (`top:50%;transform:translateY(-50%)`), width 190px, **transparent — not glass, no border, no background, no box**. Items 12px, `--ink-3`, line-height 1.9. Active item: `--ink`, weight 500, plus a 2px × 1em brand-gradient tick immediately to its left (`::before`). Smooth-scrolls; `scroll-padding-top` already accounts for the nav.

Visible when the viewport has room for it beside the content column — roughly `@media (min-width:1220px)`; below that it is display:none and the nav anchor links carry navigation as they do today. Killed in `@media print`.

Active-section tracking via `IntersectionObserver` (~15 lines), not a scroll-position calculation. Give it `aria-label="Sections"` and `aria-current="true"` on the active link.

Add `Highlights` and `Environment` to the section list.

## 3. Experiment tags — at the top

The ClearML task carries tags, some added by this template (`image:<version>`, `ul-<ultralytics version>`, the YOLO task, the model basename, the data source type, and `resume` when set) and some added by hand in the ClearML UI. They are read **at report time, which is the end of the run**, so the list reflects the final state of the experiment.

Render them in the header block, on the line directly under the run name and above the meta line. Small chips: 11px, `--ink-2`, `background:var(--band)`, `border-radius:4px`, padding `2px 7px`, gap 6px, wrapping. **No border, no accent edge, no per-tag colour** — tags are identifiers, not a taxonomy, and colouring them would invent a meaning that is not there.

Mock: `image:0.2.11` · `ul-8.4.110` · `detect` · `yolo11n` · `CVAT` · `baseline` · `fruitlet`.

If the task has no tags, the line is omitted entirely — do not print an empty rail or a "no tags" placeholder.

## 4. Environment — where it ran and how long

A new section `Environment`, placed at the end, immediately before Caveats.

Two flat column groups separated by `border-left:1px solid var(--rule)`, exactly like the v3 image-files stat row — 11px uppercase `--ink-3` headings, 13px tabular value lines. No glass, no box, no accent edge.

**WHERE**
```
Worker           gpu-server2 / NEW-gpu-machine-server2
Host             ml-node-02
GPU              NVIDIA RTX A5000 · 24 GB · 1 of 2 visible
CUDA / driver    12.4 · 550.90.07
Torch / Python   2.6.0+cu124 · 3.14.0
Image            yolo-trainer:0.2.11
```

**HOW LONG**
```
Started          2026-07-31 05:41:12 UTC
Finished         2026-07-31 06:05:38 UTC
Wall clock       24 min 26 s
Training         21 min 04 s over 3 epochs
Mean epoch       7 min 01 s  (min 6:48 · max 7:19)
Final validation 1 min 12 s
```

Add one small chart beside them: **seconds per epoch**, a thin bar or step line over the epoch axis with direct value labels, same SVG conventions as everywhere else. Three epochs is a thin chart — that is fine and honest; do not pad it.

Follow the block with one `--ink-3` note stating that wall clock includes dataset download and export while the training figure does not.

**Data provenance, for the production phase.** Split it by how reliable each source is, and degrade per-field rather than dropping the section:
- Certain, read locally on the agent: hostname (`platform.node()`), Python version, `torch.__version__`, CUDA version, GPU name / count / memory (`torch.cuda.get_device_properties`), visible-device count.
- Best effort, from the ClearML task object inside `try/except`: tags, worker id, queue name, `started`. Any field that fails prints `unknown` on its own line — never removes the row, never fails the report.
- Timing: ultralytics' `trainer.train_time_start` and the per-epoch `trainer.epoch_time` the callbacks already log (`src/yolov8/callbacks.py:290`) give training duration and the epoch series without new instrumentation. Wall clock comes from the ClearML task start; if unavailable, print training duration only and say so.

## Deliverables

1. Updated `gen_wireframe.py`, regenerated `wireframe.html`.
2. Full-page `wireframe_light.png` and `wireframe_dark.png`, plus crops `highlights.png`, `toc.png` (showing the floating TOC with an active item), `tags.png`, `environment.png`.
3. Return paths, plus any deviation with a one-line reason.
