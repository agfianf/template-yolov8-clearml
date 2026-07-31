# Wireframe brief — report redesign (liquid glass × thinkingmachines minimalism)

You are building a WIREFRAME MOCK for user review — one self-contained `wireframe.html`, not the production report. It must look finished (this is a high-fidelity wireframe: real typography, real charts as static SVG, real numbers), but it only needs ~8 representative sections, no JS beyond an optional theme toggle (<30 lines). Everything below is settled research; follow it exactly.

## The user's five revision points (binding)

1. Liquid glass + glassmorphism, themed by the Binsho logo gradient (cyan #02B1F0 → teal #74D5C2 → lime #A3E85A).
2. **NO cards with a thick colored accent edge** (top/left/right/bottom bar on a rounded card) — that pattern is banned everywhere.
3. Compact — denser type scale, less padding than a typical dashboard.
4. Feel of thinkingmachines.ai/news/inkling-small: clean, minimal, charts inline with prose, section boundaries subtle (whitespace, not rules/boxes).
5. Chart values printed directly on the chart (no hover needed). Charts are hand-authored inline SVG (this is what thinkingmachines itself does — no chart lib).

## Design tokens (paste-ready, from verified research)

```css
:root {
  color-scheme: light dark;
  --brand-cyan:#02B1F0; --brand-teal:#74D5C2; --brand-lime:#A3E85A;
  --brand-grad: linear-gradient(100deg,#02B1F0 0%,#74D5C2 52%,#A3E85A 100%);
  /* text-safe ramp on light paper (WCAG-computed): */
  --brand-cyan-ink:#017BA8; --brand-teal-ink:#457F74; --brand-lime-ink:#618B36;
  --ink:#1B1D1E; --paper:#FFFFFF;
  --ink-1:color-mix(in srgb,var(--ink) 78%,var(--paper));
  --ink-2:color-mix(in srgb,var(--ink) 58%,var(--paper));
  --ink-3:color-mix(in srgb,var(--ink) 42%,var(--paper));
  --rule:  color-mix(in srgb,var(--ink) 12%,var(--paper));
  --rule-2:color-mix(in srgb,var(--ink)  7%,var(--paper));
  --grid:  color-mix(in srgb,var(--ink) 14%,var(--paper));
  --band:  color-mix(in srgb,var(--ink)  5%,var(--paper));
  --hover: color-mix(in srgb,var(--ink)  3%,var(--paper));
  --peer:  color-mix(in srgb,var(--ink) 42%,var(--paper));
  --good:#0ca30c; --warn:#b57614; --serious:#AD2111;  /* status; never series */
  --col:720px; --col-wide:960px; --col-full:1180px; --nav-h:52px; --pad:1.3rem;
  --font-sans: ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI Variable Text","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme: dark) { :root {
  --ink:#F2F4F5; --paper:#0E1113;
  --brand-cyan-ink:var(--brand-cyan); --brand-teal-ink:var(--brand-teal); --brand-lime-ink:var(--brand-lime);
  --good:#7CCB7C; --warn:#E0A63B; --serious:#E66767; } }
/* plus :root[data-theme="dark"] / [data-theme="light"] overrides winning both ways */
```

Layering (the reconciliation of glass and paper):
- **L0 ambience** — `body::before{position:fixed;z-index:-1}` with three radial-gradient brand blobs, alpha ≤ .14 light / ≤ .20 dark, `transparent 60-64%` stops. No filter, no animation.
- **L1 sheet** — the content column background: `color-mix(in srgb,var(--paper) 93%,transparent)`, NO blur. Gradient shows at margins and faintly through.
- **L2 content** — prose/tables/charts/thumbs fully flat and opaque-on-sheet.
- **L3 glass, exactly 3 surfaces**: sticky nav, KPI strip, lightbox scrim (mock the first two; lightbox optional).

Glass recipe:
```css
.glass{background:color-mix(in srgb,var(--paper) 72%,transparent);
  -webkit-backdrop-filter:blur(14px) saturate(150%);backdrop-filter:blur(14px) saturate(150%);
  border:1px solid color-mix(in srgb,var(--ink) 10%,transparent);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.55),0 1px 2px rgba(16,24,32,.05),0 8px 24px -12px rgba(16,24,32,.14);
  border-radius:12px;}
/* dark: fill 66%, saturate 165%, border rgba(255,255,255,.10), inset highlight .09 */
/* + @supports-not / prefers-reduced-transparency / forced-colors fallbacks to opaque paper */
```

Type scale (compact): h1 26/600 -.011em; h2 19/600; h3 15/500 (or 15/400 italic); body **15/1.6**; dense 14; table **13** tabular-nums, thead 11/600 uppercase .05em; caption 12.5 `--ink-3` left, "Fig N." at 500; chart tick 11 `--ink-3`; chart panel title 13/500 `--ink-2` left (never bold centred); direct label 11/500 series-colored; KPI value 30/600 tabular -.02em; KPI label 11/500 uppercase .06em; nav 12.5. All numerals `font-variant-numeric:tabular-nums`.

Spacing: paragraph 16; block↔prose 20; h2 top/bottom **48/12**; section→section ≈72 whitespace ONLY (zero `<hr>`, zero boxes, zero background change per section); figure→caption 12; KPI strip padding 18px 20px.

## Component specs

**Nav (glass #1)**: sticky top, height 52, square corners, bottom hairline + **2px brand-gradient underline** (`::after` full width) — this is the brand signature. Left: "Binsho Solutions · Evaluation report" (11px uppercase eyebrow + run name). Right: anchor links 12.5px + theme toggle. `scroll-padding-top` set.

**Side TOC** (≥1180px): sticky, 210px, transparent (NOT glass), 12.5px, active item = `--ink` 500 + 2×1em gradient tick at left.

**Header block**: h1 run name; meta line 12.5px `--ink-3` (model · imgsz · split · date · commit · image tag) separated by "·". No box.

**KPI strip (glass #2, NOT sticky)**: single glass panel, one row grid; each KPI = 30px value + 11px uppercase label + 11px basis note (`box · conf 0.001`); separators `border-left:1px solid var(--rule)` between items; NO per-KPI box/background/icon/accent. Use real numbers from our E2E run: mAP50-95 0.0345 · mAP50 0.1336 · Precision 0.0703 · Recall 0.5760 · F1 0.1254 · τ* 0.000 · 166 images.

**Warnings**: plain lines, not banners-in-boxes: 13px, a 7px status dot (colored circle) + text, hairline `--rule-2` between lines. E.g. "‼ stack has no ground truth in the test split — silently absent from mAP" (serious), "τ* = 0.000 — F1 never rises above noise at 3 epochs" (warn).

**Charts — hand-authored static inline SVG.** Conventions (thinkingmachines-verified): transparent background; gridlines 1px `--grid` horizontal-only, `vector-effect:non-scaling-stroke`; no tick marks, tick text 11px `--ink-3`; axis line omitted (zero-gridline does the work); series = **max 2 identity colors + `--peer` grey**: hero `var(--brand-cyan-ink)`, secondary `var(--brand-teal-ink)`; lime ONLY for good/TP status; line 1.7px round cap/join; markers r4.2 with `stroke:var(--paper)` 1-1.5px; reference lines `stroke-dasharray:5 5` 1.5px `--peer` with an 11px label; **direct value labels on every mark**: 11px/500, fill = series color, `paint-order:stroke;stroke:var(--paper);stroke-width:3.5px;stroke-linejoin:round`. All colors via `fill="var(--…)"` / class + CSS — zero hex inside SVG. Width via `viewBox` + `width:100%`.

Mock these four charts with the REAL E2E data:
1. Instances per class, horizontal bars log-x: fruitlet 21,247 vs stack 58 — value labels at bar ends.
2. TIDE single stacked horizontal bar: Cls .0001 / Loc .0786 / Both .0061 / Dupe .0005 / Bkg .2527 / Miss .0970, 2px paper gaps between segments, labels in-segment when wide else outside, two dashed ceiling vlines (ΔAP_FP .450, ΔAP_FN .097) with staggered labels. Segments: use brand-cyan-ink, brand-teal-ink and greys — max 2 identities + greys; label every segment directly so color identity isn't load-bearing.
3. P/R/F1 vs confidence, 3 thin lines (cyan-ink, teal-ink, peer), direct end-of-line labels with halo, dashed vline at τ*.
4. Objects-per-image histogram, ~30 thin bars, p99.5 dashed vline labeled "p99.5 = 146"; label only the peak bar + the vline (selective labels, not every bar).

**Table (per-class)**: flat paper. `border:0`; horizontal hairlines via `td::after` 1px `--rule`; thead 11px uppercase sticky-look; row hover 3% tint; support<30 row in `--ink-3` (mock `stack  58` dimmed); best-value cell = lime **wash** `color-mix(in srgb,var(--brand-lime) 18%,var(--paper))`, never a border. Columns: Class Support AP50-95 AP50 P R F1 TP FP FN τ*. Real rows: fruitlet + stack from E2E.

**Gallery**: grid minmax(148px,1fr) gap 10px; img aspect-1 cover radius 6px, hairline via `box-shadow:0 0 0 1px`; caption 11px `--ink-3` below; outcome legend above as plain text with 9px swatches (TP green solid · FP red solid · FN amber dashed). Mock with 8 colored-placeholder `<svg>` or CSS-gradient tiles (no real photos needed).

**Caveats/footer**: 12.5px `--ink-3` bullet list + generation line.

**Print block**: `@media print` — kill ambience/glass/nav, open details, light tokens.

## Do NOT (the ban list — the wireframe must visibly honor these)

Edge-accent bars on cards (any side, any thickness); glass on tables/prose/charts/thumbnails; >3 blur surfaces; 0.12-alpha glass under text; `0 8px 32px rgba(0,0,0,.25)` shadows; boxes around every component; raw brand color as light-mode text; `<hr>` between sections; per-section background tints; bold centred chart titles; pie/donut/gauge; values only on hover; more than 2 identity colors + grey per chart; blur >20px outside the lightbox scrim.

## Deliverables

1. `wireframe.html` in the scratchpad dir — complete, self-contained, both themes (media query + data-theme override + a tiny toggle button), responsive (col 720 content, charts may break out to 960).
2. Screenshots via headless google-chrome: `wireframe_light.png` and `wireframe_dark.png` (1280×full-height, use --window-size=1280,4400 or as needed).
3. Return: file paths + any spec deviation you made with one-line why.
