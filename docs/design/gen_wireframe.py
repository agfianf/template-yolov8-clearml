"""Generate wireframe.html - high-fidelity report redesign mock (v2)."""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "wireframe.html"


def fmt(x, n=1):
    s = f"{x:.{n}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
CSS = """
:root{
  color-scheme: light dark;
  --brand-cyan:#02B1F0; --brand-teal:#74D5C2; --brand-lime:#A3E85A;
  --brand-grad:linear-gradient(100deg,#02B1F0 0%,#74D5C2 52%,#A3E85A 100%);
  --brand-cyan-ink:#017BA8; --brand-teal-ink:#457F74; --brand-lime-ink:#618B36;
  --ink:#1B1D1E; --paper:#FFFFFF;
  --ink-1:color-mix(in srgb,var(--ink) 78%,var(--paper));
  --ink-2:color-mix(in srgb,var(--ink) 58%,var(--paper));
  --ink-3:color-mix(in srgb,var(--ink) 42%,var(--paper));
  --rule:color-mix(in srgb,var(--ink) 12%,var(--paper));
  --rule-2:color-mix(in srgb,var(--ink) 7%,var(--paper));
  --grid:color-mix(in srgb,var(--ink) 14%,var(--paper));
  --band:color-mix(in srgb,var(--ink) 5%,var(--paper));
  --hover:color-mix(in srgb,var(--ink) 3%,var(--paper));
  --peer:color-mix(in srgb,var(--ink) 42%,var(--paper));
  --good:#0ca30c; --warn:#b57614; --serious:#AD2111;
  --col:720px; --col-wide:960px; --col-full:1060px; --nav-h:52px; --pad:1.3rem;
  --font-sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI Variable Text","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
  --sheet:color-mix(in srgb,var(--paper) 93%,transparent);
  --glass-fill:color-mix(in srgb,var(--paper) 72%,transparent);
  --glass-border:color-mix(in srgb,var(--ink) 10%,transparent);
  --glass-hi:rgba(255,255,255,.55);
  --glass-sat:150%;
  --amb-1:14%; --amb-2:12%; --amb-3:11%;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ink:#F2F4F5; --paper:#0E1113;
    --brand-cyan-ink:var(--brand-cyan); --brand-teal-ink:var(--brand-teal);
    --brand-lime-ink:var(--brand-lime);
    --good:#7CCB7C; --warn:#E0A63B; --serious:#E66767;
    --glass-fill:color-mix(in srgb,var(--paper) 66%,transparent);
    --glass-border:rgba(255,255,255,.10);
    --glass-hi:rgba(255,255,255,.09);
    --glass-sat:165%;
    --amb-1:20%; --amb-2:17%; --amb-3:15%;
  }
}
:root[data-theme="dark"]{
  --ink:#F2F4F5; --paper:#0E1113;
  --brand-cyan-ink:var(--brand-cyan); --brand-teal-ink:var(--brand-teal);
  --brand-lime-ink:var(--brand-lime);
  --good:#7CCB7C; --warn:#E0A63B; --serious:#E66767;
  --glass-fill:color-mix(in srgb,var(--paper) 66%,transparent);
  --glass-border:rgba(255,255,255,.10);
  --glass-hi:rgba(255,255,255,.09);
  --glass-sat:165%;
  --amb-1:20%; --amb-2:17%; --amb-3:15%;
}

*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:calc(var(--nav-h) + 20px)}
body{
  margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--font-sans);font-size:15px;line-height:1.6;
  font-variant-numeric:tabular-nums;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
}
body::before{
  content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
  background:
    radial-gradient(58% 46% at 10% -4%, color-mix(in srgb,var(--brand-cyan) var(--amb-1),transparent) 0%, transparent 62%),
    radial-gradient(52% 42% at 92% 8%, color-mix(in srgb,var(--brand-teal) var(--amb-2),transparent) 0%, transparent 60%),
    radial-gradient(64% 48% at 68% 96%, color-mix(in srgb,var(--brand-lime) var(--amb-3),transparent) 0%, transparent 64%);
}

/* ---------- glass (3 surfaces only: nav, kpi strip, lightbox scrim) ---- */
.glass{
  background:var(--glass-fill);
  -webkit-backdrop-filter:blur(14px) saturate(var(--glass-sat));
  backdrop-filter:blur(14px) saturate(var(--glass-sat));
  border:1px solid var(--glass-border);
  box-shadow:inset 0 1px 0 var(--glass-hi),0 1px 2px rgba(16,24,32,.05),
             0 8px 24px -12px rgba(16,24,32,.14);
  border-radius:12px;
}
@supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){
  .glass{background:var(--paper)}
}
@media (prefers-reduced-transparency:reduce){
  .glass{background:var(--paper);backdrop-filter:none;-webkit-backdrop-filter:none}
  body::before{display:none}
}
@media (forced-colors:active){
  .glass{background:Canvas;border:1px solid CanvasText;backdrop-filter:none}
  body::before{display:none}
}

/* ---------------------------- nav (the only navigation) ---------------- */
.nav{
  position:sticky;top:0;z-index:40;height:var(--nav-h);
  border-radius:0;border-left:0;border-right:0;border-top:0;
  border-bottom:1px solid var(--rule);
  box-shadow:inset 0 1px 0 var(--glass-hi),0 1px 2px rgba(16,24,32,.04);
}
.nav::after{
  content:"";position:absolute;left:0;right:0;bottom:-2px;height:2px;
  background:var(--brand-grad);
}
.nav-in{
  max-width:var(--col-full);margin:0 auto;padding:0 20px;height:100%;
  display:flex;align-items:center;gap:18px;
}
.brandmark{display:flex;align-items:center;gap:9px;min-width:0;flex:none}
.brandmark .dot{
  width:16px;height:16px;border-radius:5px;background:var(--brand-grad);flex:none;
}
.brand-txt{min-width:0;line-height:1.15}
.eyebrow{
  font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);display:block;
}
.brand-run{
  font-size:12.5px;font-weight:500;color:var(--ink-1);display:block;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.nav-links{
  margin-left:auto;display:flex;align-items:center;gap:15px;min-width:0;
  height:100%;overflow-x:auto;overflow-y:hidden;
  scrollbar-width:none;-ms-overflow-style:none;-webkit-overflow-scrolling:touch;
}
.nav-links::-webkit-scrollbar{display:none}
.nav-links a{
  font-size:12.5px;color:var(--ink-2);text-decoration:none;white-space:nowrap;
  flex:none;padding:17px 0;
}
.nav-links a:hover{color:var(--ink)}
.tgl{
  font:500 12.5px/1 var(--font-sans);color:var(--ink-2);cursor:pointer;
  background:transparent;border:1px solid var(--rule);border-radius:7px;
  padding:5px 9px;flex:none;
}
.tgl:hover{color:var(--ink);background:var(--hover)}

/* ---------------------------- shell ----------------------------------- */
.wrap{max-width:var(--col-full);margin:0 auto;padding:0 20px}
.sheet{
  background:var(--sheet);border-radius:14px;
  padding:34px 44px 56px;margin:26px 0 30px;
}
@media (max-width:640px){.sheet{padding:24px 18px 40px;border-radius:10px}
  .wrap{padding:0 12px}}

/* one reading column, 720 centred; only .wide breaks out */
.sheet > *,.sheet section > *{
  max-width:var(--col);margin-left:auto;margin-right:auto;
}
.sheet > section,.sheet > .wide,.sheet section > .wide{max-width:none}

/* ---------------------------- type ------------------------------------ */
h1{font-size:26px;font-weight:600;letter-spacing:-.011em;line-height:1.2;margin:0 0 8px}
h2{font-size:19px;font-weight:600;letter-spacing:-.005em;margin:48px 0 12px;line-height:1.3}
h3{font-size:15px;font-weight:500;margin:26px 0 8px;color:var(--ink-1)}
p{margin:0 0 16px}
.lede{color:var(--ink-1)}
.meta{font-size:12.5px;color:var(--ink-3);line-height:1.7;margin:0}
.meta b{font-weight:500;color:var(--ink-2)}
.dense{font-size:14px}
.note{font-size:12.5px;color:var(--ink-3);margin:0 0 16px}
code,.mono{
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:.9em;color:var(--ink-1);
}
section{scroll-margin-top:calc(var(--nav-h) + 20px)}
section + section{margin-top:24px}

/* ---------------------------- kpi strip (glass #2) --------------------- */
.kpis{padding:18px 20px;margin:22px 0 0;display:flex;flex-wrap:wrap}
.kpi{flex:1 1 0;padding:2px 10px;border-left:1px solid var(--rule);min-width:0}
.kpi:first-child{border-left:0;padding-left:0}
.kpi:last-child{padding-right:0}
.kpi-top{
  display:flex;align-items:center;justify-content:space-between;gap:6px;
  height:30px;margin-bottom:5px;
}
.ico{
  width:16px;height:16px;flex:none;display:block;overflow:visible;
  fill:none;stroke:var(--ink-3);stroke-width:1.6;
  stroke-linecap:round;stroke-linejoin:round;
}
.ring{width:30px;height:30px;flex:none;display:block}
.rk{fill:none;stroke:var(--band);stroke-width:3}
.ra{fill:none;stroke:var(--brand-cyan-ink);stroke-width:3;stroke-linecap:round}
.kpi-v{
  font-size:27px;font-weight:600;letter-spacing:-.02em;line-height:1.05;display:block;
}
.kpi-v .u{font-size:13px;font-weight:500;color:var(--ink-3);letter-spacing:0}
.kpi-l{
  font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-2);display:block;margin-top:6px;
}
.kpi-b{font-size:11px;color:var(--ink-3);display:block;margin-top:2px;line-height:1.35}
.kpi-l .tau,thead th .tau{text-transform:none;letter-spacing:0}
@media (max-width:900px){
  .kpi{flex:1 1 28%;padding-top:6px;padding-bottom:6px}
  .kpi:nth-child(3n+1){border-left:0;padding-left:0}
}

/* ---------------------------- flags ----------------------------------- */
.flags{margin:26px 0 0;padding:0;list-style:none}
.flags li{
  display:flex;gap:9px;align-items:baseline;font-size:13px;line-height:1.5;
  padding:8px 0;border-top:1px solid var(--rule-2);color:var(--ink-1);
}
.flags li:first-child{border-top:0}
.sd{width:7px;height:7px;border-radius:50%;flex:none;position:relative;top:-1px}
.sd.serious{background:var(--serious)}
.sd.warn{background:var(--warn)}
.sd.good{background:var(--good)}
.flags b{font-weight:500}

/* ---------------------------- figures ---------------------------------- */
figure{margin:20px 0 0}
.fig-t{font-size:13px;font-weight:500;color:var(--ink-2);margin:0 0 8px;text-align:left}
figcaption{
  font-size:12.5px;color:var(--ink-3);margin-top:12px;text-align:left;
  line-height:1.5;max-width:var(--col);
}
figcaption b{font-weight:500;color:var(--ink-2)}
svg.chart{width:100%;height:auto;display:block;overflow:visible}

/* chart primitives (zero hex inside svg) */
.gl{stroke:var(--grid);stroke-width:1;vector-effect:non-scaling-stroke;fill:none}
.gl-2{stroke:var(--rule-2);stroke-width:1;vector-effect:non-scaling-stroke;fill:none}
.tk{font-size:11px;fill:var(--ink-3)}
.axl{font-size:11px;fill:var(--ink-3)}
.dl{
  font-size:11px;font-weight:500;
  paint-order:stroke;stroke:var(--paper);stroke-width:3.5px;stroke-linejoin:round;
}
.dl-q{font-size:11px;font-weight:400;fill:var(--ink-3);
  paint-order:stroke;stroke:var(--paper);stroke-width:3.5px;stroke-linejoin:round;}
.ln{fill:none;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;
  vector-effect:non-scaling-stroke}
.mk{stroke:var(--paper);stroke-width:1.5}
.ref{stroke:var(--peer);stroke-width:1.5;stroke-dasharray:5 5;fill:none;
  vector-effect:non-scaling-stroke}
.lead{stroke:var(--rule);stroke-width:1;fill:none;vector-effect:non-scaling-stroke}
.f-cyan{fill:var(--brand-cyan-ink)} .s-cyan{stroke:var(--brand-cyan-ink)}
.f-teal{fill:var(--brand-teal-ink)} .s-teal{stroke:var(--brand-teal-ink)}
.f-peer{fill:var(--peer)} .s-peer{stroke:var(--peer)}
.f-ink1{fill:var(--ink-1)} .f-ink2{fill:var(--ink-2)} .f-paper{fill:var(--paper)}
.f-g1{fill:color-mix(in srgb,var(--ink) 30%,var(--paper))}
.f-g2{fill:color-mix(in srgb,var(--ink) 20%,var(--paper))}
.f-g3{fill:color-mix(in srgb,var(--ink) 13%,var(--paper))}
.seg-t{font-size:11px;font-weight:500}
.b-lbl{font-size:11px;font-weight:500;fill:var(--ink-1)}
.hit{fill:none;pointer-events:all}

/* sequential ramp for density: --band -> --brand-cyan-ink, one hue */
.hm-0{fill:var(--band)}
.hm-1{fill:color-mix(in srgb,var(--brand-cyan-ink) 16%,var(--band))}
.hm-2{fill:color-mix(in srgb,var(--brand-cyan-ink) 31%,var(--band))}
.hm-3{fill:color-mix(in srgb,var(--brand-cyan-ink) 47%,var(--band))}
.hm-4{fill:color-mix(in srgb,var(--brand-cyan-ink) 64%,var(--band))}
.hm-5{fill:color-mix(in srgb,var(--brand-cyan-ink) 82%,var(--band))}
.hm-6{fill:var(--brand-cyan-ink)}
.hm-lbl{font-size:11px;font-weight:500;fill:var(--paper)}
.hm-lbl-d{font-size:11px;font-weight:500;fill:var(--ink)}
.mx-lbl{font-size:10.5px;fill:var(--ink-2)}

/* ---------------------------- table ------------------------------------ */
.tbl-wrap{overflow-x:auto;margin:20px 0 0}
table{
  border-collapse:collapse;border:0;width:100%;font-size:13px;
  font-variant-numeric:tabular-nums;
}
thead th{
  font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink-2);text-align:right;padding:0 10px 7px 0;border:0;
  white-space:nowrap;position:relative;
}
thead th:first-child{text-align:left;padding-left:0}
thead th::after{
  content:"";position:absolute;left:0;right:0;bottom:0;height:1px;background:var(--rule);
}
tbody td{
  padding:6px 10px 6px 0;text-align:right;border:0;position:relative;white-space:nowrap;
}
tbody td:first-child{text-align:left;padding-left:0;font-weight:500}
tbody td::after{
  content:"";position:absolute;left:0;right:0;bottom:0;height:1px;background:var(--rule-2);
}
tbody tr:hover td{background:var(--hover)}
tbody tr.dim td{color:var(--ink-3);font-weight:400}
tbody tr.dim td:first-child{font-weight:500}
td.best{background:color-mix(in srgb,var(--brand-lime) 18%,var(--paper))}
tbody tr:hover td.best{background:color-mix(in srgb,var(--brand-lime) 24%,var(--paper))}
td .ex{display:block;width:40px;height:26px}
.ex-f{fill:none;stroke:var(--rule);stroke-width:1}
.ex-b{fill:none;stroke:var(--warn);stroke-width:1.3}
.ex-p{fill:none;stroke:var(--peer);stroke-width:1.3}

/* ---------------------------- co-occurrence ---------------------------- */
.cooc{display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap;margin:18px 0 0}
.cooc-stat{flex:1 1 250px;min-width:230px}
.cooc-n{font-size:27px;font-weight:600;letter-spacing:-.02em;line-height:1.1;display:block}
.cooc-n .u{font-size:14px;font-weight:500;color:var(--ink-3);letter-spacing:0}
.cooc-l{
  font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-2);display:block;margin-top:6px;
}
.cooc-p{font-size:12.5px;color:var(--ink-3);margin:12px 0 0;line-height:1.55}
.cooc-p b{font-weight:500;color:var(--ink-2)}
.cooc-mx{flex:0 0 auto;width:312px;max-width:100%}
.cooc-mx svg{width:100%;height:auto;display:block;overflow:visible}
.illus{
  font-size:11px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 4px;
}

/* ---------------------------- image-file stat row ---------------------- */
.statrow{display:flex;flex-wrap:wrap;margin:20px 0 0}
.statcol{flex:1 1 0;min-width:180px;padding:0 20px;border-left:1px solid var(--rule)}
.statcol:first-child{border-left:0;padding-left:0}
.statcol:last-child{padding-right:0}
.stat-h{
  font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 8px;
}
.stat-l{
  display:flex;justify-content:space-between;gap:16px;font-size:13px;
  color:var(--ink-1);margin:0 0 5px;font-variant-numeric:tabular-nums;
}
.stat-l b{font-weight:500;color:var(--ink)}
.stat-l.zero,.stat-l.zero b{color:var(--ink-3);font-weight:400}
@media (max-width:640px){
  .statcol{flex:1 1 100%;border-left:0;padding:0;margin-top:14px}
  .statcol:first-child{margin-top:0}
}

/* ---------------------------- gallery ---------------------------------- */
.galwrap{margin:18px 0 0}
.gal-head{
  display:flex;align-items:center;gap:18px;flex-wrap:wrap;
  font-size:12.5px;margin:0 0 10px;
}
.tabs{display:flex;align-items:center;gap:18px}
.tab{
  position:relative;background:none;border:0;padding:0 0 6px;margin:0;cursor:pointer;
  font:400 12.5px/1.4 var(--font-sans);color:var(--ink-3);
}
.tab:hover{color:var(--ink-2)}
.tab[aria-selected="true"]{color:var(--ink);font-weight:500}
.tab[aria-selected="true"]::after{
  content:"";position:absolute;left:0;right:0;bottom:0;height:2px;
  background:var(--brand-grad);
}
.tab:focus-visible{outline:2px solid var(--brand-cyan-ink);outline-offset:3px}
.gal-meta{margin-left:auto;color:var(--ink-3);white-space:nowrap}
.anno{
  display:inline-flex;align-items:center;gap:9px;background:none;border:0;padding:0;
  margin:0;cursor:pointer;font:400 12px/1 var(--font-sans);color:var(--ink-2);
}
.anno .trk{
  position:relative;width:30px;height:16px;border-radius:8px;flex:none;
  background:var(--rule);transition:background 120ms ease;
}
.anno[aria-checked="true"] .trk{background:var(--brand-grad)}
.anno .knb{
  position:absolute;top:2px;left:2px;width:12px;height:12px;border-radius:50%;
  background:var(--paper);transition:transform 120ms ease;
  box-shadow:0 1px 2px rgba(16,24,32,.3);
}
.anno[aria-checked="true"] .knb{transform:translateX(14px)}
.anno:focus-visible{outline:2px solid var(--brand-cyan-ink);outline-offset:3px}

.gal{
  --gal-mask:linear-gradient(to bottom,transparent 0,#000 40px,
              #000 calc(100% - 48px),transparent 100%);
  -webkit-mask-image:var(--gal-mask);mask-image:var(--gal-mask);
  max-height:352px;overflow-y:auto;overscroll-behavior:contain;
  scrollbar-width:none;-ms-overflow-style:none;
}
.gal::-webkit-scrollbar{display:none}
.gal.at-start{
  --gal-mask:linear-gradient(to bottom,#000 0,#000 calc(100% - 48px),transparent 100%);
}
.gal.at-end{--gal-mask:linear-gradient(to bottom,transparent 0,#000 40px,#000 100%)}
.gal.at-start.at-end{--gal-mask:none}
.grid{
  display:grid;grid-template-columns:repeat(7,1fr);gap:6px;
  margin:0;padding:0;list-style:none;
}
@media (max-width:640px){.grid{grid-template-columns:repeat(5,1fr)}}
@media (max-width:440px){.grid{grid-template-columns:repeat(4,1fr)}}
.thumb{margin:0;min-width:0}
.thumb .ph{
  position:relative;aspect-ratio:1;border-radius:5px;overflow:hidden;display:block;
  width:100%;box-shadow:0 0 0 1px var(--rule);border:0;padding:0;cursor:zoom-in;
  background:linear-gradient(145deg,
    color-mix(in srgb,var(--brand-lime) 26%,var(--ink-3)) 0%,
    color-mix(in srgb,var(--brand-teal) 20%,var(--ink-2)) 58%,
    color-mix(in srgb,var(--brand-cyan) 14%,var(--ink-1)) 100%);
}
.thumb .ph:hover{box-shadow:0 0 0 1px var(--ink-3)}
.thumb .ph:focus-visible{outline:2px solid var(--brand-cyan-ink);outline-offset:2px}
.thumb .ph svg{position:absolute;inset:0;display:block;width:100%;height:100%}
.ov{opacity:0;transition:opacity 120ms ease}
.galwrap[data-anno="on"][data-tab="outcome"] .ov-outcome,
.galwrap[data-anno="on"][data-tab="pred"] .ov-pred,
.galwrap[data-anno="on"][data-tab="gt"] .ov-gt{opacity:1}
.ob{fill:none;stroke-width:1.6;vector-effect:non-scaling-stroke}
.ob-tp{stroke:var(--good)}
.ob-fp{stroke:var(--serious)}
.ob-fn{stroke:var(--warn);stroke-dasharray:4 3}
.ob-pred{stroke:var(--brand-cyan-ink)}
.ob-gt{stroke:var(--brand-teal-ink)}
.ovt{
  font-size:6.4px;font-weight:500;
  paint-order:stroke;stroke:var(--paper);stroke-width:1.8px;stroke-linejoin:round;
}
/* the one transient blur surface: the hover caption strip (no border, no shadow) */
.thumb .cap{
  position:absolute;left:0;right:0;bottom:0;padding:3px 5px;
  font-size:10.5px;line-height:1.4;color:var(--ink-1);text-align:left;
  background:color-mix(in srgb,var(--paper) 82%,transparent);
  -webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);
  opacity:0;transition:opacity 100ms ease;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.thumb .ph:hover .cap,.thumb .ph:focus-visible .cap{opacity:1}

.gal-legend{font-size:11px;color:var(--ink-3);margin:12px 0 0;min-height:17px}
.gal-legend > span{display:none;gap:14px;align-items:center;flex-wrap:wrap}
.galwrap[data-tab="outcome"] .lg-outcome,
.galwrap[data-tab="pred"] .lg-pred,
.galwrap[data-tab="gt"] .lg-gt{display:inline-flex}
.gal-legend i{display:inline-flex;align-items:center;gap:6px;font-style:normal}
.sw{width:9px;height:9px;border-radius:2px;flex:none}
.sw.fn{width:11px;height:11px}
.sw.tp{background:var(--good)}
.sw.fp{background:var(--serious)}
.sw.fn{background:transparent;border:2px dashed var(--warn);border-radius:2px}
.sw.pd{background:var(--brand-cyan-ink)}
.sw.gt{background:var(--brand-teal-ink)}

/* ---------------------------- lightbox (glass #3) ---------------------- */
.lb{
  position:fixed;inset:0;z-index:80;display:none;
  align-items:center;justify-content:center;flex-direction:column;gap:14px;
  background:color-mix(in srgb,var(--paper) 42%,transparent);
  -webkit-backdrop-filter:blur(20px) saturate(140%);
  backdrop-filter:blur(20px) saturate(140%);
}
.lb.on{display:flex}
.lb-stage{
  position:relative;background:var(--paper);border-radius:10px;overflow:hidden;
  box-shadow:0 0 0 1px var(--rule),0 26px 60px -34px rgba(16,24,32,.55);
  width:min(76vw,760px);aspect-ratio:4/3;
}
.lb-img{position:absolute;inset:0;
  background:linear-gradient(145deg,
    color-mix(in srgb,var(--brand-lime) 26%,var(--ink-3)) 0%,
    color-mix(in srgb,var(--brand-teal) 20%,var(--ink-2)) 58%,
    color-mix(in srgb,var(--brand-cyan) 14%,var(--ink-1)) 100%);}
.lb-img svg{position:absolute;inset:0;width:100%;height:100%}
#lbc{position:absolute;inset:0;width:100%;height:100%}
.lb-bar{
  width:min(76vw,760px);display:flex;align-items:baseline;gap:16px;
  font-size:12.5px;color:var(--ink-2);
}
.lb-cap{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lb-n{margin-left:auto;color:var(--ink-3);flex:none;font-variant-numeric:tabular-nums}
.lb-btn{
  background:var(--paper);border:1px solid var(--rule);color:var(--ink-1);
  border-radius:50%;width:40px;height:40px;cursor:pointer;line-height:1;
  font-size:19px;display:flex;align-items:center;justify-content:center;padding:0;
  box-shadow:0 2px 8px -3px rgba(16,24,32,.3);
}
.lb-btn:hover{background:var(--hover)}
.lb-btn.prev{position:absolute;left:max(18px,2.5vw);top:50%;transform:translateY(-50%)}
.lb-btn.next{position:absolute;right:max(18px,2.5vw);top:50%;transform:translateY(-50%)}
.lb-btn.x{position:absolute;right:max(18px,2.5vw);top:18px;font-size:14px}

/* ---------------------------- hover tooltip ---------------------------- */
.tip{
  position:fixed;z-index:90;pointer-events:none;display:none;
  font-size:12px;line-height:1.45;color:var(--ink-1);background:var(--paper);
  border:1px solid var(--rule);border-radius:6px;padding:5px 8px;max-width:270px;
  box-shadow:0 2px 10px -4px rgba(16,24,32,.35);font-variant-numeric:tabular-nums;
}

/* ---------------------------- footer ----------------------------------- */
.caveats{font-size:12.5px;color:var(--ink-3);margin:14px 0 0;padding-left:18px}
.caveats li{margin:0 0 7px}
.gen{
  font-size:12.5px;color:var(--ink-3);margin:34px 0 0;padding-top:14px;
  border-top:1px solid var(--rule-2);
}

/* ---------------------------- print ------------------------------------ */
@media print{
  :root{--ink:#1B1D1E;--paper:#FFFFFF;--brand-cyan-ink:#017BA8;
        --brand-teal-ink:#457F74;--brand-lime-ink:#618B36;
        --good:#0ca30c;--warn:#b57614;--serious:#AD2111}
  body::before{display:none}
  .nav,.tgl,.lb,.tip{display:none!important}
  .glass{background:none;border:0;box-shadow:none;backdrop-filter:none;
         -webkit-backdrop-filter:none}
  .kpis{border-top:1px solid #ccc;border-bottom:1px solid #ccc;border-radius:0}
  .sheet{background:none;padding:0;margin:0;max-width:none}
  .wrap{display:block;max-width:none;padding:0}
  .gal{max-height:none;overflow:visible;-webkit-mask-image:none;mask-image:none}
  .ov-outcome{opacity:1}
  .anno,.tabs{display:none}
  details{display:block}
  section{break-inside:avoid}
  h2{break-after:avoid}
}
"""


# --------------------------------------------------------------------------
# 16px icons, hand-drawn thin outlines (original paths, Hugeicons-ish)
# --------------------------------------------------------------------------
ICONS = {
    "target": '<circle cx="12" cy="12" r="8.6"/><circle cx="12" cy="12" r="4.6"/>'
              '<circle cx="12" cy="12" r="1.1"/>',
    "gauge": '<path d="M3 17a9 9 0 0 1 18 0"/><path d="M12 17l4.4-5.2"/>'
             '<circle cx="12" cy="17" r="1.2"/>',
    "crosshair": '<circle cx="12" cy="12" r="7"/>'
                 '<path d="M12 1.4v6M12 16.6v6M1.4 12h6M16.6 12h6"/>',
    "recall": '<path d="M20.4 12a8.4 8.4 0 1 1-2.9-6.4"/>'
              '<path d="M20.9 3.2v4.7h-4.7"/><path d="M12 7.8v4.5l3 1.8"/>',
    "scales": '<path d="M12 5.4v14.4M8 19.8h8M4 8.6h16"/>'
              '<path d="M4 8.6l-2.6 5.8h5.2z"/><path d="M20 8.6l-2.6 5.8h5.2z"/>'
              '<circle cx="12" cy="4.2" r="1.3"/>',
    "sliders": '<path d="M3 6.8h10.4M18.6 6.8H21M3 12h4.4M12.6 12H21'
               'M3 17.2h8M17.2 17.2H21"/><circle cx="16" cy="6.8" r="2.1"/>'
               '<circle cx="10" cy="12" r="2.1"/><circle cx="14.6" cy="17.2" r="2.1"/>',
    "photo": '<rect x="3" y="5" width="18" height="14" rx="2.6"/>'
             '<circle cx="8.4" cy="9.8" r="1.5"/>'
             '<path d="M3.4 16.6l4.8-4.4 4.4 3.9 3.1-2.6 4.9 4.3"/>',
}


def icon(name):
    return f'<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">{ICONS[name]}</svg>'


def ring(frac, tip):
    r = 12.0
    circ = 2 * math.pi * r
    d = circ * max(0.0, min(1.0, frac))
    return (f'<svg class="ring" viewBox="0 0 30 30" role="img" data-tip="{tip}" '
            f'aria-label="{tip}"><title>{tip}</title>'
            f'<circle class="rk" cx="15" cy="15" r="{r}"/>'
            f'<circle class="ra" cx="15" cy="15" r="{r}" '
            f'stroke-dasharray="{d:.2f} {circ - d:.2f}" '
            f'transform="rotate(-90 15 15)"/></svg>')


# --------------------------------------------------------------------------
# Chart 1 -- instances per class, log-x horizontal bars
# --------------------------------------------------------------------------
def chart_instances():
    X0, X1 = 82, 616
    dec = 5.0  # 1 .. 100000
    sx = lambda v: X0 + (math.log10(v) / dec) * (X1 - X0)
    rows = [
        ("fruitlet", 21247, "21,247", "cyan",
         "13,476 train &#183; 3,714 valid &#183; 4,057 test",
         "fruitlet &#183; 21,247 instances &#183; 13,476 train / 3,714 valid / "
         "4,057 test"),
        ("stack", 58, "58", "teal", "48 train &#183; 10 valid &#183; 0 test",
         "stack &#183; 58 instances &#183; 48 train / 10 valid / 0 test"),
    ]
    bh, gap, top = 30, 26, 16
    s = ['<svg class="chart" viewBox="0 0 660 148" role="img" '
         'aria-label="Instances per class: fruitlet 21,247, stack 58">']
    ticks = [1, 10, 100, 1000, 10000, 100000]
    tlab = ["1", "10", "100", "1K", "10K", "100K"]
    ybot = top + 2 * bh + gap + 6
    for v in ticks:
        x = sx(v)
        s.append(f'<line class="gl-2" x1="{x:.1f}" y1="{top-6:.0f}" '
                 f'x2="{x:.1f}" y2="{ybot:.0f}"/>')
    s.append(f'<line class="gl" x1="{X0:.1f}" y1="{top-6:.0f}" '
             f'x2="{X0:.1f}" y2="{ybot:.0f}"/>')
    for v, t in zip(ticks, tlab):
        s.append(f'<text class="tk" x="{sx(v):.1f}" y="{ybot+14:.0f}" '
                 f'text-anchor="middle">{t}</text>')
    s.append(f'<text class="axl" x="{X1:.0f}" y="{ybot+32:.0f}" '
             f'text-anchor="end">instances (log scale)</text>')
    for i, (name, val, lab, col, sub, tip) in enumerate(rows):
        y = top + i * (bh + gap)
        w = sx(val) - X0
        s.append(f'<rect class="f-{col}" x="{X0:.1f}" y="{y}" width="{w:.1f}" '
                 f'height="{bh}" rx="2" data-tip="{tip}"><title>{tip}</title></rect>')
        s.append(f'<text class="b-lbl" x="{X0-10:.0f}" y="{y+bh/2+4:.0f}" '
                 f'text-anchor="end">{name}</text>')
        s.append(f'<text class="dl f-{col}" x="{X0+w+9:.1f}" y="{y+bh/2+4:.0f}">'
                 f'{lab}</text>')
        s.append(f'<text class="seg-t f-paper" x="{X0+11:.1f}" y="{y+bh/2+4:.0f}">'
                 f'{sub}</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 2 -- spatial heatmap of box centres
# --------------------------------------------------------------------------
NX, NY = 24, 16


def _hash01(c, r):
    x = (c * 374761393 + r * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFF) / 0xFFFF


def heat_data():
    """Centre-heavy density with per-cell sampling noise (no periodic texture)."""
    cells = []
    for r in range(NY):
        for c in range(NX):
            x = (c + 0.5) / NX
            y = (r + 0.5) / NY
            v = 330 * math.exp(-(((x - .46) / .34) ** 2 + ((y - .56) / .38) ** 2))
            v += 170 * math.exp(-(((x - .72) / .16) ** 2 + ((y - .31) / .18) ** 2))
            v *= .74 + .52 * _hash01(c, r)
            cells.append((c, r, max(0, round(v))))
    return cells


def chart_heat():
    cells = heat_data()
    vmax = max(v for _, _, v in cells)
    CS = 25.0
    X0, YT = 34, 10
    gw, gh = NX * CS - 1, NY * CS - 1
    s = ['<svg class="chart" viewBox="0 0 660 470" role="img" '
         'aria-label="Density of box centres over the frame">']
    for c, r, v in cells:
        step = 0 if v <= 0 else min(6, 1 + int((v / vmax) * 5.999))
        x, y = X0 + c * CS, YT + r * CS
        tip = f"column {c + 1} of {NX}, row {r + 1} of {NY} &#183; {v} box centres"
        s.append(f'<rect class="hm-{step}" x="{x:.0f}" y="{y:.0f}" '
                 f'width="{CS - 1:.0f}" height="{CS - 1:.0f}" rx="1" '
                 f'data-tip="{tip}"><title>{tip}</title></rect>')
    hot = []
    for c, r, v in sorted(cells, key=lambda t: -t[2]):
        if all(abs(c - hc) > 1 or abs(r - hr) > 1 for hc, hr, _ in hot):
            hot.append((c, r, v))
        if len(hot) == 3:
            break
    for c, r, v in hot:
        s.append(f'<text class="hm-lbl" x="{X0 + c * CS + (CS - 1) / 2:.1f}" '
                 f'y="{YT + r * CS + CS / 2 + 3:.1f}" text-anchor="middle">{v}</text>')
    s.append(f'<text class="axl" x="{X0}" y="{YT + gh + 18:.0f}">left edge</text>')
    s.append(f'<text class="axl" x="{X0 + gw:.0f}" y="{YT + gh + 18:.0f}" '
             f'text-anchor="end">right edge</text>')
    cy = YT + gh / 2
    s.append(f'<text class="axl" x="14" y="{cy:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 14 {cy:.0f})">top &#8594; bottom of frame</text>')
    lx, ly = X0, YT + gh + 36
    s.append(f'<text class="axl" x="{lx}" y="{ly + 9}">0</text>')
    for i in range(7):
        s.append(f'<rect class="hm-{i}" x="{lx + 14 + i * 21}" y="{ly}" width="20" '
                 f'height="9" rx="1"/>')
    s.append(f'<text class="axl" x="{lx + 14 + 7 * 21 + 5}" y="{ly + 9}">'
             f'{vmax} box centres in one cell</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 3 -- illustrative co-occurrence matrix
# --------------------------------------------------------------------------
MX_NAMES = ["fruitlet", "stack", "leaf", "branch", "tag", "trunk"]
MX = [
    [618, 12, 402, 274, 96, 61],
    [12, 58, 41, 33, 9, 7],
    [402, 41, 511, 318, 74, 52],
    [274, 33, 318, 396, 66, 88],
    [96, 9, 74, 66, 121, 18],
    [61, 7, 52, 88, 18, 143],
]


def chart_matrix():
    n = len(MX_NAMES)
    CS = 33.0
    X0, YT = 84, 66
    vmax = max(max(r) for r in MX)
    s = ['<svg viewBox="0 0 312 296" role="img" '
         'aria-label="Illustrative six-class co-occurrence matrix">']
    for j, nm in enumerate(MX_NAMES):
        cx = X0 + j * CS + CS / 2 - 6
        s.append(f'<text class="mx-lbl" x="{cx:.1f}" y="{YT - 7}" '
                 f'transform="rotate(-45 {cx:.1f} {YT - 7})">{nm}</text>')
    for i, nm in enumerate(MX_NAMES):
        s.append(f'<text class="mx-lbl" x="{X0 - 8}" y="{YT + i * CS + CS / 2 + 3:.1f}" '
                 f'text-anchor="end">{nm}</text>')
    for i in range(n):
        for j in range(n):
            v = MX[i][j]
            step = min(6, 1 + int((v / vmax) * 5.999)) if v else 0
            x, y = X0 + j * CS, YT + i * CS
            tip = (f"{MX_NAMES[i]} &#215; {MX_NAMES[j]} &#183; {v} images "
                   f"&#183; illustrative")
            s.append(f'<rect class="hm-{step}" x="{x:.0f}" y="{y:.0f}" '
                     f'width="{CS - 1.5:.1f}" height="{CS - 1.5:.1f}" rx="1.5" '
                     f'data-tip="{tip}"><title>{tip}</title></rect>')
            cls = "hm-lbl" if step >= 5 else "hm-lbl-d"
            s.append(f'<text class="{cls}" x="{x + (CS - 1.5) / 2:.1f}" '
                     f'y="{y + CS / 2 + 2:.1f}" text-anchor="middle">{v}</text>')
    s.append(f'<text class="axl" x="{X0}" y="{YT + n * CS + 16:.0f}">'
             'images containing both classes</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 4 -- TIDE stacked bar
# --------------------------------------------------------------------------
def chart_tide():
    X0, X1 = 8, 616
    dom = 0.55
    sc = (X1 - X0) / dom
    segs = [
        ("Bkg", 0.2527, "f-cyan", "f-paper", "in", "35,010"),
        ("Miss", 0.0970, "f-teal", "f-paper", "in", "1,703"),
        ("Loc", 0.0786, "f-g1", "f-ink1", "in", "10,291"),
        ("Both", 0.0061, "f-g2", "", "out", "1,219"),
        ("Dupe", 0.0005, "f-g3", "", "out", "409"),
        ("Cls", 0.0001, "f-g3", "", "out", "517"),
    ]
    MINW, GAP = 7.0, 2.0
    BY, BH = 48, 32
    s = ['<svg class="chart" viewBox="0 0 660 174" role="img" '
         'aria-label="TIDE error decomposition, delta AP50 per error type">']
    for val, lab, ly, tip in (
            (0.0970, "&#916;AP_FN 0.097", 15,
             "false-negative oracle &#183; ceiling &#916;AP50 0.097"),
            (0.4500, "&#916;AP_FP 0.450", 30,
             "false-positive oracle &#183; ceiling &#916;AP50 0.450")):
        x = X0 + val * sc
        s.append(f'<line class="ref" x1="{x:.1f}" y1="{ly+5}" x2="{x:.1f}" '
                 f'y2="{BY+BH+6}" data-tip="{tip}"><title>{tip}</title></line>')
        anchor = "start" if val < 0.25 else "end"
        lx = x + 6 if val < 0.25 else x - 6
        s.append(f'<text class="dl f-peer" x="{lx:.1f}" y="{ly}" '
                 f'text-anchor="{anchor}">{lab}</text>')
    x = X0
    outs = []
    for name, val, cls, tcls, mode, cnt in segs:
        w = max(val * sc, MINW)
        tip = f"{name} &#183; &#916;AP50 {val:.4f} &#183; {cnt} boxes"
        s.append(f'<rect class="{cls}" x="{x:.1f}" y="{BY}" width="{w:.1f}" '
                 f'height="{BH}" rx="1.5" data-tip="{tip}"><title>{tip}</title></rect>')
        if mode == "in":
            s.append(f'<text class="seg-t {tcls}" x="{x+9:.1f}" y="{BY+14}">{name}</text>')
            s.append(f'<text class="seg-t {tcls}" x="{x+9:.1f}" y="{BY+26}">'
                     f'{val:.4f}</text>')
        else:
            outs.append((name, val, x + w / 2))
        x += w + GAP
    s.append(f'<line class="gl" x1="{X0}" y1="{BY-10}" x2="{X0}" y2="{BY+BH+8}"/>')
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        tx = X0 + t * sc
        s.append(f'<text class="tk" x="{tx:.1f}" y="{BY+BH+22}" '
                 f'text-anchor="middle">{t:.1f}</text>')
    rows = [146, 132, 118]
    lx_end = 536
    for (name, val, cx), ly in zip(outs, rows):
        s.append(f'<path class="lead" d="M{cx:.1f} {BY+BH+2} L{cx:.1f} {ly-4} '
                 f'L{lx_end} {ly-4}"/>')
        s.append(f'<text class="dl f-ink2" x="{lx_end+6}" y="{ly}">{name} '
                 f'<tspan class="f-peer">{val:.4f}</tspan></text>')
    s.append(f'<text class="axl" x="{X0}" y="{168}">'
             '&#916;AP50 an oracle would recover by fixing only that error</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 5 -- P / R / F1 vs confidence
# --------------------------------------------------------------------------
def chart_prf():
    X0, X1, YT, YB = 56, 604, 28, 202
    xmax, ymax = 0.70, 0.62
    sx = lambda c: X0 + (c / xmax) * (X1 - X0)
    sy = lambda v: YB - (v / ymax) * (YB - YT)
    conf = [0, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .60, .70]
    rec = [.576, .470, .372, .288, .216, .158, .112, .076, .050, .031, .018, .005, .001]
    pre = [.0703, .0668, .0630, .0592, .0548, .0505, .0462, .0415, .0370,
           .0322, .0280, .0195, .0120]
    f1 = [2 * p * r / (p + r) for p, r in zip(pre, rec)]
    s = ['<svg class="chart" viewBox="0 0 660 244" role="img" '
         'aria-label="Precision, recall and F1 against confidence threshold">']
    for g in (0, .1, .2, .3, .4, .5, .6):
        y = sy(g)
        s.append(f'<line class="gl{"" if g == 0 else "-2"}" x1="{X0}" y1="{y:.1f}" '
                 f'x2="{X1}" y2="{y:.1f}"/>')
        s.append(f'<text class="tk" x="{X0-10}" y="{y+4:.1f}" text-anchor="end">'
                 f'{g:.1f}</text>')
    for t in (0, .1, .2, .3, .4, .5, .6, .7):
        s.append(f'<text class="tk" x="{sx(t):.1f}" y="{YB+18}" '
                 f'text-anchor="middle">{t:.1f}</text>')
    s.append(f'<text class="axl" x="{X1}" y="{YB+36}" text-anchor="end">'
             'confidence threshold</text>')
    s.append(f'<line class="ref" x1="{sx(0):.1f}" y1="{YT}" x2="{sx(0):.1f}" '
             f'y2="{YB}"/>')
    s.append(f'<text class="dl f-peer" x="{sx(0)+7:.1f}" y="{YT-14}">'
             '&#964;* = 0.000  (F1 peaks at the NMS floor)</text>')
    series = [
        (rec, "s-teal", "f-teal", "Recall 0.576", -8, "Recall"),
        (f1, "s-cyan", "f-cyan", "F1 0.125", -9, "F1"),
        (pre, "s-peer", "f-peer", "Precision 0.070", 15, "Precision"),
    ]
    for vals, sc, fc, lab, dy, nm in series:
        pts = " ".join(f"{sx(c):.1f},{sy(v):.1f}" for c, v in zip(conf, vals))
        s.append(f'<polyline class="ln {sc}" points="{pts}"/>')
    for vals, sc, fc, lab, dy, nm in series:
        s.append(f'<circle class="mk {fc}" cx="{sx(0):.1f}" cy="{sy(vals[0]):.1f}" '
                 f'r="4.2"/>')
        s.append(f'<text class="dl {fc}" x="{sx(0)+11:.1f}" '
                 f'y="{sy(vals[0])+dy:.1f}">{lab}</text>')
    s.append(f'<text class="dl-q" x="{sx(0.42):.1f}" y="{sy(0.30):.1f}">'
             'nothing survives above conf 0.6 &#8212;</text>')
    s.append(f'<text class="dl-q" x="{sx(0.42):.1f}" y="{sy(0.30)+14:.1f}">'
             'the scores carry no ranking signal</text>')
    for vals, sc, fc, lab, dy, nm in series:
        for c, v in zip(conf, vals):
            tip = f"{nm} {v:.4f} at conf {c:.2f}"
            s.append(f'<circle class="hit" cx="{sx(c):.1f}" cy="{sy(v):.1f}" r="7" '
                     f'data-tip="{tip}"><title>{tip}</title></circle>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 6 -- objects-per-image histogram
# --------------------------------------------------------------------------
def chart_hist():
    X0, X1, YT, YB = 44, 616, 26, 156
    counts = [6, 14, 21, 23, 19, 16, 13, 11, 9, 7, 5, 4, 3, 3, 2, 2,
              1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]
    nb = len(counts)
    slot = (X1 - X0) / nb
    bw = slot * 0.66
    ymax = 24
    sy = lambda c: YB - (c / ymax) * (YB - YT)
    s = ['<svg class="chart" viewBox="0 0 660 198" role="img" '
         'aria-label="Histogram of objects per image across 166 test images">']
    for g in (0, 5, 10, 15, 20):
        y = sy(g)
        s.append(f'<line class="gl{"" if g == 0 else "-2"}" x1="{X0}" y1="{y:.1f}" '
                 f'x2="{X1}" y2="{y:.1f}"/>')
        s.append(f'<text class="tk" x="{X0-9}" y="{y+4:.1f}" text-anchor="end">'
                 f'{g}</text>')
    for i, c in enumerate(counts):
        x = X0 + i * slot + (slot - bw) / 2
        y = sy(c)
        tip = (f"{i * 5}&#8211;{i * 5 + 5} objects &#183; {c} "
               f"image{'' if c == 1 else 's'}")
        if c:
            s.append(f'<rect class="f-cyan" x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{YB-y:.1f}" rx="1" data-tip="{tip}">'
                     f'<title>{tip}</title></rect>')
        s.append(f'<rect class="hit" x="{X0 + i * slot:.1f}" y="{YT-6}" '
                 f'width="{slot:.1f}" height="{YB - YT + 6:.1f}" data-tip="{tip}">'
                 f'<title>{tip}</title></rect>')
    pk = counts.index(max(counts))
    px = X0 + pk * slot + slot / 2
    s.append(f'<text class="dl f-cyan" x="{px:.1f}" y="{sy(23)-7:.1f}" '
             f'text-anchor="middle">23 images</text>')
    rx = X0 + (146 / 150) * (X1 - X0)
    s.append(f'<line class="ref" x1="{rx:.1f}" y1="{YT-4}" x2="{rx:.1f}" y2="{YB}"/>')
    s.append(f'<text class="dl f-peer" x="{rx-7:.1f}" y="{YT+2}" '
             f'text-anchor="end">p99.5 = 146</text>')
    for t in (0, 25, 50, 75, 100, 125, 150):
        tx = X0 + (t / 150) * (X1 - X0)
        s.append(f'<text class="tk" x="{tx:.1f}" y="{YB+18}" '
                 f'text-anchor="middle">{t}</text>')
    s.append(f'<text class="axl" x="{X0}" y="{YB+36}">objects per image '
             '&#183; bin width 5</text>')
    s.append(f'<text class="axl" x="{X1}" y="{YB+36}" text-anchor="end">'
             '166 test images</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 7 -- image resolutions, horizontal bars
# --------------------------------------------------------------------------
RESOLUTIONS = [
    ("4032 &#215; 3024", 389, "cyan", "12.2 MP &#183; landscape"),
    ("3024 &#215; 4032", 121, "cyan", "12.2 MP &#183; portrait"),
    ("1920 &#215; 1440", 58, "cyan", "2.8 MP &#183; landscape"),
    ("1280 &#215; 960", 31, "cyan", "1.2 MP &#183; landscape"),
    ("640 &#215; 480", 12, "cyan", "0.3 MP &#183; landscape"),
    ("800 &#215; 600", 5, "cyan", "0.5 MP &#183; landscape"),
    ("other (3 sizes)", 2, "peer", "three sizes with one image each"),
]


def chart_res():
    X0, X1 = 118, 560
    vmax = 400.0
    sx = lambda v: X0 + (v / vmax) * (X1 - X0)
    bh, gap, top = 18, 9, 14
    n = len(RESOLUTIONS)
    ybot = top + n * bh + (n - 1) * gap + 4
    s = ['<svg class="chart" viewBox="0 0 660 %d" role="img" '
         'aria-label="Image resolutions by count, top six sizes plus other">'
         % (ybot + 38)]
    for t in (0, 100, 200, 300, 400):
        x = sx(t)
        s.append(f'<line class="gl{"" if t == 0 else "-2"}" x1="{x:.1f}" y1="{top-8}" '
                 f'x2="{x:.1f}" y2="{ybot}"/>')
        s.append(f'<text class="tk" x="{x:.1f}" y="{ybot+15}" text-anchor="middle">'
                 f'{t}</text>')
    for i, (name, val, col, tip_extra) in enumerate(RESOLUTIONS):
        y = top + i * (bh + gap)
        w = max(sx(val) - X0, 1.5)
        tip = f"{name} &#183; {val} images &#183; {tip_extra}"
        s.append(f'<rect class="f-{col}" x="{X0:.1f}" y="{y}" width="{w:.1f}" '
                 f'height="{bh}" rx="2" data-tip="{tip}"><title>{tip}</title></rect>')
        s.append(f'<text class="b-lbl" x="{X0-10}" y="{y+bh/2+4:.0f}" '
                 f'text-anchor="end">{name}</text>')
        s.append(f'<text class="dl f-{col}" x="{X0+w+8:.1f}" y="{y+bh/2+4:.0f}">'
                 f'{val}</text>')
    s.append(f'<text class="axl" x="{X0}" y="{ybot+33}">images &#183; 618 in all '
             'three splits</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 8 -- aspect ratio, one inline stacked bar (a sentence made of pixels)
# --------------------------------------------------------------------------
def chart_aspect():
    W, BH, Y = 636.0, 14, 4
    segs = [("landscape 79%", 0.79, "f-cyan", "in", "489 images &#183; W &gt; H"),
            ("portrait 20%", 0.20, "f-teal", "in", "123 images &#183; H &gt; W"),
            ("square 1%", 0.01, "f-g1", "out", "6 images &#183; W = H")]
    GAP = 2.0
    s = ['<svg class="chart" viewBox="0 0 720 22" role="img" '
         'aria-label="Aspect ratio: 79% landscape, 20% portrait, 1% square">']
    x = 0.0
    for lab, frac, cls, mode, tip in segs:
        w = max(frac * W - GAP, 2.0)
        s.append(f'<rect class="{cls}" x="{x:.1f}" y="{Y}" width="{w:.1f}" '
                 f'height="{BH}" rx="1.5" data-tip="{lab} &#183; {tip}">'
                 f'<title>{lab} &#183; {tip}</title></rect>')
        if mode == "in":
            s.append(f'<text class="seg-t f-paper" x="{x+8:.1f}" y="{Y+10.5:.0f}">'
                     f'{lab}</text>')
        else:
            s.append(f'<text class="seg-t f-peer" x="{x+w+7:.1f}" y="{Y+10.5:.0f}">'
                     f'{lab}</text>')
        x += w + GAP
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Chart 9 -- megapixels per image histogram
# --------------------------------------------------------------------------
def chart_mp():
    X0, X1, YT, YB = 52, 616, 42, 172
    counts = [17, 0, 31, 0, 58, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 510, 0]
    nb = len(counts)
    binw = 0.6
    slot = (X1 - X0) / nb
    bw = slot * 0.66
    ymax = 540.0
    sy = lambda c: YB - (c / ymax) * (YB - YT)
    s = ['<svg class="chart" viewBox="0 0 660 214" role="img" '
         'aria-label="Histogram of megapixels per image over 618 images">']
    for g in (0, 150, 300, 450):
        y = sy(g)
        s.append(f'<line class="gl{"" if g == 0 else "-2"}" x1="{X0}" y1="{y:.1f}" '
                 f'x2="{X1}" y2="{y:.1f}"/>')
        s.append(f'<text class="tk" x="{X0-9}" y="{y+4:.1f}" text-anchor="end">'
                 f'{g}</text>')
    # the median falls inside the tallest bar, so the line is drawn under the bars
    # and reads as a stub pointing at it rather than a scar across it
    mx = X0 + (12.2 / (nb * binw)) * (X1 - X0)
    s.append(f'<line class="ref" x1="{mx:.1f}" y1="{YT-30}" x2="{mx:.1f}" y2="{YB}"/>')
    s.append(f'<text class="dl f-peer" x="{mx-7:.1f}" y="{YT-32}" '
             f'text-anchor="end">median 12.2 MP</text>')
    for i, c in enumerate(counts):
        x = X0 + i * slot + (slot - bw) / 2
        lo, hi = i * binw, (i + 1) * binw
        tip = (f"{lo:.1f}&#8211;{hi:.1f} MP &#183; {c} "
               f"image{'' if c == 1 else 's'}")
        if c:
            y = min(sy(c), YB - 2)
            s.append(f'<rect class="f-cyan" x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{YB-y:.1f}" rx="1" data-tip="{tip}">'
                     f'<title>{tip}</title></rect>')
        s.append(f'<rect class="hit" x="{X0 + i * slot:.1f}" y="{YT-8}" '
                 f'width="{slot:.1f}" height="{YB - YT + 8:.1f}" data-tip="{tip}">'
                 f'<title>{tip}</title></rect>')
    pk = counts.index(max(counts))
    px = X0 + pk * slot + (slot - bw) / 2
    s.append(f'<text class="dl f-cyan" x="{px-7:.1f}" y="{sy(510)+13:.1f}" '
             f'text-anchor="end">510 images</text>')
    for t in (0, 2, 4, 6, 8, 10, 12):
        tx = X0 + (t / (nb * binw)) * (X1 - X0)
        s.append(f'<text class="tk" x="{tx:.1f}" y="{YB+18}" '
                 f'text-anchor="middle">{t}</text>')
    s.append(f'<text class="axl" x="{X0}" y="{YB+36}">megapixels per image '
             '&#183; bin width 0.6 MP</text>')
    s.append(f'<text class="axl" x="{X1}" y="{YB+36}" text-anchor="end">'
             '618 images</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# Gallery placeholder tiles
# --------------------------------------------------------------------------
def blobs(seed, n=9):
    o = []
    rng = seed
    for _ in range(n):
        rng = (rng * 1103515245 + 12345) % 2147483648
        cx = 6 + (rng >> 7) % 88
        rng = (rng * 1103515245 + 12345) % 2147483648
        cy = 6 + (rng >> 7) % 88
        rng = (rng * 1103515245 + 12345) % 2147483648
        r = 4 + (rng >> 9) % 7
        op = 0.10 + ((rng >> 5) % 14) / 100
        o.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="var(--paper)" '
                 f'opacity="{op:.2f}"/>')
    return "".join(o)


def phg(seed):
    """The placeholder 'photograph' -- gradient comes from CSS, blobs from here."""
    return (f'<svg class="phg" viewBox="0 0 100 100" preserveAspectRatio="none" '
            f'aria-hidden="true">{blobs(seed)}</svg>')


# 28 = four full rows of seven, so the bottom fade always falls across a complete
# row: a partial row under the fade reads as a mistake, which is the thing v3 fixes.
GAL_FILES = [
    "ffb_56", "ffb_53", "ffb_34", "ffb_52", "ffb_32", "ffb_40", "ffb_41",
    "ffb_6", "ffb_18", "ffb_71", "ffb_9", "ffb_63", "ffb_27", "ffb_11",
    "ffb_48", "ffb_2", "ffb_66", "ffb_20", "ffb_15", "ffb_29", "ffb_7",
    "ffb_44", "ffb_61", "ffb_23", "ffb_38", "ffb_12", "ffb_50", "ffb_69",
]


def img_boxes(i):
    """One population of boxes per image, in 0-100 frame space.

    The three gallery tabs are three readings of this same list, never three
    different geometries -- that is the whole point of the tab strip.
    """
    rng = (i + 5) * 104729

    def nxt(m):
        nonlocal rng
        rng = (rng * 1103515245 + 12345) % 2147483648
        return (rng >> 8) % m

    kinds = ["fn", "tp", "fn", "fp", "tp", "fn", "fn", "tp", "fp", "fn"]
    out = []
    for k in range(4 + nxt(4)):
        x = 5 + nxt(62)
        y = 11 + nxt(54)
        w = 11 + nxt(13)
        h = w + nxt(6) - 3
        out.append({
            "x": x, "y": y, "w": w, "h": h,
            "k": kinds[(i + k) % len(kinds)],
            "conf": (28 + nxt(58)) / 100,
            "iou": (52 + nxt(42)) / 100,
        })
    return out


def _items():
    out = []
    for i, f in enumerate(GAL_FILES):
        bs = img_boxes(i)
        out.append({
            "f": f, "seed": (i + 3) * 17 + 11, "b": bs,
            "tp": sum(1 for b in bs if b["k"] == "tp"),
            "fp": sum(1 for b in bs if b["k"] == "fp"),
            "fn": sum(1 for b in bs if b["k"] == "fn"),
        })
    out.sort(key=lambda d: (-(d["fp"] + d["fn"]), d["f"]))
    return out


ITEMS = _items()

OV_TABS = {
    # tab -> (which outcomes it draws, box class, label fn, label colour class)
    "outcome": (("tp", "fp", "fn"), None, None, None),
    "pred": (("tp", "fp"), "ob-pred", lambda b: f'{b["conf"]:.2f}', "f-cyan"),
    "gt": (("tp", "fn"), "ob-gt", None, None),
}
LBL_MIN_W = 17  # only the boxes wide enough to carry a legible number at 128px


def ov(boxes, tab):
    """Overlay layer -- a separate SVG over the tile, never baked into it."""
    keep, cls, lab, tcls = OV_TABS[tab]
    o = [f'<svg class="ov ov-{tab}" viewBox="0 0 100 100" '
         f'preserveAspectRatio="none" aria-hidden="true">']
    for b in boxes:
        if b["k"] not in keep:
            continue
        c = cls or f'ob-{b["k"]}'
        o.append(f'<rect class="ob {c}" x="{b["x"]}" y="{b["y"]}" '
                 f'width="{b["w"]}" height="{b["h"]}" rx="1.5"/>')
        if lab and b["w"] >= LBL_MIN_W:
            tx = min(b["x"] + 1, 76)
            ty = max(b["y"] - 2.0, 6.4)
            o.append(f'<text class="ovt {tcls}" x="{tx}" y="{ty:.1f}">{lab(b)}</text>')
    o.append("</svg>")
    return "".join(o)


# --------------------------------------------------------------------------
# Lightbox mock geometry (percent of the frame)
# --------------------------------------------------------------------------
def shot_geo(i):
    it = ITEMS[i]
    boxes = []
    for b in it["b"]:
        if b["k"] == "fn":
            lab = "fruitlet \u00b7 missed"
        elif b["k"] == "fp":
            lab = f'fruitlet {b["conf"]:.2f} \u00b7 IoU 0.00'
        else:
            lab = f'fruitlet {b["conf"]:.2f} \u00b7 IoU {b["iou"]:.2f}'
        boxes.append([b["x"], b["y"], b["w"], b["h"], b["k"], lab])
    item = {"f": it["f"], "fp": it["fp"], "fn": it["fn"], "b": boxes}
    if i == 0:
        item["poly"] = {
            "pts": [[74, 28], [76, 21], [82, 16], [89, 18], [94, 24], [94, 32],
                    [89, 38], [81, 39], [76, 35]],
            "lab": "fruitlet 0.61 \u00b7 IoU 0.74 \u00b7 mask",
        }
    return item


# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
KPIS = [
    ("0.0345", "mAP50-95", "box &#183; conf 0.001", "target", 0.0345,
     "mAP50-95 0.0345 of a possible 1.00"),
    ("0.1336", "mAP50", "box &#183; conf 0.001", "gauge", 0.1336,
     "mAP50 0.1336 of a possible 1.00"),
    ("0.0703", "Precision", "at max-F1 point", "crosshair", 0.0703,
     "Precision 0.0703 of a possible 1.00"),
    ("0.5760", "Recall", "at max-F1 point", "recall", 0.5760,
     "Recall 0.5760 of a possible 1.00"),
    ("0.1254", "F1", "at max-F1 point", "scales", 0.1254,
     "F1 0.1254 of a possible 1.00"),
    ("0.000", 'Deploy <span class="tau">&#964;*</span>', "global F1-optimal",
     "sliders", None, None),
    ("166", "Images", "4,057 instances", "photo", None, None),
]

SECTIONS = [
    ("s-summary", "Summary"),
    ("s-model-card", "Model card"),
    ("s-dataset", "Dataset"),
    ("s-per-class", "Per-class"),
    ("s-threshold", "Threshold"),
    ("s-tide", "Errors"),
    ("s-strata", "Size &amp; shape"),
    ("s-galleries", "Galleries"),
    ("s-caveats", "Caveats"),
]

QUALITY = [
    ("Under 8&#8239;px on a side", "312", "1.5%",
     '<rect class="ex-f" x=".5" y=".5" width="39" height="25" rx="2"/>'
     '<rect class="ex-b" x="18" y="11" width="4" height="4" rx="1"/>',
     "smaller than the smallest stride the model sees at imgsz 640"),
    ("Touching the frame edge", "1,104", "5.2%",
     '<rect class="ex-f" x=".5" y=".5" width="39" height="25" rx="2"/>'
     '<rect class="ex-b" x="30" y="7" width="14" height="12" rx="1"/>',
     "truncated objects &#8212; half-annotated by construction"),
    ("Near-duplicate, IoU &gt; 0.9", "86", "0.4%",
     '<rect class="ex-f" x=".5" y=".5" width="39" height="25" rx="2"/>'
     '<rect class="ex-p" x="12" y="6" width="14" height="13" rx="1"/>'
     '<rect class="ex-b" x="15" y="8" width="14" height="13" rx="1"/>',
     "the same object annotated twice; trains the model to duplicate"),
    ("Aspect ratio outside 0.2&#8211;5", "47", "0.2%",
     '<rect class="ex-f" x=".5" y=".5" width="39" height="25" rx="2"/>'
     '<rect class="ex-b" x="5" y="11" width="30" height="4" rx="1"/>',
     "usually a drag-slip in the annotation tool"),
]


def kpi_cell(v, lab, basis, ic, frac, tip):
    top = icon(ic) + (ring(frac, tip) if frac is not None else "")
    return (f'    <div class="kpi"><span class="kpi-top">{top}</span>'
            f'<span class="kpi-v">{v}</span>'
            f'<span class="kpi-l">{lab}</span>'
            f'<span class="kpi-b">{basis}</span></div>')


# --------------------------------------------------------------------------
# JS: theme toggle + hover tooltip + lightbox
# (plain string: braces stay literal when interpolated into the f-string)
# --------------------------------------------------------------------------
JS = """
/* ---- theme ---- */
(function () {
  var r = document.documentElement, b = document.getElementById('tgl');
  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  function cur() { return r.getAttribute('data-theme') || (mq.matches ? 'dark' : 'light'); }
  b.addEventListener('click', function () {
    r.setAttribute('data-theme', cur() === 'dark' ? 'light' : 'dark');
    if (LB.classList.contains('on')) draw(idx);
  });
})();

/* ---- cursor-following tooltip; the printed direct labels stay as they are ---- */
(function () {
  var t = document.createElement('div');
  t.className = 'tip'; document.body.appendChild(t);
  document.addEventListener('mousemove', function (e) {
    var el = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
    if (!el) { t.style.display = 'none'; return; }
    t.innerHTML = el.getAttribute('data-tip');
    t.style.display = 'block';
    var w = t.offsetWidth, h = t.offsetHeight;
    var x = e.clientX + 14, y = e.clientY + 16;
    if (x + w > innerWidth - 8) x = e.clientX - w - 14;
    if (y + h > innerHeight - 8) y = e.clientY - h - 14;
    t.style.left = x + 'px'; t.style.top = y + 'px';
  }, { passive: true });
  addEventListener('scroll', function () { t.style.display = 'none'; }, { passive: true });
})();

/* ---- lightbox ---- */
var SHOTS = __SHOTS__;
var LB = document.getElementById('lb'), idx = 0, ANNO = true;
function css(n) {
  return getComputedStyle(document.documentElement).getPropertyValue(n).trim();
}
function draw(i) {
  var it = SHOTS[i], c = document.getElementById('lbc');
  var W = c.clientWidth || 720, H = c.clientHeight || 540, dpr = 2;
  c.width = W * dpr; c.height = H * dpr;
  var g = c.getContext('2d');
  g.scale(dpr, dpr);
  var col = { tp: css('--good'), fp: css('--serious'), fn: css('--warn'),
              mask: css('--brand-cyan'), paper: css('--paper') };
  g.font = '500 11px ' + css('--font-sans');
  g.lineJoin = 'round';
  function label(txt, x, y, color) {
    g.lineWidth = 3; g.strokeStyle = col.paper; g.strokeText(txt, x, y);
    g.fillStyle = color; g.fillText(txt, x, y);
  }
  if (ANNO) { it.b.forEach(function (b) {
    var x = b[0] / 100 * W, y = b[1] / 100 * H;
    var w = b[2] / 100 * W, h = b[3] / 100 * H, k = b[4];
    g.setLineDash(k === 'fn' ? [6, 4] : []);
    g.lineWidth = 2; g.strokeStyle = col[k]; g.strokeRect(x, y, w, h);
    g.setLineDash([]);
    label(b[5], x, y - 5, col[k]);
  }); }
  if (it.poly && ANNO) {
    g.beginPath();
    it.poly.pts.forEach(function (p, n) {
      var x = p[0] / 100 * W, y = p[1] / 100 * H;
      if (n) { g.lineTo(x, y); } else { g.moveTo(x, y); }
    });
    g.closePath();
    g.globalAlpha = 0.35; g.fillStyle = col.mask; g.fill(); g.globalAlpha = 1;
    g.lineWidth = 2; g.strokeStyle = col.mask; g.stroke();
    var xs = it.poly.pts.map(function (p) { return p[0]; });
    var ys = it.poly.pts.map(function (p) { return p[1]; });
    label(it.poly.lab, Math.min.apply(null, xs) / 100 * W - 2,
          Math.min.apply(null, ys) / 100 * H - 6, col.mask);
  }
  document.getElementById('lbimg').innerHTML = SVGS[i];
  document.getElementById('lbcap').innerHTML =
    it.f + '.jpg &#183; FP ' + it.fp + ' &#183; FN ' + it.fn + ' &#183; conf &#8805; 0.25';
  document.getElementById('lbn').textContent = (i + 1) + ' of ' + SHOTS.length;
}
function open_(i) {
  idx = (i + SHOTS.length) % SHOTS.length;
  LB.classList.add('on'); document.body.style.overflow = 'hidden';
  requestAnimationFrame(function () { draw(idx); });
}
function close_() { LB.classList.remove('on'); document.body.style.overflow = ''; }
document.querySelectorAll('.ph').forEach(function (b) {
  b.addEventListener('click', function () { open_(+b.dataset.i); });
});
document.getElementById('lbprev').addEventListener('click', function () { open_(idx - 1); });
document.getElementById('lbnext').addEventListener('click', function () { open_(idx + 1); });
document.getElementById('lbx').addEventListener('click', close_);
LB.addEventListener('click', function (e) { if (e.target === LB) close_(); });
document.addEventListener('keydown', function (e) {
  if (!LB.classList.contains('on')) { return; }
  if (e.key === 'Escape') { close_(); }
  if (e.key === 'ArrowLeft') { open_(idx - 1); }
  if (e.key === 'ArrowRight') { open_(idx + 1); }
});
addEventListener('resize', function () { if (LB.classList.contains('on')) draw(idx); });
if (/^#lb=\\d+$/.test(location.hash)) { open_(+location.hash.slice(4) - 1); }

/* ---- gallery: tab strip, global annotation switch, soft scroll edges ---- */
(function () {
  var wrap = document.getElementById('galwrap');
  var gal = document.getElementById('gal');
  var tabs = [].slice.call(wrap.querySelectorAll('[role="tab"]'));
  var sw = document.getElementById('anno');

  function edges() {
    gal.classList.toggle('at-start', gal.scrollTop <= 2);
    gal.classList.toggle('at-end',
      gal.scrollTop + gal.clientHeight >= gal.scrollHeight - 2);
  }
  gal.addEventListener('scroll', edges, { passive: true });
  addEventListener('resize', edges);
  edges();

  function sel(i) {
    tabs.forEach(function (t, n) {
      t.setAttribute('aria-selected', n === i ? 'true' : 'false');
      t.tabIndex = n === i ? 0 : -1;
    });
    wrap.dataset.tab = tabs[i].dataset.tab;
  }
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { sel(i); });
    t.addEventListener('keydown', function (e) {
      var d = e.key === 'ArrowRight' ? 1 : (e.key === 'ArrowLeft' ? -1 : 0);
      if (!d) { return; }
      e.preventDefault();
      var n = (i + d + tabs.length) % tabs.length;
      sel(n); tabs[n].focus();
    });
  });

  function anno(on) {
    ANNO = on;
    wrap.dataset.anno = on ? 'on' : 'off';
    sw.setAttribute('aria-checked', on ? 'true' : 'false');
    if (LB.classList.contains('on')) { draw(idx); }
  }
  sw.addEventListener('click', function () { anno(!ANNO); });
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'a' && e.key !== 'A') { return; }
    var t = e.target;
    if (t && /^(input|textarea|select)$/i.test(t.tagName)) { return; }
    anno(!ANNO);
  });

  /* state drivers, also used to shoot the states: #tab=pred, #anno=off */
  var h = location.hash.slice(1);
  var m = /(?:^|&)tab=(\\w+)/.exec(h);
  if (m) { tabs.forEach(function (t, i) { if (t.dataset.tab === m[1]) { sel(i); } }); }
  if (/(?:^|&)anno=off/.test(h)) { anno(false); }
})();
"""


def build():
    kpi_html = "\n".join(kpi_cell(*k) for k in KPIS)
    nav_links = "\n".join(f'      <a href="#{i}">{t}</a>' for i, t in SECTIONS)

    gal_html = "\n".join(
        f'            <div class="thumb"><button class="ph" type="button" '
        f'data-i="{n}" aria-label="{it["f"]}.jpg, {it["fp"]} false positive, '
        f'{it["fn"]} missed">{phg(it["seed"])}'
        f'{ov(it["b"], "outcome")}{ov(it["b"], "pred")}{ov(it["b"], "gt")}'
        f'<span class="cap">{it["f"]} &#183; {it["fp"]} FP &#183; '
        f'{it["fn"]} FN</span></button></div>'
        for n, it in enumerate(ITEMS))

    shots = [shot_geo(i) for i in range(len(ITEMS))]
    svgs = [f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">'
            f'{blobs(it["seed"], 30)}</svg>' for it in ITEMS]
    js = (JS.replace("__SHOTS__", json.dumps(shots, separators=(",", ":")))
            .replace("var LB = ", "var SVGS = " + json.dumps(svgs) + ";\nvar LB = "))

    q_html = "\n".join(
        f'            <tr><td style="white-space:normal">{name}</td><td>{cnt}</td>'
        f'<td>{pct}</td><td style="text-align:left"><svg class="ex" '
        f'viewBox="0 0 40 26" aria-hidden="true">{ex}</svg></td>'
        f'<td style="text-align:left;white-space:normal">{why}</td></tr>'
        for name, cnt, pct, ex, why in QUALITY)

    html = f"""<!doctype html>
<html lang="en" data-theme="">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evaluation report &#183; yolo-train report-e2e &#8212; wireframe</title>
<style>{CSS}</style>
</head>
<body>

<header class="nav glass">
  <div class="nav-in">
    <span class="brandmark">
      <span class="dot" aria-hidden="true"></span>
      <span class="brand-txt">
        <span class="eyebrow">Evaluation report</span>
        <span class="brand-run">yolo-train report-e2e (detect)</span>
      </span>
    </span>
    <nav class="nav-links" aria-label="Sections">
{nav_links}
    </nav>
    <button class="tgl" id="tgl" type="button">Theme</button>
  </div>
</header>

<div class="wrap">
  <article class="sheet">

    <section id="s-summary">
      <h1>yolo-train report-e2e (detect)</h1>
      <p class="meta"><b>yolo11n.pt</b> &#183; imgsz 640 &#183; split <b>test</b>
      (166 images) &#183; 3 epochs &#183; generated 2026-07-31 06:05 UTC &#183;
      commit <span class="mono">46de57c</span> &#183;
      <span class="mono">yolo-trainer:0.2.11</span> &#183; ultralytics 8.4.110</p>

      <div class="kpis glass wide">
{kpi_html}
      </div>

      <ul class="flags">
        <li><span class="sd serious" aria-hidden="true"></span>
          <span><b>stack</b> has no ground truth in the test split &#8212; the class is
          silently absent from mAP, so the headline number describes one class, not
          two.</span></li>
        <li><span class="sd warn" aria-hidden="true"></span>
          <span><b>&#964;* = 0.000</b> &#8212; F1 never rises above noise at 3 epochs.
          There is no threshold to deploy at; the scores carry no ranking signal
          yet.</span></li>
        <li><span class="sd warn" aria-hidden="true"></span>
          <span>The final validation pass departed from the ultralytics defaults:
          <span class="mono">conf=0.001</span>, <span class="mono">batch=32</span>,
          <span class="mono">split=test</span>.</span></li>
        <li><span class="sd good" aria-hidden="true"></span>
          <span>Split is a clean partition &#8212; 361 train / 91 valid / 166 test,
          no image in two slices.</span></li>
      </ul>
    </section>

    <section id="s-model-card">
      <h2>Model card</h2>
      <p class="lede">Three confidence thresholds coexist in this task and they are not
      alternatives. mAP and every curve are computed at the NMS floor,
      <span class="mono">0.001</span> &#8212; anything below it never reaches the metrics
      at all. The confusion matrix and every gallery are drawn at the display threshold,
      <span class="mono">0.25</span>, because a matrix at the floor is all background.
      The threshold you would deploy at is the F1-optimal one, which for this run is
      <span class="mono">0.000</span>.</p>
      <p class="note">Intended use and out-of-scope notes are not stated for this run.
      Set <span class="mono">8_Visualization/report_intended_use</span> to record
      them.</p>
    </section>

    <section id="s-dataset">
      <h2>Dataset and split composition</h2>
      <p>Read from the label files on disk, after filtering and after the split. The axis
      is logarithmic because the class you have to worry about is the sliver, not the
      bulk: <b>stack</b> carries 58 instances against fruitlet's 21,247, a ratio of
      366&#215;, and none of those 58 landed in the test slice.</p>

      <figure class="wide">
        <p class="fig-t">Instances per class, all splits</p>
        {chart_instances()}
        <figcaption><b>Fig 1.</b> Sorted so the class with no test-split ground truth
        reads first. Counted from the label files, not from the CVAT task metadata
        &#8212; class indices come from the run's own class map.</figcaption>
      </figure>

      <p style="margin-top:36px">Where the boxes sit in the frame, not how many there
      are: a model only learns the corners it is shown, and this one is shown the
      middle.</p>

      <figure>
        <p class="fig-t">Box centres over the frame &#183; 24 &#215; 16 cells</p>
        {chart_heat()}
        <figcaption><b>Fig 2.</b> One cell is a twenty-fourth of the frame across by a
        sixteenth down, pooled over all 21,247 boxes. The three densest cells carry their
        count; the ramp is a single hue, so the reading is magnitude and never
        category.</figcaption>
      </figure>

      <p style="margin-top:36px">Which classes share an image decides whether a confusion
      between them is even possible &#8212; here it almost never is.</p>

      <figure>
        <p class="fig-t">Class co-occurrence</p>
        <div class="cooc">
          <div class="cooc-stat">
            <span class="cooc-n">12<span class="u"> / 618</span></span>
            <span class="cooc-l">Images holding both classes</span>
            <p class="cooc-p">Of 618 annotated images, 12 contain a <b>fruitlet</b> and a
            <b>stack</b> together &#8212; 1.9%. 548 are fruitlet-only, 58 stack-only.
            With two classes the whole matrix is one number, so it is printed rather than
            drawn.</p>
          </div>
          <div class="cooc-mx">
            <p class="illus">Illustrative &#183; not this run</p>
            {chart_matrix()}
          </div>
        </div>
        <figcaption><b>Fig 3.</b> The six-class matrix is a shape demonstration for
        multi-class runs and its numbers are invented; only the figure on the left
        describes this dataset.</figcaption>
      </figure>

      <p style="margin-top:36px">Four cheap geometric checks over the label files. None
      of them is an error on its own; each is a reason to open the image before blaming
      the model.</p>

      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th style="text-align:left">Annotation flag</th><th>Boxes</th>
            <th>Share</th><th style="text-align:left">Example</th>
            <th style="text-align:left">Why it matters</th>
          </tr></thead>
          <tbody>
{q_html}
          </tbody>
        </table>
      </div>
      <p class="note" style="margin-top:12px">Counted over all 21,305 boxes in every
      split. The border-touching population is the one worth reading twice: it is also
      where the background error of Fig 7 concentrates.</p>

      <h3 style="margin-top:40px">Image files</h3>
      <p>Everything above reads the label files. This block reads the images themselves
      &#8212; header only, <span class="mono">.size</span>,
      <span class="mono">.mode</span> and <span class="mono">.format</span>, with no
      pixel decode &#8212; so it is cheap enough to cover all 618 images rather than a
      sample. Six sizes account for 616 of them; what happens to those pixels on the way
      into the model is the rest of this block.</p>

      <figure>
        <p class="fig-t">Image resolutions, top six by count</p>
        {chart_res()}
        <figcaption><b>Fig 4.</b> Two of the six are the same sensor held the other way
        up. A resolution counter that admits unbounded distinct keys would grow with the
        dataset, so past a few thousand sizes the rest is tallied as
        <span class="mono">other</span>.</figcaption>
      </figure>

      <p style="margin-top:34px">Orientation is the same fact viewed sideways: four
      fifths of the set is landscape, a fifth portrait, and a handful square.</p>
      {chart_aspect()}

      <p style="margin-top:26px">Pixel count matters more than either, because it is what
      survives the resize. At <span class="mono">imgsz 640</span> a 12&#8239;MP frame is
      thrown away at a ratio of about 30 to 1 in area, and every small object goes
      with it.</p>

      <figure>
        <p class="fig-t">Pixels per image</p>
        {chart_mp()}
        <figcaption><b>Fig 5.</b> Discrete rather than smooth &#8212; seven sensor sizes
        and nothing between them, with 83% of the set sitting in the single bin at
        12.2&#8239;MP. The bars below 3&#8239;MP are the 115 images that were already
        small before the resize.</figcaption>
      </figure>

      <div class="statrow">
        <div class="statcol">
          <p class="stat-h">Colour mode</p>
          <p class="stat-l"><span>RGB</span><b>616</b></p>
          <p class="stat-l"><span>L (grey)</span><b>2</b></p>
          <p class="stat-l zero"><span>RGBA</span><b>0</b></p>
        </div>
        <div class="statcol">
          <p class="stat-h">File format</p>
          <p class="stat-l"><span>JPEG</span><b>614</b></p>
          <p class="stat-l"><span>PNG</span><b>4</b></p>
        </div>
        <div class="statcol">
          <p class="stat-h">Fit at imgsz&#8239;=&#8239;640</p>
          <p class="stat-l"><span>downscaled</span><b>612</b></p>
          <p class="stat-l"><span>upscaled</span><b>6</b></p>
          <p class="stat-l"><span>short side &lt; 640</span><b>6</b></p>
        </div>
      </div>
      <p class="note" style="margin-top:14px">Two greyscale images in an otherwise RGB
      set is the kind of thing that survives all the way to a confusing per-class number,
      and a short side under <span class="mono">imgsz</span> means the loader is
      inventing detail rather than discarding it.</p>
    </section>

    <section id="s-per-class">
      <h2>Per-class evaluation</h2>
      <p>Sorted by support ascending, so the unreliable rows sit at the top where they
      belong. P, R and F1 are each class's own max-F1 point; TP, FP and FN are counted at
      IoU 0.50 and the display threshold 0.25 &#8212; which is why they are all zero here,
      since &#964;* is 0.003 and nothing scores that high.</p>

      <div class="tbl-wrap wide">
        <table>
          <thead><tr>
            <th>Class</th><th>Support</th><th>AP50-95</th><th>AP50</th><th>P</th>
            <th>R</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th>
            <th><span class="tau">&#964;*</span></th>
          </tr></thead>
          <tbody>
            <tr class="dim"><td>stack</td><td>58</td><td>&#8212;</td><td>&#8212;</td>
              <td>&#8212;</td><td>&#8212;</td><td>&#8212;</td><td>&#8212;</td>
              <td>&#8212;</td><td>&#8212;</td><td>&#8212;</td></tr>
            <tr><td>fruitlet</td><td>4,057</td><td>0.0345</td><td>0.1336</td>
              <td>0.0703</td><td class="best">0.5760</td><td>0.1254</td><td>0</td>
              <td>0</td><td>4,057</td><td>0.0030</td></tr>
          </tbody>
        </table>
      </div>
      <p class="note">Built from <span class="mono">metrics.summary()</span>, which
      resolves names through <span class="mono">ap_class_index</span> &#8212; the only
      index-safe source. Classes under 30 instances are dimmed: their AP is not reliable.
      The washed cell is the best value in its column.</p>
    </section>

    <section id="s-threshold">
      <h2>Operating threshold</h2>
      <p>&#964;* is the F1-optimal threshold, and it is none of the other three
      confidences in this task. Here F1 is highest at the very floor and falls
      monotonically from there: raising the threshold discards recall without buying
      precision, which is the signature of scores that do not rank. Read it next to
      recall &#8212; this curve is computed over predictions only, so it is blind to
      false negatives.</p>

      <figure class="wide">
        <p class="fig-t">Precision, recall and F1 against confidence</p>
        {chart_prf()}
        <figcaption><b>Fig 6.</b> Values printed at &#964;*, the operating point the
        headline P/R/F1 are quoted at. Precision is grey because it is context here:
        the decision belongs to F1.</figcaption>
      </figure>
    </section>

    <section id="s-tide">
      <h2>Error decomposition</h2>
      <p>Six error types, each with the AP50 an oracle that fixed only that error would
      recover. Background detections dominate at
      <b>&#916;AP 0.2527</b> over 35,010 boxes &#8212; on a set this sparse that usually
      means <b>missing annotations</b> before it means a weak model. The six values are
      independent oracles and do not sum to the total headroom.</p>

      <figure class="wide">
        <p class="fig-t">&#916;AP50 by error type &#183; baseline AP50 0.1354</p>
        {chart_tide()}
        <figcaption><b>Fig 7.</b> Matched at IoU 0.5 for foreground and 0.1 for
        background, box IoU only &#8212; even on a segmentation run. The two dashed
        ceilings are the false-positive and false-negative oracles, which are not
        segments of the bar.</figcaption>
      </figure>

      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Type</th><th>Count</th><th>&#916;AP50</th>
            <th style="text-align:left">First lever</th></tr></thead>
          <tbody>
            <tr><td>Bkg</td><td>35,010</td><td class="best">0.2527</td>
              <td style="text-align:left">audit the annotations before the model</td></tr>
            <tr><td>Miss</td><td>1,703</td><td>0.0970</td>
              <td style="text-align:left">lower the deploy threshold, raise
              <span class="mono">max_det</span></td></tr>
            <tr><td>Loc</td><td>10,291</td><td>0.0786</td>
              <td style="text-align:left">raise <span class="mono">imgsz</span> or the
              <span class="mono">box</span> loss gain</td></tr>
            <tr><td>Both</td><td>1,219</td><td>0.0061</td>
              <td style="text-align:left">usually a very rare class</td></tr>
            <tr><td>Dupe</td><td>409</td><td>0.0005</td>
              <td style="text-align:left">raise
              <span class="mono">5_Testing/iou</span></td></tr>
            <tr><td>Cls</td><td>517</td><td>0.0001</td>
              <td style="text-align:left">check the class-map order</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section id="s-strata">
      <h2>Size and shape</h2>
      <p>Where the misses actually are. Object sizes use this dataset's own terciles at
      38px and 51px at imgsz 640, not COCO's 32/96 absolutes. Density matters more than
      size here: the median image carries about twenty fruitlets and the tail runs past
      a hundred and forty, well beyond the default
      <span class="mono">max_det</span> headroom once recall improves.</p>

      <figure class="wide">
        <p class="fig-t">Objects per image, test split</p>
        {chart_hist()}
        <figcaption><b>Fig 8.</b> Only the modal bin and the p99.5 cut are labelled;
        printing thirty numbers would bury the shape they are meant to
        describe.</figcaption>
      </figure>
    </section>

    <section id="s-galleries">
      <h2>Galleries</h2>
      <p>Ranked by false positives plus false negatives, whole image, at
      conf &#8805; 0.25. The three tabs are three readings of the same twenty-eight
      thumbnails &#8212; only the overlay layer changes, never the thumbnail. Click any
      of them for the full frame.</p>

      <div class="galwrap wide" id="galwrap" data-tab="outcome" data-anno="on">
        <div class="gal-head">
          <div class="tabs" role="tablist" aria-label="Overlay layer">
            <button class="tab" type="button" role="tab" data-tab="outcome"
              aria-selected="true" aria-controls="gal">Outcome</button>
            <button class="tab" type="button" role="tab" data-tab="pred"
              aria-selected="false" tabindex="-1" aria-controls="gal">Prediction</button>
            <button class="tab" type="button" role="tab" data-tab="gt"
              aria-selected="false" tabindex="-1" aria-controls="gal">Ground truth</button>
          </div>
          <span class="gal-meta">28 of 166 &#183; worst FP+FN</span>
          <button class="anno" id="anno" type="button" role="switch" aria-checked="true">
            <span>annotations</span>
            <span class="trk" aria-hidden="true"><span class="knb"></span></span>
          </button>
        </div>
        <div class="gal at-start" id="gal">
          <div class="grid">
{gal_html}
          </div>
        </div>
        <p class="gal-legend">
          <span class="lg-outcome">
            <i><span class="sw tp" aria-hidden="true"></span>true positive</i>
            <i><span class="sw fp" aria-hidden="true"></span>false positive</i>
            <i><span class="sw fn" aria-hidden="true"></span>missed ground truth</i>
          </span>
          <span class="lg-pred">
            <i><span class="sw pd" aria-hidden="true"></span>detection &#183;
            conf &#8805; 0.25</i>
          </span>
          <span class="lg-gt">
            <i><span class="sw gt" aria-hidden="true"></span>annotated instance</i>
          </span>
        </p>
      </div>
    </section>

    <section id="s-caveats">
      <h2>Caveats</h2>
      <ul class="caveats">
        <li>mAP and every curve are computed at <span class="mono">conf=0.001</span>;
        the confusion matrix and every gallery are drawn at
        <span class="mono">conf=0.25</span>. The two disagree by construction, not by
        mistake.</li>
        <li>One class had no ground truth in the test split and is absent from mAP
        entirely &#8212; the headline is a single-class number.</li>
        <li><span class="mono">save_crop</span> and
        <span class="mono">label_smoothing</span> were set but do nothing in
        ultralytics 8.4.110.</li>
        <li>Three epochs on a 90% fraction. Nothing in this report should be read as a
        statement about the achievable ceiling.</li>
      </ul>
      <p class="gen">Generated 2026-07-31T06:05:38+00:00 &#183; report schema 1 &#183;
      template 0.2.11 &#183; ClearML task
      <span class="mono">9dceaf4b8cf7</span>.</p>
    </section>

  </article>
</div>

<div class="lb" id="lb" role="dialog" aria-modal="true" aria-label="Gallery viewer">
  <div class="lb-stage">
    <div class="lb-img" id="lbimg"></div>
    <canvas id="lbc"></canvas>
  </div>
  <div class="lb-bar">
    <span class="lb-cap" id="lbcap"></span>
    <span class="lb-n" id="lbn"></span>
  </div>
  <button class="lb-btn prev" id="lbprev" type="button"
    aria-label="Previous image">&#8249;</button>
  <button class="lb-btn next" id="lbnext" type="button"
    aria-label="Next image">&#8250;</button>
  <button class="lb-btn x" id="lbx" type="button" aria-label="Close">&#10005;</button>
</div>

<script>
{js}
</script>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html.encode()) / 1024:.1f} KB)")


build()
