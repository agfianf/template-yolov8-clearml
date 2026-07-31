/* Report runtime. The figures are already in the document as inline SVG, so nothing
   here draws a chart; what is left is the interaction the markup cannot express.

   Four contracts the Python side depends on:
   - a lazy thumbnail is `img[data-thumb="<key>"]` and its key is a key of BLOB.thumbs;
   - a gallery tile is `figure[data-gid][data-idx]` and those index BLOB.grids, which is
     where its overlay boxes come from -- the three overlay layers are built here rather
     than server-rendered, because 5 grids x 24 tiles x 3 layers of boxes is thousands of
     nodes of markup for something only ever seen a tile at a time;
   - the annotation switch is one `body[data-anno]` attribute, and the per-grid tab is
     one `.galwrap[data-tab]` attribute -- CSS picks the visible layer, never JS;
   - filters are applied by rewriting one <style> element, never by touching each item. */
(function () {
  "use strict";
  var el = document.getElementById("report-data");
  var BLOB = {};
  try { BLOB = JSON.parse(el.textContent); } catch (e) { BLOB = {}; }
  var THUMBS = BLOB.thumbs || {};
  var GRIDS = BLOB.grids || {};
  var SVG_NS = "http://www.w3.org/2000/svg";

  /* --- theme --------------------------------------------------------------------- */
  function currentTheme() {
    return document.documentElement.dataset.theme ||
      (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark" : "light");
  }
  try {
    var saved = localStorage.getItem("report-theme");
    if (saved) { document.documentElement.dataset.theme = saved; }
  } catch (e) { /* private mode */ }

  var themeBtn = document.getElementById("theme");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("report-theme", next); } catch (e) { /* ignore */ }
      /* The charts are SVG naming CSS custom properties, so they re-colour themselves.
         Nothing to redraw. */
    });
  }

  /* --- cursor-following tooltip; the printed direct labels stay as they are -------- */
  (function () {
    var tip = document.createElement("div");
    tip.className = "tip";
    document.body.appendChild(tip);
    document.addEventListener("mousemove", function (e) {
      var node = e.target && e.target.closest ? e.target.closest("[data-tip]") : null;
      if (!node) { tip.style.display = "none"; return; }
      tip.textContent = node.getAttribute("data-tip");
      tip.style.display = "block";
      var x = e.clientX + 14, y = e.clientY + 16;
      if (x + tip.offsetWidth > innerWidth - 8) { x = e.clientX - tip.offsetWidth - 14; }
      if (y + tip.offsetHeight > innerHeight - 8) { y = e.clientY - tip.offsetHeight - 14; }
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    }, { passive: true });
    addEventListener("scroll", function () { tip.style.display = "none"; },
      { passive: true });
  })();

  /* --- overlays: three layers over one thumbnail, built from the blob -------------- */
  var TABS = {
    outcome: { keep: { tp: 1, fp: 1, fn: 1 }, cls: null, label: null },
    pred: { keep: { tp: 1, fp: 1 }, cls: "ob-pred", label: "conf" },
    gt: { keep: { gt: 1, fn: 1 }, cls: "ob-gt", label: "cls" }
  };
  var LABEL_MIN = 0.17;   /* narrower than this and the number is illegible at 128px */

  function layer(boxes, tab) {
    var spec = TABS[tab];
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "ov ov-" + tab);
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("aria-hidden", "true");
    (boxes || []).forEach(function (b) {
      var kind = b[4];
      if (!spec.keep[kind]) { return; }
      var w = (b[2] - b[0]) * 100, h = (b[3] - b[1]) * 100;
      if (w <= 0 || h <= 0) { return; }
      var rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("class", "ob " + (spec.cls || ("ob-" + kind)));
      rect.setAttribute("x", (b[0] * 100).toFixed(2));
      rect.setAttribute("y", (b[1] * 100).toFixed(2));
      rect.setAttribute("width", w.toFixed(2));
      rect.setAttribute("height", h.toFixed(2));
      rect.setAttribute("rx", "1.5");
      svg.appendChild(rect);
      if (!spec.label || (b[2] - b[0]) < LABEL_MIN) { return; }
      var text = spec.label === "conf"
        ? (b[6] == null ? "" : Number(b[6]).toFixed(2))
        : String(b[5] == null ? "" : b[5]);
      if (!text) { return; }
      var node = document.createElementNS(SVG_NS, "text");
      node.setAttribute("class", "ovt " + (tab === "pred" ? "f-cyan" : "f-teal"));
      node.setAttribute("x", Math.min(b[0] * 100 + 1, 76).toFixed(2));
      node.setAttribute("y", Math.max(b[1] * 100 - 2, 6.4).toFixed(2));
      node.textContent = text;
      svg.appendChild(node);
    });
    return svg;
  }

  function fillTile(fig) {
    var grid = GRIDS[fig.dataset.gid];
    var item = grid && grid.items && grid.items[+fig.dataset.idx];
    if (!item || fig.dataset.filled) { return; }
    fig.dataset.filled = "1";
    var boxes = item.boxes || [];
    Object.keys(TABS).forEach(function (tab) { fig.appendChild(layer(boxes, tab)); });
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) { return; }
      var node = entry.target;
      io.unobserve(node);
      var b64 = THUMBS[node.dataset.thumb];
      if (b64) { node.src = "data:image/jpeg;base64," + b64; }
      var fig = node.closest(".grid figure");
      if (fig) { fillTile(fig); }
    });
  }, { rootMargin: "200px" });

  document.querySelectorAll("img[data-thumb]").forEach(function (n) { io.observe(n); });

  /* --- gallery: per-grid tabs, one global annotation switch ------------------------ */
  var ANNO = true;
  function setAnno(on) {
    ANNO = on;
    document.body.dataset.anno = on ? "on" : "off";
    document.querySelectorAll(".anno").forEach(function (sw) {
      sw.setAttribute("aria-checked", on ? "true" : "false");
    });
    /* `lb` is hoisted but not assigned until later in this IIFE, and setAnno runs
       once at startup -- so this has to tolerate it being undefined. */
    if (lb && lb.classList.contains("on")) { drawLightbox(); }
  }
  document.querySelectorAll(".anno").forEach(function (sw) {
    sw.addEventListener("click", function () { setAnno(!ANNO); });
  });
  /* Set here rather than in the template so the markup carries no state: nothing is
     visible before this script runs anyway, because the overlay layers are built here
     too. */
  setAnno(true);
  document.addEventListener("keydown", function (e) {
    if (e.key !== "a" && e.key !== "A") { return; }
    /* Bare 'a' only -- without this, Ctrl+A / Cmd+A to select the page text also
       cleared every overlay in the gallery. */
    if (e.ctrlKey || e.metaKey || e.altKey) { return; }
    var t = e.target;
    if (t && /^(input|textarea|select)$/i.test(t.tagName)) { return; }
    if (t && t.isContentEditable) { return; }
    setAnno(!ANNO);
  });

  document.querySelectorAll(".galwrap").forEach(function (wrap) {
    var tabs = [].slice.call(wrap.querySelectorAll('[role="tab"]'));
    var gal = wrap.querySelector(".gal");
    function select(i) {
      tabs.forEach(function (t, n) {
        t.setAttribute("aria-selected", n === i ? "true" : "false");
        t.tabIndex = n === i ? 0 : -1;
      });
      wrap.dataset.tab = tabs[i].dataset.tab;
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(i); });
      t.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" ? 1 : (e.key === "ArrowLeft" ? -1 : 0);
        if (!d) { return; }
        e.preventDefault();
        var n = (i + d + tabs.length) % tabs.length;
        select(n);
        tabs[n].focus();
      });
    });
    if (!gal) { return; }
    function edges() {
      gal.classList.toggle("at-start", gal.scrollTop <= 2);
      gal.classList.toggle("at-end",
        gal.scrollTop + gal.clientHeight >= gal.scrollHeight - 2);
    }
    gal.addEventListener("scroll", edges, { passive: true });
    addEventListener("resize", edges);
    edges();
  });

  /* --- confusion matrix: counts <-> row-normalised, by attribute ------------------- */
  var cmBtn = document.getElementById("cm-toggle");
  if (cmBtn) {
    cmBtn.addEventListener("click", function () {
      var wrap = document.getElementById("cm-wrap");
      var norm = wrap.dataset.cm === "norm";
      wrap.dataset.cm = norm ? "counts" : "norm";
      cmBtn.textContent = norm ? "Show row-normalised" : "Show counts";
    });
  }

  /* --- filters -------------------------------------------------------------------- */
  var rules = document.getElementById("filter-rules");
  function esc(v) { return window.CSS && CSS.escape ? CSS.escape(v) : v.replace(/"/g, ""); }
  function setFilter(kind, value) {
    var body = document.body;
    var attr = kind === "pair" ? "pairFilter" : "classFilter";
    if (body.dataset[attr] === value) { value = ""; }
    body.dataset.classFilter = "";
    body.dataset.pairFilter = "";
    if (value) { body.dataset[attr] = value; }
    applyFilter();
  }
  function applyFilter() {
    var cls = document.body.dataset.classFilter;
    var pair = document.body.dataset.pairFilter;
    var css = "";
    if (cls) {
      css += 'body[data-class-filter] .grid figure:not([data-classes~="' +
        esc(cls) + '"]){display:none}';
    }
    if (pair) {
      css += 'body[data-pair-filter] .grid figure:not([data-pairs~="' +
        esc(pair) + '"]){display:none}';
    }
    rules.textContent = css;
    var bar = document.getElementById("filterbar");
    var text = document.getElementById("filtertext");
    if (cls || pair) {
      text.textContent = (cls ? "Class: " + cls : "Confusion: " + pair) +
        " - galleries filtered";
      bar.hidden = false;
    } else { bar.hidden = true; }
    document.querySelectorAll(".grid").forEach(function (g) {
      var total = g.children.length, shown = 0;
      Array.prototype.forEach.call(g.children, function (fig) {
        if (fig.offsetParent !== null || !(cls || pair)) { shown++; }
      });
      var out = g.closest(".galwrap").querySelector(".shown");
      if (out) { out.textContent = shown + " of " + total; }
    });
    document.querySelectorAll("tr.on").forEach(function (r) { r.classList.remove("on"); });
    if (cls) {
      document.querySelectorAll('tr[data-class="' + cls.replace(/"/g, "") + '"]')
        .forEach(function (r) { r.classList.add("on"); });
    }
  }
  var clear = document.getElementById("filterclear");
  if (clear) { clear.addEventListener("click", function () { setFilter("class", ""); }); }

  /* --- lightbox: one overlay, one reused <img>, never :target ---------------------- */
  var lb = document.getElementById("lb");
  var lbImg = document.getElementById("lbimg");
  var lbOvl = document.getElementById("lbovl");
  var lbCap = document.getElementById("lbcap");
  var lbN = document.getElementById("lbn");
  var order = [];
  var at = 0;
  var opener = null;

  document.querySelectorAll(".grid figure").forEach(function (fig) {
    order.push([fig.dataset.gid, +fig.dataset.idx]);
  });

  function drawLightbox() {
    var gid = order[at][0], idx = order[at][1];
    var grid = GRIDS[gid];
    var item = grid && grid.items && grid.items[idx];
    if (!item) { return; }
    lbImg.src = "data:image/jpeg;base64," + (THUMBS[item.thumb] || "");
    lbOvl.textContent = "";
    if (ANNO) {
      var tab = "outcome";
      var wrap = document.querySelector('.galwrap[data-gid="' + gid + '"]');
      if (wrap && wrap.dataset.tab) { tab = wrap.dataset.tab; }
      var built = layer(item.boxes, tab);
      while (built.firstChild) { lbOvl.appendChild(built.firstChild); }
    }
    lbCap.textContent = item.label + " - " + grid.basis + " - " +
      (grid.thumb_px || 192) + "px thumbnail, enlarged";
    lbN.textContent = (at + 1) + " of " + order.length;
  }
  function openLightbox(i) {
    if (!order.length) { return; }
    at = (i + order.length) % order.length;
    lb.classList.add("on");
    lb.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    drawLightbox();
  }
  function closeLightbox() {
    lb.classList.remove("on");
    lb.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    lbImg.src = "";
    if (opener) { opener.focus(); opener = null; }
  }
  document.getElementById("lbclose").addEventListener("click", closeLightbox);
  document.getElementById("lbprev").addEventListener("click",
    function () { openLightbox(at - 1); });
  document.getElementById("lbnext").addEventListener("click",
    function () { openLightbox(at + 1); });
  lb.addEventListener("click", function (e) {
    if (e.target === lb) { closeLightbox(); }
  });
  document.addEventListener("keydown", function (e) {
    if (!lb.classList.contains("on")) { return; }
    if (e.key === "Escape") { closeLightbox(); }
    if (e.key === "ArrowLeft") { openLightbox(at - 1); }
    if (e.key === "ArrowRight") { openLightbox(at + 1); }
  });
  /* Range-check the deep link rather than letting the cycling modulo wrap it: #lb=0
     used to open the last image and #lb=99 an unrelated one, both silently. */
  if (/^#lb=\d+$/.test(location.hash)) {
    var want = +location.hash.slice(4);
    if (want >= 1 && want <= order.length) { openLightbox(want - 1); }
  }

  /* --- one delegated click listener for rows and gallery items --------------------- */
  document.addEventListener("click", function (e) {
    var fig = e.target.closest && e.target.closest(".grid figure");
    if (fig) {
      opener = fig;
      var key = fig.dataset.gid + "/" + fig.dataset.idx;
      for (var i = 0; i < order.length; i++) {
        if (order[i][0] + "/" + order[i][1] === key) { openLightbox(i); return; }
      }
      return;
    }
    var row = e.target.closest && e.target.closest("tr[data-class],tr[data-pair]");
    if (row) {
      if (row.dataset.pair) { setFilter("pair", row.dataset.pair); }
      else { setFilter("class", row.dataset.class); }
    }
  });

  /* --- active section, on both link sets ------------------------------------------ */
  /* The TOC is hidden below 1400px, and on a 1280-wide laptop that left a thirteen-
     section report with no position marker at all -- so the nav links track too. */
  (function () {
    var links = [].slice.call(document.querySelectorAll(".toc a, .nav-links a"));
    if (!links.length || !window.IntersectionObserver) { return; }
    var ids = [].slice.call(document.querySelectorAll(".toc a")).map(function (a) {
      return a.getAttribute("href").slice(1);
    });
    var seen = {};
    function paint() {
      var id = ids.filter(function (s) { return seen[s]; })[0] || ids[0];
      links.forEach(function (a) {
        var on = a.getAttribute("href").slice(1) === id;
        a.classList.toggle("on", on);
        if (on) { a.setAttribute("aria-current", "true"); }
        else { a.removeAttribute("aria-current"); }
      });
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { seen[e.target.id] = e.isIntersecting; });
      paint();
    }, { rootMargin: "-56px 0px -62% 0px" });
    ids.forEach(function (s) {
      var node = document.getElementById(s);
      if (node) { obs.observe(node); }
    });
    paint();
  })();

  applyFilter();
})();
