/* Report runtime. Everything is driven from the single JSON blob; the markup carries
   only ids, so the page stays inside its DOM-node budget while the data stays readable.

   Four contracts the Python side depends on:
   - a figure div is `div[data-fig="<id>"]` and its id is a key of BLOB.figures;
   - a lazy thumbnail is `img[data-thumb="<key>"]` and its key is a key of BLOB.thumbs;
   - a figure is never inside `display:none` or a closed <details> when it is observed,
     because Plotly renders zero-width there and never recovers -- the collapsed
     appendix observes its figures from its own `toggle` handler;
   - filters are applied by rewriting one <style> element, never by touching each item. */
(function () {
  "use strict";
  var el = document.getElementById("report-data");
  var BLOB = {};
  try { BLOB = JSON.parse(el.textContent); } catch (e) { BLOB = {}; }
  var FIGS = BLOB.figures || {};
  var THUMBS = BLOB.thumbs || {};
  var GRIDS = BLOB.grids || {};
  var TPL = BLOB.plotly_template || {};
  var rendered = {};

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
      /* Bounded: only figures already on screen are touched. Plotly.relayout cannot
         swap a template's colorway reliably, so purge and re-plot. */
      Object.keys(rendered).forEach(function (id) {
        var div = document.querySelector('[data-fig="' + id + '"]');
        if (!div) { return; }
        if (window.Plotly) { Plotly.purge(div); }
        delete rendered[id];
        plot(div, id);
      });
    });
  }

  /* --- figures ------------------------------------------------------------------- */
  function plot(div, id) {
    var f = FIGS[id];
    if (!f) {
      div.className = "fig err";
      div.textContent = "This figure was not built for this run - see the caveats.";
      return;
    }
    if (!window.Plotly) { return; }
    var layout = JSON.parse(JSON.stringify(f.layout || {}));
    layout.template = TPL[currentTheme()] || TPL.light;
    Plotly.newPlot(div, f.data || [], layout, {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    });
    rendered[id] = true;
    if (id === "f_confusion") { wireConfusion(div); }
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) { return; }
      var node = entry.target;
      io.unobserve(node);
      if (node.dataset.fig) { plot(node, node.dataset.fig); }
      else if (node.dataset.thumb) {
        var b64 = THUMBS[node.dataset.thumb];
        if (b64) { node.src = "data:image/jpeg;base64," + b64; }
      }
    });
  }, { rootMargin: "200px" });

  function observeAll(root) {
    (root || document).querySelectorAll("[data-fig],[data-thumb]").forEach(function (n) {
      if (!n.dataset.observed) { n.dataset.observed = "1"; io.observe(n); }
    });
  }
  /* Figures inside a closed <details> are claimed *before* the first sweep and handed
     back only when it opens: Plotly renders zero-width in a hidden container and never
     recovers from it. */
  document.querySelectorAll("details.lazy [data-fig]").forEach(function (n) {
    n.dataset.observed = "deferred";
  });
  document.querySelectorAll("details.lazy").forEach(function (d) {
    d.addEventListener("toggle", function () {
      if (!d.open) { return; }
      d.querySelectorAll('[data-observed="deferred"]').forEach(function (n) {
        n.dataset.observed = "";
      });
      observeAll(d);
    });
  });
  observeAll(document.querySelector("main"));

  /* --- confusion matrix: counts <-> row-normalised -------------------------------- */
  function wireConfusion(div) {
    var btn = document.getElementById("cm-toggle");
    var f = FIGS.f_confusion;
    if (btn && f && f.alt_z && !btn.dataset.wired) {
      btn.dataset.wired = "1";
      var counts = (f.data && f.data[0] && f.data[0].z) || [];
      btn.addEventListener("click", function () {
        var norm = btn.dataset.mode === "norm";
        Plotly.restyle(div, { z: [norm ? counts : f.alt_z] }, [0]);
        btn.dataset.mode = norm ? "counts" : "norm";
        btn.textContent = norm ? "Show row-normalised" : "Show counts";
      });
    }
    div.on && div.on("plotly_click", function (ev) {
      var pt = ev.points && ev.points[0];
      if (!pt) { return; }
      setFilter("pair", String(pt.y) + ">" + String(pt.x));
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
      var out = g.parentNode.querySelector(".shown");
      if (out) { out.textContent = shown + " of " + total + " shown"; }
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
  var opener = null;

  function openLightbox(gid, idx) {
    var grid = GRIDS[gid];
    if (!grid || !grid.items[idx]) { return; }
    var item = grid.items[idx];
    lbImg.src = "data:image/jpeg;base64," + (THUMBS[item.thumb] || "");
    lbOvl.textContent = "";
    (item.boxes || []).forEach(function (b) {
      var d = document.createElement("div");
      d.className = "bx o-" + b[4];
      d.style.left = (b[0] * 100) + "%";
      d.style.top = (b[1] * 100) + "%";
      d.style.width = ((b[2] - b[0]) * 100) + "%";
      d.style.height = ((b[3] - b[1]) * 100) + "%";
      var s = document.createElement("span");
      s.className = "lbl";
      s.textContent = b[4].toUpperCase() + " " + (b[5] || "") +
        (b[6] == null ? "" : " " + Number(b[6]).toFixed(2)) +
        (b[7] == null ? "" : " IoU " + Number(b[7]).toFixed(2));
      d.appendChild(s);
      lbOvl.appendChild(d);
    });
    lbCap.textContent = item.label + "  -  192px thumbnail, enlarged; " + grid.basis;
    lb.hidden = false;
  }
  function closeLightbox() {
    lb.hidden = true;
    lbImg.src = "";
    if (opener) { opener.focus(); opener = null; }
  }
  document.getElementById("lbclose").addEventListener("click", closeLightbox);
  lb.addEventListener("click", function (e) {
    if (e.target === lb || e.target.id === "lbstage") { closeLightbox(); }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !lb.hidden) { closeLightbox(); }
  });

  /* --- one delegated click listener for rows, chips and gallery items -------------- */
  document.addEventListener("click", function (e) {
    var fig = e.target.closest && e.target.closest(".grid figure");
    if (fig) {
      opener = fig;
      openLightbox(fig.dataset.gid, +fig.dataset.idx);
      return;
    }
    var row = e.target.closest && e.target.closest("tr[data-class],tr[data-pair]");
    if (row) {
      if (row.dataset.pair) { setFilter("pair", row.dataset.pair); }
      else { setFilter("class", row.dataset.class); }
    }
  });

  /* --- nav highlight -------------------------------------------------------------- */
  var links = {};
  document.querySelectorAll("#nav a").forEach(function (a) {
    links[a.getAttribute("href").slice(1)] = a;
  });
  var navObs = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      var a = links[entry.target.id];
      if (a && entry.isIntersecting) {
        document.querySelectorAll("#nav a.active").forEach(function (n) {
          n.classList.remove("active");
        });
        a.classList.add("active");
      }
    });
  }, { rootMargin: "-64px 0px -70% 0px" });
  document.querySelectorAll("main section").forEach(function (s) { navObs.observe(s); });

  applyFilter();
})();
