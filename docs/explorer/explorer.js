// Hydroscope Aguascalientes · Explorer
// Every number shown here is read from window.HYDROSCOPE_DATA (data.js), which is written by
// scripts/make_explorer_data.py from experiments/results/*.json. The page computes nothing that
// the paper reports: it only formats those values (3 decimals, like the paper) and, in the spatial
// split demo, counts cells for the random split it draws itself. Images (images.js) are optional.
(function () {
  "use strict";

  const DATA = window.HYDROSCOPE_DATA;
  const SCHEMA_VERSION = 1;
  const REPO_BLOB = "https://github.com/alejandro-serrano/ags-hydroscope/blob/master/";
  const COLOR = {
    eurosat: "#213E63", zeroShot: "#BB4B3E", head: "#4777B4", lastBlock: "#0D7F91",
    base: "#D5DCE6", dropBand: "#FAECEB", finetune: "#4777B4", test: "#D29D41", touching: "#BB4B3E",
  };
  const STAGES = [
    { key: "eurosat", label: "EuroSAT test" },
    { key: "zero_shot", label: "Zero-shot" },
    { key: "head", label: "head" },
    { key: "last_block", label: "last_block" },
  ];
  const BANDS_LABEL = { rgb: "RGB", ms13: "13 bands" };
  // Paper figures: MODEL_COLORS in scripts/make_results.py (Fig. 2, convergence).
  const PAPER = { colors: { resnet50: "#2a78d6", efficientnet_b0: "#eb6834", vit_small: "#1baf7a" }, ink: "#222222", gray: "#777777" };
  // Presentation figures (dumbbell and cost scatter slides).
  const SLIDE = {
    ink: "#13232B", soft: "#4A5560", grid: "#D8D1C2", teal: "#1E7A88", tealLight: "#7FB3BB", orange: "#C7741A",
    orangeText: "#A85F12", gray: "#8A949C", dropBand: "#F6E3C8",
    model: { resnet50: "#1E7A88", efficientnet_b0: "#7FB3BB", vit_small: "#C7741A" },
  };
  const paperColor = (m) => PAPER.colors[m] || "#68758D";
  const slideColor = (m) => SLIDE.model[m] || "#68758D";

  const fmt3 = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(3));
  const pct = (v) => (v * 100).toFixed(1) + " %";
  const pretty = (name) => name.replace(/([a-z])([A-Z])/g, "$1 $2");
  const el = (tag, attrs, html) => {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (html !== undefined) node.innerHTML = html;
    return node;
  };

  // ------------------------------------------------------------------ data checks
  if (!DATA) {
    document.getElementById("main").innerHTML =
      '<div class="tarjeta error-card"><div class="rotulo">Data not found</div><h2>data.js is missing</h2>' +
      '<p class="lead">Generate it from the repository root with <code>python scripts/make_results.py</code> ' +
      "(or <code>python scripts/make_explorer_data.py</code>) and reload this page.</p></div>";
    document.querySelector(".tabs").hidden = true;
    return;
  }
  const warnings = [];
  if (DATA.schema_version !== SCHEMA_VERSION) {
    warnings.push(`data.js has schema ${DATA.schema_version}; this page expects ${SCHEMA_VERSION}.`);
  }
  const IMG = window.HYDROSCOPE_IMAGES || null;
  const IMG_SCHEMA_VERSION = 1;
  if (IMG && IMG.schema_version !== IMG_SCHEMA_VERSION) {
    warnings.push(`images.js has schema ${IMG.schema_version}; this page expects ${IMG_SCHEMA_VERSION}.`);
  }

  const runsById = new Map(DATA.runs.map((r) => [r.run_id, r]));
  const presentIn = (order, key) => {
    const seen = [...new Set(DATA.runs.map((r) => r[key]))];
    return [...(order || []).filter((v) => seen.includes(v)), ...seen.filter((v) => !(order || []).includes(v))];
  };
  const MODELS = presentIn(DATA.order.model, "model");
  const BANDS = presentIn(DATA.order.bands, "bands");
  const INITS = presentIn(DATA.order.init, "init");
  const nameOf = (kind, v) => (kind === "bands" ? BANDS_LABEL[v] : null) || (DATA.names[kind] || {})[v] || v;
  const selected = (DATA.selection && DATA.selection.all && DATA.selection.all.winner) || DATA.runs[0].run_id;
  const bestImageNet = DATA.selection && DATA.selection.imagenet && DATA.selection.imagenet.winner;
  const excludedClasses = DATA.classes.filter((c) => !DATA.macro_f1_classes.includes(c));

  // ------------------------------------------------------------------ state
  const state = {
    tab: "data", run: runsById.has(selected) ? selected : DATA.runs[0].run_id,
    stage: "zero_shot", view: "count", cell: null, formulas: false,
    splitMode: "blocks", seed: 1, step: 0, basemap: true,
    curveBands: "rgb", curveMetric: "val_macro_f1", costX: "patches_per_s", costY: "ags_last_block",
  };
  const renderers = {};
  function setState(patch) {
    Object.assign(state, patch);
    Object.values(renderers).forEach((fn) => fn());
  }

  // ------------------------------------------------------------------ tooltip
  const tip = document.getElementById("tooltip");
  function showTip(event, html) {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px";
    tip.style.top = event.clientY + 14 + "px";
  }
  const hideTip = () => (tip.style.opacity = 0);

  // ------------------------------------------------------------------ header and footer
  (function header() {
    const inputs = DATA.inputs || {};
    const commits = inputs.result_commits || [];
    const pills = [
      `<span class="pill"><span class="dot"></span>${inputs.n_evaluations} evaluations · ${DATA.runs.length} runs</span>`,
      `<span class="pill">protocol ${(inputs.protocol_versions || []).join(", ")}</span>`,
      `<span class="pill" title="${commits.join(", ")}">results @ ${commits.length <= 2 ? commits.join(", ") : commits.length + " commits"}</span>`,
    ];
    document.getElementById("pills").innerHTML = pills.join("");
    document.getElementById("footer").innerHTML =
      "Every number on this page is read from <code>docs/explorer/data.js</code>, written by " +
      "<code>scripts/make_explorer_data.py</code> from <code>experiments/results/*.json</code> " +
      "(it runs with <code>python scripts/make_results.py</code>). One seed per configuration. " +
      `Code: <a href="https://github.com/alejandro-serrano/ags-hydroscope">github.com/alejandro-serrano/ags-hydroscope</a>.` +
      (IMG ? "<br>Image patches: <code>docs/explorer/images.js</code>, written by <code>scripts/make_explorer_images.py</code>. " +
        "EuroSAT patches: Helber et al. (MIT license). Aguascalientes patches contain modified Copernicus Sentinel data (2024)." : "") +
      (warnings.length ? `<br><strong>Warning:</strong> ${warnings.join(" ")}` : "");
  })();

  // ------------------------------------------------------------------ tabs
  const TABS = ["data", "eurosat", "runs", "confusion", "cost", "gradcam", "split", "code"];
  const demoLabel = (tab) => `Demo ${TABS.indexOf(tab) + 1}`;
  function showTab(tab) {
    if (!TABS.includes(tab)) tab = TABS[0];
    document.querySelectorAll(".tab").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === tab));
    TABS.forEach((t) => (document.getElementById("demo-" + t).hidden = t !== tab));
    if (location.hash.slice(1) !== tab) history.replaceState(null, "", "#" + tab);
    window.scrollTo(0, 0);
    setState({ tab });
  }
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

  // ------------------------------------------------------------------ shared controls
  function segmented(container, options, onPick) {
    const wrap = el("div", { class: "seg", role: "group" });
    const buttons = options.map((opt) => {
      const b = el("button", { type: "button", "aria-pressed": "false" }, opt.label);
      b.addEventListener("click", () => onPick(opt.value));
      wrap.appendChild(b);
      return { b, opt };
    });
    container.appendChild(wrap);
    return function update(current, isEnabled) {
      buttons.forEach(({ b, opt }) => {
        b.setAttribute("aria-pressed", String(opt.value === current));
        const enabled = isEnabled ? isEnabled(opt.value) : true;
        b.disabled = !enabled;
        // a disabled button says why in its tooltip; an enabled one has no title
        if (enabled) b.removeAttribute("title");
        else b.title = opt.value === "ssl4eo"
          ? "SSL4EO-S12 weights exist only for ResNet-50 and ViT-S with 13 bands"
          : "No run with this combination";
      });
    };
  }

  function findRun(model, bands, init) {
    return DATA.runs.find((r) => r.model === model && r.bands === bands && r.init === init);
  }
  function pickRun(model, bands, init) {
    const run = findRun(model, bands, init) ||
      DATA.runs.find((r) => r.model === model && r.bands === bands) ||
      DATA.runs.find((r) => r.model === model && r.init === init) ||
      DATA.runs.find((r) => r.model === model);
    if (run) setState({ run: run.run_id, cell: null });
  }

  function runPicker(container) {
    const groups = [];
    [["model", MODELS, "Model"], ["bands", BANDS, "Input bands"], ["init", INITS, "Pre-trained weights"]].forEach(([kind, values, title]) => {
      const box = el("div", { class: "control" });
      box.appendChild(el("span", { class: "rotulo" }, title));
      const update = segmented(box, values.map((v) => ({ value: v, label: nameOf(kind, v) })), (v) => {
        const run = runsById.get(state.run);
        const next = { model: run.model, bands: run.bands, init: run.init, [kind]: v };
        pickRun(next.model, next.bands, next.init);
      });
      container.appendChild(box);
      groups.push([kind, update]);
    });
    const badges = el("div", { class: "badges" });
    container.appendChild(badges);
    return function update() {
      const run = runsById.get(state.run);
      groups.forEach(([kind, fn]) => {
        if (kind === "model") fn(run.model);
        if (kind === "bands") fn(run.bands, (v) => DATA.runs.some((r) => r.model === run.model && r.bands === v));
        if (kind === "init") fn(run.init, (v) => Boolean(findRun(run.model, run.bands, v)));
      });
      const items = [];
      if (run.run_id === selected) items.push('<span class="badge sel">★ Selected model, by the rule fixed in advance</span>');
      if (run.run_id === bestImageNet && bestImageNet !== selected) items.push('<span class="badge imnet">Best with ImageNet weights</span>');
      badges.innerHTML = items.join("");
    };
  }

  // ================================================================== demo 1 · results on EuroSAT
  // Same layout and colors as Fig. 2 of the paper (scripts/make_results.py: plot_convergence).
  (function demoEuroSAT() {
    const root = document.getElementById("demo-eurosat");
    const curves = DATA.curves || null;
    const tr = DATA.training || {};
    const testF1 = DATA.runs.map((r) => r.evaluations.eurosat.macro_f1);
    const paperLabel = (r) => `${nameOf("model", r.model)} / ${(DATA.names.init || {})[r.init] || r.init}`;
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("eurosat")} · Training on EuroSAT</div>
        <h2>On EuroSAT, the ${DATA.runs.length} runs end up almost tied</h2>
        <p class="lead">Every run was trained on the same EuroSAT split with one shared protocol${tr.max_epochs ? `: up to ${tr.max_epochs} epochs, with early stopping (patience ${tr.patience}) on validation macro-F1` : ""}.
        The curves follow validation during training; the dots below are the final score on the EuroSAT test set.</p>
      </div>
      <div class="grid">
        <div class="tarjeta panel" id="eu-panel"></div>
        <div class="stack">
          <div class="tarjeta"><h3 id="eu-curve-title"></h3><div id="eu-curves" class="panels"></div></div>
          <div class="tarjeta"><h3>EuroSAT test set <span class="hint" style="font-weight:400">· click a row to select that run in the other demos</span></h3>
            <div id="eu-test"></div>
            <div class="legend"><span><i style="background:#222"></i>Macro-F1 (${DATA.classes.length} classes)</span><span><i style="background:#fff;border:2px solid #222"></i>Accuracy</span></div>
          </div>
        </div>
      </div>`;

    const panel = document.getElementById("eu-panel");
    panel.appendChild(el("div", { class: "tile" },
      `<div class="v">${fmt3(d3.min(testF1))} – ${fmt3(d3.max(testF1))}</div><div class="k">EuroSAT test macro-F1, all ${DATA.runs.length} runs</div>`));
    const metricBox = el("div", { class: "control" });
    metricBox.appendChild(el("span", { class: "rotulo" }, "Validation metric"));
    const updateMetric = segmented(metricBox, [{ value: "val_macro_f1", label: "Macro-F1" }, { value: "val_accuracy", label: "Accuracy" }], (v) => setState({ curveMetric: v }));
    panel.appendChild(metricBox);
    panel.appendChild(el("p", { class: "hint" },
      "One panel per input, with the same vertical axis. Solid lines: ImageNet weights; dashed: SSL4EO-S12. " +
      "A curve that ends before the last epoch was stopped by early stopping. Hover a panel to read every run at one epoch."));

    if (!curves) {
      document.getElementById("eu-curves").innerHTML =
        '<div class="callout warm">The training curves are not in <code>data.js</code> yet. Run <code>python scripts/make_results.py</code> and reload.</div>';
    }
    const maxEpoch = curves ? d3.max(Object.values(curves), (c) => d3.max(c.epoch)) : tr.max_epochs || 1;
    const PW = 440, PH = 380, pm = { l: 56, r: 12, t: 30, b: 44 };
    const x = d3.scaleLinear().domain([0.5, maxEpoch + 0.5]).range([pm.l, PW - pm.r]);
    const y = d3.scaleLinear().range([PH - pm.b, pm.t]);
    const panels = curves ? BANDS.map((bands) => {
      const svg = d3.select("#eu-curves").append("svg").attr("viewBox", `0 0 ${PW} ${PH}`)
        .attr("role", "img").attr("aria-label", `Validation curves on EuroSAT, ${nameOf("bands", bands)} input`);
      svg.append("text").attr("x", (pm.l + PW - pm.r) / 2).attr("y", 18).attr("text-anchor", "middle").attr("font-size", 14)
        .attr("fill", PAPER.ink).text(`${(DATA.names.bands || {})[bands] || bands} input`);
      const grid = svg.append("g");
      const gx = svg.append("g").attr("class", "axis paper").attr("transform", `translate(0,${PH - pm.b})`);
      const gy = svg.append("g").attr("class", "axis paper").attr("transform", `translate(${pm.l},0)`);
      svg.append("rect").attr("x", pm.l).attr("y", pm.t).attr("width", PW - pm.r - pm.l).attr("height", PH - pm.b - pm.t)
        .attr("fill", "none").attr("stroke", PAPER.gray).attr("stroke-width", 0.9);
      svg.append("text").attr("x", (pm.l + PW - pm.r) / 2).attr("y", PH - 8).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", PAPER.ink).text("Epoch");
      const yTitle = svg.append("text").attr("transform", `translate(14,${(PH - pm.b + pm.t) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", PAPER.ink);
      const linesG = svg.append("g"), dotsG = svg.append("g"), legendG = svg.append("g");
      const hover = svg.append("line").attr("stroke", "#9AA5B4").attr("stroke-dasharray", "3 3").attr("y1", pm.t).attr("y2", PH - pm.b).attr("opacity", 0);
      const runs = DATA.runs.filter((r) => r.bands === bands && curves[r.run_id]);
      svg.append("rect").attr("x", pm.l).attr("y", pm.t).attr("width", PW - pm.r - pm.l).attr("height", PH - pm.b - pm.t).attr("fill", "transparent")
        .on("mousemove", (e) => {
          const [mx] = d3.pointer(e);
          const ep = Math.max(1, Math.min(maxEpoch, Math.round(x.invert(mx))));
          hover.attr("x1", x(ep)).attr("x2", x(ep)).attr("opacity", 1);
          const rows = runs.map((r) => {
            const c = curves[r.run_id], i = c.epoch.indexOf(ep);
            return `<span style="color:${paperColor(r.model)}">●</span> ${paperLabel(r)}: <b>${i >= 0 ? fmt3(c[state.curveMetric][i]) : "stopped"}</b>`;
          });
          showTip(e, `${(DATA.names.bands || {})[bands] || bands} input · epoch ${ep}<br>${rows.join("<br>")}`);
        })
        .on("mouseleave", () => { hover.attr("opacity", 0); hideTip(); });
      return { bands, runs, svg, grid, gx, gy, yTitle, linesG, dotsG, legendG };
    }) : [];

    function drawCurves() {
      const metric = state.curveMetric;
      // shared y-limits with a 0.02 margin, like convergence_limits() in make_results.py
      const values = DATA.runs.filter((r) => curves[r.run_id]).flatMap((r) => curves[r.run_id][metric]);
      y.domain([Math.max(0, d3.min(values) - 0.02), Math.min(1, d3.max(values) + 0.02)]);
      document.getElementById("eu-curve-title").textContent =
        `Validation ${metric === "val_macro_f1" ? "macro-F1" : "accuracy"} per epoch`;
      panels.forEach((p) => {
        const t = p.svg.transition().duration(600);
        p.gx.call(d3.axisBottom(x).tickValues(d3.range(2, maxEpoch + 1, 2)).tickSizeOuter(0));
        p.gy.transition(t).call(d3.axisLeft(y).ticks(8).tickFormat(d3.format(".2f")).tickSizeOuter(0));
        const gl = [...y.ticks(8).map((v) => ({ k: "y" + v, x1: pm.l, x2: PW - pm.r, y1: y(v), y2: y(v) })),
          ...d3.range(2, maxEpoch + 1, 2).map((v) => ({ k: "x" + v, x1: x(v), x2: x(v), y1: pm.t, y2: PH - pm.b }))];
        p.grid.selectAll("line").data(gl, (d) => d.k).join("line").attr("stroke", "#B0B0B0").attr("stroke-opacity", 0.3)
          .attr("x1", (d) => d.x1).attr("x2", (d) => d.x2).attr("y1", (d) => d.y1).attr("y2", (d) => d.y2);
        p.yTitle.text(metric === "val_macro_f1" ? "Validation macro-F1" : "Validation accuracy");
        const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
        const pts = (r) => curves[r.run_id].epoch.map((ep, i) => [ep, curves[r.run_id][metric][i], r]);
        p.linesG.selectAll("path").data(p.runs, (r) => r.run_id).join("path").attr("fill", "none").attr("stroke-width", 1.7)
          .attr("stroke", (r) => paperColor(r.model)).attr("stroke-dasharray", (r) => (r.init === "imagenet" ? null : "6 4"))
          .transition(t).attr("d", (r) => line(pts(r)));
        p.dotsG.selectAll("circle").data(p.runs.flatMap(pts), (d) => d[2].run_id + ":" + d[0]).join("circle").attr("r", 2.6)
          .attr("fill", (d) => paperColor(d[2].model))
          .transition(t).attr("cx", (d) => x(d[0])).attr("cy", (d) => y(d[1]));
        // legend inside, lower right, with a frame (matplotlib loc="lower right", frameon=True)
        const lh = 17, lw = 198;
        const lx = PW - pm.r - lw - 6, ly = PH - pm.b - p.runs.length * lh - 14;
        p.legendG.selectAll("*").remove();
        p.legendG.append("rect").attr("x", lx).attr("y", ly).attr("width", lw).attr("height", p.runs.length * lh + 8)
          .attr("rx", 3).attr("fill", "#FFFFFF").attr("fill-opacity", 0.85).attr("stroke", "#D0D0D0");
        p.runs.forEach((r, i) => {
          const yy = ly + 12 + i * lh;
          p.legendG.append("line").attr("x1", lx + 8).attr("x2", lx + 30).attr("y1", yy).attr("y2", yy)
            .attr("stroke", paperColor(r.model)).attr("stroke-width", 1.7).attr("stroke-dasharray", r.init === "imagenet" ? null : "6 4");
          p.legendG.append("circle").attr("cx", lx + 19).attr("cy", yy).attr("r", 2.6).attr("fill", paperColor(r.model));
          p.legendG.append("text").attr("x", lx + 36).attr("y", yy + 4).attr("font-size", 11.5).attr("fill", PAPER.ink).text(paperLabel(r));
        });
      });
    }

    // --- test set dot plot (all runs), paper colors
    const tW = 860, tRow = 30, tL = 250, tR = 70, tTop = 6;
    const allVals = DATA.runs.flatMap((r) => [r.evaluations.eurosat.macro_f1, r.evaluations.eurosat.accuracy]);
    const tx = d3.scaleLinear().domain([Math.floor(d3.min(allVals) * 100 - 0.5) / 100, 1]).range([tL, tW - tR]);
    const tH = tTop + DATA.runs.length * tRow + 30;
    const tsvg = d3.select("#eu-test").append("svg").attr("viewBox", `0 0 ${tW} ${tH}`)
      .attr("role", "img").attr("aria-label", "Macro-F1 and accuracy of every run on the EuroSAT test set");
    tsvg.append("g").attr("class", "axis").attr("transform", `translate(0,${tTop + DATA.runs.length * tRow + 2})`)
      .call(d3.axisBottom(tx).ticks(6).tickFormat(d3.format(".3f")).tickSizeOuter(0));
    const trow = tsvg.selectAll("g.trow").data(DATA.runs).join("g").attr("class", "trow clickable")
      .attr("transform", (d, i) => `translate(0,${tTop + i * tRow})`).on("click", (e, d) => setState({ run: d.run_id, cell: null }));
    trow.append("rect").attr("class", "hl").attr("width", tW).attr("height", tRow - 2).attr("rx", 8);
    trow.append("text").attr("x", 12).attr("y", tRow / 2 + 4).attr("font-size", 13).text((d) => (d.run_id === selected ? "★ " : "") + d.label);
    trow.append("line").attr("x1", tL).attr("x2", tW - tR).attr("y1", tRow / 2).attr("y2", tRow / 2).attr("stroke", "#EEF1F6");
    trow.append("circle").attr("cx", (d) => tx(d.evaluations.eurosat.accuracy)).attr("cy", tRow / 2).attr("r", 6)
      .attr("fill", "#fff").attr("stroke", (d) => paperColor(d.model)).attr("stroke-width", 2);
    trow.append("circle").attr("cx", (d) => tx(d.evaluations.eurosat.macro_f1)).attr("cy", tRow / 2).attr("r", 5.5)
      .attr("fill", (d) => paperColor(d.model));
    trow.append("text").attr("x", tW - tR + 12).attr("y", tRow / 2 + 4).attr("font-size", 12.5).attr("font-family", "IBM Plex Mono")
      .text((d) => fmt3(d.evaluations.eurosat.macro_f1));
    trow.on("mousemove", (e, d) => showTip(e, `${d.label}<br>macro-F1 <b>${fmt3(d.evaluations.eurosat.macro_f1)}</b> · accuracy <b>${fmt3(d.evaluations.eurosat.accuracy)}</b>` +
      (d.summary.epochs_to_95 ? `<br>95 % of best validation macro-F1 at epoch ${d.summary.epochs_to_95}` : ""))).on("mouseleave", hideTip);

    let lastKey = null;
    renderers.eurosat = function () {
      updateMetric(state.curveMetric);
      trow.select("rect.hl").attr("fill", (d) => (d.run_id === state.run ? "#EDF2F9" : "transparent"));
      trow.select("text").attr("font-weight", (d) => (d.run_id === state.run ? 600 : 400));
      if (curves && state.curveMetric !== lastKey) { lastKey = state.curveMetric; drawCurves(); }
    };
  })();

  // ================================================================== demo 4 · CNN vs. ViT and cost
  // Same colors and markers as the presentation's "CNN vs. ViT, and what each one costs" slide.
  (function demoCost() {
    const root = document.getElementById("demo-cost");
    const XS = [
      { value: "patches_per_s", label: "Throughput", title: "Inference throughput (patches/s, EuroSAT test)", get: (s) => s.patches_per_s, fmt: (v) => v.toFixed(0) },
      { value: "params", label: "Parameters", title: "Parameters (millions)", get: (s) => s.params / 1e6, fmt: (v) => v.toFixed(1) },
      { value: "time_per_epoch_s", label: "Time per epoch", title: "Training time per EuroSAT epoch (s)", get: (s) => s.time_per_epoch_s, fmt: (v) => v.toFixed(0) },
      { value: "peak_mem_mb", label: "Peak memory", title: "Peak training memory (GB)", get: (s) => s.peak_mem_mb / 1024, fmt: (v) => v.toFixed(2) },
    ];
    const YS = [
      { value: "ags_last_block", label: "Ags · last_block", title: "Aguascalientes macro-F1 after last_block" },
      { value: "ags_head", label: "Ags · head", title: "Aguascalientes macro-F1 after head" },
      { value: "ags_zero_shot", label: "Ags · zero-shot", title: "Aguascalientes macro-F1, zero-shot" },
      { value: "eurosat_macro_f1", label: "EuroSAT", title: "EuroSAT test macro-F1" },
    ];
    // marker per (bands, init), as in the slide: circle RGB·ImageNet, square 13 bands·ImageNet, star 13 bands·SSL4EO
    const markerOf = (r) => (r.bands === "rgb" ? d3.symbolCircle : r.init === "imagenet" ? d3.symbolSquare : r.init === "ssl4eo" ? d3.symbolStar : d3.symbolDiamond);
    const sizeOf = (r) => (r.init === "ssl4eo" ? 420 : 230);
    const share = DATA.last_block_share || {};
    const devices = (DATA.inputs && DATA.inputs.devices) || [];
    const combos = [...new Set(DATA.runs.map((r) => r.bands + "|" + r.init))];
    const comboLabel = (k) => { const [b, i] = k.split("|"); return `${nameOf("bands", b)} · ${(DATA.names.init || {})[i] || i}`; };
    const icon = (sym, size, fill, stroke) => `<svg width="18" height="18" viewBox="-9 -9 18 18"><path d="${d3.symbol(sym, size)()}" fill="${fill}" stroke="${stroke}" stroke-width="1"/></svg>`;
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("cost")} · Architecture and cost</div>
        <h2>CNN vs. ViT, and what each one costs</h2>
        <p class="lead">Each point is one run. Choose the score on the vertical axis and a cost on the horizontal one.</p>
      </div>
      <div class="grid">
        <div class="tarjeta panel" id="cost-panel"></div>
        <div class="stack">
          <div class="tarjeta"><h3 id="cost-title"></h3><div id="cost-scatter"></div>
            <div class="legend">
              ${MODELS.map((mm) => `<span>${icon(d3.symbolCircle, 110, slideColor(mm), slideColor(mm))}${nameOf("model", mm)}</span>`).join("")}
              ${combos.map((k) => { const [b, i] = k.split("|"); const r = { bands: b, init: i }; return `<span>${icon(markerOf(r), r.init === "ssl4eo" ? 160 : 100, "#fff", SLIDE.ink)}${comboLabel(k)}</span>`; }).join("")}
            </div>
          </div>
          <div class="tarjeta" id="cost-says"></div>
          <div class="tarjeta"><h3>Cost of every run</h3><div style="overflow-x:auto"><table class="cost" id="cost-table"></table></div></div>
        </div>
      </div>`;
    const panel = document.getElementById("cost-panel");
    const yBox = el("div", { class: "control" });
    yBox.appendChild(el("span", { class: "rotulo" }, "Vertical axis: score"));
    const updateY = segmented(yBox, YS.map((o) => ({ value: o.value, label: o.label })), (v) => setState({ costY: v }));
    panel.appendChild(yBox);
    const xBox = el("div", { class: "control" });
    xBox.appendChild(el("span", { class: "rotulo" }, "Horizontal axis: cost"));
    const updateX = segmented(xBox, XS.map((o) => ({ value: o.value, label: o.label })), (v) => setState({ costX: v }));
    panel.appendChild(xBox);
    if (Object.keys(share).length) {
      panel.appendChild(el("div", { class: "callout warm" },
        "<b>Not a firm ranking:</b> one seed, and last_block trains " +
        Object.entries(share).map(([mm, v]) => `${Math.round(v * 100)} % of ${nameOf("model", mm)}`).join(", ") + "."));
    }
    panel.appendChild(el("p", { class: "hint" }, `Costs were measured on one machine${devices.length ? ` (device: ${devices.join(", ")})` : ""}.`));

    const W = 860, H = 460, m = { l: 66, r: 24, t: 14, b: 54 };
    const svg = d3.select("#cost-scatter").append("svg").attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Score against cost for every run");
    const x = d3.scaleLinear().range([m.l, W - m.r]), y = d3.scaleLinear().range([H - m.b, m.t]);
    const grid = svg.append("g");
    const gx = svg.append("g").attr("class", "axis slide").attr("transform", `translate(0,${H - m.b})`);
    const gy = svg.append("g").attr("class", "axis slide").attr("transform", `translate(${m.l},0)`);
    const xTitle = svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 10).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", SLIDE.soft);
    const yTitle = svg.append("text").attr("transform", `translate(16,${(H - m.b + m.t) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", SLIDE.soft);
    const pts = svg.append("g").selectAll("path").data(DATA.runs).join("path").attr("class", "clickable")
      .attr("d", (r) => d3.symbol(markerOf(r), sizeOf(r))())
      .attr("fill", (r) => slideColor(r.model)).attr("stroke", SLIDE.ink).attr("stroke-width", 1)
      .on("click", (e, r) => setState({ run: r.run_id, cell: null }));
    const ring = svg.append("circle").attr("r", 17).attr("fill", "none").attr("stroke", SLIDE.ink).attr("stroke-dasharray", "3 3").attr("pointer-events", "none");
    const label = svg.append("text").attr("font-size", 12.5).attr("font-weight", 600).attr("fill", SLIDE.ink);

    const table = d3.select("#cost-table");
    const cols = [["Run", (r) => r.label], ["Params (M)", (r) => XS[1].fmt(XS[1].get(r.summary))], ["Time / epoch (s)", (r) => XS[2].fmt(XS[2].get(r.summary))],
      ["Epochs", (r) => r.summary.epochs], ["Peak mem. (GB)", (r) => XS[3].fmt(XS[3].get(r.summary))], ["Patches/s", (r) => XS[0].fmt(XS[0].get(r.summary))]];
    table.append("thead").append("tr").selectAll("th").data(cols).join("th").text((c) => c[0]);
    const trs = table.append("tbody").selectAll("tr").data(DATA.runs).join("tr").attr("class", "clickable").on("click", (e, r) => setState({ run: r.run_id, cell: null }));
    trs.selectAll("td").data((r) => cols.map((c) => c[1](r))).join("td").text((v) => v);

    function best(runs, key) {
      return runs.reduce((a, b) => (b.summary[key] > a.summary[key] ? b : a), runs[0]);
    }

    renderers.cost = function () {
      updateX(state.costX);
      updateY(state.costY);
      const xo = XS.find((o) => o.value === state.costX), yo = YS.find((o) => o.value === state.costY);
      const xv = DATA.runs.map((r) => xo.get(r.summary)), yv = DATA.runs.map((r) => r.summary[yo.value]);
      const [x0, x1] = d3.extent(xv), [y0, y1] = d3.extent(yv);
      x.domain([x0 - (x1 - x0) * 0.1 - 1e-9, x1 + (x1 - x0) * 0.1 + 1e-9]).nice();
      y.domain([Math.max(0, y0 - (y1 - y0) * 0.25 - 0.005), Math.min(1, y1 + (y1 - y0) * 0.25 + 0.005)]).nice();
      const t = svg.transition().duration(650);
      gx.transition(t).call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));
      gy.transition(t).call(d3.axisLeft(y).ticks(8).tickFormat(d3.format(".2f")).tickSizeOuter(0));
      const gl = [...y.ticks(8).map((v) => ({ k: "y" + v, x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) })),
        ...x.ticks(7).map((v) => ({ k: "x" + v, x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }))];
      grid.selectAll("line").data(gl, (d) => d.k).join("line").attr("stroke", SLIDE.grid)
        .attr("x1", (d) => d.x1).attr("x2", (d) => d.x2).attr("y1", (d) => d.y1).attr("y2", (d) => d.y2);
      xTitle.text(xo.title); yTitle.text(yo.title);
      document.getElementById("cost-title").textContent = `${yo.title} vs. ${xo.title.charAt(0).toLowerCase() + xo.title.slice(1)}`;
      pts.transition(t).attr("transform", (r) => `translate(${x(xo.get(r.summary))},${y(r.summary[yo.value])})`);
      pts.on("mousemove", (e, r) => showTip(e, `${r.label}<br>${yo.title}: <b>${fmt3(r.summary[yo.value])}</b><br>${xo.title}: <b>${xo.fmt(xo.get(r.summary))}</b>`)).on("mouseleave", hideTip);
      const sel = runsById.get(state.run);
      const sx = x(xo.get(sel.summary)), sy = y(sel.summary[yo.value]);
      ring.transition(t).attr("cx", sx).attr("cy", sy);
      label.text(`${sel.label} · ${fmt3(sel.summary[yo.value])}`).attr("text-anchor", sx > W - 320 ? "end" : "start")
        .transition(t).attr("x", sx > W - 320 ? sx - 22 : sx + 22).attr("y", sy + 4);
      trs.classed("sel", (r) => r.run_id === state.run);
      // what the chart says, recomputed from data.js for the chosen score
      const lines = [];
      const top = best(DATA.runs, yo.value);
      lines.push(`Highest of all runs: <b>${top.label}</b>, ${fmt3(top.summary[yo.value])}.`);
      combos.forEach((k) => {
        const group = DATA.runs.filter((r) => r.bands + "|" + r.init === k);
        if (group.length > 1) {
          const g = best(group, yo.value);
          lines.push(`Among the ${comboLabel(k)} runs: <b>${nameOf("model", g.model)}</b>, ${fmt3(g.summary[yo.value])}.`);
        }
      });
      document.getElementById("cost-says").innerHTML = `<h3>${yo.title}: highest observed</h3>` +
        lines.map((l) => `<p style="margin:0 0 6px">${l}</p>`).join("") +
        '<p class="hint">One seed per run: differences of a few hundredths are not a firm ranking.</p>';
    };
  })();

  // ================================================================== demo 2 · drop and recovery
  (function demoRuns() {
    const root = document.getElementById("demo-runs");
    const n = DATA.macro_f1_classes.length;
    const S = DATA.runs.map((r) => r.summary);
    const runLabel = (r) => `${nameOf("model", r.model)} · ${nameOf("bands", r.bands)} · ${(DATA.names.init || {})[r.init] || r.init}`;
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("runs")} · From EuroSAT to Aguascalientes</div>
        <h2>In Aguascalientes, all ${DATA.runs.length} runs fell</h2>
        <p class="lead">Drop: <b style="color:${SLIDE.orangeText}">${fmt3(d3.min(S, (s) => s.drop))} – ${fmt3(d3.max(S, (s) => s.drop))}</b> of macro-F1 ·
        last_block recovered at most <b style="color:${SLIDE.teal}">${fmt3(d3.max(S, (s) => s.recovery_last_block))}</b> of the drop; head at most ${fmt3(d3.max(S, (s) => s.recovery_head))}.</p>
      </div>
      <div class="tarjeta" style="margin-bottom:20px">
        <div id="runs-all"></div>
        <p class="hint" style="margin-top:4px">Macro-F1 over the same ${n} classes in both datasets. Click a row to open that run below.</p>
      </div>
      <div class="grid">
        <div class="tarjeta panel" id="runs-panel"></div>
        <div class="stack">
          <div class="tarjeta">
            <h3 id="runs-title"></h3>
            <div id="runs-track"></div>
            <div class="legend">
              <span><i class="sq" style="background:${SLIDE.dropBand};border:1px solid #EFD3AE"></i>Drop: what was lost from EuroSAT to Aguascalientes</span>
              <span><i class="sq" style="background:${SLIDE.teal}"></i>Recovered by fine-tuning</span>
            </div>
          </div>
          <div class="tarjeta">
            <div class="tiles" id="runs-tiles"></div>
            <div class="formulas" id="runs-formulas" style="margin-top:14px" hidden></div>
          </div>
        </div>
      </div>`;

    const panel = document.getElementById("runs-panel");
    panel.appendChild(el("h3", {}, "Choose a run"));
    const updatePicker = runPicker(panel);
    const sw = el("label", { class: "switch" }, '<input type="checkbox" id="runs-formula-toggle"> Show formulas');
    panel.appendChild(sw);
    sw.querySelector("input").addEventListener("change", (e) => setState({ formulas: e.target.checked }));
    const riverRows = excludedClasses.map((c) => {
      const i = DATA.classes.indexOf(c);
      const conf = DATA.runs[0].evaluations.zero_shot.confusion;
      return `${pretty(c)}, ${d3.sum(conf[i])} test patches`;
    });
    panel.appendChild(el("p", { class: "hint" },
      `Macro-F1 over the ${n} classes of the Aguascalientes test${riverRows.length ? ` (left out: ${riverRows.join("; ")})` : ""}. ` +
      "The EuroSAT value uses the same classes, so the drop compares like with like."));

    // --- all runs: the presentation's dumbbell chart, click a row to select it
    const aW = 1240, aLeft = 330, aRight = 40, aRow = 46, aTop = 46;
    const ax = d3.scaleLinear().domain([0, 1.02]).range([aLeft, aW - aRight]);
    const aH = aTop + DATA.runs.length * aRow + 44;
    const asvg = d3.select("#runs-all").append("svg").attr("viewBox", `0 0 ${aW} ${aH}`)
      .attr("role", "img").attr("aria-label", "EuroSAT, zero-shot and last_block macro-F1 for every run");
    const legendItems = [["circle", SLIDE.gray, "EuroSAT test"], ["circle", SLIDE.orange, "Ags zero-shot"], ["diamond", SLIDE.teal, "Ags after last_block"]];
    const lg = asvg.append("g").attr("transform", `translate(${(aLeft + aW - aRight) / 2 - 260},16)`);
    legendItems.forEach(([shape, c, t], i) => {
      const gx = i * 190;
      lg.append("path").attr("transform", `translate(${gx},0)`).attr("d", d3.symbol(shape === "diamond" ? d3.symbolDiamond2 : d3.symbolCircle, 170)()).attr("fill", c);
      lg.append("text").attr("x", gx + 16).attr("y", 5).attr("font-size", 15).attr("fill", SLIDE.ink).text(t);
    });
    asvg.append("g").selectAll("line").data(d3.range(0, 1.01, 0.2)).join("line")
      .attr("x1", (d) => ax(d)).attr("x2", (d) => ax(d)).attr("y1", aTop - 6).attr("y2", aTop + DATA.runs.length * aRow).attr("stroke", SLIDE.grid);
    asvg.append("g").attr("class", "axis slide big").attr("transform", `translate(0,${aTop + DATA.runs.length * aRow})`)
      .call(d3.axisBottom(ax).tickValues(d3.range(0, 1.01, 0.2)).tickFormat(d3.format(".1f")).tickSizeOuter(0));
    asvg.append("text").attr("x", (aLeft + aW - aRight) / 2).attr("y", aH - 4).attr("text-anchor", "middle").attr("font-size", 15).attr("fill", SLIDE.soft).text(`Macro-F1 (${n} classes)`);
    const arow = asvg.selectAll("g.arow").data(DATA.runs).join("g").attr("class", "arow clickable")
      .attr("transform", (d, i) => `translate(0,${aTop + i * aRow})`)
      .on("click", (e, d) => setState({ run: d.run_id, cell: null }));
    arow.append("rect").attr("class", "hl").attr("x", 0).attr("width", aW).attr("height", aRow - 4).attr("y", 2).attr("rx", 10);
    arow.append("text").attr("x", aLeft - 14).attr("y", aRow / 2 + 5).attr("text-anchor", "end").attr("font-size", 16).attr("fill", SLIDE.ink)
      .text((d) => (d.run_id === selected ? "★ " : "") + runLabel(d));
    arow.append("line").attr("x1", (d) => ax(d.summary.ags_zero_shot)).attr("x2", (d) => ax(d.summary.eurosat_f1_macro_classes))
      .attr("y1", aRow / 2).attr("y2", aRow / 2).attr("stroke", SLIDE.grid).attr("stroke-width", 3.5);
    arow.append("line").attr("x1", (d) => ax(d.summary.ags_zero_shot)).attr("x2", (d) => ax(d.summary.ags_last_block))
      .attr("y1", aRow / 2).attr("y2", aRow / 2).attr("stroke", SLIDE.teal).attr("stroke-width", 3.5);
    const dot = (key, shape, c) => arow.append("path").attr("d", d3.symbol(shape, 200)())
      .attr("transform", (d) => `translate(${ax(d.summary[key])},${aRow / 2})`).attr("fill", c)
      .on("mousemove", (e, d) => showTip(e, `${d.label}<br>${key === "eurosat_f1_macro_classes" ? "EuroSAT test" : key === "ags_zero_shot" ? "Ags zero-shot" : "Ags after last_block"}: <b>${fmt3(d.summary[key])}</b>`))
      .on("mouseleave", hideTip);
    dot("eurosat_f1_macro_classes", d3.symbolCircle, SLIDE.gray);
    dot("ags_zero_shot", d3.symbolCircle, SLIDE.orange);
    dot("ags_last_block", d3.symbolDiamond2, SLIDE.teal);

    // --- track chart for the selected run: four bars on a shared 0–1 macro-F1 axis
    const W = 860, rowH = 50, top = 46, left = 130, right = 190;
    const rows = [
      { key: "eurosat", label: "EuroSAT test", color: SLIDE.gray },
      { key: "zero", label: "Zero-shot", color: SLIDE.orange },
      { key: "head", label: "head", color: SLIDE.tealLight },
      { key: "last", label: "last_block", color: SLIDE.teal },
    ];
    const H = top + rows.length * rowH + 34;
    const x = d3.scaleLinear().domain([0, 1]).range([left, W - right]);
    const svg = d3.select("#runs-track").append("svg").attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Macro-F1 on EuroSAT, zero-shot, after head and after last_block");
    svg.append("g").selectAll("line").data(x.ticks(5)).join("line").attr("stroke", SLIDE.grid)
      .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", top - 6).attr("y2", top + rows.length * rowH);
    svg.append("g").attr("class", "axis slide").attr("transform", `translate(0,${top + rows.length * rowH + 4})`)
      .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
    const band = svg.append("rect").attr("y", top - 6).attr("height", rows.length * rowH + 6).attr("fill", SLIDE.dropBand).attr("fill-opacity", 0.75);
    const bandLabel = svg.append("text").attr("y", top - 16).attr("text-anchor", "middle")
      .attr("font-weight", 600).attr("fill", SLIDE.orangeText).attr("font-size", 14);
    const g = svg.selectAll("g.row").data(rows).join("g").attr("class", "row")
      .attr("transform", (d, i) => `translate(0,${top + i * rowH})`);
    g.append("text").attr("x", left - 12).attr("y", rowH / 2 + 5).attr("text-anchor", "end").attr("font-size", 14).attr("fill", SLIDE.ink).text((d) => d.label);
    g.append("rect").attr("class", "base").attr("x", x(0)).attr("y", 12).attr("height", rowH - 24).attr("rx", 4);
    g.append("rect").attr("class", "gain").attr("y", 12).attr("height", rowH - 24).attr("rx", 4);
    g.append("text").attr("class", "val mono").attr("y", rowH / 2 + 5).attr("font-size", 13.5).attr("font-family", "IBM Plex Mono").attr("fill", SLIDE.ink);
    g.append("text").attr("class", "note").attr("x", W - right + 66).attr("y", rowH / 2 + 5).attr("font-size", 12.5).attr("fill", SLIDE.soft);

    function updateTrack(run) {
      const s = run.summary;
      const values = { eurosat: s.eurosat_f1_macro_classes, zero: s.ags_zero_shot, head: s.ags_head, last: s.ags_last_block };
      const t = svg.transition().duration(650).ease(d3.easeCubicOut);
      band.transition(t).attr("x", x(s.ags_zero_shot)).attr("width", Math.max(0, x(s.eurosat_f1_macro_classes) - x(s.ags_zero_shot)));
      bandLabel.transition(t).attr("x", (x(s.ags_zero_shot) + x(s.eurosat_f1_macro_classes)) / 2);
      bandLabel.text(`Drop ${fmt3(s.drop)}`);
      g.select("rect.base").transition(t)
        .attr("width", (d) => {
          const v = values[d.key];
          if (d.key === "eurosat" || d.key === "zero") return x(v) - x(0);
          return x(Math.min(v, s.ags_zero_shot)) - x(0);
        })
        .attr("fill", (d) => (d.key === "eurosat" || d.key === "zero" ? d.color : "#E4E0D6"));
      g.select("rect.gain").transition(t)
        .attr("x", (d) => x(Math.min(values[d.key], s.ags_zero_shot)))
        .attr("width", (d) => (d.key === "head" || d.key === "last" ? Math.abs(x(values[d.key]) - x(s.ags_zero_shot)) : 0))
        .attr("fill", (d) => (values[d.key] >= s.ags_zero_shot ? d.color : SLIDE.orange));
      g.select("text.val").transition(t).attr("x", (d) => x(values[d.key]) + 8);
      g.select("text.val").text((d) => fmt3(values[d.key]));
      g.select("text.note").text((d) => {
        if (d.key === "head") return `recovery ${fmt3(s.recovery_head)}`;
        if (d.key === "last") return `recovery ${fmt3(s.recovery_last_block)}`;
        return "";
      });
    }

    renderers.runs = function () {
      const run = runsById.get(state.run);
      updatePicker();
      document.getElementById("runs-title").textContent = run.label;
      updateTrack(run);
      const s = run.summary;
      const tiles = [
        ["EuroSAT test F1", s.eurosat_f1_macro_classes, ""],
        ["Zero-shot F1", s.ags_zero_shot, ""],
        ["After head", s.ags_head, ""],
        ["After last_block", s.ags_last_block, ""],
        ["Drop", s.drop, "drop"],
        ["Recovery · head", s.recovery_head, "rec"],
        ["Recovery · last_block", s.recovery_last_block, "rec"],
      ];
      document.getElementById("runs-tiles").innerHTML = tiles
        .map(([k, v, cls]) => `<div class="tile ${cls}"><div class="v">${fmt3(v)}</div><div class="k">${k}</div></div>`).join("");
      const f = document.getElementById("runs-formulas");
      f.hidden = !state.formulas;
      f.innerHTML =
        `<span class="name">Drop</span> = F1<sub>EuroSAT</sub> − F1<sub>zero-shot</sub> = ${fmt3(s.eurosat_f1_macro_classes)} − ${fmt3(s.ags_zero_shot)} = <b>${fmt3(s.drop)}</b><br>` +
        `<span class="name">Recovery</span> = (F1<sub>fine-tuned</sub> − F1<sub>zero-shot</sub>) / Drop<br>` +
        `&nbsp;&nbsp;head: (${fmt3(s.ags_head)} − ${fmt3(s.ags_zero_shot)}) / ${fmt3(s.drop)} = <b>${fmt3(s.recovery_head)}</b><br>` +
        `&nbsp;&nbsp;last_block: (${fmt3(s.ags_last_block)} − ${fmt3(s.ags_zero_shot)}) / ${fmt3(s.drop)} = <b>${fmt3(s.recovery_last_block)}</b><br>` +
        `<span class="name">1 would mean the fine-tuned model recovered everything it lost.</span>`;
      arow.select("rect.hl").attr("fill", (d) => (d.run_id === state.run ? "#EDF2F9" : "transparent"));
      arow.select("text").attr("font-weight", (d) => (d.run_id === state.run ? 700 : 400));
    };
  })();

  // ================================================================== demo 2 · confusion matrix
  (function demoConfusion() {
    const root = document.getElementById("demo-confusion");
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("confusion")} · Where the errors go</div>
        <h2>Confusion matrix of any run, before and after fine-tuning</h2>
        <p class="lead">Rows are the true class and columns the predicted class. Switch between stages to see how the
        errors move: zero-shot, most Aguascalientes patches fall into one or two classes.</p>
      </div>
      <div class="grid">
        <div class="tarjeta panel" id="conf-panel"></div>
        <div class="stack">
          <div class="tarjeta"><h3 id="conf-title"></h3><div id="conf-matrix"></div></div>
          <div class="tarjeta" id="conf-detail"></div>
        </div>
      </div>`;
    const panel = document.getElementById("conf-panel");
    panel.appendChild(el("h3", {}, "Choose a run"));
    const updatePicker = runPicker(panel);
    const stageBox = el("div", { class: "control" });
    stageBox.appendChild(el("span", { class: "rotulo" }, "Test set and stage"));
    const updateStage = segmented(stageBox, STAGES.map((s) => ({ value: s.key, label: s.label })), (v) => setState({ stage: v, cell: null }));
    panel.appendChild(stageBox);
    const viewBox = el("div", { class: "control" });
    viewBox.appendChild(el("span", { class: "rotulo" }, "Show in each cell"));
    const updateView = segmented(viewBox, [{ value: "share", label: "Share of the row" }, { value: "count", label: "Patch count" }], (v) => setState({ view: v }));
    panel.appendChild(viewBox);
    const info = el("div", { class: "tiles" });
    panel.appendChild(info);

    const K = DATA.classes.length, cell = 48, left = 200, top = 150, W = left + K * cell + 70, H = top + K * cell + 16;
    const color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]); // same as cmap="Blues" in the slide figure
    const svg = d3.select("#conf-matrix").append("svg").attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Confusion matrix: rows are true classes, columns predicted classes");
    svg.append("text").attr("x", left + (K * cell) / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", "#5B6B85").text("Predicted class →");
    svg.append("text").attr("transform", `translate(14,${top + (K * cell) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", "#5B6B85").text("True class →");
    svg.append("text").attr("x", left + K * cell + 12).attr("y", top - 10).attr("font-size", 12).attr("fill", "#5B6B85").text("patches");
    const colLabels = svg.selectAll("text.col").data(DATA.classes).join("text").attr("class", "col").attr("font-size", 12.5)
      .attr("transform", (d, j) => `translate(${left + j * cell + cell / 2 + 4},${top - 8}) rotate(-45)`).text((d) => pretty(d));
    const rowLabels = svg.selectAll("text.rowl").data(DATA.classes).join("text").attr("class", "rowl").attr("font-size", 12.5)
      .attr("x", left - 10).attr("y", (d, i) => top + i * cell + cell / 2 + 4).attr("text-anchor", "end");
    const totals = svg.selectAll("text.tot").data(DATA.classes).join("text").attr("class", "tot mono").attr("font-size", 12)
      .attr("font-family", "IBM Plex Mono").attr("fill", "#5B6B85").attr("x", left + K * cell + 12).attr("y", (d, i) => top + i * cell + cell / 2 + 4);
    const cellsData = [];
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) cellsData.push({ i, j });
    const cg = svg.selectAll("g.cell").data(cellsData).join("g").attr("class", "cell clickable")
      .attr("transform", (d) => `translate(${left + d.j * cell},${top + d.i * cell})`)
      .on("click", (e, d) => setState({ cell: [d.i, d.j] }));
    cg.append("rect").attr("width", cell - 2).attr("height", cell - 2).attr("rx", 5);
    cg.append("text").attr("x", (cell - 2) / 2).attr("y", cell / 2 + 3).attr("text-anchor", "middle").attr("font-size", 12)
      .attr("font-family", "IBM Plex Mono");
    const selRect = svg.append("rect").attr("width", cell - 2).attr("height", cell - 2).attr("rx", 5)
      .attr("fill", "none").attr("stroke", "#BB4B3E").attr("stroke-width", 3).attr("pointer-events", "none");

    function mainError(conf) {
      let best = null;
      conf.forEach((row, i) => row.forEach((v, j) => {
        if (i !== j && (!best || v > best.v)) best = { i, j, v };
      }));
      return best;
    }

    renderers.confusion = function () {
      const run = runsById.get(state.run);
      updatePicker();
      updateStage(state.stage);
      updateView(state.view);
      const ev = run.evaluations[state.stage];
      const conf = ev.confusion;
      const rowSums = conf.map((r) => d3.sum(r));
      const total = d3.sum(rowSums);
      const isEuroSAT = state.stage === "eurosat";
      const stageLabel = STAGES.find((s) => s.key === state.stage).label;
      document.getElementById("conf-title").textContent = `${run.label} · ${stageLabel}`;
      const nMacro = isEuroSAT ? DATA.classes.length : DATA.macro_f1_classes.length;
      info.innerHTML =
        `<div class="tile"><div class="v">${fmt3(ev.accuracy)}</div><div class="k">Accuracy</div></div>` +
        `<div class="tile"><div class="v">${fmt3(ev.macro_f1)}</div><div class="k">Macro-F1 (${nMacro} classes)</div></div>` +
        `<div class="tile" style="grid-column:1/-1"><div class="v">${total.toLocaleString("en-US")}</div><div class="k">test patches (${isEuroSAT ? "EuroSAT" : "Aguascalientes"})</div></div>`;
      const t = svg.transition().duration(550);
      cg.select("rect").transition(t).attr("fill", (d) => (rowSums[d.i] ? color(conf[d.i][d.j] / rowSums[d.i]) : "#F4F7FB"))
        .attr("stroke", (d) => (d.i === d.j ? "#0D7F91" : "none")).attr("stroke-width", 2);
      cg.select("text")
        .text((d) => {
          const v = conf[d.i][d.j];
          if (state.view === "count") return v ? v : "";
          return v ? Math.round((100 * v) / rowSums[d.i]) : "";
        })
        .attr("fill", (d) => (rowSums[d.i] && conf[d.i][d.j] / rowSums[d.i] > 0.55 ? "#FFFFFF" : "#13232B"));
      cg.on("mousemove", (e, d) => showTip(e, `True <b>${pretty(DATA.classes[d.i])}</b> → predicted <b>${pretty(DATA.classes[d.j])}</b><br>` +
        `${conf[d.i][d.j]} of ${rowSums[d.i]} patches${rowSums[d.i] ? ` (${pct(conf[d.i][d.j] / rowSums[d.i])})` : ""}`)).on("mouseleave", hideTip);
      rowLabels.text((d) => pretty(d) + (!isEuroSAT && excludedClasses.includes(d) ? " *" : ""))
        .attr("font-style", (d) => (!isEuroSAT && excludedClasses.includes(d) ? "italic" : "normal"));
      totals.text((d, i) => rowSums[i].toLocaleString("en-US"));
      const err = mainError(conf);
      const target = state.cell || (err ? [err.i, err.j] : [0, 0]);
      selRect.transition(t).attr("x", left + target[1] * cell).attr("y", top + target[0] * cell);
      const [ti, tj] = target;
      const trueName = pretty(DATA.classes[ti]), predName = pretty(DATA.classes[tj]);
      const share = rowSums[ti] ? conf[ti][tj] / rowSums[ti] : 0;
      const f1 = ev.f1_per_class[DATA.classes[ti]];
      const detail = document.getElementById("conf-detail");
      detail.innerHTML =
        `<div class="rotulo">${state.cell ? "Selected cell" : "Largest error in this matrix"}</div>` +
        `<h3 style="margin-top:6px">${trueName} → ${predName}</h3>` +
        `<p style="margin:0 0 8px">${ti === tj
          ? `<b>${conf[ti][tj]}</b> of the ${rowSums[ti]} ${trueName} patches were classified correctly (${pct(share)}).`
          : `<b>${conf[ti][tj]}</b> of the ${rowSums[ti]} patches labeled ${trueName} were predicted as ${predName} (${pct(share)}).`}</p>` +
        `<p class="hint">F1 of ${trueName} in this evaluation: <b class="mono">${fmt3(f1)}</b>.` +
        (!isEuroSAT && excludedClasses.length ? ` * ${excludedClasses.map(pretty).join(", ")}: left out of the Aguascalientes macro-F1 (too few test patches).` : "") +
        " Click any cell to inspect it.</p>";
    };
  })();

  // ================================================================== demo 3 · spatial split
  (function demoSplit() {
    const root = document.getElementById("demo-split");
    const sp = DATA.split;
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("split")} · Why we split by blocks</div>
        <h2>Neighboring cells look alike, so they should not be split apart</h2>
        <p class="lead">Each square is one labeled 640 m cell of Aguascalientes. If a test cell has a fine-tune cell next to it,
        the model has practically seen it, and the test score looks better than it is.</p>
      </div>`;
    if (!sp) {
      root.insertAdjacentHTML("beforeend",
        '<div class="tarjeta callout warm"><b>Split data not available.</b> <code>data/ags/labels.csv</code> is not in git: ' +
        "it comes with release v1.0 (<code>python scripts/download_release.py</code>). Then run " +
        "<code>python scripts/make_results.py</code> and reload this page.</div>");
      renderers.split = () => {};
      return;
    }
    root.insertAdjacentHTML("beforeend", `
      <div class="grid">
        <div class="tarjeta panel" id="split-panel"></div>
        <div class="tarjeta"><div id="split-map"></div>
          <div class="legend">
            <span><i class="sq" style="background:${COLOR.finetune}"></i>Fine-tune cell</span>
            <span><i class="sq" style="background:${COLOR.test}"></i>Test cell</span>
            <span><i class="sq" style="background:${COLOR.touching}"></i>Test cell touching a fine-tune cell</span>
            <span><i class="sq" style="background:#fff;border:1px solid #B9C4D4"></i>Block of ${sp.block_cells} × ${sp.block_cells} cells</span>
          </div>
          ${window.HYDROSCOPE_BASEMAP ? '<p class="hint" style="margin-top:6px">Background: our April 2024 Sentinel-2 composite, true color, the same previews used for labeling. Contains modified Copernicus Sentinel-2 data (2024).</p>' : ""}
        </div>
      </div>`);
    const panel = document.getElementById("split-panel");
    const modeBox = el("div", { class: "control" });
    modeBox.appendChild(el("span", { class: "rotulo" }, "How the cells are split"));
    const updateMode = segmented(modeBox, [{ value: "blocks", label: `Our split: ${sp.block_cells} × ${sp.block_cells} blocks` }, { value: "random", label: "Random cells" }], (v) => setState({ splitMode: v }));
    panel.appendChild(modeBox);
    if (window.HYDROSCOPE_BASEMAP) {
      const sw = el("label", { class: "switch" }, '<input type="checkbox" checked> Show the April 2024 composite');
      sw.querySelector("input").addEventListener("change", (e) => setState({ basemap: e.target.checked }));
      modeBox.appendChild(sw);
    }
    const reshuffle = el("button", { class: "btn", type: "button" }, "Reshuffle the random split");
    reshuffle.addEventListener("click", () => setState({ splitMode: "random", seed: state.seed + 1 }));
    panel.appendChild(reshuffle);
    const counter = el("div", { class: "control" });
    panel.appendChild(counter);
    panel.appendChild(el("p", { class: "hint" },
      `"Touching" means one of the ${sp.neighborhood} surrounding cells is in the fine-tune split. ` +
      `Both splits keep the same sizes: ${sp.n_finetune} fine-tune and ${sp.n_test} test cells.`));
    panel.appendChild(el("div", { class: "callout warm" },
      `With blocks, <b>${sp.test_touching_finetune}</b> test cells still touch a fine-tune cell across a block edge. ` +
      "This is a known limitation: the split has no buffer zone between blocks."));

    // geometry: the map covers the labeled cells and, if present, the background image
    const BM = window.HYDROSCOPE_BASEMAP && window.HYDROSCOPE_BASEMAP.schema_version === 1 ? window.HYDROSCOPE_BASEMAP : null;
    const rows = sp.cells.map((c) => c[0]), cols = sp.cells.map((c) => c[1]);
    let r0 = d3.min(rows), r1 = d3.max(rows) + 1, c0 = d3.min(cols), c1 = d3.max(cols) + 1; // r1, c1 exclusive
    if (BM) {
      r0 = Math.min(r0, BM.row0); c0 = Math.min(c0, BM.col0);
      r1 = Math.max(r1, BM.row0 + BM.n_rows); c1 = Math.max(c1, BM.col0 + BM.n_cols);
    }
    const size = 5.5, pad = 8;
    const W = (c1 - c0) * size + 2 * pad, H = (r1 - r0) * size + 2 * pad;
    const X = (col) => pad + (col - c0) * size, Y = (row) => pad + (row - r0) * size;
    const svg = d3.select("#split-map").append("svg").attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Map of the labeled cells of Aguascalientes colored by split");
    const image = BM ? svg.append("image").attr("href", BM.file).attr("x", X(BM.col0)).attr("y", Y(BM.row0))
      .attr("width", BM.n_cols * size).attr("height", BM.n_rows * size).attr("preserveAspectRatio", "none") : null;
    // One square per block that holds labeled cells, colored by the block's split.
    const blockSide = new Map();
    sp.cells.forEach((c) => blockSide.set(Math.floor(c[0] / sp.block_cells) + ":" + Math.floor(c[1] / sp.block_cells), c[2]));
    const blocks = [...blockSide.entries()].map(([k, side]) => ({ by: +k.split(":")[0], bx: +k.split(":")[1], side }));
    const blockRects = svg.append("g").selectAll("rect").data(blocks).join("rect")
      .attr("x", (d) => X(d.bx * sp.block_cells)).attr("y", (d) => Y(d.by * sp.block_cells))
      .attr("width", sp.block_cells * size).attr("height", sp.block_cells * size);
    const grid = svg.append("g").attr("stroke-width", 0.6);
    for (let b = Math.floor(c0 / sp.block_cells); b <= Math.ceil(c1 / sp.block_cells); b++) {
      grid.append("line").attr("x1", X(b * sp.block_cells)).attr("x2", X(b * sp.block_cells)).attr("y1", pad).attr("y2", H - pad);
    }
    for (let b = Math.floor(r0 / sp.block_cells); b <= Math.ceil(r1 / sp.block_cells); b++) {
      grid.append("line").attr("y1", Y(b * sp.block_cells)).attr("y2", Y(b * sp.block_cells)).attr("x1", pad).attr("x2", W - pad);
    }
    const key = (r, c) => r * 100000 + c;
    const blockId = (r, c) => `b${String(Math.floor(r / sp.block_cells)).padStart(3, "0")}_${String(Math.floor(c / sp.block_cells)).padStart(3, "0")}`;
    const cellId = (r, c) => `r${String(r).padStart(4, "0")}_c${String(c).padStart(4, "0")}`;
    const offsets = [];
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if ((dr || dc) && (sp.neighborhood === 8 || dr === 0 || dc === 0)) offsets.push([dr, dc]);
    }
    function touchingSet(sides) {
      const side = new Map(sp.cells.map((c, i) => [key(c[0], c[1]), sides[i]]));
      return sp.cells.map((c, i) => sides[i] === "test" && offsets.some(([dr, dc]) => side.get(key(c[0] + dr, c[1] + dc)) === "finetune"));
    }
    function mulberry32(a) {
      return function () {
        a |= 0; a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    function randomSides(seed) {
      const idx = d3.shuffler(mulberry32(seed))(d3.range(sp.cells.length));
      const sides = new Array(sp.cells.length).fill("finetune");
      idx.slice(0, sp.n_test).forEach((i) => (sides[i] = "test"));
      return sides;
    }
    const ourSides = sp.cells.map((c) => c[2]);
    const ourTouching = touchingSet(ourSides);
    if (d3.sum(ourTouching) !== sp.test_touching_finetune) {
      console.warn("touching count differs from data.js", d3.sum(ourTouching), sp.test_touching_finetune);
    }
    const rects = svg.append("g").selectAll("rect").data(sp.cells).join("rect")
      .attr("x", (d) => X(d[1]) + 0.4).attr("y", (d) => Y(d[0]) + 0.4)
      .attr("width", size - 0.8).attr("height", size - 0.8);

    let lastSplitKey = null;
    renderers.split = function () {
      const splitKey = state.splitMode + ":" + state.seed + ":" + state.basemap;
      if (splitKey === lastSplitKey) return;
      lastSplitKey = splitKey;
      updateMode(state.splitMode);
      reshuffle.disabled = false;
      const sides = state.splitMode === "blocks" ? ourSides : randomSides(state.seed);
      const touching = state.splitMode === "blocks" ? ourTouching : touchingSet(sides);
      const n = state.splitMode === "blocks" ? sp.test_touching_finetune : d3.sum(touching);
      counter.innerHTML = '<span class="rotulo">Test cells touching a fine-tune cell</span>' +
        `<div class="big" style="color:${state.splitMode === "blocks" ? "#0B6574" : "#803128"}">${n}</div>` +
        `<span class="hint">of ${sp.n_test} test cells (${pct(n / sp.n_test)})${state.splitMode === "random" ? ` · random split #${state.seed}` : ""}</span>`;
      const onImage = Boolean(image) && state.basemap;
      if (image) image.transition().duration(400).attr("opacity", onImage ? 1 : 0);
      blockRects.transition().duration(500)
        .attr("fill", (d) => {
          if (state.splitMode !== "blocks") return onImage ? "none" : "#F6F8FB";
          if (onImage) return d.side === "test" ? COLOR.test : COLOR.finetune;
          return d.side === "test" ? "#FAF1E1" : "#E6EEF8";
        })
        .attr("fill-opacity", onImage ? 0.3 : 1)
        .attr("stroke", onImage && state.splitMode === "blocks" ? (d) => (d.side === "test" ? COLOR.test : COLOR.finetune) : "none")
        .attr("stroke-width", 0.8);
      grid.attr("stroke", onImage ? "rgba(255,255,255,0.45)" : "#DCE3EC");
      rects.attr("stroke", onImage ? "#FFFFFF" : "none").attr("stroke-width", 0.7);
      rects.transition().duration(500).delay((d) => (d[0] - r0) * 3)
        .attr("fill", (d, i) => (touching[i] ? COLOR.touching : sides[i] === "test" ? COLOR.test : COLOR.finetune));
      rects.on("mousemove", (e, d) => {
        const i = sp.cells.indexOf(d);
        showTip(e, `${cellId(d[0], d[1])} · block ${blockId(d[0], d[1])}<br>${sides[i]}${touching[i] ? " · touches a fine-tune cell" : ""}`);
      }).on("mouseleave", hideTip);
    };
  })();

  // ================================================================== demo 4 · code tour
  (function demoCode() {
    const root = document.getElementById("demo-code");
    const STEP_INFO = {
      earth_engine: ["Earth Engine composite", "The April 2024 Sentinel-2 L1C images of the state are cloud-masked with QA60 and reduced to the per-pixel median. The result keeps the 13 bands in EuroSAT order, clipped to the state."],
      grid: ["640 m grid and 8 × 8 blocks", "Each 640 m cell is exactly one 64 × 64 patch at 10 m. Cells are grouped into blocks of 8 × 8 cells (5.12 km); a block goes whole to fine-tune or to test."],
      build_model: ["Three models from timm", "One function builds ResNet-50, EfficientNet-B0 or ViT-S/16 with RGB or 13-band input and ImageNet or SSL4EO-S12 weights, always with a new 10-class head."],
      first_layer: ["From 3 to 13 input bands", "ImageNet weights expect 3 channels. The RGB filters are copied to B04, B03 and B02, and every other band starts from the mean of the three."],
      last_block: ["What last_block fine-tunes", "In last_block mode the last block and the classifier learn; everything else stays frozen. The share of the model this represents differs a lot between architectures."],
      class_weights: ["Class-weighted loss", "The fine-tune split is imbalanced, so the cross-entropy gives each class the weight n / (K · n_c): rare classes count more."],
      freeze: ["Freezing everything else", "All parameters are frozen first; then only the modules of the chosen mode (head or last_block) are unfrozen."],
      drop_recovery: ["Drop and recovery", `How much each run loses in Aguascalientes, and how much of that loss fine-tuning wins back. ${demoLabel("runs")} shows both for every run.`],
      selection: ["Selection rule, fixed in advance", "The best Aguascalientes macro-F1 after last_block wins; runs within 0.01 of the top go to the higher throughput."],
    };
    const steps = [];
    DATA.code.forEach((c) => {
      let s = steps.find((x) => x.step === c.step);
      if (!s) steps.push((s = { step: c.step, snippets: [] }));
      s.snippets.push(c);
    });
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("code")} · The code behind the results</div>
        <h2>From the satellite to the selected model, in ${steps.length} functions</h2>
        <p class="lead">Each step shows the real function from the repository, read when <code>data.js</code> was generated,
        with a link to the same lines on GitHub.</p>
      </div>
      <div class="grid">
        <div class="tarjeta panel"><h3>Steps</h3><ol class="steps" id="code-steps"></ol></div>
        <div class="tarjeta" id="code-body"></div>
      </div>`;
    const list = document.getElementById("code-steps");
    steps.forEach((s, k) => {
      const li = el("li");
      const b = el("button", { type: "button" }, `<span class="num">${k + 1}</span><span>${(STEP_INFO[s.step] || [s.snippets[0].function.split(".").pop()])[0]}</span>`);
      b.addEventListener("click", () => setState({ step: k }));
      li.appendChild(b);
      list.appendChild(li);
    });

    function widget(step) {
      if (step === "last_block" && DATA.last_block_share) {
        const items = Object.entries(DATA.last_block_share);
        return '<div class="control" style="margin:4px 0 14px"><span class="rotulo">Share of parameters trained in last_block</span>' +
          items.map(([m, v]) => `<div style="display:flex;align-items:center;gap:10px"><span style="width:130px">${nameOf("model", m)}</span>` +
            `<span style="flex:1;background:#EEF1F6;border-radius:6px;height:14px;overflow:hidden"><span style="display:block;height:100%;width:${(v * 100).toFixed(1)}%;background:${SLIDE.teal}"></span></span>` +
            `<span class="mono" style="width:56px;text-align:right">${Math.round(v * 100)} %</span></div>`).join("") + "</div>";
      }
      if (step === "first_layer") {
        // EuroSAT / TorchGeo band order, fixed by the data contract (docs/contracts.md).
        const bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"];
        const rgb = { B04: ["R", "#FBE3E0", "#803128"], B03: ["G", "#E3F3EA", "#20654E"], B02: ["B", "#E3ECF8", "#2E5486"] };
        return '<div class="control" style="margin:4px 0 14px"><span class="rotulo">First-layer filters per input band</span><div class="chips">' +
          bands.map((b) => (rgb[b] ? `<span class="chip" style="background:${rgb[b][1]};color:${rgb[b][2]};border-color:transparent"><b>${b}</b> ← ${rgb[b][0]}</span>` :
            `<span class="chip">${b} ← mean</span>`)).join("") + "</div></div>";
      }
      if (step === "selection" && DATA.selection) {
        const line = (name, sel) => sel ? `<div><b>${name}:</b> ${runsById.get(sel.winner).label}, ${fmt3(sel.margin)} above ${runsById.get(sel.runner_up).label}; tie-break ${sel.tie_break_applied ? "applied" : "not needed"}.</div>` : "";
        return `<div class="callout" style="margin:4px 0 14px">${line("All runs", DATA.selection.all)}${line("ImageNet runs only", DATA.selection.imagenet)}</div>`;
      }
      if (step === "grid" && DATA.split) {
        return '<div class="callout" style="margin:4px 0 14px"><div class="rotulo" style="margin-bottom:6px">Try it</div>' +
          '<label class="hint">row <input id="g-row" type="range" min="0" max="150" value="74"></label> ' +
          '<label class="hint">col <input id="g-col" type="range" min="0" max="170" value="158"></label>' +
          '<div class="mono" id="g-out" style="margin-top:6px"></div></div>';
      }
      if (step === "drop_recovery") {
        return '<p style="margin:4px 0 14px"><button class="btn" type="button" id="go-demo1">Open demo 2 with these formulas</button></p>';
      }
      return "";
    }

    let lastStep = null;
    renderers.code = function () {
      const k = Math.max(0, Math.min(state.step, steps.length - 1));
      if (k === lastStep) return;
      lastStep = k;
      const s = steps[k];
      list.querySelectorAll("button").forEach((b, i) => (i === k ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
      const [title, text] = STEP_INFO[s.step] || [s.snippets[0].function.split(".").pop(), ""];
      const body = document.getElementById("code-body");
      body.innerHTML = `<div class="rotulo">Step ${k + 1} of ${steps.length}</div><h2 style="font-size:23px">${title}</h2>` +
        `<p class="lead" style="margin-bottom:14px">${text}</p>` + widget(s.step) +
        s.snippets.map((c) => `<div class="codehead"><code title="${c.function}"><b>${c.function.split(".").pop()}()</b></code>` +
          `<a href="${REPO_BLOB}${c.file}#L${c.start_line}-L${c.end_line}" target="_blank" rel="noopener">${c.file} · lines ${c.start_line}–${c.end_line}</a></div>` +
          `<pre class="code"><code class="language-python"></code></pre>`).join("") +
        `<div class="nav-steps"><button class="btn" type="button" id="prev-step" ${k === 0 ? "disabled" : ""}>← Previous</button>` +
        `<button class="btn primary" type="button" id="next-step" ${k === steps.length - 1 ? "disabled" : ""}>Next →</button></div>`;
      body.querySelectorAll("pre.code code").forEach((node, i) => {
        node.textContent = s.snippets[i].source.replace(/\n$/, "");
        if (window.Prism && Prism.languages.python) Prism.highlightElement(node);
      });
      body.querySelector("#prev-step").addEventListener("click", () => setState({ step: k - 1 }));
      body.querySelector("#next-step").addEventListener("click", () => setState({ step: k + 1 }));
      const go = body.querySelector("#go-demo1");
      if (go) go.addEventListener("click", () => { state.formulas = true; document.getElementById("runs-formula-toggle").checked = true; showTab("runs"); });
      const gr = body.querySelector("#g-row");
      if (gr) {
        const gc = body.querySelector("#g-col"), out = body.querySelector("#g-out");
        const b = DATA.split.block_cells;
        const paint = () => {
          const r = +gr.value, c = +gc.value;
          out.innerHTML = `cell_id = r${String(r).padStart(4, "0")}_c${String(c).padStart(4, "0")} &nbsp;→&nbsp; block_id = ` +
            `b${String(Math.floor(r / b)).padStart(3, "0")}_${String(Math.floor(c / b)).padStart(3, "0")}`;
        };
        gr.addEventListener("input", paint); gc.addEventListener("input", paint); paint();
      }
    };
  })();

  // ------------------------------------------------------------------ images (optional: images.js)
  // window.HYDROSCOPE_IMAGES is written by scripts/make_explorer_images.py. Each tile is a T x T
  // square of a PNG sprite; the page only cuts tiles out and, for Grad-CAM, blends the map over
  // the RGB patch with the chosen opacity. Without images.js both demos show a notice.
  const spriteCache = {};
  function loadSprite(file) {
    if (!spriteCache[file]) {
      spriteCache[file] = new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`${file} could not be loaded`));
        img.src = file;
      });
    }
    return spriteCache[file];
  }
  function patchCanvas(T, label) {
    return el("canvas", { width: T, height: T, class: "patch", role: "img", "aria-label": label });
  }
  // Draw one tile; with `over`, blend a second tile on top: (1 - alpha) * base + alpha * over.
  function drawTile(canvas, sprite, T, base, over, alpha) {
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, T, T);
    ctx.drawImage(sprite, base.x, base.y, T, T, 0, 0, T, T);
    if (over && alpha > 0) {
      ctx.globalAlpha = alpha;
      ctx.drawImage(sprite, over.x, over.y, T, T, 0, 0, T, T);
      ctx.globalAlpha = 1;
    }
  }
  const imagesNotice = (what) =>
    `<div class="callout warm"><b>${what} not available.</b> Generate the images from the repository root with ` +
    "<code>python scripts/make_explorer_images.py</code> (it needs EuroSAT and the v1.0 release files) and reload this page.</div>";
  const tipOn = (node, html) => {
    node.addEventListener("mousemove", (e) => showTip(e, html));
    node.addEventListener("mouseleave", hideTip);
  };

  // ================================================================== demo · the two datasets
  (function demoData() {
    const root = document.getElementById("demo-data");
    const S = IMG && IMG.samples;
    const T = (IMG && IMG.tile) || 64;
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("data")} · The two datasets</div>
        <h2>The same ${DATA.classes.length} classes, in two very different landscapes</h2>
        <p class="lead">Every model was trained on EuroSAT, a benchmark of Sentinel-2 patches from Europe, and then tested on
        cells of Aguascalientes that we labeled by hand with the same classes. Each row shows random patches of one class, in true color.</p>
      </div>
      <div id="data-body"></div>`;
    const body = document.getElementById("data-body");
    if (!S || (!S.eurosat && !S.ags)) {
      body.innerHTML = `<div class="tarjeta">${imagesNotice("Sample patches")}</div>`;
      return;
    }
    const classes = S.classes || DATA.classes;
    const euCount = (c) => (S.eurosat && S.eurosat.count[c]) || 0;
    const agsCount = (c) => (S.ags && S.ags.count[c]) || { finetune: 0, test: 0 };
    const agsTotal = (c) => agsCount(c).finetune + agsCount(c).test;
    const most = d3.greatest(classes, agsTotal), fewest = d3.least(classes, agsTotal);

    const bar = el("div", { class: "tarjeta controls-bar" });
    const tiles = el("div", { class: "tiles", style: "flex:1 1 520px" });
    if (S.eurosat) {
      tiles.appendChild(el("div", { class: "tile" },
        `<div class="v">${d3.sum(classes, euCount).toLocaleString("en-US")}</div><div class="k">EuroSAT test patches</div>`));
    }
    if (S.ags) {
      tiles.appendChild(el("div", { class: "tile" },
        `<div class="v">${d3.sum(classes, agsTotal).toLocaleString("en-US")}</div><div class="k">Aguascalientes labeled cells (fine-tune + test)</div>`));
      tiles.appendChild(el("div", { class: "tile drop" },
        `<div class="v">${agsTotal(most)} vs. ${agsTotal(fewest)}</div><div class="k">Labeled cells: ${pretty(most)} vs. ${pretty(fewest)}</div>`));
    }
    bar.appendChild(tiles);
    const sizeBox = el("div", { class: "control" });
    sizeBox.appendChild(el("span", { class: "rotulo" }, "Patch size on screen"));
    let size = 72;
    const updateSize = segmented(sizeBox, [{ value: 52, label: "Small" }, { value: 72, label: "Medium" }, { value: 112, label: "Large" }], (v) => {
      size = v;
      table.style.setProperty("--patch", v + "px");
      updateSize(size);
    });
    updateSize(size);
    bar.appendChild(sizeBox);
    bar.appendChild(el("p", { class: "hint", style: "flex-basis:100%" },
      `One patch is ${T} × ${T} pixels at 10 m, so ${(T * 10).toLocaleString("en-US")} m on a side: the same size in both datasets. ` +
      "True color from bands B04, B03 and B02, stretched separately for each dataset. Hover a patch to see where it comes from."));
    body.appendChild(bar);

    const card = el("div", { class: "tarjeta" });
    const table = el("div", { class: "datasets", style: `--patch:${size}px` });
    table.appendChild(el("div", { class: "ds-head" },
      `<span>Class</span><span>EuroSAT <span class="hint">· Europe, test split</span></span>` +
      `<span>Aguascalientes <span class="hint">· labeled by us, both splits</span></span>`));
    const pending = []; // [canvas, sprite position]
    classes.forEach((c) => {
      const row = el("div", { class: "ds-row" });
      const ac = agsCount(c);
      const excluded = !DATA.macro_f1_classes.includes(c);
      row.appendChild(el("div", { class: "ds-name" },
        `${pretty(c)}<div class="ds-meta">EuroSAT test: ${euCount(c)}<br>Ags fine-tune: ${ac.finetune}<br>Ags test: ${ac.test}</div>` +
        (excluded ? '<span class="note-chip">Left out of the Aguascalientes macro-F1: too few test cells</span>' : "")));
      [["eurosat", "EuroSAT"], ["ags", "Aguascalientes"]].forEach(([key, name]) => {
        const side = el("div", { class: "side" });
        side.appendChild(el("div", { class: "side-cap" }, name));
        const strip = el("div", { class: "patches" });
        const list = (S[key] && S[key].tiles[c]) || [];
        if (!S[key]) strip.appendChild(el("span", { class: "empty" }, "Not generated"));
        else if (!list.length) strip.appendChild(el("span", { class: "empty" }, "No labeled cells of this class"));
        list.forEach((t) => {
          const where = key === "eurosat" ? `EuroSAT test patch #${t.index}` : `Cell ${t.cell_id} · ${t.split === "test" ? "test" : "fine-tune"} split`;
          const canvas = patchCanvas(T, `${pretty(c)}, ${where}`);
          tipOn(canvas, `<b>${pretty(c)}</b><br>${where}`);
          strip.appendChild(canvas);
          pending.push([canvas, t]);
        });
        side.appendChild(strip);
        row.appendChild(side);
      });
      table.appendChild(row);
    });
    card.appendChild(table);
    body.appendChild(card);
    loadSprite(S.file).then(
      (sprite) => pending.forEach(([canvas, t]) => drawTile(canvas, sprite, T, t)),
      () => card.insertAdjacentHTML("afterbegin", imagesNotice(`<code>${S.file}</code>`)),
    );
  })();

  // ================================================================== demo · Grad-CAM
  (function demoGradcam() {
    const root = document.getElementById("demo-gradcam");
    const G = IMG && IMG.gradcam;
    const T = (IMG && IMG.tile) || 64;
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">${demoLabel("gradcam")} · Where the models look</div>
        <h2>Grad-CAM: which part of the patch drove each prediction</h2>
        <p class="lead">Grad-CAM colors each patch by how much that region pushed the model toward the class it predicted:
        red is strong, blue is weak. Every model sees the same patches, chosen at random.</p>
      </div>
      <div id="gc-body"></div>`;
    const body = document.getElementById("gc-body");
    if (!G || !G.figures || !G.figures.length) {
      body.innerHTML = `<div class="tarjeta">${imagesNotice("Grad-CAM maps")}</div>`;
      return;
    }
    const FIG = {
      gradcam_eurosat: ["EuroSAT test", "The models trained on EuroSAT, on EuroSAT test patches."],
      gradcam_ags: ["Aguascalientes, zero-shot", "The same models, without any training on Aguascalientes, on Aguascalientes test cells."],
      gradcam_selected: ["Selected model, before and after fine-tuning", "The selected model before and after last_block fine-tuning, on Aguascalientes test cells."],
    };
    const figLabel = (f) => (FIG[f.name] || [f.name])[0];
    const isAgs = (f) => String(f.dataset).startsWith("ags");
    body.innerHTML = `
      <div class="grid">
        <div class="tarjeta panel" id="gc-panel"></div>
        <div class="tarjeta" style="overflow-x:auto"><h3 id="gc-title"></h3><p class="hint" id="gc-lead" style="margin-bottom:14px"></p><div id="gc-grid"></div></div>
      </div>`;
    const panel = document.getElementById("gc-panel");
    const figBox = el("div", { class: "control" });
    figBox.appendChild(el("span", { class: "rotulo" }, "Patches and models"));
    let current = G.figures[0].name;
    const updateFig = segmented(figBox, G.figures.map((f) => ({ value: f.name, label: figLabel(f) })), (v) => {
      current = v;
      updateFig(current);
      build();
    });
    panel.appendChild(figBox);

    let alpha = 0.5;
    const alphaBox = el("div", { class: "control" });
    alphaBox.innerHTML = `<span class="rotulo">Grad-CAM opacity: <span id="gc-alpha-v">50 %</span></span>` +
      '<input type="range" id="gc-alpha" min="0" max="100" step="5" value="50" aria-label="Grad-CAM opacity">' +
      '<div class="jet" aria-hidden="true"></div><div class="jet-labels"><span>weak</span><span>strong</span></div>';
    panel.appendChild(alphaBox);
    const scoreBox = el("div", { class: "tiles" });
    panel.appendChild(scoreBox);
    const rulesBox = el("div", {});
    panel.appendChild(rulesBox);
    panel.appendChild(el("p", { class: "hint" },
      "Each map explains the class the model predicted. ✓ means the prediction matches the true class; ✗ that it does not. " +
      "At 0 % you see the plain true-color patch."));

    const colTitle = (f, col) => {
      const run = runsById.get(col.base_run_id);
      const name = run ? `${nameOf("model", run.model)}</b><br>${nameOf("bands", run.bands)} · ${(DATA.names.init || {})[run.init] || run.init}` : `${col.base_run_id}</b>`;
      const stage = !isAgs(f) ? "trained on EuroSAT" : col.stage === "last_block" ? "after last_block" : "zero-shot";
      return `<b>${name}<br><span class="${col.stage === "last_block" ? "ok" : ""}">${stage}</span>`;
    };

    let maps = []; // [canvas, rgb position, map position]
    function paint(sprite) {
      maps.forEach(([canvas, rgb, map]) => drawTile(canvas, sprite, T, rgb, map, alpha));
    }
    function build() {
      const f = G.figures.find((x) => x.name === current) || G.figures[0];
      document.getElementById("gc-title").textContent = figLabel(f);
      document.getElementById("gc-lead").textContent = (FIG[f.name] || [null, ""])[1];
      // score per column, counted from the maps
      scoreBox.innerHTML = f.columns.map((col, j) => {
        const ok = f.rows.filter((r) => r.maps[j] && r.maps[j].correct).length;
        const run = runsById.get(col.base_run_id);
        const who = run ? nameOf("model", run.model) : col.base_run_id;
        const stage = isAgs(f) && f.columns.some((c) => c.stage !== col.stage) ? ` · ${col.stage}` : "";
        return `<div class="tile ${ok === f.rows.length ? "rec" : ""}"><div class="v">${ok} / ${f.rows.length}</div><div class="k">${who}${stage}: correct</div></div>`;
      }).join("");
      const rules = [...new Set(f.rows.map((r) => r.selection_rule))];
      const ruleText = (rule) => {
        const n = f.rows.filter((r) => r.selection_rule === rule).length;
        const t = pretty(rule.replace(/gradcam_\w+/g, (m) => `“${FIG[m] ? FIG[m][0] : m}”`));
        return `${t.charAt(0).toUpperCase()}${t.slice(1)} (${n} ${n === 1 ? "patch" : "patches"}).`;
      };
      rulesBox.innerHTML = '<span class="rotulo">How the patches were chosen</span>' +
        `<ul class="rules">${rules.map((r) => `<li>${ruleText(r)}</li>`).join("")}</ul>`;

      const grid = document.getElementById("gc-grid");
      grid.innerHTML = "";
      grid.className = "gc-grid";
      grid.style.gridTemplateColumns = `var(--label) repeat(${f.columns.length + 1}, var(--patch))`;
      grid.appendChild(el("div", { class: "gc-colhead" }, "<b>True class</b>"));
      grid.appendChild(el("div", { class: "gc-colhead" }, "<b>True color</b><br>no map"));
      f.columns.forEach((col) => grid.appendChild(el("div", { class: "gc-colhead" }, colTitle(f, col))));
      maps = [];
      const plain = [];
      f.rows.forEach((r) => {
        const where = isAgs(f) ? `cell ${r.cell_id}` : `EuroSAT test patch #${r.index}`;
        grid.appendChild(el("div", { class: "gc-rowhead" }, `${pretty(r.true_label)}<div class="ds-meta">${where}</div>`));
        const rgbCanvas = patchCanvas(T, `${pretty(r.true_label)}, true color`);
        tipOn(rgbCanvas, `<b>${pretty(r.true_label)}</b><br>${where}`);
        grid.appendChild(el("figure", { class: "gc-cell" })).appendChild(rgbCanvas);
        plain.push([rgbCanvas, r.rgb]);
        r.maps.forEach((m) => {
          const fig = el("figure", { class: "gc-cell" });
          const canvas = patchCanvas(T, `Grad-CAM, predicted ${m.predicted_label}`);
          tipOn(canvas, `True: <b>${pretty(r.true_label)}</b><br>Predicted: <b>${pretty(m.predicted_label)}</b> ${m.correct ? "✓" : "✗"}`);
          fig.appendChild(canvas);
          fig.appendChild(el("figcaption", { class: m.correct ? "ok" : "bad" }, `${m.correct ? "✓" : "✗"} ${pretty(m.predicted_label)}`));
          grid.appendChild(fig);
          maps.push([canvas, r.rgb, m]);
        });
      });
      loadSprite(G.file).then(
        (sprite) => {
          plain.forEach(([canvas, pos]) => drawTile(canvas, sprite, T, pos));
          paint(sprite);
        },
        () => (grid.innerHTML = imagesNotice(`<code>${G.file}</code>`)),
      );
    }
    const slider = document.getElementById("gc-alpha");
    slider.addEventListener("input", () => {
      alpha = +slider.value / 100;
      document.getElementById("gc-alpha-v").textContent = `${slider.value} %`;
      loadSprite(G.file).then(paint, () => {});
    });
    updateFig(current);
    build();
  })();

  // ------------------------------------------------------------------ start
  showTab(location.hash.slice(1) || TABS[0]);
})();
