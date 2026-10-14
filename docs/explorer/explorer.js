// Hydroscope Aguascalientes · Explorer
// Every number shown here is read from window.HYDROSCOPE_DATA (data.js), which is written by
// scripts/make_explorer_data.py from experiments/results/*.json. The page computes nothing that
// the paper reports: it only formats those values (3 decimals, like the paper) and, in demo 3,
// counts cells for the random split it draws itself.
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
    tab: "runs", run: runsById.has(selected) ? selected : DATA.runs[0].run_id,
    stage: "zero_shot", view: "share", cell: null, formulas: false,
    splitMode: "blocks", seed: 1, step: 0,
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
      (warnings.length ? `<br><strong>Warning:</strong> ${warnings.join(" ")}` : "");
  })();

  // ------------------------------------------------------------------ tabs
  const TABS = ["runs", "confusion", "split", "code"];
  function showTab(tab) {
    if (!TABS.includes(tab)) tab = "runs";
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
        b.disabled = isEnabled ? !isEnabled(opt.value) : false;
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

  // ================================================================== demo 1 · drop and recovery
  (function demoRuns() {
    const root = document.getElementById("demo-runs");
    const n = DATA.macro_f1_classes.length;
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">Demo 1 · RQ-b and RQ-c</div>
        <h2>From Europe to Aguascalientes: how much is lost, how much comes back</h2>
        <p class="lead">Each run was trained on EuroSAT, tested on Aguascalientes without local training (zero-shot),
        and then fine-tuned on 561 local patches in two ways: <b>head</b> (classifier only) and <b>last_block</b>.</p>
      </div>
      <div class="grid">
        <div class="tarjeta panel" id="runs-panel"></div>
        <div class="stack">
          <div class="tarjeta">
            <h3 id="runs-title"></h3>
            <div id="runs-track"></div>
            <div class="legend">
              <span><i class="sq" style="background:${COLOR.dropBand};border:1px solid #F1D2CE"></i>Drop: what was lost from EuroSAT to Aguascalientes</span>
              <span><i class="sq" style="background:${COLOR.lastBlock}"></i>Recovered by fine-tuning</span>
            </div>
          </div>
          <div class="tarjeta">
            <div class="tiles" id="runs-tiles"></div>
            <div class="formulas" id="runs-formulas" style="margin-top:14px" hidden></div>
          </div>
          <div class="tarjeta">
            <h3>All runs at a glance <span class="hint" style="font-weight:400">· click a row to select it</span></h3>
            <div id="runs-all"></div>
            <div class="legend">
              <span><i style="background:${COLOR.eurosat}"></i>EuroSAT test</span>
              <span><i style="background:${COLOR.zeroShot}"></i>Aguascalientes zero-shot</span>
              <span><i style="background:${COLOR.lastBlock}"></i>after last_block</span>
            </div>
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

    // --- track chart: four bars on a shared 0–1 macro-F1 axis
    const W = 860, rowH = 50, top = 46, left = 130, right = 190;
    const rows = [
      { key: "eurosat", label: "EuroSAT test", color: COLOR.eurosat },
      { key: "zero", label: "Zero-shot", color: COLOR.zeroShot },
      { key: "head", label: "head", color: COLOR.head },
      { key: "last", label: "last_block", color: COLOR.lastBlock },
    ];
    const H = top + rows.length * rowH + 34;
    const x = d3.scaleLinear().domain([0, 1]).range([left, W - right]);
    const svg = d3.select("#runs-track").append("svg").attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Macro-F1 on EuroSAT, zero-shot, after head and after last_block");
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${top + rows.length * rowH + 4})`)
      .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
    svg.append("g").selectAll("line").data(x.ticks(5)).join("line").attr("class", "gridline")
      .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", top - 6).attr("y2", top + rows.length * rowH);
    const band = svg.append("rect").attr("y", top - 6).attr("height", rows.length * rowH + 6).attr("fill", COLOR.dropBand);
    const bandLabel = svg.append("text").attr("y", top - 16).attr("text-anchor", "middle")
      .attr("font-weight", 600).attr("fill", "#803128").attr("font-size", 14);
    const g = svg.selectAll("g.row").data(rows).join("g").attr("class", "row")
      .attr("transform", (d, i) => `translate(0,${top + i * rowH})`);
    g.append("text").attr("x", left - 12).attr("y", rowH / 2 + 5).attr("text-anchor", "end").attr("font-size", 14).text((d) => d.label);
    g.append("rect").attr("class", "base").attr("x", x(0)).attr("y", 12).attr("height", rowH - 24).attr("rx", 4);
    g.append("rect").attr("class", "gain").attr("y", 12).attr("height", rowH - 24).attr("rx", 4);
    g.append("text").attr("class", "val mono").attr("y", rowH / 2 + 5).attr("font-size", 13.5).attr("font-family", "IBM Plex Mono");
    g.append("text").attr("class", "note").attr("x", W - right + 66).attr("y", rowH / 2 + 5).attr("font-size", 12.5).attr("fill", "#5B6B85");

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
        .attr("fill", (d) => (d.key === "eurosat" || d.key === "zero" ? d.color : COLOR.base));
      g.select("rect.gain").transition(t)
        .attr("x", (d) => x(Math.min(values[d.key], s.ags_zero_shot)))
        .attr("width", (d) => (d.key === "head" || d.key === "last" ? Math.abs(x(values[d.key]) - x(s.ags_zero_shot)) : 0))
        .attr("fill", (d) => (values[d.key] >= s.ags_zero_shot ? d.color : COLOR.zeroShot));
      g.select("text.val").transition(t).attr("x", (d) => x(values[d.key]) + 8);
      g.select("text.val").text((d) => fmt3(values[d.key]));
      g.select("text.note").text((d) => {
        if (d.key === "head") return `recovery ${fmt3(s.recovery_head)}`;
        if (d.key === "last") return `recovery ${fmt3(s.recovery_last_block)}`;
        return "";
      });
    }

    // --- all runs: dumbbells, click to select
    const aW = 860, aLeft = 250, aRight = 40, aRow = 34, aTop = 8;
    const ax = d3.scaleLinear().domain([0, 1]).range([aLeft, aW - aRight]);
    const aH = aTop + DATA.runs.length * aRow + 30;
    const asvg = d3.select("#runs-all").append("svg").attr("viewBox", `0 0 ${aW} ${aH}`)
      .attr("role", "img").attr("aria-label", "EuroSAT, zero-shot and last_block macro-F1 for every run");
    asvg.append("g").attr("class", "axis").attr("transform", `translate(0,${aTop + DATA.runs.length * aRow + 2})`)
      .call(d3.axisBottom(ax).ticks(5).tickSizeOuter(0));
    const arow = asvg.selectAll("g.arow").data(DATA.runs).join("g").attr("class", "arow clickable")
      .attr("transform", (d, i) => `translate(0,${aTop + i * aRow})`)
      .on("click", (e, d) => setState({ run: d.run_id, cell: null }));
    arow.append("rect").attr("class", "hl").attr("x", 0).attr("width", aW).attr("height", aRow - 2).attr("rx", 8);
    arow.append("text").attr("x", 12).attr("y", aRow / 2 + 4).attr("font-size", 13)
      .text((d) => (d.run_id === selected ? "★ " : "") + d.label);
    arow.append("line").attr("x1", (d) => ax(d.summary.ags_zero_shot)).attr("x2", (d) => ax(d.summary.eurosat_f1_macro_classes))
      .attr("y1", aRow / 2).attr("y2", aRow / 2).attr("stroke", "#E7B9B3").attr("stroke-width", 3);
    [["eurosat_f1_macro_classes", COLOR.eurosat], ["ags_zero_shot", COLOR.zeroShot], ["ags_last_block", COLOR.lastBlock]].forEach(([k, c]) => {
      arow.append("circle").attr("cx", (d) => ax(d.summary[k])).attr("cy", aRow / 2).attr("r", 6).attr("fill", c)
        .on("mousemove", (e, d) => showTip(e, `${d.label}<br>${k.replace(/_/g, " ")}: <b>${fmt3(d.summary[k])}</b>`))
        .on("mouseleave", hideTip);
    });

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
      arow.select("text").attr("font-weight", (d) => (d.run_id === state.run ? 600 : 400));
    };
  })();

  // ================================================================== demo 2 · confusion matrix
  (function demoConfusion() {
    const root = document.getElementById("demo-confusion");
    root.innerHTML = `
      <div class="demo-head">
        <div class="rotulo">Demo 2 · Where the errors go</div>
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
    const color = d3.scaleSequential(d3.interpolateRgbBasis(["#F4F7FB", "#B4C9E4", "#4777B4", "#213E63"])).domain([0, 1]);
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
        .attr("fill", (d) => (rowSums[d.i] && conf[d.i][d.j] / rowSums[d.i] > 0.5 ? "#FFFFFF" : "#1E2A45"));
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
        <div class="rotulo">Demo 3 · Why we split by blocks</div>
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
        </div>
      </div>`);
    const panel = document.getElementById("split-panel");
    const modeBox = el("div", { class: "control" });
    modeBox.appendChild(el("span", { class: "rotulo" }, "How the cells are split"));
    const updateMode = segmented(modeBox, [{ value: "blocks", label: `Our split: ${sp.block_cells} × ${sp.block_cells} blocks` }, { value: "random", label: "Random cells" }], (v) => setState({ splitMode: v }));
    panel.appendChild(modeBox);
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
      "The paper reports this as a limitation: the split has no buffer zone."));

    // geometry
    const rows = sp.cells.map((c) => c[0]), cols = sp.cells.map((c) => c[1]);
    const r0 = d3.min(rows), r1 = d3.max(rows), c0 = d3.min(cols), c1 = d3.max(cols);
    const size = 5.5, pad = 8;
    const W = (c1 - c0 + 1) * size + 2 * pad, H = (r1 - r0 + 1) * size + 2 * pad;
    const svg = d3.select("#split-map").append("svg").attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Map of the labeled cells of Aguascalientes colored by split");
    const bx0 = Math.floor(c0 / sp.block_cells), bx1 = Math.floor(c1 / sp.block_cells);
    const by0 = Math.floor(r0 / sp.block_cells), by1 = Math.floor(r1 / sp.block_cells);
    const grid = svg.append("g").attr("stroke", "#DCE3EC").attr("stroke-width", 0.6);
    for (let b = bx0; b <= bx1 + 1; b++) {
      const xx = pad + (b * sp.block_cells - c0) * size;
      grid.append("line").attr("x1", xx).attr("x2", xx).attr("y1", pad).attr("y2", H - pad);
    }
    for (let b = by0; b <= by1 + 1; b++) {
      const yy = pad + (b * sp.block_cells - r0) * size;
      grid.append("line").attr("y1", yy).attr("y2", yy).attr("x1", pad).attr("x2", W - pad);
    }
    // One background square per block that holds labeled cells, colored by the block's split.
    const blockSide = new Map();
    sp.cells.forEach((c) => blockSide.set(Math.floor(c[0] / sp.block_cells) + ":" + Math.floor(c[1] / sp.block_cells), c[2]));
    const blocks = [...blockSide.entries()].map(([k, side]) => ({ by: +k.split(":")[0], bx: +k.split(":")[1], side }));
    const blockRects = svg.insert("g", "g").selectAll("rect").data(blocks).join("rect")
      .attr("x", (d) => pad + (d.bx * sp.block_cells - c0) * size).attr("y", (d) => pad + (d.by * sp.block_cells - r0) * size)
      .attr("width", sp.block_cells * size).attr("height", sp.block_cells * size);
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
      .attr("x", (d) => pad + (d[1] - c0) * size + 0.4).attr("y", (d) => pad + (d[0] - r0) * size + 0.4)
      .attr("width", size - 0.8).attr("height", size - 0.8);

    let lastSplitKey = null;
    renderers.split = function () {
      const splitKey = state.splitMode + ":" + state.seed;
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
      blockRects.transition().duration(500)
        .attr("fill", (d) => (state.splitMode === "blocks" ? (d.side === "test" ? "#FAF1E1" : "#E6EEF8") : "#F6F8FB"));
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
      drop_recovery: ["Drop and recovery", "The two numbers behind RQ-b and RQ-c, exactly as the paper defines them. Demo 1 shows them for every run."],
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
        <div class="rotulo">Demo 4 · The code behind the results</div>
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
            `<span style="flex:1;background:#EEF1F6;border-radius:6px;height:14px;overflow:hidden"><span style="display:block;height:100%;width:${(v * 100).toFixed(1)}%;background:${COLOR.lastBlock}"></span></span>` +
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
        return '<p style="margin:4px 0 14px"><button class="btn" type="button" id="go-demo1">Open demo 1 with these formulas</button></p>';
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

  // ------------------------------------------------------------------ start
  showTab(location.hash.slice(1) || "runs");
})();
