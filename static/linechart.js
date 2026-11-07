// A small multi-line chart on a 0–100% axis, drawn like the Stats learning curve: hairline grid, 2px lines, end dots
// with direct end labels (kept 16px apart), and a crosshair whose tooltip lists every series at that x.
//   LineChart.draw(box, { series: [{ name, color, points: [{ x, y }] }], xLabel: "clip", height: 260, label: "…" })
// Colors are theme tokens passed in by the caller; text stays in ink tokens.
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; };
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  const lineKey = (color) => { const k = el("i", "lk"); k.style.background = color; return k; };

  function ticks(lo, hi, max = 6) {
    const span = Math.max(1, hi - lo), raw = span / max, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw && Number.isInteger(s)) || Math.ceil(raw);
    const out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push(t);
    return out;
  }

  function draw(box, { series, xLabel = "", height = 260, label = "" }) {
    const W = box.clientWidth, H = height, M = { l: 40, r: 150, t: 14, b: 30 }, iw = W - M.l - M.r, ih = H - M.t - M.b;
    if (iw < 40 || !series.some((s) => s.points.length)) return;
    const xs = series.flatMap((s) => s.points.map((p) => p.x)), x0 = Math.min(...xs), x1 = Math.max(...xs);
    const X = (x) => M.l + (x1 === x0 ? iw / 2 : ((x - x0) / (x1 - x0)) * iw), Y = (v) => M.t + (1 - v) * ih;
    const root = svg("svg", { class: "chart", width: W, height: H, viewBox: `0 0 ${W} ${H}`, tabindex: 0, role: "img", "aria-label": label });
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      root.append(svg("line", { x1: M.l, x2: M.l + iw, y1: Y(v), y2: Y(v), stroke: v === 0 ? "var(--hair-2)" : "var(--hair)" }));
      root.append(svg("text", { x: M.l - 8, y: Y(v) + 4, "text-anchor": "end" }, `${v * 100}%`));
    }
    for (const t of ticks(x0, x1)) root.append(svg("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, String(t)));
    if (xLabel) root.append(svg("text", { x: M.l + iw + 10, y: H - 8 }, xLabel));
    for (const s of [...series].reverse()) {
      root.append(svg("polyline", { points: s.points.map((p) => `${X(p.x)},${Y(p.y)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
    const ends = series.filter((s) => s.points.length).map((s) => { const p = s.points[s.points.length - 1]; return { s, x: X(p.x), y: Y(p.y), v: p.y }; }).sort((a, b) => a.y - b.y);
    for (let k = 1; k < ends.length; k++) ends[k].ly = Math.max(ends[k].y, (ends[k - 1].ly ?? ends[k - 1].y) + 16);
    if (ends.length) ends[0].ly = ends[0].y;
    const over = ends.length ? ends[ends.length - 1].ly - (M.t + ih) : 0;
    if (over > 0) ends.forEach((e) => { e.ly -= over; });
    for (const e of ends) {
      if (Math.abs(e.ly - e.y) > 1) root.append(svg("polyline", { points: `${M.l + iw + 6},${e.y} ${M.l + iw + 12},${e.ly}`, fill: "none", stroke: "var(--ink-4)" }));
      root.append(svg("circle", { cx: e.x, cy: e.y, r: 4, fill: e.s.color, stroke: "var(--card)", "stroke-width": 2 }));
      const t = svg("text", { class: "end", x: M.l + iw + 14, y: e.ly + 4 });
      t.append(svg("tspan", { class: "v" }, pct(e.v)), svg("tspan", {}, ` ${e.s.name}`));
      root.append(t);
    }
    // crosshair at the nearest x; one tooltip, every series
    const allX = [...new Set(xs)].sort((a, b) => a - b);
    const cross = svg("line", { y1: M.t, y2: M.t + ih, stroke: "var(--ink-4)", visibility: "hidden" });
    const dots = series.map((s) => svg("circle", { r: 4, fill: s.color, stroke: "var(--card)", "stroke-width": 2, visibility: "hidden" }));
    const hit = svg("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent" });
    root.append(cross, ...dots, hit);
    const tip = el("div", "event tip");
    tip.hidden = true;
    let at = -1;
    const show = (k) => {
      at = k;
      if (k < 0) { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); return; }
      const x = allX[k], cx = X(x);
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
      const rows = [el("span", null, `${xLabel} ${x}`)];
      series.forEach((s, j) => {
        const p = s.points.find((q) => q.x === x);
        dots[j].setAttribute("visibility", p ? "visible" : "hidden");
        if (p) { dots[j].setAttribute("cx", cx); dots[j].setAttribute("cy", Y(p.y)); }
        const r = el("span"); r.append(lineKey(s.color), el("b", null, pct(p?.y)), s.name); rows.push(r);
      });
      tip.replaceChildren(...rows);
      tip.hidden = false;
      const tw = tip.offsetWidth;
      tip.style.left = `${cx + 14 + tw > W ? cx - 14 - tw : cx + 14}px`;
      tip.style.top = `${M.t}px`;
    };
    hit.addEventListener("pointermove", (e) => {
      const x = x0 + ((e.clientX - root.getBoundingClientRect().left - M.l) / iw) * (x1 - x0);
      let k = 0;
      allX.forEach((v, j) => { if (Math.abs(v - x) < Math.abs(allX[k] - x)) k = j; });
      show(k);
    });
    hit.addEventListener("pointerleave", () => show(-1));
    root.addEventListener("focus", () => show(at >= 0 ? at : allX.length - 1));
    root.addEventListener("blur", () => show(-1));
    root.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") show(Math.max(0, at - 1));
      else if (e.key === "ArrowRight") show(Math.min(allX.length - 1, at + 1));
      else return;
      e.preventDefault();
    });
    box.replaceChildren(root, tip);
  }

  window.LineChart = { draw, lineKey };
})();
