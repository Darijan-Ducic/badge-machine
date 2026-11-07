// VTT vs STT, word by word. VTT = video-to-text (lip reading), STT = speech-to-text (the label).
// Words compare after the backend's normalization (lowercase, punctuation stripped; preheat._normalize) but show as
// written. Ops: eq (same word), sub (VTT said another word), del (STT word VTT missed), ins (extra VTT word).
// Classes (theme.css) are prefixed so page classes (.col, .sub, .row) never restyle them:
// .vw-sub = highlighted VTT word, .vw-del = struck-through STT word, .vw-ins = marked extra VTT word.
(() => {
  const NONE = "∅";
  const key = (w) => w.toLowerCase().replace(/[^a-z0-9']/g, "");
  const words = (text) => (text || "").split(/\s+/).filter((w) => key(w));

  function diff(stt, vtt) {
    const a = words(stt), b = vtt === NONE ? [] : words(vtt), A = a.map(key), B = b.map(key);
    const L = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));   // LCS table
    for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--)
      L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    const raw = [];
    let i = 0, j = 0;
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && A[i] === B[j]) raw.push({ op: "eq", stt: a[i++], vtt: b[j++] });
      else if (j < b.length && (i === a.length || L[i][j + 1] >= L[i + 1][j])) raw.push({ op: "ins", vtt: b[j++] });
      else raw.push({ op: "del", stt: a[i++] });
    }
    // between two matches, pair missed STT words with extra VTT words as substitutions
    const out = [];
    for (let k = 0; k < raw.length;) {
      if (raw[k].op === "eq") { out.push(raw[k++]); continue; }
      const dels = [], ins = [];
      for (; k < raw.length && raw[k].op !== "eq"; k++) (raw[k].op === "del" ? dels : ins).push(raw[k]);
      const n = Math.min(dels.length, ins.length);
      for (let m = 0; m < n; m++) out.push({ op: "sub", stt: dels[m].stt, vtt: ins[m].vtt });
      out.push(...dels.slice(n), ...ins.slice(n));
    }
    return out;
  }

  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const same = (stt, vtt) => vtt !== NONE && vtt != null && stt != null && words(stt).map(key).join(" ") === words(vtt).map(key).join(" ");

  // Two aligned lines: each column holds an STT word over the VTT word it was matched with. Without both texts
  // (no STT, no read, or ∅ won) the lines are plain; under ∅ every STT word counts as missed.
  function pair(stt, vtt, { vttLabel = "VTT" } = {}) {
    const box = el("div", "vtt-diff");
    box.setAttribute("aria-label", `STT: ${stt ?? "none"}. ${vttLabel}: ${vtt === NONE ? "none of these" : vtt ?? "none"}.`);
    const top = el("span", "vtt-lbl", "STT"), bot = el("span", "vtt-lbl", vttLabel);
    top.style.gridRow = "1"; bot.style.gridRow = "2";
    if (stt == null || vtt == null || vtt === NONE) {
      const r1 = el("span", "vtt-line"), r2 = el("span", "vtt-line");
      r1.style.gridRow = "1"; r2.style.gridRow = "2";
      if (stt == null) r1.append(el("span", "vw vw-none", "no STT"));
      else for (const w of words(stt)) r1.append(el("span", vtt === NONE ? "vw vw-del" : "vw", w), " ");
      r2.append(vtt == null ? el("span", "vw vw-none", "no read") : vtt === NONE ? el("span", "vw vw-none", "∅ none of these") : vtt);
      box.append(top, r1, bot, r2);
      return box;
    }
    const cols = el("span", "vtt-cols");
    cols.style.gridRow = "1 / span 2";
    for (const o of diff(stt, vtt)) {
      const col = el("span", "vtt-col");
      col.append(el("span", o.op === "del" ? "vw vw-del" : "vw", o.stt ?? " "),
                 el("span", o.op === "sub" ? "vw vw-sub" : o.op === "ins" ? "vw vw-ins" : "vw", o.vtt ?? " "));
      cols.append(col);
    }
    box.append(top, bot, cols);
    return box;
  }

  // One line: the model's words against STT, with STT words it missed shown struck through in place.
  function line(stt, vtt) {
    const box = el("span", "vtt-line1");
    if (vtt == null) { box.append(el("span", "vw vw-none", "no read")); return box; }
    if (vtt === NONE) { box.append(el("span", "vw vw-none", "∅ none of these")); return box; }
    if (stt == null) { box.append(vtt); return box; }
    for (const o of diff(stt, vtt)) {
      box.append(el("span", o.op === "eq" ? "vw" : `vw vw-${o.op}`, o.op === "del" ? o.stt : o.vtt), " ");
    }
    return box;
  }

  // VTT · Muse pick for a clip: the most common top hypothesis across valid samples, ties to the first seen
  // (candidates.featurize → muse_pick).
  function musePick(samples) {
    const tops = (samples || []).map((s) => s.output?.hypotheses?.length ? s.output.hypotheses.reduce((a, b) => (b.s > a.s ? b : a)).text : null).filter(Boolean);
    const votes = new Map();
    for (const t of tops) { const k = words(t).map(key).join(" "); votes.set(k, (votes.get(k) || 0) + 1); }
    let best = null;
    for (const t of tops) if (best == null || votes.get(words(t).map(key).join(" ")) > votes.get(words(best).map(key).join(" "))) best = t;
    return best;
  }

  // The key for the three marks, as a .legend row.
  function legend() {
    const box = el("p", "legend");
    for (const [cls, word, text] of [["sub", "about", "VTT said another word"], ["del", "with", "STT word VTT missed"], ["ins", "more", "extra VTT word"]]) {
      const item = el("span");
      item.append(el("span", `vw vw-${cls}`, word), ` ${text}`);
      box.append(item);
    }
    return box;
  }

  window.VTT = { NONE, diff, pair, line, same, musePick, legend };
})();
