// Metrics for the tracker lab (tools/tracker_eval.mjs): how closely a tracker's mouth box follows SAM's
// ground-truth lips box, frame by frame. Pure functions on boxes [x, y, w, h]; no DOM, no I/O.
// Units: errors in GT mouth widths, speeds in mouth widths per second, times in ms.

export const centre = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];

// The mouth box inside a crop that pads it by `pad` around the same centre (tracker crops are mouth x CROP_PAD).
export function mouthFromCrop([x, y, w, h], pad) {
  const mw = w / pad, mh = h / pad;
  return [x + (w - mw) / 2, y + (h - mh) / 2, mw, mh];
}

// Per frame: {centre, sizeW, sizeH} of our mouth box against the GT mouth, in GT mouth widths;
// null where there is no GT mouth; Infinity everywhere when we have no box but GT has one (a total miss).
export function frameErrors(ours, gt) {
  return gt.map((g, i) => {
    if (!g) return null;
    const o = ours[i];
    if (!o) return { centre: Infinity, sizeW: Infinity, sizeH: Infinity };
    const [ox, oy] = centre(o), [gx, gy] = centre(g);
    return { centre: Math.hypot(ox - gx, oy - gy) / g[2], sizeW: Math.abs(o[2] - g[2]) / g[2], sizeH: Math.abs(o[3] - g[3]) / g[2] };
  });
}

// Largest symmetric half-window <= k around i with a GT box on every frame in it.
function halfWindow(gt, i, k) {
  let d = 0;
  while (d < k && i - d - 1 >= 0 && i + d + 1 < gt.length && gt[i - d - 1] && gt[i + d + 1]) d++;
  return d;
}

// Head speed per frame (mouth widths per second): the GT mouth centre averaged over +-k frames (k ~ 0.1 s) so
// syllable-rate lip motion cancels, then a central difference over +-k frames. null where it cannot be measured.
export function headSpeed(gt, fps, k = Math.max(1, Math.round(0.1 * fps))) {
  const n = gt.length, smooth = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    if (!gt[i]) continue;
    const d = halfWindow(gt, i, k);
    let sx = 0, sy = 0;
    for (let j = i - d; j <= i + d; j++) { const [x, y] = centre(gt[j]); sx += x; sy += y; }
    smooth[i] = [sx / (2 * d + 1), sy / (2 * d + 1)];
  }
  return gt.map((g, i) => {
    if (!g) return null;
    let d = 0;
    while (d < k && i - d - 1 >= 0 && i + d + 1 < n && smooth[i - d - 1] && smooth[i + d + 1]) d++;
    if (!d) return null;
    const a = smooth[i - d], b = smooth[i + d];
    return Math.hypot(b[0] - a[0], b[1] - a[1]) / ((2 * d) / fps) / g[2];
  });
}

// The delay (ms, whole frames from 0 to maxShift) that best aligns our centres with the GT centres on masked frames:
// argmin over s of mean |ours[i] - gt[i - s]|.
export function lagMs(ours, gt, mask, fps, maxShift = 10) {
  let best = 0, bestErr = Infinity;
  for (let s = 0; s <= maxShift; s++) {
    let sum = 0, n = 0;
    for (let i = 0; i < ours.length; i++) {
      if (!mask[i] || !ours[i] || !gt[i - s]) continue;
      sum += Math.hypot(ours[i][0] - gt[i - s][0], ours[i][1] - gt[i - s][1]);
      n++;
    }
    if (n && sum / n < bestErr) { bestErr = sum / n; best = s; }
  }
  return (best / fps) * 1000;
}

// Frames where the GT mouth centre jumps more than `widths` mouth widths from the previous GT frame (a cut or a jerk).
export function jumpEvents(gt, widths = 1) {
  const out = [];
  let prev = null;
  gt.forEach((g, i) => {
    if (!g) return;
    if (prev) {
      const [ax, ay] = centre(prev), [bx, by] = centre(g);
      if (Math.hypot(bx - ax, by - ay) > widths * g[2]) out.push(i);
    }
    prev = g;
  });
  return out;
}

// For each event frame: ms until the centre error is under `tol` for `hold` frames in a row (Infinity if never).
// A frame without an error (no GT) never counts as recovered.
export function recoveryMs(errors, events, fps, { tol = 0.5, hold = 3 } = {}) {
  return events.map((e) => {
    for (let j = e; j + hold <= errors.length; j++) {
      let ok = true;
      for (let k = j; k < j + hold; k++) if (!(errors[k] != null && errors[k] < tol)) { ok = false; break; }
      if (ok) return ((j - e) / fps) * 1000;
    }
    return Infinity;
  });
}

// Quality-gate confusion counts: flagged[i] (any tracker flag) vs bad[i] (occluded/blurred label; null = unlabeled).
export function gatePR(flagged, bad) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  bad.forEach((b, i) => {
    if (b == null) return;
    if (flagged[i]) b ? tp++ : fp++;
    else b ? fn++ : tn++;
  });
  return { tp, fp, fn, tn, precision: tp + fp ? tp / (tp + fp) : NaN, recall: tp + fn ? tp / (tp + fn) : NaN };
}

// Is box a entirely within box b?
export const inside = (a, b) => a[0] >= b[0] && a[1] >= b[1] && a[0] + a[2] <= b[0] + b[2] && a[1] + a[3] <= b[1] + b[3];

// Share of box a's area that lies within box b (how much of the GT mouth a crop keeps).
export function areaInside(a, b) {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  return (ix * iy) / (a[2] * a[3]);
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// Per-coordinate median of the boxes (numpy's median, as crop.py takes it), skipping missing frames; null if none.
export function medianBox(boxes) {
  const bs = boxes.filter(Boolean);
  return bs.length ? [0, 1, 2, 3].map((k) => median(bs.map((b) => b[k]))) : null;
}

export const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
export const quantile = (xs, q) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
