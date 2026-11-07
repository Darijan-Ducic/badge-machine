// Mouth-crop tracker. SAM (paid, injected as `samFace`) re-anchors on a few keyframes; in between, a
// classical tracker follows a RIGID anchor — the nose-tip-to-chin patch of the face box — with
// coarse-to-fine normalized cross-correlation (NCC) over position and scale. The mouth crop sits at a
// fixed offset from the anchor and is Kalman-smoothed so jitter never reaches the lip-reading models.
//
// Pure ES module, no DOM: the browser and Node import it unchanged.
// Frame: grayscale {data: Uint8Array|Uint8ClampedArray (width*height), width, height, t (seconds)}.
// Box: [x, y, w, h] in frame pixels, top-left origin. Emitted boxes are integers; internal math is float.

// Every threshold, ratio and smoothing parameter — tune these offline (tools/replay.mjs --sweep).
export const CONFIG = Object.freeze({
  // re-anchor triggers
  T_CONF: 0.5, // NCC peak below this = tracker lost
  T_MOTION: 0.3, // fraction of the frame that changed = heavy motion
  T_OUTSIDE: 0.1, // fraction of the area outside the (padded) face that changed = new face / motion outside
  OUTSIDE_PAD: 0.5, // face box grows by this fraction of its size on every side before "outside" is measured
  T_SCALE: 0.15, // anchor size change since its keyframe (relative)
  T_REFRESH: 2, // s; drift insurance
  SAM_MIN_GAP: 0.25, // s; min time between triggered SAM calls (cost cap; forced calls ignore it)
  MOTION_PIX: 16, // gray levels; a quarter-res pixel changing more than this counts as motion
  // occlusion + quality gate
  T_STILL: 0.1, // conf < T_CONF while motion < T_STILL = 'occluded?'
  OCC_IOU: 0.3, // SAM box overlapping the held anchor this much confirms the occlusion (clean template is kept)
  T_BAD: 0.2, // flagged-frame ratio above this = ask the user to repeat (before paying for Gemini/Muse)
  // NCC search
  SEARCH_R: 16, // px; largest per-frame anchor displacement searched
  SCALE_STEP: 1.04, // scale ratio tried on each side of the current scale (finest level uses its square root)
  TPL_MAX_PX: 12000, // the finest search level is the first whose template fits this many pixels (caps cost)
  // geometry, as fractions of a SAM face box (which includes the ears and runs ~13% below the chin).
  // Mouth: SAM "lips" on a 269-frame talking head measured (cx .56, cy .74, w .30, h .14); 9 stills
  // (.46, .67, .32, .11). cx uses the frontal-symmetric .5 (the two sources straddle it); the rest
  // use the talking-head values. Anchor: nose tip (~.63) to just below the chin (~.87), ~1.5 mouth widths.
  ANCHOR_X: 0.28, ANCHOR_Y: 0.63, ANCHOR_W: 0.44, ANCHOR_H: 0.27,
  MOUTH_CX: 0.5, MOUTH_CY: 0.74, MOUTH_W: 0.3, MOUTH_H: 0.14,
  CROP_PAD: 1.6, // crop = mouth box scaled by this around its center
  // smoothing (One-Euro on crop center and size)
  SMOOTH_MIN_CUTOFF: 0.5, // Hz at rest; lower = steadier, laggier
  SMOOTH_BETA: 3, // cutoff gain per (crop width / s) of speed; higher = less lag when moving
  SMOOTH_D_CUTOFF: 1, // Hz; smoothing of the speed estimate
  SMOOTH_RESET: 0.5, // a jump of more than this many crop widths (re-anchor after loss) is taken as-is
});

const EPS = 1e-6;
const MIN_TPL = 12; // px; coarsest template side

export function withConfig(overrides = {}) {
  for (const k of Object.keys(overrides)) if (!(k in CONFIG)) throw new Error(`unknown tracker config key: ${k}`);
  return { ...CONFIG, ...overrides };
}

// ---------- geometry ----------
export const center = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
export const roundBox = (b) => b.map(Math.round);

export function anchorFromFace([x, y, w, h], cfg = CONFIG) {
  return [x + cfg.ANCHOR_X * w, y + cfg.ANCHOR_Y * h, cfg.ANCHOR_W * w, cfg.ANCHOR_H * h];
}

export function faceFromAnchor([x, y, w, h], cfg = CONFIG) {
  const fw = w / cfg.ANCHOR_W, fh = h / cfg.ANCHOR_H;
  return [x - cfg.ANCHOR_X * fw, y - cfg.ANCHOR_Y * fh, fw, fh];
}

export function mouthFromAnchor(anchor, cfg = CONFIG) {
  const [x, y, w, h] = faceFromAnchor(anchor, cfg);
  const mw = cfg.MOUTH_W * w * cfg.CROP_PAD, mh = cfg.MOUTH_H * h * cfg.CROP_PAD;
  return [x + cfg.MOUTH_CX * w - mw / 2, y + cfg.MOUTH_CY * h - mh / 2, mw, mh];
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter);
}

// Relative change in linear size between two boxes.
export function scaleJump(box, ref) {
  return Math.abs(Math.sqrt((box[2] * box[3]) / (ref[2] * ref[3])) - 1);
}

// The enrolled user among SAM's faces: nearest to where they were, else (first keyframe) the largest.
export function pickFace(faces, prevFace) {
  if (!faces?.length) return null;
  if (prevFace) {
    const [px, py] = center(prevFace);
    let best = null, bd = Infinity;
    for (const f of faces) {
      const [cx, cy] = center(f), d = Math.hypot(cx - px, cy - py);
      if (d < bd) { bd = d; best = f; }
    }
    return best;
  }
  return faces.reduce((a, b) => (b[2] * b[3] > a[2] * a[3] ? b : a));
}

const grow = ([x, y, w, h], p) => [x - p * w, y - p * h, w * (1 + 2 * p), h * (1 + 2 * p)];

// ---------- image pyramid (memoized per frame object) ----------
const PYRAMIDS = new WeakMap();

export function pyramid(frame) {
  let pyr = PYRAMIDS.get(frame);
  if (pyr) return pyr;
  const { width: w, height: h } = frame;
  const base = new Float32Array(w * h);
  base.set(frame.data);
  pyr = [{ data: base, w, h }];
  while (pyr.length < 5) {
    const p = pyr.at(-1), nw = p.w >> 1, nh = p.h >> 1;
    if (nw < 16 || nh < 16) break;
    const d = new Float32Array(nw * nh), s = p.data, sw = p.w;
    for (let y = 0; y < nh; y++) {
      let o = 2 * y * sw, k = y * nw;
      for (let x = 0; x < nw; x++, o += 2, k++) d[k] = (s[o] + s[o + 1] + s[o + sw] + s[o + sw + 1]) * 0.25;
    }
    pyr.push({ data: d, w: nw, h: nh });
  }
  PYRAMIDS.set(frame, pyr);
  return pyr;
}

// ---------- template ----------
// The anchor patch cut from each pyramid level it will be searched at.
export function cutAnchor(frame, anchor, cfg = CONFIG) {
  const pyr = pyramid(frame);
  const [ax, ay, aw, ah] = anchor;
  let fine = 0;
  while (fine < pyr.length - 1 && (aw * ah) / 4 ** fine > cfg.TPL_MAX_PX) fine++;
  let coarse = fine;
  while (coarse < fine + 2 && coarse < pyr.length - 1 && Math.min(aw, ah) / 2 ** (coarse + 1) >= MIN_TPL) coarse++;
  const levels = [];
  for (let L = fine; L <= coarse; L++) {
    const f = 2 ** L, img = pyr[L];
    const x0 = Math.min(img.w - 4, Math.max(0, Math.round(ax / f))), y0 = Math.min(img.h - 4, Math.max(0, Math.round(ay / f)));
    const x1 = Math.max(x0 + 4, Math.min(img.w, Math.round((ax + aw) / f))), y1 = Math.max(y0 + 4, Math.min(img.h, Math.round((ay + ah) / f)));
    const w = x1 - x0, h = y1 - y0, data = new Float32Array(w * h);
    for (let j = 0; j < h; j++) data.set(img.data.subarray((y0 + j) * img.w + x0, (y0 + j) * img.w + x1), j * w);
    // anchor center in template pixel coords (pixel i spans [i, i+1))
    levels[L] = { data, w, h, ocx: (ax + aw / 2) / f - x0, ocy: (ay + ah / 2) / f - y0, cache: new Map() };
  }
  return { box: anchor.slice(), fine, coarse, levels };
}

// The level-L template resampled by exactly scale s (integer size only sets coverage), zero-mean,
// with its anchor center in resampled coords.
function scaled(tpl, L, s) {
  const T = tpl.levels[L], key = Math.round(s * 1e4);
  let R = T.cache.get(key);
  if (R) return R;
  if (T.cache.size > 64) T.cache.clear();
  const w = Math.max(4, Math.floor(T.w * s)), h = Math.max(4, Math.floor(T.h * s));
  const sx = s, sy = s, z = new Float32Array(w * h), src = T.data;
  let sum = 0;
  for (let j = 0; j < h; j++) {
    const v = Math.min(T.h - 1, Math.max(0, (j + 0.5) / sy - 0.5));
    const j0 = Math.min(T.h - 2, Math.floor(v)), fv = v - j0;
    for (let i = 0; i < w; i++) {
      const u = Math.min(T.w - 1, Math.max(0, (i + 0.5) / sx - 0.5));
      const i0 = Math.min(T.w - 2, Math.floor(u)), fu = u - i0, k = j0 * T.w + i0;
      const val = (src[k] * (1 - fu) + src[k + 1] * fu) * (1 - fv) + (src[k + T.w] * (1 - fu) + src[k + T.w + 1] * fu) * fv;
      z[j * w + i] = val;
      sum += val;
    }
  }
  const mean = sum / (w * h);
  let ss = 0;
  for (let k = 0; k < z.length; k++) { z[k] -= mean; ss += z[k] * z[k]; }
  R = { z, w, h, n: w * h, norm: Math.sqrt(ss), cx: T.ocx * sx, cy: T.ocy * sy, s: Math.sqrt(sx * sy) };
  T.cache.set(key, R);
  return R;
}

// NCC of template R placed with its top-left at integer (x, y) of image img; -2 if it does not fit.
function ncc(img, x, y, R) {
  if (x < 0 || y < 0 || x + R.w > img.w || y + R.h > img.h) return -2;
  const d = img.data, z = R.z, iw = img.w, tw = R.w;
  let sI = 0, sII = 0, sIT = 0;
  for (let j = 0; j < R.h; j++) {
    let o = (y + j) * iw + x, k = j * tw;
    for (let i = 0; i < tw; i++, o++, k++) {
      const v = d[o];
      sI += v;
      sII += v * v;
      sIT += v * z[k];
    }
  }
  const varI = sII - (sI * sI) / R.n;
  if (varI <= EPS || R.norm <= EPS) return 0;
  return sIT / (Math.sqrt(varI) * R.norm);
}

function bestInWindow(img, R, x0, y0, r) {
  let bx = x0, by = y0, best = -Infinity;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const sc = ncc(img, x0 + dx, y0 + dy, R);
      if (sc > best) { best = sc; bx = x0 + dx; by = y0 + dy; }
    }
  }
  return { x: bx, y: by, score: best };
}

// Hill-climb to a local NCC max; returns its 3x3 neighborhood for sub-pixel refinement.
function climb(img, R, x, y) {
  const memo = new Map(), at = (px, py) => {
    const k = py * 100000 + px;
    if (!memo.has(k)) memo.set(k, ncc(img, px, py, R));
    return memo.get(k);
  };
  for (let it = 0; it < 6; it++) {
    let bx = 0, by = 0, best = at(x, y);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const v = at(x + dx, y + dy); if (v > best) { best = v; bx = dx; by = dy; } }
    if (!bx && !by) break;
    x += bx; y += by;
  }
  return { x, y, score: at(x, y), l: at(x - 1, y), r: at(x + 1, y), u: at(x, y - 1), d: at(x, y + 1) };
}

// Vertex offset of a parabola through (-1, a), (0, b), (1, c), clamped to half a step.
function vertex(a, b, c) {
  const den = a - 2 * b + c;
  if (a < -1 || c < -1 || den >= 0) return 0;
  return Math.max(-0.5, Math.min(0.5, (a - c) / (2 * den)));
}

// One tracking step: where the template's anchor is in `frame`, searched around prevBox.
// Returns {box (float anchor box), conf (NCC peak, -1..1)}.
export function trackAnchor(frame, tpl, prevBox, cfg = CONFIG) {
  const pyr = pyramid(frame);
  let [cx, cy] = center(prevBox);
  let s = prevBox[2] / tpl.box[2];
  const step = cfg.SCALE_STEP;
  // coarse to fine: exhaustive window at the coarsest level, +-2 px below it, 3 scales each
  for (let L = tpl.coarse; L >= Math.min(tpl.coarse, tpl.fine + 1); L--) {
    const f = 2 ** L, img = pyr[L], r = L === tpl.coarse ? Math.max(2, Math.ceil(cfg.SEARCH_R / f)) : 2;
    let best = null;
    for (const sc of [s / step, s, s * step]) {
      const R = scaled(tpl, L, sc);
      const b = bestInWindow(img, R, Math.round(cx / f - R.cx), Math.round(cy / f - R.cy), r);
      if (!best || b.score > best.score) best = { ...b, R, sc };
    }
    cx = (best.x + best.R.cx) * f;
    cy = (best.y + best.R.cy) * f;
    s = best.sc;
  }
  // finest level: hill-climb in scale (steps of sqrt(SCALE_STEP)) until the NCC peak is bracketed,
  // each scale at its own local position max; then sub-pixel position and parabolic scale
  const f = 2 ** tpl.fine, img = pyr[tpl.fine], q = Math.sqrt(step);
  const at = (sc) => {
    const R = scaled(tpl, tpl.fine, sc);
    return { sc, R, ...climb(img, R, Math.round(cx / f - R.cx), Math.round(cy / f - R.cy)) };
  };
  const cands = [at(s / q), at(s), at(s * q)];
  for (let it = 0; it < 4; it++) {
    if (cands[0].score > cands[1].score && cands[0].score > cands[2].score) cands.unshift(at(cands[0].sc / q)) && cands.pop();
    else if (cands[2].score > cands[1].score) cands.push(at(cands[2].sc * q)) && cands.shift();
    else break;
  }
  const k = cands[1].score >= cands[0].score && cands[1].score >= cands[2].score ? 1 : cands[0].score > cands[2].score ? 0 : 2;
  const c = cands[k];
  if (c.score <= -1) return { box: prevBox.slice(), conf: 0 };
  const px = (c.x + vertex(c.l, c.score, c.r) + c.R.cx) * f, py = (c.y + vertex(c.u, c.score, c.d) + c.R.cy) * f;
  let scale = c.R.s;
  if (k === 1) {
    const [a, b, d] = cands.map((o) => Math.log(o.R.s)), t = vertex(cands[0].score, c.score, cands[2].score);
    scale = Math.exp(t < 0 ? b + t * (b - a) : b + t * (d - b));
  }
  const w = tpl.box[2] * scale, h = tpl.box[3] * scale;
  return { box: [px - w / 2, py - h / 2, w, h], conf: c.score };
}

// Fraction of (quarter-res) pixels that changed between frames; with excludeBox, only those outside it.
export function frameMotion(frame, prevFrame, excludeBox = null, cfg = CONFIG) {
  if (!prevFrame) return 0;
  const pa = pyramid(frame), pb = pyramid(prevFrame);
  let L = 0;
  while (L < pa.length - 1 && pa[L].w > 200) L++;
  const a = pa[L], b = pb[L];
  if (!b || a.w !== b.w || a.h !== b.h) return 1;
  const f = 2 ** L;
  let ex0 = 0, ey0 = 0, ex1 = 0, ey1 = 0;
  if (excludeBox) {
    ex0 = Math.floor(excludeBox[0] / f); ey0 = Math.floor(excludeBox[1] / f);
    ex1 = Math.ceil((excludeBox[0] + excludeBox[2]) / f); ey1 = Math.ceil((excludeBox[1] + excludeBox[3]) / f);
  }
  let changed = 0, total = 0;
  const da = a.data, db = b.data, thr = cfg.MOTION_PIX;
  for (let y = 0; y < a.h; y++) {
    const inRows = y >= ey0 && y < ey1;
    for (let x = 0, k = y * a.w; x < a.w; x++, k++) {
      if (inRows && x >= ex0 && x < ex1) continue;
      total++;
      if (Math.abs(da[k] - db[k]) > thr) changed++;
    }
  }
  return total ? changed / total : 0;
}

// ---------- smoothing ----------
// One-Euro filter (Casiez et al., CHI 2012) on crop center x, y, width, height: an exponential smoother
// whose cutoff rises with speed, so a still head gets heavy smoothing and a moving one little lag.
// Chosen over a constant-velocity Kalman, which at equal lag kept more jitter and overshot re-anchor steps.
// Speeds are measured in crop widths per second, so the knobs do not depend on face size.
export function createSmoother(cfg = CONFIG) {
  let s = null, ds = null, lastT = 0;
  const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
  return {
    update(box, t) {
      const z = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]];
      if (s && Math.hypot(z[0] - s[0], z[1] - s[1]) > cfg.SMOOTH_RESET * s[2]) s = null; // re-anchor jump
      if (!s) {
        s = z;
        ds = [0, 0, 0, 0];
      } else {
        const dt = Math.max(1e-3, t - lastT), ad = alpha(cfg.SMOOTH_D_CUTOFF, dt), w = s[2];
        for (let i = 0; i < 4; i++) {
          ds[i] += ad * ((z[i] - s[i]) / dt - ds[i]);
          s[i] += alpha(cfg.SMOOTH_MIN_CUTOFF + (cfg.SMOOTH_BETA * Math.abs(ds[i])) / w, dt) * (z[i] - s[i]);
        }
      }
      lastT = t;
      return [s[0] - s[2] / 2, s[1] - s[3] / 2, s[2], s[3]];
    },
    reset() { s = null; },
  };
}

// ---------- quality gate ----------
export const badFrameRatio = (flagsPerFrame) =>
  flagsPerFrame.length ? flagsPerFrame.filter((f) => f.length > 0).length / flagsPerFrame.length : 0;

export function qualityGate(flagsPerFrame, cfg = CONFIG) {
  const ratio = badFrameRatio(flagsPerFrame);
  return { ok: ratio <= cfg.T_BAD, ratio };
}

// ---------- the per-frame loop ----------
// samFace(frame) -> Promise<Array<[x,y,w,h]>> (every face SAM found). step() returns, per frame:
// {t, conf, motion, outside, anchor, face, crop (smoothed mouth crop), rawCrop (float, unsmoothed),
//  sam: null | {reasons, faces, error?}, flags: [] | ['occluded?'] | ['lost'] | ['no-face']}.
export function createTracker({ samFace, config = {} } = {}) {
  const cfg = withConfig(config);
  const smoother = createSmoother(cfg);
  let tpl = null, anchor = null, prevFrame = null, lastAnchorT = -Infinity, lastSamT = -Infinity, samCalls = 0;

  async function step(frame, { force = false } = {}) {
    const t = frame.t, tracking = !!tpl;
    let conf = 0, tracked = null;
    if (tracking) ({ box: tracked, conf } = trackAnchor(frame, tpl, anchor, cfg));
    const motion = frameMotion(frame, prevFrame, null, cfg);
    const outside = anchor ? frameMotion(frame, prevFrame, grow(faceFromAnchor(anchor, cfg), cfg.OUTSIDE_PAD), cfg) : 0;
    const lost = conf < cfg.T_CONF, held = anchor;
    if (tracking && !lost) anchor = tracked;

    const reasons = [];
    if (!tracking) reasons.push('init');
    else {
      if (lost) reasons.push('lost');
      if (motion > cfg.T_MOTION) reasons.push('motion');
      if (outside > cfg.T_OUTSIDE) reasons.push('outside');
      if (scaleJump(anchor, tpl.box) > cfg.T_SCALE) reasons.push('scale');
      if (t - lastAnchorT >= cfg.T_REFRESH - EPS) reasons.push('refresh');
    }
    if (force) reasons.push('forced');

    let occluded = tracking && lost && motion < cfg.T_STILL, fixed = false, sam = null;
    const flags = [];
    if (reasons.length && (force || t - lastSamT >= cfg.SAM_MIN_GAP - EPS)) {
      samCalls++;
      lastSamT = t;
      sam = { reasons, faces: 0 };
      let faces = null;
      try { faces = await samFace(frame); } catch (e) { sam.error = String(e?.message ?? e); }
      if (faces) {
        sam.faces = faces.length;
        const face = pickFace(faces, held && faceFromAnchor(held, cfg));
        if (face) {
          const a = anchorFromFace(face, cfg);
          // SAM sees the face where we held it, yet NCC failed: a real occlusion, so keep the clean
          // template (later occluded frames keep scoring low and stay flagged). Otherwise re-cut it.
          if (!(occluded && iou(a, held) >= cfg.OCC_IOU)) { occluded = false; tpl = cutAnchor(frame, a, cfg); }
          anchor = a;
          lastAnchorT = t;
          fixed = true;
        } else flags.push('no-face');
      }
    }
    if (occluded) flags.push('occluded?');
    else if (tracking && lost && !fixed) flags.push('lost');
    if (!anchor && !flags.length) flags.push('no-face');

    let crop = null, rawCrop = null;
    if (anchor) {
      rawCrop = mouthFromAnchor(anchor, cfg);
      crop = roundBox(smoother.update(rawCrop, t));
    }
    prevFrame = frame;
    return {
      t, conf, motion, outside, sam, flags, crop, rawCrop,
      anchor: anchor && roundBox(anchor), face: anchor && roundBox(faceFromAnchor(anchor, cfg)),
    };
  }

  return { step, config: cfg, get samCalls() { return samCalls; } };
}

// Runs a whole clip (iterable or async iterable of frames) and applies the quality gate.
export async function trackClip(frames, samFace, config = {}) {
  const tracker = createTracker({ samFace, config });
  const results = [];
  for await (const f of frames) results.push(await tracker.step(f));
  return { results, samCalls: tracker.samCalls, gate: qualityGate(results.map((r) => r.flags), tracker.config) };
}
