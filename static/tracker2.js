// Mouth tracker v2: lips-anchored and lip-aware. SAM keyframes ask for the LIPS (not the face), so the mouth box
// sits where the lips really are, whatever the head pose. Between keyframes v1's rigid nose-to-chin NCC anchor
// (lips masked) carries the box, and two per-session models learned from SAM's lips box correct it every frame:
// where the mouth sits on the anchor (re-learned at each keyframe) and the lips' own colours (a lips-vs-skin
// histogram, back-projected around the mouth), which follow the lips through head turns, show the mouth opening,
// and notice a hand over the mouth. v1 (static/tracker.js) is untouched and still selectable; v2 reuses its
// image machinery.
//
// Pure ES module, no DOM: the browser and Node import it unchanged.
// Frame: {data: gray Uint8(Clamped)Array (w*h), width, height, t (s), rgba?: Uint8ClampedArray (w*h*4)}.
// Without rgba the colour model is off and v2 runs on geometry alone.
// sam(frame, prompt) -> Promise<Array<[x, y, w, h]>> for prompt 'lips' (keyframes) or 'face' (when no lips show).
import {
  CONFIG as V1, center, roundBox, anchorFromFace, faceFromAnchor, iou, scaleJump, pickFace, cutAnchor, trackAnchor,
  frameMotion, createSmoother, qualityGate,
} from './tracker.js';

export const VERSION = 'v2';

// v1's knobs (image search, smoothing, triggers) plus v2's own; tune offline with tools/tracker_eval.mjs on the dev split.
export const CONFIG = Object.freeze({
  ...V1,
  SAM_PROMPT: 'lips', // what a keyframe asks SAM for (image mode found lips on 60/60 sampled 640x480 frames)
  FACE_ASPECT: 1.12, // height / width of a SAM face box: builds the anchor (via a virtual face) around a lips box
  CROP_ASPECT: 0.52, // crop height / width: v1's framing (MOUTH_H * FACE_ASPECT / MOUTH_W), so the readers see the same shape
  OFFSET_ALPHA: 0.5, // weight of a new keyframe in the learned mouth offset/size (the rest is the running average)
  SNAP_MOUTH: 0.35, // mouth widths: a confident track whose mouth is this close to SAM's lips keeps its anchor
  PREDICT: 1, // 1 = search where constant velocity puts the anchor; 0 = where it was
  VEL_ALPHA: 0.5, // weight of the newest frame in the velocity estimate
  SMOOTH_BETA: 10, // One-Euro speed gain (v1: 3); less lag at speed, the server crop is a median anyway
  // lip colour model (needs frame.rgba)
  COLOR: 1, // 1 = measure the lips every frame and pull the mouth onto them; 0 = geometry only
  COLOR_ALPHA: 0.5, // weight of a new keyframe's histograms in the session colour model
  COLOR_GAIN: 0.5, // share of the gap to the measured lips closed per frame (scaled by the measurement's confidence)
  COLOR_GATE: 0.6, // mouth widths: a lips measurement farther than this from the anchor's mouth is ignored
  COLOR_Y: 1, // multiplier of COLOR_GAIN for the vertical pull (1 = same as horizontal)
  COLOR_H: 0, // multiplier of COLOR_GAIN for following the lips' height (dev: 1 raised size error 0.08 -> 0.40; dropped)
  MASS_LO: 0.4, MASS_HI: 0.7, // lip mass / keyframe mass: confidence ramps from 0 at LO to 1 at HI
  HIDDEN_MASS: 0.3, // lip mass / keyframe mass below this = lips not visible: ask SAM whether the mouth is covered
  BLUR_RATIO: 0.55, // mouth-region sharpness / its running level below this = 'blurred?'
});

export function withConfig(overrides = {}) {
  for (const k of Object.keys(overrides)) if (!(k in CONFIG)) throw new Error(`unknown tracker2 config key: ${k}`);
  return { ...CONFIG, ...overrides };
}

// The anchor whose implied mouth (v1 ratios) is centred on a lips box: v1's anchor around a virtual face that is
// MOUTH_W wide per lips width, FACE_ASPECT tall, with its mouth point (MOUTH_CX, MOUTH_CY) on the lips centre.
export function anchorFromLips([x, y, w, h], cfg = CONFIG) {
  const fw = w / cfg.MOUTH_W, fh = cfg.FACE_ASPECT * fw;
  return anchorFromFace([x + w / 2 - cfg.MOUTH_CX * fw, y + h / 2 - cfg.MOUTH_CY * fh, fw, fh], cfg);
}

// The mouth point an anchor implies with v1's ratios (before the learned offset).
function mouthPoint(anchor, cfg) {
  const [fx, fy, fw, fh] = faceFromAnchor(anchor, cfg);
  return [fx + cfg.MOUTH_CX * fw, fy + cfg.MOUTH_CY * fh];
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// The user's lips among SAM's: nearest the expected mouth, else (no expectation) the largest (the nearest person).
function pickLips(boxes, expected) {
  if (!boxes?.length) return null;
  if (expected) return boxes.reduce((a, b) => (dist(center(b), expected) < dist(center(a), expected) ? b : a));
  return boxes.reduce((a, b) => (b[2] * b[3] > a[2] * a[3] ? b : a));
}

// ---------- lip colour model ----------
// Bins: normalized chromaticity r = R/(R+G+B), g = G/(R+G+B) over the range skin and lips occupy, times 4 brightness
// levels (teeth, lips, shadowed mouth interior differ mostly in brightness).
const RB = 32, GB = 32, YB = 4, R0 = 0.2, R1 = 0.75, G0 = 0.12, G1 = 0.45, NB = RB * GB * YB;
function binOf(px, p) {
  const r = px[p], g = px[p + 1], b = px[p + 2], s = r + g + b + 1;
  const ri = clamp(Math.floor(((r / s - R0) / (R1 - R0)) * RB), 0, RB - 1), gi = clamp(Math.floor(((g / s - G0) / (G1 - G0)) * GB), 0, GB - 1);
  return (ri * GB + gi) * YB + Math.min(YB - 1, (r * 77 + g * 150 + b * 29) >> 14);
}

// The search window around a mouth box: 2.4 widths by 3.2 heights (cheeks, philtrum, chin), clipped to the frame.
function windowOf([x, y, w, h], W, H, gx = 2.4, gy = 3.2) {
  const cx = x + w / 2, cy = y + h / 2;
  const x0 = clamp(Math.floor(cx - (gx * w) / 2), 0, W - 1), y0 = clamp(Math.floor(cy - (gy * h) / 2), 0, H - 1);
  return [x0, y0, clamp(Math.ceil(cx + (gx * w) / 2), x0 + 1, W), clamp(Math.ceil(cy + (gy * h) / 2), y0 + 1, H)];
}

// Lip-ness per pixel of a window: 2 P(lips | colour) - 1, floored at 0, with equal priors (normalized histograms).
function lipWeights(frame, model, [x0, y0, x1, y1]) {
  const { rgba, width: W } = frame, out = new Float32Array((x1 - x0) * (y1 - y0)), lut = model.lut;
  for (let y = y0, k = 0; y < y1; y++) for (let x = x0, p = (y * W + x0) * 4; x < x1; x++, p += 4, k++) out[k] = lut[binOf(rgba, p)];
  return out;
}

// Mean shift of the lip-ness inside a box kernel (half the guess size plus 30%) from the guess centre:
// the centroid, the weighted spread (sx, sy) and the mass of lip-ness in the converged kernel.
function blobIn(weights, [x0, y0, x1, y1], guess) {
  const ww = x1 - x0, hx = 0.8 * guess[2], hy = 0.9 * guess[3];
  let [cx, cy] = center(guess), m = 0, sxx = 0, syy = 0;
  for (let it = 0; it < 6; it++) {
    const xa = Math.max(x0, Math.floor(cx - hx)), xb = Math.min(x1, Math.ceil(cx + hx));
    const ya = Math.max(y0, Math.floor(cy - hy)), yb = Math.min(y1, Math.ceil(cy + hy));
    let s = 0, sx = 0, sy = 0, qx = 0, qy = 0;
    for (let y = ya; y < yb; y++) {
      for (let x = xa, k = (y - y0) * ww + (xa - x0); x < xb; x++, k++) {
        const v = weights[k];
        if (!v) continue;
        const px = x + 0.5, py = y + 0.5;
        s += v; sx += v * px; sy += v * py; qx += v * px * px; qy += v * py * py;
      }
    }
    if (s <= 1e-6) return { cx, cy, sx: 0, sy: 0, mass: 0 };
    const nx = sx / s, ny = sy / s, moved = Math.hypot(nx - cx, ny - cy);
    cx = nx; cy = ny; m = s;
    sxx = Math.max(0, qx / s - nx * nx); syy = Math.max(0, qy / s - ny * ny);
    if (moved < 0.25) break;
  }
  return { cx, cy, sx: Math.sqrt(sxx), sy: Math.sqrt(syy), mass: m };
}

// Learns (or, given `prev`, updates by `alpha`) the lips-vs-surroundings colour model from a SAM lips box:
// lips = the ellipse inscribed in the box, surroundings = its search window minus the box grown by 25%.
// Also calibrates the blob measurement on this frame: lips box = blob centre + (dx, dy), size (kx sx, ky sy).
export function learnLips(frame, lips, prev = null, alpha = 1) {
  const { rgba, width: W, height: H } = frame, [lx, ly, lw, lh] = lips, cx = lx + lw / 2, cy = ly + lh / 2;
  const win = windowOf(lips, W, H), fg = new Float32Array(NB), bg = new Float32Array(NB);
  let nf = 0, nb = 0;
  for (let y = win[1]; y < win[3]; y++) {
    for (let x = win[0], p = (y * W + win[0]) * 4; x < win[2]; x++, p += 4) {
      const ex = (x + 0.5 - cx) / (lw / 2), ey = (y + 0.5 - cy) / (lh / 2);
      if (ex * ex + ey * ey < 1) { fg[binOf(rgba, p)]++; nf++; }
      else if (Math.abs(ex) > 1.25 || Math.abs(ey) > 1.25) { bg[binOf(rgba, p)]++; nb++; }
    }
  }
  const model = { fg: new Float32Array(NB), bg: new Float32Array(NB), lut: new Float32Array(NB) };
  for (let i = 0; i < NB; i++) {
    const f = nf ? fg[i] / nf : 0, b = nb ? bg[i] / nb : 0;
    model.fg[i] = prev ? prev.fg[i] + alpha * (f - prev.fg[i]) : f;
    model.bg[i] = prev ? prev.bg[i] + alpha * (b - prev.bg[i]) : b;
    const pf = model.fg[i], pb = model.bg[i];
    model.lut[i] = pf + pb > 0 ? Math.max(0, (2 * pf) / (pf + pb + 1e-3 / NB) - 1) : 0;
  }
  const blob = lipBlob(frame, model, lips);
  Object.assign(model, {
    mass: blob.mass, dx: (cx - blob.cx) / lw, dy: (cy - blob.cy) / lw,
    kx: blob.sx > 0 ? lw / blob.sx : 0, ky: blob.sy > 0 ? lh / blob.sy : 0,
  });
  return model;
}

// Measures the lips near `guess` (a mouth box): {cx, cy, sx, sy, mass}; centre and spread in pixels.
export function lipBlob(frame, model, guess) {
  const win = windowOf(guess, frame.width, frame.height);
  return blobIn(lipWeights(frame, model, win), win, guess);
}

// RMS gray gradient inside box, per axis {x, y}. Motion blur along an axis spreads each edge over the blur length,
// which lowers the RMS gradient along it ~1/sqrt(length) (a mean |gradient| would not move: blur keeps total variation).
export function sharpness(frame, box) {
  const { data, width: W, height: H } = frame;
  const x0 = clamp(Math.floor(box[0]), 1, W - 2), y0 = clamp(Math.floor(box[1]), 1, H - 2);
  const x1 = clamp(Math.ceil(box[0] + box[2]), x0 + 1, W - 1), y1 = clamp(Math.ceil(box[1] + box[3]), y0 + 1, H - 1);
  let sx = 0, sy = 0, n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0, k = y * W + x0; x < x1; x++, k++) {
      const gx = data[k + 1] - data[k - 1], gy = data[k + W] - data[k - W];
      sx += gx * gx; sy += gy * gy; n++;
    }
  }
  return n ? { x: Math.sqrt(sx / n), y: Math.sqrt(sy / n) } : { x: 0, y: 0 };
}

// How sharp s is relative to ref, along its blurrier axis (1 = as sharp; motion blur in any direction lowers it).
export const blurRatio = (s, ref) => Math.min(ref.x > 0 ? s.x / ref.x : 1, ref.y > 0 ? s.y / ref.y : 1);

// ---------- the per-frame loop ----------
export function createTracker({ sam, config = {} } = {}) {
  const cfg = withConfig(config);
  const smoother = createSmoother(cfg);
  // geometry learned at keyframes, in anchor widths: mouth centre = mouthPoint(anchor) + off * aw; size = (rw, rh) * aw
  let tpl = null, anchor = null, off = [0, 0], rw = 0, rh = 0, colour = null, sharpRef = null;
  let prevFrame = null, lastT = 0, vel = [0, 0], lastAnchorT = -Infinity, lastSamT = -Infinity, samCalls = 0;
  let occluded = false, wasHidden = false;

  const mouthOf = (a) => {
    const [px, py] = mouthPoint(a, cfg), aw = a[2], w = rw * aw, h = rh * aw;
    const cx = px + off[0] * aw, cy = py + off[1] * aw;
    return [cx - w / 2, cy - h / 2, w, h];
  };

  function learn(lips, a, fresh) {
    const aw = a[2], [px, py] = mouthPoint(a, cfg), [lx, ly] = center(lips);
    const o = [(lx - px) / aw, (ly - py) / aw], w = lips[2] / aw, h = lips[3] / aw, k = fresh ? 1 : cfg.OFFSET_ALPHA;
    off = [off[0] + k * (o[0] - off[0]), off[1] + k * (o[1] - off[1])];
    rw += k * (w - rw);
    rh += k * (h - rh);
  }

  async function ask(frame, prompt, info) {
    info.prompts.push(prompt);
    try { return await sam(frame, prompt); } catch (e) { info.error = String(e?.message ?? e); return null; }
  }

  // The lips measured by colour near the anchor's mouth, pulled into the learned geometry by its confidence
  // (only on frames worth learning from: not while the match is broken or the mouth is known to be covered).
  function measure(frame, adapt) {
    const pred = mouthOf(anchor), blob = lipBlob(frame, colour, pred), ratio = colour.mass > 0 ? blob.mass / colour.mass : 0;
    const w = pred[2], lips = [blob.cx + colour.dx * w, blob.cy + colour.dy * w];
    const gap = dist(lips, center(pred)) / Math.max(1, w);
    const q = gap > cfg.COLOR_GATE ? 0 : clamp((ratio - cfg.MASS_LO) / (cfg.MASS_HI - cfg.MASS_LO), 0, 1);
    if (adapt && q > 0) {
      const g = cfg.COLOR_GAIN * q, aw = anchor[2], [cx, cy] = center(pred);
      off = [off[0] + (g * (lips[0] - cx)) / aw, off[1] + (cfg.COLOR_Y * g * (lips[1] - cy)) / aw];
      if (blob.sy > 0) rh += cfg.COLOR_H * g * ((colour.ky * blob.sy) / aw - rh);
    }
    return { ratio, q };
  }

  async function step(frame, { force = false } = {}) {
    const t = frame.t, tracking = !!tpl, dt = Math.max(1e-3, t - lastT), useColour = cfg.COLOR && !!frame.rgba;
    let conf = 0, tracked = null;
    if (tracking) {
      const guess = cfg.PREDICT ? [anchor[0] + vel[0] * dt, anchor[1] + vel[1] * dt, anchor[2], anchor[3]] : anchor;
      ({ box: tracked, conf } = trackAnchor(frame, tpl, guess, cfg));
    }
    const motion = frameMotion(frame, prevFrame, null, cfg);
    const lost = tracking && conf < cfg.T_CONF, held = anchor;
    if (tracking && !lost) {
      const [ax, ay] = center(anchor), [bx, by] = center(tracked), a = cfg.VEL_ALPHA;
      vel = [vel[0] + a * ((bx - ax) / dt - vel[0]), vel[1] + a * ((by - ay) / dt - vel[1])];
      anchor = tracked;
      tpl.proven = true;
    } else if (lost) vel = [0, 0];
    const seen = tracking && useColour && colour ? measure(frame, !lost && !occluded) : null;
    // lips not visible: the colour model sees too little lip, or the match broke
    const hidden = lost || (!!seen && seen.ratio < cfg.HIDDEN_MASS);

    const reasons = [];
    if (!tracking) reasons.push('init');
    else {
      if (lost) reasons.push('lost');
      if (seen && hidden && !occluded) reasons.push('hidden');
      if (motion > cfg.T_MOTION) reasons.push('motion');
      if (scaleJump(anchor, tpl.box) > cfg.T_SCALE) reasons.push('scale');
      if (t - lastAnchorT >= cfg.T_REFRESH - 1e-6) reasons.push('refresh');
      if (occluded) reasons.push('occluded');
    }
    if (force) reasons.push('forced');
    // an occlusion that hid the lips (or broke the match) and now no longer does: look again at once
    const uncovered = occluded && wasHidden && !hidden;
    wasHidden = hidden;

    let fixed = false, sam = null;
    const flags = [];
    if (reasons.length && (force || uncovered || t - lastSamT >= cfg.SAM_MIN_GAP - 1e-6)) {
      samCalls++;
      lastSamT = t;
      sam = { reasons, prompts: [], faces: 0 };
      const expected = tracking ? center(mouthOf(anchor)) : null;
      const lipsAll = await ask(frame, cfg.SAM_PROMPT, sam);
      const lips = pickLips(lipsAll, expected);
      if (lips) {
        sam.faces = lipsAll.length;
        occluded = false;
        const near = tracking && !lost && dist(center(lips), expected) <= cfg.SNAP_MOUTH * Math.max(1, rw * anchor[2]);
        if (near) {
          // a healthy track that agrees with SAM: keep the anchor, refresh its look, re-learn where the lips sit on it
          learn(lips, anchor, false);
          tpl = cutAnchor(frame, anchor, cfg);
        } else {
          anchor = anchorFromLips(lips, cfg);
          off = [0, 0];
          learn(lips, anchor, true);
          tpl = cutAnchor(frame, anchor, cfg);
          vel = [0, 0];
        }
        if (useColour) colour = learnLips(frame, lips, colour, colour ? cfg.COLOR_ALPHA : 1);
        lastAnchorT = t;
        fixed = true;
      } else if (lipsAll) {
        // no lips: is the face there (mouth covered) or gone?
        samCalls++;
        const faces = await ask(frame, 'face', sam);
        const face = pickFace(faces, held && faceFromAnchor(held, cfg));
        sam.faces = faces?.length ?? 0;
        if (face) {
          const a = anchorFromFace(face, cfg);
          if (!tracking || !tpl.proven || iou(a, held) < cfg.OCC_IOU) {
            // nothing to hold on to (start, or the face moved away): v1's face geometry until the lips show again
            anchor = a;
            off = [0, 0];
            rw = cfg.MOUTH_W / cfg.ANCHOR_W;
            rh = (cfg.MOUTH_H / cfg.ANCHOR_W) * cfg.FACE_ASPECT;
            tpl = cutAnchor(frame, a, cfg);
            vel = [0, 0];
          }
          occluded = tracking; // a known face without visible lips is a covered mouth
          lastAnchorT = t;
          fixed = true;
        } else flags.push('no-face');
      }
    }

    let mouth = null, rawMouth = null, crop = null, rawCrop = null, sharp = null;
    if (anchor) {
      rawMouth = mouthOf(anchor);
      const [cx, cy] = center(rawMouth), cw = cfg.CROP_PAD * rawMouth[2], ch = cw * cfg.CROP_ASPECT;
      rawCrop = [cx - cw / 2, cy - ch / 2, cw, ch];
      const s = smoother.update(rawMouth, t); // smoothed mouth box; the crop is its padded, fixed-aspect frame
      const [sx, sy] = center(s), sw = cfg.CROP_PAD * s[2];
      mouth = s;
      crop = roundBox([sx - sw / 2, sy - (sw * cfg.CROP_ASPECT) / 2, sw, sw * cfg.CROP_ASPECT]);
      sharp = sharpness(frame, rawCrop);
    }
    // motion blur: the mouth region much less sharp than its running level. The level follows clear frames (fast)
    // and blurred ones (slowly, so a lasting change of light or focus re-normalises in a few seconds); never
    // frames where the mouth is covered or the match broke.
    const ratio = sharp && sharpRef ? blurRatio(sharp, sharpRef) : 1, blurred = ratio < cfg.BLUR_RATIO;
    if (sharp && !occluded && !lost) {
      if (!sharpRef) sharpRef = { ...sharp };
      else {
        const k = blurred ? 0.02 : 0.1;
        sharpRef = { x: sharpRef.x + k * (sharp.x - sharpRef.x), y: sharpRef.y + k * (sharp.y - sharpRef.y) };
      }
    }

    if (occluded) flags.push('occluded?');
    else if (lost && !fixed) flags.push('lost');
    if (blurred && !occluded) flags.push('blurred?');
    if (!anchor && !flags.length) flags.push('no-face');
    prevFrame = frame;
    lastT = t;
    return {
      t, conf, motion, outside: 0, sam, flags, crop, rawCrop, mouth, rawMouth, version: VERSION,
      lipMass: seen ? seen.ratio : null, sharpness: ratio,
      anchor: anchor && roundBox(anchor), face: anchor && roundBox(faceFromAnchor(anchor, cfg)),
    };
  }

  return { step, config: cfg, version: VERSION, get samCalls() { return samCalls; } };
}

// Runs a whole clip (iterable or async iterable of frames) and applies the quality gate.
export async function trackClip(frames, sam, config = {}) {
  const tracker = createTracker({ sam, config });
  const results = [];
  for await (const f of frames) results.push(await tracker.step(f));
  return { results, samCalls: tracker.samCalls, gate: qualityGate(results.map((r) => r.flags), tracker.config) };
}

export { qualityGate };
