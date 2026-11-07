// Mouth tracker v2: lips-anchored. SAM keyframes ask for the LIPS (not the face), so the mouth box sits where the
// lips really are, whatever the head pose; between keyframes v1's rigid nose-to-chin NCC anchor (lips masked) carries
// the box, and the mouth's offset and size relative to that anchor are re-learned at every keyframe.
// v1 (static/tracker.js) is untouched and still selectable; this module reuses its image machinery.
//
// Pure ES module, no DOM: the browser and Node import it unchanged.
// Frame: {data: gray Uint8(Clamped)Array (w*h), width, height, t (s), rgba?: Uint8ClampedArray (w*h*4)}.
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

// The user's lips among SAM's: nearest the expected mouth, else (no expectation) the largest (the nearest person).
function pickLips(boxes, expected) {
  if (!boxes?.length) return null;
  if (expected) return boxes.reduce((a, b) => (dist(center(b), expected) < dist(center(a), expected) ? b : a));
  return boxes.reduce((a, b) => (b[2] * b[3] > a[2] * a[3] ? b : a));
}

export function createTracker({ sam, config = {} } = {}) {
  const cfg = withConfig(config);
  const smoother = createSmoother(cfg);
  // geometry learned at keyframes, in anchor widths: mouth centre = mouthPoint(anchor) + off * aw; size = (rw, rh) * aw
  let tpl = null, anchor = null, off = [0, 0], rw = 0, rh = 0;
  let prevFrame = null, lastT = 0, vel = [0, 0], lastAnchorT = -Infinity, lastSamT = -Infinity, samCalls = 0;
  let occluded = false, wasLost = false;

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

  async function step(frame, { force = false } = {}) {
    const t = frame.t, tracking = !!tpl, dt = Math.max(1e-3, t - lastT);
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

    const reasons = [];
    if (!tracking) reasons.push('init');
    else {
      if (lost) reasons.push('lost');
      if (motion > cfg.T_MOTION) reasons.push('motion');
      if (scaleJump(anchor, tpl.box) > cfg.T_SCALE) reasons.push('scale');
      if (t - lastAnchorT >= cfg.T_REFRESH - 1e-6) reasons.push('refresh');
      if (occluded) reasons.push('occluded');
    }
    if (force) reasons.push('forced');
    // an occlusion that hurt the match and now no longer does: look again at once (SAM_MIN_GAP is a cost cap)
    const uncovered = occluded && wasLost && !lost;
    wasLost = lost;

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
    if (occluded) flags.push('occluded?');
    else if (lost && !fixed) flags.push('lost');
    if (!anchor && !flags.length) flags.push('no-face');

    let mouth = null, rawMouth = null, crop = null, rawCrop = null;
    if (anchor) {
      rawMouth = mouthOf(anchor);
      const [cx, cy, w] = [...center(rawMouth), rawMouth[2]], cw = cfg.CROP_PAD * w, ch = cw * cfg.CROP_ASPECT;
      rawCrop = [cx - cw / 2, cy - ch / 2, cw, ch];
      const s = smoother.update(rawMouth, t); // smoothed mouth box; the crop is its padded, fixed-aspect frame
      const [sx, sy] = center(s), sw = cfg.CROP_PAD * s[2];
      mouth = s;
      crop = roundBox([sx - sw / 2, sy - (sw * cfg.CROP_ASPECT) / 2, sw, sw * cfg.CROP_ASPECT]);
    }
    prevFrame = frame;
    lastT = t;
    return {
      t, conf, motion, outside: 0, sam, flags, crop, rawCrop, mouth, rawMouth, version: VERSION,
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
