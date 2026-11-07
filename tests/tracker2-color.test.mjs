// Tracker v2's per-session lip colour model (learned from a SAM lips box) on synthetic colour frames:
// warm skin, red lips, a dark mouth opening. Truth is where the lips were painted, never tracker output.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { learnLips, lipBlob, sharpness, createTracker } from '../static/tracker2.js';
import { rng, makeFace, makeBackground, render, faceBox, mouthBox } from './synth.mjs';

const W = 640, H = 480, FPS = 30;
const mid = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const SKIN = [1.1, 0.85, 0.7], LIP = [158, 72, 80];

// Gray synthetic scene -> warm-tinted RGBA with lips painted as a red elliptical ring around a dark opening;
// the gray plane is recomputed from the colour, as static/capture.js does.
function colorFrame(f, lips, { open = 0.35, patch = null } = {}) {
  const { width: w, height: h } = f, rgba = new Uint8ClampedArray(w * h * 4), data = new Uint8ClampedArray(w * h);
  const [lx, ly, lw, lh] = lips, cx = lx + lw / 2, cy = ly + lh / 2;
  for (let y = 0, k = 0; y < h; y++) {
    for (let x = 0; x < w; x++, k++) {
      const g = f.data[k], p = 4 * k;
      let c = [g * SKIN[0], g * SKIN[1], g * SKIN[2]];
      const ex = (x + 0.5 - cx) / (lw / 2), ey = (y + 0.5 - cy) / (lh / 2), r = ex * ex + ey * ey;
      if (r < 1) c = ey * ey * 1 < open * open && Math.abs(ex) < 0.8 ? [40, 25, 28] : LIP;
      if (patch && x >= patch[0] && x < patch[0] + patch[2] && y >= patch[1] && y < patch[1] + patch[3]) c = [150 * SKIN[0], 150 * SKIN[1], 150 * SKIN[2]];
      rgba[p] = c[0]; rgba[p + 1] = c[1]; rgba[p + 2] = c[2]; rgba[p + 3] = 255;
      data[k] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
    }
  }
  return { data, rgba, width: w, height: h, t: f.t };
}

const face = makeFace(180, 220, 7), bg = makeBackground(W, H, 3);
const L0 = { face, x: 230, y: 130 }, LIPS0 = (() => { const m = mouthBox({ ...L0, mouthOpen: 0.9 }); return [m[0] - 4, m[1] - 4, m[2] + 8, m[3] + 8]; })();
const scene = (lips, opts = {}, t = 0) => colorFrame(render(W, H, bg, [L0], { t, rand: rng(3) }), lips, opts);

test('learnLips + lipBlob find painted lips that moved 12 px right and 5 px up from where we looked', () => {
  const model = learnLips(scene(LIPS0), LIPS0);
  const moved = [LIPS0[0] + 12, LIPS0[1] - 5, LIPS0[2], LIPS0[3]];
  const blob = lipBlob(scene(moved), model, LIPS0);
  assert.ok(dist([blob.cx, blob.cy], mid(moved)) <= 1.5, `blob at ${blob.cx.toFixed(1)},${blob.cy.toFixed(1)}; lips at ${mid(moved)}`);
  assert.ok(blob.mass >= 0.8 * model.mass, `mass ${blob.mass.toFixed(0)} vs learned ${model.mass.toFixed(0)}`);
});

test('lipBlob height grows with the lips: a 40% taller mouth measures at least 25% taller', () => {
  const model = learnLips(scene(LIPS0), LIPS0);
  const tall = [LIPS0[0], LIPS0[1] - 0.2 * LIPS0[3], LIPS0[2], 1.4 * LIPS0[3]];
  const a = lipBlob(scene(LIPS0), model, LIPS0), b = lipBlob(scene(tall), model, LIPS0);
  assert.ok(a.sy >= 0.15 * LIPS0[3], `the lips measure ${a.sy.toFixed(2)} px tall: no blob found`);
  assert.ok(b.sy >= 1.25 * a.sy, `sy ${a.sy.toFixed(2)} -> ${b.sy.toFixed(2)}`);
});

test('a skin-coloured hand over the lips takes away most of the lip mass', () => {
  const model = learnLips(scene(LIPS0), LIPS0);
  const hand = [LIPS0[0] - 15, LIPS0[1] - 10, LIPS0[2] + 30, LIPS0[3] + 20];
  const blob = lipBlob(scene(LIPS0, { patch: hand }), model, LIPS0), clear = lipBlob(scene(LIPS0), model, LIPS0);
  assert.ok(model.mass > 0 && clear.mass >= 0.8 * model.mass, `uncovered lips keep only ${clear.mass} of ${model.mass}`);
  assert.ok(blob.mass <= 0.3 * model.mass, `covered lips keep ${(100 * blob.mass / model.mass).toFixed(0)}% of their mass`);
});

function blurH(f, len) {
  const { width: w, height: h } = f, d = new Uint8ClampedArray(w * h), half = len >> 1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let k = -half; k < len - half; k++) s += f.data[y * w + Math.min(w - 1, Math.max(0, x + k))];
    d[y * w + x] = s / len;
  }
  return { ...f, data: d };
}

test('sharpness of the mouth region halves under a 12 px motion blur but not under a 3 px one', () => {
  const f = scene(LIPS0), box = [LIPS0[0] - 20, LIPS0[1] - 20, LIPS0[2] + 40, LIPS0[3] + 40];
  const s0 = sharpness(f, box), s3 = sharpness(blurH(f, 3), box), s12 = sharpness(blurH(f, 12), box);
  assert.ok(s0 > 0, 'a textured mouth region has some sharpness');
  assert.ok(s12 <= 0.5 * s0, `12 px blur keeps ${(s12 / s0).toFixed(2)} of the sharpness`);
  assert.ok(s3 >= 0.7 * s0, `3 px blur keeps only ${(s3 / s0).toFixed(2)}`);
});

test('with colour on, the mouth follows lips that slide 18 px across a still face (a head turn); without, it stays behind', async () => {
  const n = 40, at = (i) => [LIPS0[0] + Math.min(18, Math.max(0, (i - 5) * 1.2)), LIPS0[1], LIPS0[2], LIPS0[3]];
  const frames = Array.from({ length: n }, (_, i) => Object.assign(colorFrame(render(W, H, bg, [L0], { t: i / FPS, rand: rng(9 + i) }), at(i)), { t: i / FPS, i }));
  const sam = async (frame, prompt) => (prompt === 'face' ? [faceBox(L0).map(Math.round)] : [at(frame.i).map(Math.round)]);
  const errAt = async (color) => {
    const tr = createTracker({ sam, config: { T_REFRESH: Infinity, COLOR: color } });
    const out = [];
    for (const f of frames) out.push(await tr.step(f));
    return out.slice(30).map((r, k) => dist(mid(r.mouth), mid(at(30 + k))) / LIPS0[2]);
  };
  const on = await errAt(1), off = await errAt(0);
  assert.ok(Math.max(...on) <= 0.12, `colour on: mouth up to ${Math.max(...on).toFixed(3)} lip widths off`);
  assert.ok(Math.min(...off) >= 0.2, `colour off is already within ${Math.min(...off).toFixed(3)}; the scene does not test colour`);
});
