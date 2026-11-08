// Behaviour of tracker v2 (static/tracker2.js) on synthetic frames whose true face and mouth boxes are known.
// Expected values come from the scene geometry (tests/synth.mjs), never from tracker output.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { VERSION, CONFIG, createTracker } from '../static/tracker2.js';
import * as V1 from '../static/tracker.js';
import { rng, makeFace, makeBackground, render, faceBox, mouthBox } from './synth.mjs';

const W = 640, H = 480, FPS = 30;
const mid = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// RGBA alongside the gray frame (a warm tint; lips are not coloured here, so these tests run v2 on geometry alone)
function withRgba(f) {
  const rgba = new Uint8ClampedArray(f.width * f.height * 4);
  for (let k = 0, p = 0; k < f.data.length; k++, p += 4) { const g = f.data[k]; rgba[p] = Math.min(255, g * 1.1); rgba[p + 1] = g * 0.9; rgba[p + 2] = g * 0.75; rgba[p + 3] = 255; }
  return { ...f, rgba };
}

function clip(n, layersAt, occludedAt = () => []) {
  const bg = makeBackground(W, H, 3), rand = rng(11), frames = [];
  for (let i = 0; i < n; i++) {
    const f = withRgba(render(W, H, bg, layersAt(i), { t: i / FPS, rand, occluders: occludedAt(i) }));
    f.i = i;
    frames.push(f);
  }
  return frames;
}

// SAM stand-in answering 'face' and 'lips' prompts per frame; records every call as [frame, prompt].
function fakeSam(answers) {
  const calls = [];
  const sam = async (frame, prompt) => {
    calls.push([frame.i, prompt]);
    return (answers(frame.i, prompt) ?? []).map((b) => b.map(Math.round));
  };
  return { sam, calls };
}

async function run(frames, sam, config = {}) {
  const tracker = createTracker({ sam, config });
  const out = [];
  for (const f of frames) out.push(await tracker.step(f));
  return { out, tracker };
}

const QUIET = { T_REFRESH: Infinity };

test('VERSION names the tracker for the upload headers', () => {
  assert.equal(VERSION, 'v2');
});

test('a: keyframes on SAM lips put the mouth on the lips even when the face box is off-centre (turned head)', async () => {
  const face = makeFace(180, 220, 7), L = { face, x: 230, y: 130, mouthOpen: 0.4 };
  const fb = faceBox(L), mouth = mouthBox(L);
  const shiftedFace = [fb[0] - 0.1 * fb[2], fb[1], fb[2], fb[3]]; // a turned head: the face box sits 0.1 face widths off the mouth
  const frames = clip(20, () => [L]);
  const { sam } = fakeSam((i, p) => (p === 'face' ? [shiftedFace] : [mouth]));
  const { out } = await run(frames, sam, QUIET);
  out.forEach((r, i) => {
    const err = dist(mid(r.mouth), mid(mouth)) / mouth[2];
    assert.ok(err <= 0.05, `frame ${i}: v2 mouth ${err.toFixed(3)} mouth widths off the lips`);
  });
  // the same SAM face box sends v1's face-ratio mouth well off the lips, so this scene does test the geometry
  const v1 = V1.createTracker({ samFace: async () => [shiftedFace.map(Math.round)], config: QUIET });
  const r1 = await v1.step(frames[0]);
  assert.ok(dist(mid(r1.crop), mid(mouth)) / mouth[2] > 0.2, 'v1 is not fooled; the scene is too easy');
});

test('b: between keyframes the raw mouth follows a face moving up to 8 px/frame to within 2 px', async () => {
  const face = makeFace(180, 220, 7), r = rng(5), pos = [[230, 130]];
  for (let k = 0; k < 40; k++) {
    let [x, y] = pos.at(-1);
    const dx = (r() * 2 - 1) * 8, dy = (r() * 2 - 1) * 8;
    x = x + dx < 20 || x + dx > W - 200 ? x - dx : x + dx;
    y = y + dy < 20 || y + dy > H - 240 ? y - dy : y + dy;
    pos.push([x, y]);
  }
  const layers = pos.map(([x, y]) => ({ face, x, y, mouthOpen: 0.4 }));
  const frames = clip(layers.length, (i) => [layers[i]]);
  const { sam, calls } = fakeSam((i, p) => (p === 'face' ? [faceBox(layers[i])] : [mouthBox(layers[i])]));
  const { out } = await run(frames, sam, QUIET);
  assert.deepEqual(calls.map((c) => c[0]), [0], 'only the first frame may use SAM; the rest is tracking');
  out.forEach((res, i) => {
    const err = dist(mid(res.rawMouth), mid(mouthBox(layers[i])));
    assert.ok(err <= 2, `frame ${i}: raw mouth off by ${err.toFixed(2)} px`);
  });
});

test('c: the smoothed mouth trails a steady 6 px/frame glide by at most 3 px once settled', async () => {
  const face = makeFace(160, 200, 7), layers = Array.from({ length: 50 }, (_, i) => ({ face, x: 100 + 6 * i, y: 130, mouthOpen: 0.4 }));
  const frames = clip(layers.length, (i) => [layers[i]]);
  const { sam } = fakeSam((i, p) => (p === 'face' ? [faceBox(layers[i])] : [mouthBox(layers[i])]));
  const { out } = await run(frames, sam, QUIET);
  for (let i = 20; i < 50; i++) {
    const lag = mid(mouthBox(layers[i]))[0] - mid(out[i].mouth)[0];
    assert.ok(Math.abs(lag) <= 3, `frame ${i}: smoothed mouth ${lag.toFixed(2)} px behind`);
  }
});

test('f: refresh keyframes whose lips sit a little off a healthy track move the mouth onto them and keep the anchor', async () => {
  const face = makeFace(180, 220, 7), L = { face, x: 230, y: 130, mouthOpen: 0.4 }, m = mouthBox(L);
  const off = [m[0] + 0.2 * m[2], m[1], m[2], m[3]]; // SAM's lips 0.2 mouth widths right of where the first keyframe put them
  const frames = clip(40, () => [L]);
  const { sam, calls } = fakeSam((i, p) => (p === 'face' ? [faceBox(L)] : [i === 0 ? m : off]));
  const { out } = await run(frames, sam, { T_REFRESH: 0.5, COLOR: 0 }); // geometry alone: colour calibration would also get there
  assert.ok(calls.length >= 3, 'fewer than two refresh keyframes; the test is vacuous');
  const err = dist(mid(out.at(-1).rawMouth), mid(off)) / m[2];
  assert.ok(err <= 0.06, `after the refreshes the mouth is still ${err.toFixed(3)} mouth widths off SAM's lips`);
  assert.ok(dist(mid(out.at(-1).anchor), mid(out[1].anchor)) <= 1.5, `the anchor moved from ${out[1].anchor} to ${out.at(-1).anchor}: re-placed, not kept`);
});

const STILL = { x: 230, y: 130 };
function coverScene(n, covered) {
  const face = makeFace(180, 220, 7), L = { face, ...STILL, mouthOpen: 0.4 }, m = mouthBox(L);
  const frames = clip(n, () => [L], (i) => (covered(i) ? [{ box: [m[0] - 20, m[1] - 15, m[2] + 40, m[3] + 50], seed: 42 }] : []));
  const { sam, calls } = fakeSam((i, p) => (p === 'face' ? [faceBox(L)] : covered(i) ? [] : [m]));
  return { frames, sam, calls };
}

test('d: a covered mouth (SAM sees the face but no lips) is flagged occluded?, and the flag clears once uncovered', async () => {
  const covered = (i) => i >= 12 && i < 24;
  const { frames, sam } = coverScene(45, covered);
  const { out } = await run(frames, sam, QUIET);
  const occ = out.map((r) => r.flags.includes('occluded?'));
  const hits = occ.filter((o, i) => o && covered(i)).length, wrong = occ.filter((o, i) => o && !covered(i) && i < 24 + 5).length;
  assert.ok(hits >= 10, `only ${hits}/12 covered frames flagged occluded?`);
  assert.equal(wrong, 0, 'a frame before the cover was flagged occluded?');
  const m = mouthBox({ face: makeFace(180, 220, 7), ...STILL, mouthOpen: 0.4 });
  out.forEach((r, i) => { if (covered(i)) assert.ok(dist(mid(r.rawMouth), mid(m)) <= 3, `frame ${i}: the mouth wandered to ${r.rawMouth.map(Math.round)} under the cover`); });
  out.slice(24 + 5).forEach((r, k) => assert.deepEqual(r.flags, [], `frame ${24 + 5 + k} still flagged ${r.flags}`));
});

test('e: a changed look with the lips still in view (a head turn) is re-learned, not taken for an occlusion', async () => {
  const a = makeFace(180, 220, 7), b = makeFace(180, 220, 8); // same place, new appearance from frame 10
  const L = (i) => ({ face: i < 10 ? a : b, ...STILL, mouthOpen: 0.4 });
  const frames = clip(40, (i) => [L(i)]);
  const { sam } = fakeSam((i, p) => (p === 'face' ? [faceBox(L(i))] : [mouthBox(L(i))]));
  const { out } = await run(frames, sam, QUIET);
  out.forEach((r, i) => assert.ok(!r.flags.includes('occluded?'), `frame ${i} flagged occluded? after a look change`));
  out.slice(20).forEach((r, k) => assert.ok(r.conf >= CONFIG.T_CONF && r.flags.length === 0, `frame ${20 + k}: conf ${r.conf.toFixed(2)}, flags ${r.flags}`));
});

test('g: a head speeding up to 24 px/frame (past the 16 px search radius) is followed without SAM', async () => {
  const face = makeFace(160, 200, 7), xs = [20];
  for (let i = 1; i < 20; i++) xs.push(xs.at(-1) + Math.min(24, 2 * i));
  const layers = xs.map((x) => ({ face, x, y: 130, mouthOpen: 0.4 }));
  assert.ok(Math.max(...xs) + 160 < W, 'the face leaves the frame');
  const frames = clip(layers.length, (i) => [layers[i]]);
  const { sam, calls } = fakeSam((i, p) => (p === 'face' ? [faceBox(layers[i])] : [mouthBox(layers[i])]));
  const { out } = await run(frames, sam, { ...QUIET, T_MOTION: 1 });
  assert.deepEqual(calls.map((c) => c[0]), [0], 'SAM had to rescue the track');
  out.forEach((r, i) => assert.ok(dist(mid(r.rawMouth), mid(mouthBox(layers[i]))) <= 2.5, `frame ${i}: raw mouth ${dist(mid(r.rawMouth), mid(mouthBox(layers[i]))).toFixed(1)} px off`));
});
