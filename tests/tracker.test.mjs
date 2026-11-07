// Behavior of static/tracker.js on synthetic frames whose true face boxes are known.
// Expected values come from the scene geometry (true boxes + CONFIG ratios), never from tracker output.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CONFIG, trackClip, createSmoother, qualityGate } from '../static/tracker.js';
import { rng, gauss, makeFace, makeBackground, render, faceBox } from './synth.mjs';

const W = 640, H = 480, FPS = 30;
const C = CONFIG;
// Only the triggers under test may call SAM; the rest are switched off with these overrides.
const QUIET = { T_REFRESH: Infinity, T_MOTION: 1, T_OUTSIDE: 1 };

const mid = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
// The anchor (nose-tip-to-chin patch) center of a true face box, straight from the CONFIG ratios.
const trueAnchorCenter = ([x, y, w, h]) => [x + (C.ANCHOR_X + C.ANCHOR_W / 2) * w, y + (C.ANCHOR_Y + C.ANCHOR_H / 2) * h];
const trueAnchorBox = ([x, y, w, h]) => [x + C.ANCHOR_X * w, y + C.ANCHOR_Y * h, C.ANCHOR_W * w, C.ANCHOR_H * h];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// SAM stand-in: returns the boxes `facesAt(i)` gives for the frame, and records each call.
function fakeSam(facesAt) {
  const calls = [];
  const sam = async (frame) => {
    const faces = facesAt(frame.i).map((b) => b.map(Math.round));
    calls.push({ i: frame.i, faces });
    return faces;
  };
  return { sam, calls };
}

function clip(n, layersAt, occludedAt = () => []) {
  const bg = makeBackground(W, H, 3), rand = rng(11), frames = [];
  for (let i = 0; i < n; i++) {
    const f = render(W, H, bg, layersAt(i), { t: i / FPS, rand, occluders: occludedAt(i) });
    f.i = i;
    frames.push(f);
  }
  return frames;
}

test('a: anchor center stays within 2 px of truth for translations up to 8 px/frame', async () => {
  const face = makeFace(180, 220, 7), r = rng(5);
  const steps = [[8, 8], [-8, 8], [8, -8], [-8, -8], [0, 8], [8, 0], [0, -8], [-8, 0], [7.5, -7.5], [-6.25, 3.5], [2.5, -0.5], [0.25, 0.75]];
  for (let k = 0; k < 28; k++) steps.push([(r() * 2 - 1) * 8, (r() * 2 - 1) * 8]);
  const pos = [[230, 130]];
  for (const [dx, dy] of steps) {
    let [x, y] = pos.at(-1);
    x = x + dx < 20 || x + dx > W - 200 ? x - dx : x + dx; // bounce off the frame edges, same speed
    y = y + dy < 20 || y + dy > H - 240 ? y - dy : y + dy;
    pos.push([x, y]);
  }
  const truth = pos.map(([x, y]) => [x, y, 180, 220]);
  const frames = clip(truth.length, (i) => [{ face, x: truth[i][0], y: truth[i][1] }]);
  const { sam, calls } = fakeSam((i) => [truth[i]]);
  const { results } = await trackClip(frames, sam, QUIET);
  assert.deepEqual(calls.map((c) => c.i), [0], 'only the initial keyframe may use SAM; the rest is tracking');
  results.forEach((r, i) => {
    const err = dist(mid(r.anchor), trueAnchorCenter(truth[i]));
    assert.ok(err <= 2, `frame ${i}: anchor center off by ${err.toFixed(2)} px`);
  });
});

const STILL = [230, 130, 180, 220];
function occlusionClip(n, hidden) {
  const face = makeFace(180, 220, 7);
  const [ax, ay, aw, ah] = trueAnchorBox(STILL);
  return clip(n, () => [{ face, x: STILL[0], y: STILL[1] }],
    (i) => (hidden(i) ? [{ box: [ax - 4, ay - 4, aw + 8, ah + 8], seed: 42 }] : []));
}

test('b1: covering the anchor with noise drops conf below T_CONF and triggers a SAM re-anchor', async () => {
  const hidden = (i) => i >= 10 && i < 15;
  const { sam } = fakeSam(() => [STILL]);
  const { results } = await trackClip(occlusionClip(25, hidden), sam, { T_REFRESH: Infinity });
  results.forEach((r, i) => {
    if (i === 0) return; // keyframe from SAM, nothing to track yet
    if (hidden(i)) assert.ok(r.conf < C.T_CONF, `frame ${i} occluded but conf ${r.conf.toFixed(3)}`);
    else assert.ok(r.conf >= C.T_CONF, `frame ${i} clear but conf ${r.conf.toFixed(3)}`);
  });
  assert.ok(results[10].sam?.reasons.includes('lost'), `frame 10 should re-anchor as lost, got ${JSON.stringify(results[10].sam)}`);
});

test('b2: exactly the occluded frames are flagged occluded?', async () => {
  const hidden = (i) => i >= 10 && i < 15;
  const { sam } = fakeSam(() => [STILL]);
  const { results } = await trackClip(occlusionClip(25, hidden), sam, { T_REFRESH: Infinity });
  const flagged = results.flatMap((r, i) => (r.flags.includes('occluded?') ? [i] : []));
  assert.deepEqual(flagged, [10, 11, 12, 13, 14]);
});

function jitterRun(n, truthAt, sigma, seed) {
  const r = rng(seed), sm = createSmoother(C), raw = [], out = [];
  for (let i = 0; i < n; i++) {
    const [x, y, w, h] = truthAt(i);
    const z = [x + sigma * gauss(r), y + sigma * gauss(r), w, h];
    raw.push(mid(z));
    out.push(mid(sm.update(z, i / FPS)));
  }
  return { raw, out };
}
const std = (xs) => { const m = xs.reduce((a, b) => a + b, 0) / xs.length; return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length); };

test('c1: smoothing halves crop-center jitter (sigma 3 px) on a static mouth', () => {
  const { raw, out } = jitterRun(300, () => [270, 275, 100, 50], 3, 21);
  for (const k of [0, 1]) {
    const rawStd = std(raw.slice(30).map((p) => p[k])), smStd = std(out.slice(30).map((p) => p[k]));
    assert.ok(smStd <= rawStd / 2, `axis ${k}: smoothed std ${smStd.toFixed(2)} vs raw ${rawStd.toFixed(2)}`);
  }
});

test('c2: smoothed crop follows a real 20 px move to within 2 px inside 1 s', () => {
  const { out } = jitterRun(160, (i) => [i < 100 ? 270 : 290, 275, 100, 50], 0, 1);
  const err = dist(out[130], [340, 300]);
  assert.ok(err <= 2, `1 s after the move the crop center is still ${err.toFixed(2)} px away`);
});

test('d: a face scaling past T_SCALE since the last keyframe fires a scale re-anchor', async () => {
  const face = makeFace(150, 180, 7);
  const HOLD = 10; // still until the SAM_MIN_GAP cost cap after the init keyframe has passed, then zoom 3%/frame
  assert.ok(HOLD / FPS >= C.SAM_MIN_GAP, 'scenario must not cross T_SCALE inside the SAM cost-cap window');
  const n = HOLD + 12, scaleAt = (i) => 1.03 ** Math.max(0, i - HOLD);
  const layer = (i) => ({ face, s: scaleAt(i), x: 320 - 75 * scaleAt(i), y: 240 - 90 * scaleAt(i) });
  const frames = clip(n, (i) => [layer(i)]);
  const { sam, calls } = fakeSam((i) => [faceBox(layer(i))]);
  const { results } = await trackClip(frames, sam, QUIET);
  const expected = [...Array(n).keys()].find((i) => scaleAt(i) - 1 > C.T_SCALE); // first frame past the threshold
  const first = calls[1]?.i;
  assert.ok(first !== undefined && Math.abs(first - expected) <= 1, `scale re-anchor at frame ${first}, truth crosses T_SCALE at ${expected}`);
  assert.deepEqual(results[first].sam.reasons, ['scale']);
});

test('e: T_REFRESH re-anchors on schedule on a still face', async () => {
  const face = makeFace(180, 220, 7);
  const frames = clip(150, () => [{ face, x: STILL[0], y: STILL[1] }]);
  const { sam, calls } = fakeSam(() => [STILL]);
  const { results } = await trackClip(frames, sam, { T_REFRESH: 2 });
  assert.deepEqual(calls.map((c) => c.i), [0, 2 * FPS, 4 * FPS]);
  assert.deepEqual(results[2 * FPS].sam.reasons, ['refresh']);
});

test('f1: the quality gate rejects a clip flagged above T_BAD and accepts one below', async () => {
  const { sam } = fakeSam(() => [STILL]);
  const heavy = await trackClip(occlusionClip(40, (i) => i >= 10 && i < 20), sam, { T_REFRESH: Infinity });
  const light = await trackClip(occlusionClip(40, (i) => i >= 10 && i < 14), sam, { T_REFRESH: Infinity });
  assert.equal(heavy.gate.ratio, 10 / 40);
  assert.equal(heavy.gate.ok, 10 / 40 <= C.T_BAD);
  assert.equal(light.gate.ratio, 4 / 40);
  assert.equal(light.gate.ok, 4 / 40 <= C.T_BAD);
  assert.notEqual(heavy.gate.ok, light.gate.ok, 'T_BAD must sit between the two clips for this test to mean anything');
});

test('f2: a flagged ratio exactly at T_BAD passes, one frame more fails', () => {
  const n = 100, k = Math.floor(C.T_BAD * n);
  const flags = (bad) => Array.from({ length: n }, (_, i) => (i < bad ? ['occluded?'] : []));
  assert.equal(qualityGate(flags(k), C).ok, k / n <= C.T_BAD);
  assert.equal(qualityGate(flags(k + 1), C).ok, false);
});

test('g1: a multi-face keyframe keeps the face nearest the previous box, not the larger one', async () => {
  const me = makeFace(160, 200, 7), other = makeFace(200, 250, 8);
  const ME = [80, 120, 160, 200], OTHER = [380, 80, 200, 250];
  const frames = clip(45, (i) => [{ face: me, x: ME[0], y: ME[1] }, ...(i >= 5 ? [{ face: other, x: OTHER[0], y: OTHER[1] }] : [])]);
  const { sam, calls } = fakeSam((i) => (i >= 5 ? [OTHER, ME] : [ME])); // the far, larger face listed first
  const { results } = await trackClip(frames, sam, { T_REFRESH: 0.5 });
  assert.ok(calls.some((c) => c.faces.length === 2), 'no keyframe saw both faces; test is vacuous');
  results.forEach((r, i) => assert.ok(dist(mid(r.face), mid(ME)) <= 3, `frame ${i}: face box ${r.face} left the enrolled user`));
});

test('g2: with no previous box the first keyframe takes the largest face', async () => {
  const me = makeFace(160, 200, 7), other = makeFace(200, 250, 8);
  const ME = [80, 120, 160, 200], OTHER = [380, 80, 200, 250];
  const frames = clip(3, () => [{ face: me, x: ME[0], y: ME[1] }, { face: other, x: OTHER[0], y: OTHER[1] }]);
  const { sam } = fakeSam(() => [ME, OTHER]);
  const { results } = await trackClip(frames, sam, QUIET);
  assert.ok(dist(mid(results[0].face), mid(OTHER)) <= 1, `picked ${results[0].face}, largest is ${OTHER}`);
});

test('h: a talking mouth does not drag the anchor (center within 1.5 px, scale within 2%)', async () => {
  const face = makeFace(180, 220, 7);
  const frames = clip(45, (i) => [{ face, x: STILL[0], y: STILL[1], mouthOpen: 0.5 + 0.5 * Math.sin((2 * Math.PI * 3 * i) / FPS) }]);
  const { sam, calls } = fakeSam(() => [STILL]);
  const { results } = await trackClip(frames, sam, QUIET);
  assert.deepEqual(calls.map((c) => c.i), [0], 'SAM must not rescue the tracker here');
  const want = trueAnchorBox(STILL);
  results.forEach((r, i) => {
    const err = dist(mid(r.anchor), mid(want)), ds = r.anchor[2] / want[2] - 1;
    assert.ok(err <= 1.5 && Math.abs(ds) <= 0.02, `frame ${i}: anchor off ${err.toFixed(2)} px, scale off ${(100 * ds).toFixed(1)}%`);
  });
});

// SAM face boxes swing with head pose (an ear comes into the box), so a keyframe box can sit off the face's
// true features. A confident track within SNAP_TOL keeps its own anchor; beyond it, SAM wins (drift insurance).
async function keyframeShift(shiftFrac) {
  const face = makeFace(180, 220, 7);
  const frames = clip(40, () => [{ face, x: STILL[0], y: STILL[1] }]);
  const shifted = [STILL[0] + shiftFrac * STILL[2], STILL[1], STILL[2], STILL[3]];
  const { sam, calls } = fakeSam((i) => [i === 0 ? STILL : shifted]);
  const { results } = await trackClip(frames, sam, { ...QUIET, T_REFRESH: 0.5 });
  assert.ok(calls.length >= 2, 'no refresh keyframe happened; test is vacuous');
  return { results, shifted };
}

test('i1: a refresh keyframe whose SAM box is off by half of SNAP_TOL keeps the tracked anchor', async () => {
  const { results } = await keyframeShift(C.SNAP_TOL / 2);
  results.forEach((r, i) => assert.ok(dist(mid(r.anchor), trueAnchorCenter(STILL)) <= 1.5, `frame ${i}: anchor moved to ${r.anchor}`));
});

test('i2: a refresh keyframe whose SAM box is off by twice SNAP_TOL re-places the anchor on SAM', async () => {
  const { results, shifted } = await keyframeShift(2 * C.SNAP_TOL);
  const k = Math.round(0.5 * FPS);
  results.slice(k).forEach((r, i) => assert.ok(dist(mid(r.anchor), trueAnchorCenter(shifted)) <= 1.5, `frame ${k + i}: anchor ${r.anchor} did not follow SAM`));
});
