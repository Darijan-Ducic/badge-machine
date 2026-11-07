// Stress variants of real footage (tools/tracker_augment.mjs): the warped ground truth must land where the warped
// pixels land, and the bad-frame labels must be exactly the strongly blurred or mostly covered frames.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { plan, warp, warpBox, applyFrame } from '../tools/tracker_augment.mjs';

const W = 160, H = 120;
function frameWithRect([x, y, w, h], value = 250) {
  const rgb = new Uint8ClampedArray(W * H * 3).fill(20);
  for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) rgb.fill(value, (j * W + i) * 3, (j * W + i) * 3 + 3);
  return rgb;
}
function brightBBox(rgb) {
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) if (rgb[(j * W + i) * 3] > 135) { x0 = Math.min(x0, i); y0 = Math.min(y0, j); x1 = Math.max(x1, i); y1 = Math.max(y1, j); }
  return [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
}

test('warp and warpBox agree: a bright rectangle lands inside 1 px of its warped box', () => {
  const box = [50, 40, 30, 20];
  for (const m of [{ s: 1, tx: 13, ty: -7 }, { s: 1.25, tx: 0, ty: 0 }, { s: 0.85, tx: -10.5, ty: 6.25 }]) {
    const got = brightBBox(warp(frameWithRect(box), W, H, m)), want = warpBox(box, m, W, H);
    for (let k = 0; k < 4; k++) assert.ok(Math.abs(got[k] - want[k]) <= 1.01, `motion ${JSON.stringify(m)}: pixels ${got} vs box ${want}`);
  }
});

const MOUTHS = Array.from({ length: 300 }, () => [60, 70, 40, 16]);
const FACES = Array.from({ length: 300 }, () => [40, 20, 80, 90]);

test('plan labels as bad exactly the frames whose mouth is mostly covered or strongly blurred', () => {
  const spec = { occlude: [[2, 3, 'skin']], blur: [[5, 5.5, 16], [7, 7.5, 3]] };
  const p = plan(spec, MOUTHS, FACES, 30, W, H);
  p.forEach((f, i) => {
    const t = i / 30, strongBlur = t >= 5 && t < 5.5, covered = f.cover >= 0.5;
    assert.equal(f.bad, strongBlur || covered, `frame ${i} (t ${t.toFixed(2)}): bad ${f.bad}, cover ${f.cover}, blur ${f.blur}`);
    if (t >= 2.2 && t < 2.8) assert.ok(covered, `frame ${i}: mid-occlusion but cover ${f.cover}`);
    if (t >= 7 && t < 7.5) assert.equal(f.bad, false, `frame ${i}: a 3 px blur is not a bad frame`);
  });
  assert.ok(p.some((f) => f.bad) && p.some((f) => !f.bad));
});

test('plan jumps move the true mouth by at least a mouth width in one frame, and the motion is seeded', () => {
  const spec = { move: { seed: 3 }, jumps: [4] };
  const a = plan(spec, MOUTHS, FACES, 30, W, H), b = plan(spec, MOUTHS, FACES, 30, W, H);
  assert.deepEqual(a.map((f) => f.motion), b.map((f) => f.motion), 'same seed, same path');
  const c = (f) => [f.mouth[0] + f.mouth[2] / 2, f.mouth[1] + f.mouth[3] / 2];
  const step = (i) => Math.hypot(c(a[i])[0] - c(a[i - 1])[0], c(a[i])[1] - c(a[i - 1])[1]);
  assert.ok(step(120) >= a[120].mouth[2], `jump at 4 s moved ${step(120)} px, mouth width ${a[120].mouth[2]}`);
  assert.ok(step(60) < 0.5 * a[60].mouth[2], 'an ordinary frame is not a jump');
});

test('applyFrame covers the mouth with the occluder and hides it from the simulated SAM lips answer', () => {
  const spec = { occlude: [[0, 1, 'object']] };
  const p = plan(spec, MOUTHS, FACES, 30, W, H), f = p[15];
  const rgb = frameWithRect([60, 70, 40, 16], 200);
  const out = applyFrame(rgb, W, H, f);
  let changed = 0;
  for (let j = 70; j < 86; j++) for (let i = 60; i < 100; i++) if (Math.abs(out[(j * W + i) * 3] - 200) > 30) changed++;
  assert.ok(changed / (40 * 16) >= 0.5, `only ${changed} of the mouth pixels were covered`);
  assert.equal(f.samMouth, null, 'SAM cannot see covered lips');
  assert.deepEqual(p[45].samMouth, p[45].mouth, 'after the occlusion SAM sees the lips again');
});
