// Per-frame cost of tracker v2 (colour model on) against v1 on the same 640x480 frames, SAM excluded.
// The bar: v2's median step at most 2x v1's and at most 5 ms.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import * as V1 from '../static/tracker.js';
import * as V2 from '../static/tracker2.js';
import { rng, makeFace, makeBackground, render, faceBox, mouthBox } from './synth.mjs';

const W = 640, H = 480, face = makeFace(180, 220, 7), bg = makeBackground(W, H, 3), rand = rng(1);
const layers = Array.from({ length: 240 }, (_, i) => ({ face, x: 230 + 40 * Math.sin(i / 15), y: 130 + 12 * Math.sin(i / 11), mouthOpen: 0.5 + 0.5 * Math.sin(i / 2) }));
const frames = layers.map((L, i) => {
  const f = render(W, H, bg, [L], { t: i / 30, rand }), rgba = new Uint8ClampedArray(W * H * 4);
  for (let k = 0, p = 0; k < f.data.length; k++, p += 4) { const g = f.data[k]; rgba[p] = Math.min(255, g * 1.1); rgba[p + 1] = g * 0.85; rgba[p + 2] = g * 0.7; rgba[p + 3] = 255; }
  return { ...f, rgba, i };
});

async function median(make) {
  let samMs = 0;
  const tracker = make(async (frame, prompt) => {
    const t0 = performance.now(), L = layers[frame.i], box = prompt === 'face' ? faceBox(L) : mouthBox(L);
    samMs += performance.now() - t0;
    return [box.map(Math.round)];
  });
  const ms = [];
  for (const f of frames) {
    samMs = 0;
    const t0 = performance.now();
    await tracker.step({ ...f });
    ms.push(performance.now() - t0 - samMs);
  }
  const s = ms.slice(1).sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

test('v2 median step (colour on) is at most 2x v1 and at most 5 ms', async (t) => {
  const m1 = await median((sam) => V1.createTracker({ samFace: (f) => sam(f, 'face') }));
  const m2 = await median((sam) => V2.createTracker({ sam }));
  t.diagnostic(`median step: v1 ${m1.toFixed(2)} ms, v2 ${m2.toFixed(2)} ms (${(m2 / m1).toFixed(2)}x)`);
  assert.ok(m2 <= 2 * m1 && m2 <= 5, `v2 ${m2.toFixed(2)} ms vs v1 ${m1.toFixed(2)} ms`);
});
