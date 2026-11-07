// Per-frame cost of the tracker loop on 640x480 frames, SAM excluded (the stub answers instantly).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { createTracker } from '../static/tracker.js';
import { rng, makeFace, makeBackground, render, faceBox } from './synth.mjs';

test('median tracker step on a 640x480 frame is at most 8 ms', async (t) => {
  const W = 640, H = 480, face = makeFace(180, 220, 7), bg = makeBackground(W, H, 3), rand = rng(1);
  const layers = Array.from({ length: 240 }, (_, i) => ({ face, x: 230 + 40 * Math.sin(i / 15), y: 130 + 12 * Math.sin(i / 11), mouthOpen: 0.5 + 0.5 * Math.sin(i / 2) }));
  const frames = layers.map((L, i) => render(W, H, bg, [L], { t: i / 30, rand }));
  let samMs = 0;
  const tracker = createTracker({
    samFace: async (frame) => { const t0 = performance.now(); const box = faceBox(layers[Math.round(frame.t * 30)]); samMs += performance.now() - t0; return [box.map(Math.round)]; },
  });
  const ms = [];
  for (const f of frames) {
    samMs = 0;
    const t0 = performance.now();
    await tracker.step(f);
    ms.push(performance.now() - t0 - samMs);
  }
  const sorted = ms.slice(1).sort((a, b) => a - b); // frame 0 has no template yet
  const median = sorted[Math.floor(sorted.length / 2)], p95 = sorted[Math.floor(sorted.length * 0.95)];
  t.diagnostic(`step ms: median ${median.toFixed(2)}, p95 ${p95.toFixed(2)}, max ${sorted.at(-1).toFixed(2)} over ${sorted.length} frames`);
  assert.ok(median <= 8, `median ${median.toFixed(2)} ms`);
});
