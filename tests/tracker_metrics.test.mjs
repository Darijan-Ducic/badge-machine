// The tracker-lab metrics (tools/tracker_metrics.mjs) on hand-built trajectories whose right answers follow from
// the geometry: errors are in GT mouth widths, speeds in mouth widths per second, times in ms at 30 fps.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  mouthFromCrop, frameErrors, headSpeed, lagMs, jumpEvents, recoveryMs, gatePR, inside, medianBox,
} from '../tools/tracker_metrics.mjs';

const FPS = 30;
const close = (a, b, tol = 1e-9, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} ${a} vs ${b}`);

test('mouthFromCrop undoes the crop padding around the same centre', () => {
  assert.deepEqual(mouthFromCrop([100, 200, 160, 80], 1.6), [130, 215, 100, 50]);
});

test('frameErrors: a mouth 6 px right of a 60 px wide GT mouth is 0.1 mouth widths off; size errors per axis', () => {
  const gt = [[100, 100, 60, 30], null];
  const ours = [[106, 103, 66, 24], [0, 0, 10, 10]]; // same centre height (103 + 12 = 100 + 15)
  const [e, none] = frameErrors(ours, gt);
  close(e.centre, (6 + 3) / 60, 1e-9, 'centre'); // centre moves by 6 + half the 6 px wider box
  close(e.sizeW, 6 / 60, 1e-9, 'sizeW');
  close(e.sizeH, 6 / 60, 1e-9, 'sizeH');
  assert.equal(none, null, 'no GT mouth -> no error');
  assert.equal(frameErrors([null], [[0, 0, 10, 10]])[0].centre, Infinity, 'a frame without a box is a total miss');
});

test('headSpeed: a mouth gliding 3 px/frame with width 60 moves 1.5 mouth widths per second', () => {
  const gt = Array.from({ length: 40 }, (_, i) => [100 + 3 * i, 50, 60, 30]);
  const v = headSpeed(gt, FPS);
  for (let i = 5; i < 35; i++) close(v[i], 1.5, 1e-9, `frame ${i}`);
});

test('headSpeed ignores lip motion: a mouth opening and closing around a fixed centre-top is (nearly) still', () => {
  // only the height oscillates at 5 Hz (syllables); the top lip stays put, so the raw centre bobs +-4 px
  const gt = Array.from({ length: 60 }, (_, i) => [100, 50, 60, 22 + 8 * Math.sin((2 * Math.PI * 5 * i) / FPS)]);
  const v = headSpeed(gt, FPS);
  for (let i = 10; i < 50; i++) assert.ok(v[i] < 0.5, `frame ${i}: speed ${v[i]} from lip motion alone`);
});

test('lagMs: a track that replays the GT path 4 frames late lags 133 ms', () => {
  const path = (i) => 300 + 80 * Math.sin(i / 6);
  const gt = Array.from({ length: 90 }, (_, i) => [path(i), 50]);
  const ours = Array.from({ length: 90 }, (_, i) => [path(i - 4), 50]);
  const mask = gt.map((_, i) => i >= 10);
  close(lagMs(ours, gt, mask, FPS), (4 / FPS) * 1000, 1e-6);
});

test('jumpEvents: frames whose GT mouth centre jumps more than a mouth width from the previous GT frame', () => {
  const gt = [[0, 0, 50, 20], [5, 0, 50, 20], null, [80, 0, 50, 20], [82, 0, 50, 20], [20, 0, 50, 20]];
  assert.deepEqual(jumpEvents(gt, 1), [3, 5]);
});

test('recoveryMs: time from the event until the error stays under tol for `hold` frames; Infinity if never', () => {
  const err = [0.1, 0.1, 2, 2, 2, 0.3, 0.9, 0.3, 0.3, 0.3, 0.2, 3, 3, 3, 3];
  // event at 2: under 0.5 at 5, but 6 breaks it; holds 7, 8, 9 -> recovered at 7 = 5 frames
  assert.deepEqual(recoveryMs(err, [2, 11], FPS, { tol: 0.5, hold: 3 }), [(5 / FPS) * 1000, Infinity]);
});

test('gatePR counts flagged frames against bad frames; frames without a label are ignored', () => {
  const flagged = [true, true, false, false, true, true];
  const bad = [true, false, true, false, null, true];
  assert.deepEqual(gatePR(flagged, bad), { tp: 2, fp: 1, fn: 1, tn: 1, precision: 2 / 3, recall: 2 / 3 });
});

test('inside: a GT mouth fully within the crop counts, one poking out by a pixel does not', () => {
  assert.equal(inside([10, 10, 30, 20], [0, 0, 100, 50]), true);
  assert.equal(inside([71, 10, 30, 20], [0, 0, 100, 50]), false);
});

test('medianBox: per-coordinate median of the boxes, skipping missing frames', () => {
  assert.deepEqual(medianBox([[0, 0, 10, 10], null, [4, 2, 30, 12], [2, 8, 20, 11]]), [2, 2, 20, 11]);
});
