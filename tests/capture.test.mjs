// static/capture.js tracker selection and the version tag on the uploaded flags (pure helpers; openCapture itself is
// covered by the headless-Chrome e2e pages).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_TRACKER, trackerChoice, withVersion, makeTracker } from '../static/capture.js';

test('?tracker= picks v1 or v2; anything else falls back to the default', () => {
  assert.equal(trackerChoice('?tracker=v1&user=x'), 'v1');
  assert.equal(trackerChoice('?user=x&tracker=v2'), 'v2');
  assert.equal(trackerChoice('?tracker=v9'), DEFAULT_TRACKER);
  assert.equal(trackerChoice(''), DEFAULT_TRACKER);
  assert.ok(['v1', 'v2'].includes(DEFAULT_TRACKER));
});

test('the flags header names the tracker first while the take keeps one flag list per frame', () => {
  const flags = withVersion([[], ['occluded?'], []], 'v2');
  assert.equal(flags.length, 3, 'pages count frames with take.flags.length');
  assert.equal(JSON.stringify(flags), '["tracker:v2",[],["occluded?"],[]]');
});

test('makeTracker asks SAM for the face with v1 and for the lips with v2', async () => {
  const frame = { data: new Uint8ClampedArray(64 * 48).fill(100), width: 64, height: 48, t: 0 };
  for (const [version, prompt] of [['v1', 'face'], ['v2', 'lips']]) {
    const asked = [];
    const tracker = makeTracker(version, async (_f, p) => { asked.push(p); return []; });
    await tracker.step(frame);
    assert.equal(asked[0], prompt, `${version} asked ${asked}`);
  }
});
