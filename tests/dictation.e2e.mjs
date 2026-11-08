// Headless-Chrome check of static/dictation.html on a real talking face AND its voice (the talking head in
// /tmp/badge-sam as a 640x480 30 fps fake camera plus a 48 kHz mono fake mic; see e2e-kit.mjs). /dictation,
// /sessions and /sessions/<id> are stubs: a new session is "processing" for PROCESS_MS, then "done" with turns.
// Checks: lock-on; a held take uploads a 640x480 webm WITH a non-silent Opus track, lasting the hold, with
// X-Tracker-Flags and X-Tracker-Boxes (one per recorded frame, on the mouth); the STT text shows at once with a
// "learning" chip; the sessions list polls every 3 s while processing, then shows per-turn VTT vs STT with clip
// links, flips the chip to "learned" and stops polling; with no face the take still uploads (flags, no gate);
// every page shares the rail with Dictation right after Record; no console errors.
//   node tests/dictation.e2e.mjs [out-dir]   (SOURCE=talking.mp4 GT=talking.gt.json CHROME=/path/to/chrome)
import { createServer } from 'node:http';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { audioMatch, boxesOnMouth, fakeCamera, launch, probeTake, readBody, sendJSON, serveStatic } from './e2e-kit.mjs';
import { TRACKER, trackerFlags } from './e2e-kit.mjs';
import { DEFAULT_TRACKER } from '../static/capture.js';

const SAM_DELAY_MS = Number(process.env.SAM_DELAY_MS ?? 300), PROCESS_MS = 6500;
const work = await mkdtemp(join(tmpdir(), 'dictation-e2e-'));
const OUT = process.argv[2] ?? join(work, 'out');
await mkdir(OUT, { recursive: true });
const cam = await fakeCamera(work, { audio: true });

// ---------- stub server ----------
const STT = 'we are going to build it together';
const stub = { sam: 'gt', uploads: [], lists: [], details: 0, sessions: [] };
const turn = (i, start, end, stt, clip, vtt, correct) => ({ i, start, end, stt, status: 'processed', reason: null, clip_id: clip, vtt, correct });
stub.sessions.push({ id: 's-0', t: Date.now() / 1000 - 3600, user_id: 'e2e', dataset: 'dictation', created: 0, turns: [
  turn(0, 0.4, 2.1, 'good morning', 'c-0-0', { text: 'good morning', p: 0.82 }, { vtt: true, gemini: true, muse: false }),
  { i: 1, start: 2.5, end: 2.8, stt: 'uh', status: 'skipped', reason: 'too short', clip_id: null, vtt: null, correct: null },
] });
const DONE_TURNS = [
  turn(0, 0.2, 1.9, 'we are going to', 'c-1-0', { text: 'we are going to', p: 0.74 }, { vtt: true, gemini: false, muse: true }),
  turn(1, 2.1, 3.4, 'build it together', 'c-1-1', { text: 'bill it together', p: 0.41 }, { vtt: false, gemini: false, muse: false }),
];
function view(s) { // a new session reads as processing (turns pending) until PROCESS_MS after its upload
  if (!s.created || Date.now() - s.created >= PROCESS_MS) return { ...s, status: 'done' };
  return { ...s, status: 'processing', turns: s.turns.map((t) => ({ ...t, status: 'pending', clip_id: null, vtt: null, correct: null })) };
}
const summary = (s) => ({ id: s.id, t: s.t, user_id: s.user_id, dataset: s.dataset, status: s.status, turns: s.turns.length,
  processed: s.turns.filter((t) => t.status === 'processed').length, skipped: s.turns.filter((t) => t.status === 'skipped').length,
  vtt_correct: s.turns.filter((t) => t.correct?.vtt).length });

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost'), path = url.pathname;
  if (req.method === 'POST' && path === '/sam') {
    const jpeg = await readBody(req);
    if (stub.sam === 'empty') return sendJSON(res, { objects: [] });
    await sleep(SAM_DELAY_MS);
    return sendJSON(res, { objects: await cam.samObjects(jpeg, url.searchParams.get('prompt')) });
  }
  if (req.method === 'POST' && path === '/dictation') {
    const data = await readBody(req), id = `s-${stub.sessions.length}`, file = join(OUT, `${id}.webm`);
    await writeFile(file, data);
    stub.uploads.push({ id, at: Date.now(), user: url.searchParams.get('user'), type: req.headers['content-type'], file, size: data.length,
      flags: req.headers['x-tracker-flags'], boxes: req.headers['x-tracker-boxes'] });
    stub.sessions.push({ id, t: Date.now() / 1000, user_id: 'e2e', dataset: 'dictation', created: Date.now(), turns: DONE_TURNS });
    return sendJSON(res, { session: id, stt: STT, turns: DONE_TURNS.length });
  }
  if (req.method === 'GET' && path === '/sessions') {
    stub.lists.push(Date.now());
    const sessions = stub.sessions.map(view).map(summary);
    return sendJSON(res, { users: [{ id: 'e2e', sessions: sessions.length }], sessions });
  }
  const m = path.match(/^\/sessions\/([^/]+)$/);
  if (req.method === 'GET' && m) {
    stub.details++;
    const s = stub.sessions.find((x) => x.id === decodeURIComponent(m[1]));
    return s ? sendJSON(res, view(s)) : sendJSON(res, { error: 'no such session' });
  }
  return serveStatic(path, res);
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const BASE = `http://127.0.0.1:${server.address().port}`;

const checks = {};
const check = (name, ok, evidence) => { checks[name] = { ok: !!ok, ...evidence }; };
let browser = null;

async function main() {
  browser = await launch({ camera: cam.camera, wav: cam.wav, profile: join(work, 'profile'), base: BASE });
  const { evaluate, errors, waitFor, go, holdSpace } = browser;
  const shot = (name) => browser.shot(join(OUT, `${name}.png`));
  const session = (id) => `document.querySelector('#sessions li.session[data-id="${id}"]')`;
  const links = (id) => `[...${session(id)}.querySelectorAll('a.turn')].map((a) => a.getAttribute('href'))`;

  // ---- 1. lock-on, and the past session with its turns ----
  await go('dictation.html?user=e2e');
  const locked = await waitFor('(window.__dict?.frames ?? 0) > 45 && window.__dict.quality === "ok" && !!window.__dict.last?.crop', 25000);
  const pastShown = await waitFor(`!!${session('s-0')}?.querySelector('a.turn')`, 5000);
  const live = await evaluate(`(() => { const c = document.getElementById('overlay'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let ink = 0; for (let k = 3; k < d.length; k += 4) if (d[k]) ink++;
    return { frames: window.__dict.frames, chip: document.getElementById('quality').textContent, overlayPixels: ink, conf: window.__dict.last.conf,
      audioTracks: document.getElementById('preview').srcObject.getAudioTracks().length }; })()`);
  check('lockOn', locked && pastShown && live.overlayPixels > 0 && live.audioTracks === 1,
    { ...live, pastLinks: await evaluate(links('s-0')), pastText: await evaluate(`${session('s-0')}.textContent`), screenshot: await shot('1-lock-on') });

  // ---- 2. a held take: full frame with audio, flags + boxes; STT at once with a learning chip ----
  const heldMs = await holdSpace(3500);
  const shownStt = await waitFor(`document.querySelector('#said .tip-text')?.textContent === ${JSON.stringify(STT)}`, 10000);
  const said = await evaluate(`(() => { const c = document.getElementById('said'); return { text: c.textContent, chip: c.querySelector('.chip')?.textContent }; })()`);
  const up = stub.uploads.at(-1);
  const { version, frames: flags } = trackerFlags(up?.flags), boxes = JSON.parse(up?.boxes ?? 'null');
  check('trackerVersion', version === (TRACKER ?? DEFAULT_TRACKER), { version, expected: TRACKER ?? DEFAULT_TRACKER });
  const take = await probeTake(up.file), audioCorr = take.audio.length ? +(await audioMatch(up.file, cam.wav)).toFixed(3) : null;
  check('upload', shownStt && said.chip === 'learning' && stub.uploads.length === 1 && up.user === 'e2e'
    && take.video.length === 1 && take.video[0].width === 640 && take.video[0].height === 480
    && take.audio.length === 1 && take.audio[0].codec_name === 'opus' && take.meanVolumeDb > -50 && audioCorr > 0.5
    && Math.abs(take.duration * 1000 - heldMs) < 300 && Array.isArray(flags) && Array.isArray(boxes),
  { heldMs, said, contentType: up.type, bytes: up.size, streams: take.streams, durationS: +take.duration.toFixed(3), meanVolumeDb: take.meanVolumeDb, audioCorrWithWav: audioCorr,
    webmFrames: take.pts.length, flags: flags?.length, flagged: flags?.filter((f) => f.length).length, boxes: boxes?.length, firstBoxes: boxes?.slice(0, 2),
    screenshot: await shot('2-result-learning') });
  check('boxesPerFrame', Math.abs(boxes.length - take.pts.length) <= 2 && Math.abs(flags.length - take.pts.length) <= 2
    && boxes.every((b) => b.length === 5 && b.every(Number.isInteger)), { webmFrames: take.pts.length, boxes: boxes.length, flags: flags.length });
  const onMouth = await boxesOnMouth(cam, up.file, take.pts, boxes, join(OUT, '2b-box-on-recorded-frame.png'));
  check('boxesOnMouth', onMouth.ok, onMouth);

  // ---- 3. sessions poll every 3 s while processing, then show VTT vs STT per turn and stop ----
  const processingSeen = await waitFor(`${session(up.id)}?.querySelector('.chip')?.textContent === 'processing'`, 3000);
  const done = await waitFor(`${session(up.id)}?.querySelector('.head .chip')?.textContent === 'done' && ${session(up.id)}.querySelectorAll('a.turn').length === 2`, PROCESS_MS + 8000);
  const doneAt = Date.now(), polled = stub.lists.filter((t) => t >= up.at && t <= doneAt);
  const gaps = polled.slice(1).map((t, i) => t - polled[i]);
  const newTurns = await evaluate(`[...${session(up.id)}.querySelectorAll('.turn')].map((t) => t.textContent)`);
  const doneView = { links: await evaluate(links(up.id)), turns: newTurns, chip: await evaluate(`document.querySelector('#said .chip').textContent`) };
  const doneShot = await shot('3-sessions-done');
  await sleep(7000);
  const listsAfterDone = stub.lists.filter((t) => t > doneAt).length;
  check('sessionsPoll', processingSeen && done && doneView.chip === 'learned'
    && JSON.stringify(doneView.links) === JSON.stringify(['clips.html?clip=c-1-0', 'clips.html?clip=c-1-1'])
    && newTurns[1].includes('build it together') && newTurns[1].includes('bill it together') && newTurns[1].includes('VTT ✗')
    && gaps.length >= 2 && gaps.every((g) => g >= 2800 && g <= 4000) && listsAfterDone === 0,
  { ...doneView, listGapsMs: gaps, listsAfterDone, screenshot: doneShot });
  const dictErrors = errors.splice(0);

  // ---- 4. no face: no gate here, the take still uploads with its flags ----
  stub.sam = 'empty';
  await go('dictation.html?user=e2e');
  await waitFor('(window.__dict?.frames ?? 0) > 30', 20000);
  const before = stub.uploads.length;
  await holdSpace(1500);
  await waitFor(`document.querySelector('#said .tip-text')?.textContent === ${JSON.stringify(STT)}`, 10000);
  const noFace = stub.uploads.at(-1), nfFlags = trackerFlags(noFace.flags).frames, nfBoxes = JSON.parse(noFace.boxes);
  check('noGate', stub.uploads.length === before + 1 && nfFlags.length > 30 && nfFlags.every((f) => f.includes('no-face')) && nfBoxes.length === 0,
    { flags: nfFlags.length, sampleFlags: nfFlags.slice(0, 2), boxes: nfBoxes.length, quality: await evaluate('window.__dict.quality') });
  const noFaceErrors = errors.splice(0);

  // ---- 5. every page: the same rail, Dictation right after Record ----
  const pages = {};
  for (const p of ['index.html', 'lipmode.html', 'track.html', 'preheat.html', 'clips.html', 'stats.html', 'training.html', 'pipeline.html']) {
    await go(p);
    await sleep(1200);
    pages[p] = await evaluate('[...document.querySelectorAll("nav.rail a")].map((a) => a.textContent).join(",")');
  }
  const pageErrors = errors.splice(0);
  const exceptions = pageErrors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e.text));
  check('rail', Object.values(pages).every((r) => r.startsWith('Record,Dictation,Lip mode,')) && exceptions.length === 0,
    { rail: pages['index.html'], stubbed404s: pageErrors.length - exceptions.length, exceptions });
  check('noConsoleErrors', dictErrors.length === 0 && noFaceErrors.length === 0, { dictErrors, noFaceErrors });
}

let ok = false;
try {
  await main();
  ok = Object.values(checks).every((c) => c.ok);
} catch (e) {
  console.error(e);
} finally {
  browser?.close();
  server.close();
  await rm(join(work, 'profile'), { recursive: true, force: true }).catch(() => {});
  await rm(cam.camera, { force: true }).catch(() => {});
}
console.log(JSON.stringify(checks, null, 2));
for (const [k, v] of Object.entries(checks)) console.log(`${v.ok ? 'PASS' : 'FAIL'} ${k}`);
console.log(`out: ${OUT}`);
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
