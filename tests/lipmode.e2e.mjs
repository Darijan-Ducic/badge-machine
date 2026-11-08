// Headless-Chrome check of static/lipmode.html on a real talking face (the public-domain talking head in
// /tmp/badge-sam, cut to a 640x480 30 fps fake camera). The /sam stub answers with the SAM ground-truth face box
// of whichever source frame the uploaded JPEG is (nearest 32x24 thumbnail), so the tracker locks on as it would
// live. /vtt, /vtt/<id>/label and /readiness are stubs that record what the page sends.
// Checks: lock-on; a held take uploads a 640x480 video-only webm lasting the hold with X-Tracker-Flags and
// X-Tracker-Boxes (one box per recorded frame, on the mouth of the matching source frame); the auto, confirm
// and none paths and the undo/confirm/edit label bodies; Space typed into the input does not record; a 10 s hold
// stops itself at 8 s, uploads that take while still held and notes the cap in its card; the readiness panel; the quality gate blocks the upload when SAM finds no face; no console errors anywhere.
//   node tests/lipmode.e2e.mjs [out-dir]   (SOURCE=talking.mp4 GT=talking.gt.json CHROME=/path/to/chrome)
import { createServer } from 'node:http';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { boxesOnMouth, fakeCamera, launch, probeTake, readBody, sendJSON, serveStatic } from './e2e-kit.mjs';
import { TRACKER, trackerFlags } from './e2e-kit.mjs';
import { DEFAULT_TRACKER } from '../static/capture.js';

const SAM_DELAY_MS = Number(process.env.SAM_DELAY_MS ?? 300); // the hosted SAM call's latency
const work = await mkdtemp(join(tmpdir(), 'lipmode-e2e-'));
const OUT = process.argv[2] ?? join(work, 'out');
await mkdir(OUT, { recursive: true });
const cam = await fakeCamera(work);

// ---------- stub server ----------
const stub = { sam: 'gt', mode: 'auto', samHits: 0, vtt: [], labels: [], n: 0 }; // sam: 'gt' | 'empty'; mode: a READS key
const READINESS = { phrases: [
  { text: 'turn on the lights', n: 14, vtt_acc: 0.86, ready: true },
  { text: 'call mom', n: 6, vtt_acc: 0.5, ready: false },
  { text: "i'm hungry", n: 0, vtt_acc: null, ready: false },
] };
const READS = {
  auto: { vtt: { text: 'turn on the lights', p: 0.91 }, action: 'auto', alternatives: [{ text: 'turn on the lights', p: 0.91 }, { text: 'turn off the lights', p: 0.06 }], latency_ms: 640 },
  confirm: { vtt: { text: 'call mom', p: 0.48 }, action: 'confirm', alternatives: [{ text: 'call mom', p: 0.48 }, { text: 'call tom', p: 0.31 }, { text: 'all done', p: 0.12 }], latency_ms: 710 },
  none: { vtt: { text: '', p: 0.08 }, action: 'none', alternatives: [], latency_ms: 520 },
};
const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost'), path = url.pathname;
  if (req.method === 'POST' && path === '/sam') {
    stub.samHits++;
    const jpeg = await readBody(req);
    if (stub.sam === 'empty') return sendJSON(res, { objects: [] });
    await sleep(SAM_DELAY_MS); // frames arriving meanwhile must still be recorded
    return sendJSON(res, { objects: await cam.samObjects(jpeg, url.searchParams.get('prompt')) });
  }
  if (req.method === 'POST' && path === '/vtt') {
    const data = await readBody(req), clip = `clip-${++stub.n}`, file = join(OUT, `${clip}.webm`);
    await writeFile(file, data);
    stub.vtt.push({ clip, user: url.searchParams.get('user'), type: req.headers['content-type'], file, size: data.length,
      flags: req.headers['x-tracker-flags'], boxes: req.headers['x-tracker-boxes'] });
    return sendJSON(res, { clip, ...READS[stub.mode] });
  }
  const m = path.match(/^\/vtt\/([^/]+)\/label$/);
  if (req.method === 'POST' && m) {
    stub.labels.push({ clip: decodeURIComponent(m[1]), ...JSON.parse(await readBody(req)) });
    return sendJSON(res, { ok: true });
  }
  if (req.method === 'GET' && path === '/readiness') return sendJSON(res, READINESS);
  return serveStatic(path, res);
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const BASE = `http://127.0.0.1:${server.address().port}`;

const checks = {};
const check = (name, ok, evidence) => { checks[name] = { ok: !!ok, ...evidence }; };
let browser = null;

async function main() {
  browser = await launch({ camera: cam.camera, profile: join(work, 'profile'), base: BASE });
  const { cdp, evaluate, key, errors, waitFor, go, holdSpace } = browser;
  const shot = (name) => browser.shot(join(OUT, `${name}.png`));
  const holdButton = (ms) => browser.holdButton('hold', ms);
  const top = 'document.querySelector("#feed .read")', cards = 'document.querySelectorAll("#feed .read").length'; // newest read card, count
  const clickIn = (sel, text) => evaluate(`(() => { const b = [...${top}.querySelectorAll(${JSON.stringify(sel)})].find((x) => x.textContent.includes(${JSON.stringify(text)})); b.click(); return !!b; })()`);

  // ---- 1. lock-on on the real face ----
  await go('lipmode.html?user=e2e');
  const locked = await waitFor('(window.__lip?.frames ?? 0) > 45 && window.__lip.quality === "ok" && !!window.__lip.last?.crop', 25000);
  await sleep(500);
  const live = await evaluate(`(() => { const ink = (id) => { const c = document.getElementById(id), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0; for (let k = 3; k < d.length; k += 4) if (d[k]) n++; return n; };
    const l = window.__lip; return { frames: l.frames, quality: l.quality, chip: document.getElementById('quality').textContent, conf: l.last.conf,
      face: l.last.face, crop: l.last.crop, overlayPixels: ink('overlay'), cropPixels: ink('crop'), fps: document.getElementById('fps').textContent }; })()`);
  check('lockOn', locked && live.overlayPixels > 0 && live.cropPixels > 0 && live.chip === 'Quality OK', { ...live, screenshot: await shot('1-lock-on') });

  // ---- 2. auto: a held take uploads the full-frame webm with flags + boxes ----
  stub.mode = 'auto';
  // 4.5 s: past T_REFRESH (4 s), so a (slow, SAM_DELAY_MS) SAM re-anchor lands inside the take; plus a 400 ms
  // main-thread stall that swallows frame callbacks. Every recorded frame must still get flags and a box.
  const t0 = Date.now();
  await key('keyDown');
  await sleep(1500);
  await evaluate('(() => { const end = performance.now() + 400; while (performance.now() < end); })()');
  await sleep(4500 - (Date.now() - t0));
  await key('keyUp');
  const heldMs = Date.now() - t0;
  const gotAuto = await waitFor(`${cards} === 1 && ${top}.querySelector('.chip').textContent === 'Sent' && !!${top}.querySelector('h2')`, 10000);
  const up = stub.vtt.at(-1);
  const { version, frames: flags } = trackerFlags(up?.flags), boxes = JSON.parse(up?.boxes ?? 'null');
  check('trackerVersion', version === (TRACKER ?? DEFAULT_TRACKER), { version, expected: TRACKER ?? DEFAULT_TRACKER });
  const take = await probeTake(up.file), pts = take.pts, duration = take.duration;
  const vstreams = take.video, astreams = take.audio;
  check('autoUpload', gotAuto && stub.vtt.length === 1 && up.user === 'e2e' && Array.isArray(flags) && Array.isArray(boxes)
    && vstreams.length === 1 && vstreams[0].width === 640 && vstreams[0].height === 480 && astreams.length === 0
    && Math.abs(duration * 1000 - heldMs) < 300,
  { heldMs, contentType: up.type, bytes: up.size, streams: take.streams, webmFrames: pts.length, durationS: +duration.toFixed(3),
    flagsFrames: flags.length, flagged: flags.filter((f) => f.length).length, boxesFrames: boxes.length, firstBoxes: boxes.slice(0, 3), samHitsSoFar: stub.samHits,
    flagsHeaderBytes: up.flags.length, boxesHeaderBytes: up.boxes.length });
  check('boxesPerFrame', Math.abs(boxes.length - pts.length) <= 2 && Math.abs(flags.length - pts.length) <= 2
    && boxes.every((b) => b.length === 5 && b.every(Number.isInteger)) && boxes.every((b, i) => !i || b[0] >= boxes[i - 1][0]),
  { webmFrames: pts.length, boxes: boxes.length, flags: flags.length, tFirstLast: [boxes[0]?.[0], boxes.at(-1)?.[0]],
    largestGapMs: Math.max(...boxes.slice(1).map((b, i) => b[0] - boxes[i][0])) });

  // the boxes sit on the mouth of the frame they belong to: match each recorded frame to its source frame
  const onMouth = await boxesOnMouth(cam, up.file, pts, boxes, join(OUT, '2b-box-on-recorded-frame.png'));
  check('boxesOnMouth', onMouth.ok, onMouth);
  const autoView = await evaluate(`(() => { const c = ${top}; return { chip: c.querySelector('.chip').textContent, text: c.querySelector('h2').textContent,
    undo: [...c.querySelectorAll('button')].map((b) => b.textContent) }; })()`);
  const autoShot = await shot('2-auto-sent');
  await clickIn('button', 'Undo');
  const undone = await waitFor(`${top}.querySelector('.chip').textContent === 'Undone' && !!${top}.querySelector('.ask')`, 5000);
  const undoLabel = stub.labels.at(-1);
  check('autoAndUndo', autoView.text === 'turn on the lights' && autoView.chip === 'Sent' && autoView.undo.includes('Undo') && undone
    && undoLabel?.clip === up.clip && undoLabel.text === 'turn on the lights' && undoLabel.source === 'undo',
  { autoView, undoLabel, afterUndo: await evaluate(`[...${top}.querySelectorAll('.ask button')].map((b) => b.textContent)`), screenshot: autoShot });

  // ---- 3. confirm via the press-and-hold button; pick an alternative ----
  stub.mode = 'confirm';
  await holdButton(1500);
  const gotConfirm = await waitFor(`${cards} === 2 && ${top}.querySelector('.chip').textContent === 'Confirm' && ${top}.querySelectorAll('.ask .opts button').length === 3`, 10000);
  const confirmView = await evaluate(`(() => { const c = ${top}; return { alts: [...c.querySelectorAll('.ask .opts button')].map((b) => b.textContent),
    placeholder: c.querySelector('.typeit input').placeholder }; })()`);
  const confirmShot = await shot('3-confirm');
  await clickIn('.ask .opts button', 'call tom');
  const confirmed = await waitFor(`${top}.querySelector('.chip').textContent === 'Sent' && ${top}.querySelector('h2')?.textContent === 'call tom'`, 5000);
  check('confirmPick', gotConfirm && confirmed && stub.vtt.length === 2 && JSON.stringify(stub.labels.at(-1)) === JSON.stringify({ clip: stub.vtt[1].clip, text: 'call tom', source: 'confirm' }),
    { confirmView, label: stub.labels.at(-1), screenshot: confirmShot });

  // ---- 4. confirm, then "none of these — type it" (Space inside the input must not start a take) ----
  await holdSpace(1200);
  const gotConfirm2 = await waitFor(`${cards} === 3 && ${top}.querySelector('.chip').textContent === 'Confirm'`, 10000);
  const vttBefore = stub.vtt.length;
  await evaluate(`${top}.querySelector('.typeit input').focus()`);
  await cdp('Input.insertText', { text: 'call' });
  await key('keyDown'); await key('keyUp');
  await cdp('Input.insertText', { text: 'ron' });
  await sleep(800);
  const typed = await evaluate(`${top}.querySelector('.typeit input').value`), recordingWhileTyping = await evaluate('document.getElementById("rec").hidden === false');
  await clickIn('.typeit button', 'Send');
  const edited = await waitFor(`${top}.querySelector('h2')?.textContent === 'call ron'`, 5000);
  check('confirmEdit', gotConfirm2 && edited && typed === 'call ron' && !recordingWhileTyping && stub.vtt.length === vttBefore
    && JSON.stringify(stub.labels.at(-1)) === JSON.stringify({ clip: stub.vtt.at(-1).clip, text: 'call ron', source: 'edit' }), { typed, label: stub.labels.at(-1) });

  // ---- 5. none: straight to the type-it input ----
  stub.mode = 'none';
  await evaluate('document.activeElement.blur()');
  await holdSpace(1200);
  const gotNone = await waitFor(`${cards} === 4 && ${top}.querySelector('.chip').textContent === 'Type it'`, 10000);
  const noneView = await evaluate(`(() => { const c = ${top}; return { alts: c.querySelectorAll('.ask .opts button').length, placeholder: c.querySelector('.typeit input')?.placeholder }; })()`);
  const noneShot = await shot('5-none');
  await evaluate(`(() => { const i = ${top}.querySelector('.typeit input'); i.value = 'good night'; i.form.requestSubmit(); })()`);
  const noneDone = await waitFor(`${top}.querySelector('h2')?.textContent === 'good night'`, 5000);
  check('none', gotNone && noneDone && noneView.alts === 0 && noneView.placeholder === 'Type what you said'
    && JSON.stringify(stub.labels.at(-1)) === JSON.stringify({ clip: stub.vtt.at(-1).clip, text: 'good night', source: 'edit' }), { noneView, label: stub.labels.at(-1), screenshot: noneShot });

  // ---- 5b. a ~10 s hold stops itself at 8 s (Muse sees at most 192 frames = 8 s at 24 fps) and uploads that take ----
  stub.mode = 'auto';
  await evaluate('document.activeElement.blur()');
  const vttCap = stub.vtt.length, tc = Date.now();
  await key('keyDown');
  await sleep(9500);
  const whileHeld = { uploads: stub.vtt.length - vttCap, recording: await evaluate('document.getElementById("rec").hidden === false') };
  await sleep(10000 - (Date.now() - tc));
  await key('keyUp');
  const capHeldMs = Date.now() - tc;
  const gotCap = await waitFor(`${cards} === 5 && ${top}.querySelector('.chip').textContent === 'Sent'`, 10000);
  await sleep(1000); // the release after the cap must not send a second take
  const capUp = stub.vtt.at(-1), capFlags = trackerFlags(capUp?.flags ?? '[]').frames, capBoxes = JSON.parse(capUp?.boxes ?? '[]');
  const capTake = await probeTake(capUp.file), capPts = capTake.pts;
  const capNote = await evaluate(`(() => { const n = [...${top}.querySelectorAll('*')].find((e) => !e.children.length && /capped at 8 s/i.test(e.textContent));
    return n ? { text: n.textContent, visible: n.checkVisibility() && n.getBoundingClientRect().height > 0 } : null; })()`);
  check('capAt8s', gotCap && whileHeld.uploads === 1 && !whileHeld.recording && stub.vtt.length === vttCap + 1
    && capTake.duration <= 8.1 && capTake.duration > 7.7
    && Math.abs(capBoxes.length - capPts.length) <= 2 && Math.abs(capFlags.length - capPts.length) <= 2 && !!capNote?.visible,
  { capHeldMs, whileHeld, uploads: stub.vtt.length - vttCap, durationS: +capTake.duration.toFixed(3), webmFrames: capPts.length,
    boxes: capBoxes.length, flags: capFlags.length, note: capNote, head: await evaluate(`${top}.querySelector('.eyebrow').textContent`), screenshot: await shot('5b-capped') });

  // ---- 6. readiness ----
  const ready = await evaluate(`[...document.querySelectorAll('#ready li')].map((li) => [...li.children].map((c) => c.textContent))`);
  check('readiness', JSON.stringify(ready) === JSON.stringify([['turn on the lights', '14 clips', 'VTT 86%', 'ready'], ['call mom', '6 clips', 'VTT 50%', 'learning'], ["i'm hungry", '0 clips', 'VTT —', 'learning']]),
    { rows: ready, summary: await evaluate('document.getElementById("ready-sum").textContent') });
  const lipErrors = errors.splice(0);

  // ---- 7. quality gate: SAM finds no face -> "Please repeat", no upload ----
  stub.sam = 'empty';
  await go('lipmode.html?user=e2e');
  await waitFor('(window.__lip?.frames ?? 0) > 30', 20000);
  const vttGate = stub.vtt.length;
  await holdSpace(1500);
  const repeated = await waitFor(`${cards} === 1 && ${top}.textContent.includes('Please repeat — mouth hidden or blurred')`, 5000);
  await sleep(1500);
  check('qualityGate', repeated && stub.vtt.length === vttGate, { quality: await evaluate('window.__lip.quality'), vttRequests: stub.vtt.length - vttGate,
    card: await evaluate(`${top}.textContent`), note: await evaluate('document.getElementById("note").textContent'), screenshot: await shot('7-please-repeat') });
  const gateErrors = errors.splice(0);

  // ---- 8. the other pages still load, all with the same rail including Lip mode ----
  const pages = {};
  for (const p of ['index.html', 'track.html', 'preheat.html', 'clips.html', 'stats.html', 'training.html', 'pipeline.html']) {
    await go(p);
    await sleep(1200);
    pages[p] = { rail: await evaluate('[...document.querySelectorAll("nav.rail a")].map((a) => a.textContent).join(",")'),
      current: await evaluate('document.querySelector("nav.rail a.on")?.textContent ?? null') };
  }
  const pageErrors = errors.splice(0);
  const exceptions = pageErrors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e.text));
  const railOk = Object.values(pages).every((v) => v.rail.split(',').includes('Lip mode') && v.rail === pages['index.html'].rail);
  check('otherPages', railOk && exceptions.length === 0, { pages, stubbed404s: pageErrors.length - exceptions.length, exceptions,
    trackerScreenshot: (await go('track.html'), await sleep(2500), await shot('8-tracker-page')) });
  check('noConsoleErrors', lipErrors.length === 0 && gateErrors.length === 0, { lipErrors, gateErrors });
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
