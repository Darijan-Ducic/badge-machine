// Headless-Chrome check of static/lipmode.html on a real talking face (the public-domain talking head in
// /tmp/badge-sam, cut to a 640x480 30 fps fake camera). The /sam stub answers with the SAM ground-truth face box
// of whichever source frame the uploaded JPEG is (nearest 32x24 thumbnail), so the tracker locks on as it would
// live. /vtt, /vtt/<id>/label and /readiness are stubs that record what the page sends.
// Checks: lock-on; a held take uploads a 640x480 video-only webm lasting the hold with X-Tracker-Flags and
// X-Tracker-Boxes (one box per recorded frame, on the mouth of the matching source frame); the auto, confirm
// and none paths and the undo/confirm/edit label bodies; Space typed into the input does not record; the
// readiness panel; the quality gate blocks the upload when SAM finds no face; no console errors anywhere.
//   node tests/lipmode.e2e.mjs [out-dir]   (SOURCE=talking.mp4 GT=talking.gt.json CHROME=/path/to/chrome)
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { extname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

const SOURCE = process.env.SOURCE ?? '/tmp/badge-sam/talking.mp4';
const GT = process.env.GT ?? '/tmp/badge-sam/talking.gt.json';
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const STATIC = new URL('../static/', import.meta.url);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.md': 'text/markdown' };
const SAM_DELAY_MS = Number(process.env.SAM_DELAY_MS ?? 300);
const TW = 32, TH = 24; // thumbnail size for frame matching
// source 1280x720 -> center 960x720 -> 640x480
const CROP_X = 160, SCALE = 2 / 3;
const toCam = ([x, y, w, h]) => [(x - CROP_X) * SCALE, y * SCALE, w * SCALE, h * SCALE];

const work = await mkdtemp(join(tmpdir(), 'lipmode-e2e-'));
const OUT = process.argv[2] ?? join(work, 'out');
await mkdir(OUT, { recursive: true });

function run(cmd, args, input) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args), out = [], err = [];
    p.stdout.on('data', (d) => out.push(d));
    p.stderr.on('data', (d) => err.push(d));
    p.on('close', (code) => (code === 0 ? resolve(Buffer.concat(out)) : reject(new Error(`${cmd} ${code}: ${Buffer.concat(err)}`))));
    if (input) p.stdin.end(input); else p.stdin.end();
  });
}
const thumbs = (buf) => Array.from({ length: buf.length / (TW * TH) }, (_, i) => buf.subarray(i * TW * TH, (i + 1) * TW * TH));
function nearest(t, refs) {
  let best = -1, bd = Infinity;
  refs.forEach((r, i) => { let d = 0; for (let k = 0; k < t.length; k++) d += (t[k] - r[k]) ** 2; if (d < bd) { bd = d; best = i; } });
  return best;
}

// ---------- the fake camera and its ground truth ----------
const camera = join(work, 'talking-640x480.y4m');
await run('ffmpeg', ['-v', 'error', '-y', '-t', '10', '-i', SOURCE, '-an', '-vf', 'crop=960:720:160:0,scale=640:480,fps=30', '-pix_fmt', 'yuv420p', camera]);
const camThumbs = thumbs(await run('ffmpeg', ['-v', 'error', '-i', camera, '-vf', `scale=${TW}:${TH}`, '-f', 'rawvideo', '-pix_fmt', 'gray', '-']));
const gt = JSON.parse(await readFile(GT, 'utf8'));
const gtAt = (j) => gt.frames[Math.min(gt.frames.length - 1, Math.round((j / 30) * gt.fps))]; // camera frame -> source frame
const matchImage = async (buf, fmt) => nearest(await run('ffmpeg', ['-v', 'error', '-f', fmt, '-i', '-', '-vf', `scale=${TW}:${TH}`, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], buf), camThumbs);

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
const body = (req) => new Promise((resolve) => { const c = []; req.on('data', (d) => c.push(d)); req.on('end', () => resolve(Buffer.concat(c))); });
const send = (res, obj) => { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(obj)); };
const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost'), path = url.pathname;
  if (req.method === 'POST' && path === '/sam') {
    stub.samHits++;
    const jpeg = await body(req);
    if (stub.sam === 'empty') return send(res, { objects: [] });
    await sleep(SAM_DELAY_MS); // the hosted SAM call takes this long: frames arriving meanwhile must still be recorded
    const j = await matchImage(jpeg, 'jpeg_pipe');
    return send(res, { objects: [{ id: 1, box: toCam(gtAt(j).face).map(Math.round) }] });
  }
  if (req.method === 'POST' && path === '/vtt') {
    const data = await body(req), clip = `clip-${++stub.n}`, file = join(OUT, `${clip}.webm`);
    await writeFile(file, data);
    stub.vtt.push({ clip, user: url.searchParams.get('user'), type: req.headers['content-type'], file, size: data.length,
      flags: req.headers['x-tracker-flags'], boxes: req.headers['x-tracker-boxes'] });
    return send(res, { clip, ...READS[stub.mode] });
  }
  const m = path.match(/^\/vtt\/([^/]+)\/label$/);
  if (req.method === 'POST' && m) {
    stub.labels.push({ clip: decodeURIComponent(m[1]), ...JSON.parse(await body(req)) });
    return send(res, { ok: true });
  }
  if (req.method === 'GET' && path === '/readiness') return send(res, READINESS);
  try {
    const file = await readFile(new URL(`.${path === '/' ? '/index.html' : path}`, STATIC));
    res.writeHead(200, { 'Content-Type': TYPES[extname(path)] ?? 'application/octet-stream' });
    res.end(file);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const BASE = `http://127.0.0.1:${server.address().port}`;

const chrome = spawn(CHROME, ['--headless=new', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  `--use-file-for-fake-video-capture=${camera}`, '--remote-debugging-port=0', `--user-data-dir=${join(work, 'profile')}`,
  '--window-size=1280,1000', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });

const checks = {};
const check = (name, ok, evidence) => { checks[name] = { ok: !!ok, ...evidence }; };

async function main() {
  const wsUrl = await new Promise((resolve, reject) => {
    let log = '';
    chrome.stderr.on('data', (d) => { log += d; const m = log.match(/DevTools listening on (ws:\/\/\S+)/); if (m) resolve(m[1]); });
    chrome.on('exit', (code) => reject(new Error(`chrome exited (${code})`)));
    setTimeout(() => reject(new Error('chrome did not start')), 20000);
  });
  const targets = await (await fetch(`${wsUrl.replace(/^ws:\/\/([^/]+).*/, 'http://$1')}/json/list`)).json();
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve) => ws.addEventListener('open', resolve, { once: true }));

  let nextId = 0, page = 'about:blank';
  const pending = new Map(), errors = [];
  ws.addEventListener('message', ({ data }) => {
    const msg = JSON.parse(data);
    if (pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method === 'Runtime.consoleAPICalled' && ['error', 'assert', 'warning'].includes(msg.params.type)) {
      errors.push({ page, text: `console.${msg.params.type}: ${msg.params.args.map((a) => a.value ?? a.description).join(' ')}` });
    } else if (msg.method === 'Runtime.exceptionThrown') {
      errors.push({ page, text: `exception: ${msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text}` });
    } else if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') {
      errors.push({ page, text: `log: ${msg.params.entry.text} ${msg.params.entry.url ?? ''}` });
    }
  });
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(`evaluate: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
    return r.result.value;
  };
  const waitFor = async (expression, ms) => {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(100)) if (await evaluate(expression)) return true;
    return false;
  };
  const go = async (path) => { page = path; await cdp('Page.navigate', { url: `${BASE}/${path}` }); await sleep(300); };
  const shot = async (name) => {
    const { cssContentSize: size } = await cdp('Page.getLayoutMetrics');
    const s = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width: size.width, height: size.height, scale: 1 } });
    const file = join(OUT, `${name}.png`);
    await writeFile(file, Buffer.from(s.data, 'base64'));
    return file;
  };
  const key = (type) => cdp('Input.dispatchKeyEvent', { type, key: ' ', code: 'Space', windowsVirtualKeyCode: 32, nativeVirtualKeyCode: 32, ...(type === 'keyDown' ? { text: ' ' } : {}) });
  const holdSpace = async (ms) => { const t0 = Date.now(); await key('keyDown'); await sleep(ms); await key('keyUp'); return Date.now() - t0; };
  const holdButton = async (ms) => {
    const [x, y] = await evaluate('(() => { const r = document.getElementById("hold").getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()');
    const t0 = Date.now();
    await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', buttons: 1, clickCount: 1 });
    await sleep(ms);
    await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', buttons: 0, clickCount: 1 });
    return Date.now() - t0;
  };
  const top = 'document.querySelector("#feed .read")', cards = 'document.querySelectorAll("#feed .read").length'; // newest read card, count
  const clickIn = (sel, text) => evaluate(`(() => { const b = [...${top}.querySelectorAll(${JSON.stringify(sel)})].find((x) => x.textContent.includes(${JSON.stringify(text)})); b.click(); return !!b; })()`);

  await cdp('Runtime.enable');
  await cdp('Log.enable');
  await cdp('Page.enable');

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
  const flags = JSON.parse(up?.flags ?? 'null'), boxes = JSON.parse(up?.boxes ?? 'null');
  const probe = JSON.parse((await run('ffprobe', ['-v', 'error', '-show_entries', 'stream=codec_type,codec_name,width,height', '-of', 'json', up.file])).toString()).streams;
  const pts = (await run('ffprobe', ['-v', 'error', '-select_streams', 'v', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', up.file])).toString().trim().split('\n').map(Number);
  const duration = pts.at(-1) + (pts.at(-1) - pts[0]) / Math.max(1, pts.length - 1);
  const vstreams = probe.filter((s) => s.codec_type === 'video'), astreams = probe.filter((s) => s.codec_type === 'audio');
  check('autoUpload', gotAuto && stub.vtt.length === 1 && up.user === 'e2e' && Array.isArray(flags) && Array.isArray(boxes)
    && vstreams.length === 1 && vstreams[0].width === 640 && vstreams[0].height === 480 && astreams.length === 0
    && Math.abs(duration * 1000 - heldMs) < 300,
  { heldMs, contentType: up.type, bytes: up.size, streams: probe, webmFrames: pts.length, durationS: +duration.toFixed(3),
    flagsFrames: flags.length, flagged: flags.filter((f) => f.length).length, boxesFrames: boxes.length, firstBoxes: boxes.slice(0, 3), samHitsSoFar: stub.samHits,
    flagsHeaderBytes: up.flags.length, boxesHeaderBytes: up.boxes.length });
  check('boxesPerFrame', Math.abs(boxes.length - pts.length) <= 2 && Math.abs(flags.length - pts.length) <= 2
    && boxes.every((b) => b.length === 5 && b.every(Number.isInteger)) && boxes.every((b, i) => !i || b[0] >= boxes[i - 1][0]),
  { webmFrames: pts.length, boxes: boxes.length, flags: flags.length, tFirstLast: [boxes[0]?.[0], boxes.at(-1)?.[0]],
    largestGapMs: Math.max(...boxes.slice(1).map((b, i) => b[0] - boxes[i][0])) });

  // the boxes sit on the mouth of the frame they belong to: match the recorded frame to its source frame
  const recThumbs = thumbs(await run('ffmpeg', ['-v', 'error', '-i', up.file, '-vf', `scale=${TW}:${TH}`, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']));
  const samples = [];
  for (const i of [0, Math.floor(boxes.length / 4), Math.floor(boxes.length / 2), Math.floor((3 * boxes.length) / 4), boxes.length - 1]) {
    const [t, x, y, w, h] = boxes[i];
    let f = 0;
    pts.forEach((p, k) => { if (Math.abs(p * 1000 - t) < Math.abs(pts[f] * 1000 - t)) f = k; });
    const j = nearest(recThumbs[f], camThumbs), g = gtAt(j), face = toCam(g.face), mouth = g.mouth && toCam(g.mouth);
    const [cx, cy] = [x + w / 2, y + h / 2], inside = (px, py, b) => px >= b[0] && px <= b[0] + b[2] && py >= b[1] && py <= b[1] + b[3];
    const mc = mouth && [mouth[0] + mouth[2] / 2, mouth[1] + mouth[3] / 2];
    samples.push({ i, t, box: [x, y, w, h], webmFrame: f, sourceFrame: j, gtFace: face.map(Math.round), gtMouth: mouth?.map(Math.round) ?? null,
      boxCenterInFace: inside(cx, cy, face), gtMouthCenterInBox: mc ? inside(mc[0], mc[1], [x, y, w, h]) : null,
      mouthOffsetPx: mc ? Math.round(Math.hypot(cx - mc[0], cy - mc[1])) : null });
    if (i === Math.floor(boxes.length / 2)) {
      await run('ffmpeg', ['-v', 'error', '-y', '-i', up.file, '-vf', `select=eq(n\\,${f}),drawbox=x=${x}:y=${y}:w=${w}:h=${h}:color=lime:t=3${mouth ? `,drawbox=x=${Math.round(mouth[0])}:y=${Math.round(mouth[1])}:w=${Math.round(mouth[2])}:h=${Math.round(mouth[3])}:color=red:t=2` : ''}`,
        '-frames:v', '1', join(OUT, '2b-box-on-recorded-frame.png')]);
    }
  }
  check('boxesOnMouth', samples.every((s) => s.boxCenterInFace && s.gtMouthCenterInBox !== false), { samples, image: join(OUT, '2b-box-on-recorded-frame.png') });
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

  // ---- 8. the other pages still load, with Lip mode right after Record in the rail ----
  const pages = {};
  for (const p of ['index.html', 'track.html', 'preheat.html', 'clips.html', 'stats.html', 'training.html', 'pipeline.html']) {
    await go(p);
    await sleep(1200);
    pages[p] = { rail: await evaluate('[...document.querySelectorAll("nav.rail a")].map((a) => a.textContent).join(",")'),
      current: await evaluate('document.querySelector("nav.rail a.on")?.textContent ?? null') };
  }
  const pageErrors = errors.splice(0);
  const exceptions = pageErrors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e.text));
  const railOk = Object.values(pages).every((v) => v.rail.startsWith('Record,Lip mode,'));
  check('otherPages', railOk && exceptions.length === 0, { pages, stubbed404s: pageErrors.length - exceptions.length, exceptions,
    trackerScreenshot: (await go('track.html'), await sleep(2500), await shot('8-tracker-page')) });
  check('noConsoleErrors', lipErrors.length === 0 && gateErrors.length === 0, { lipErrors, gateErrors });
  ws.close();
}

let ok = false;
try {
  await main();
  ok = Object.values(checks).every((c) => c.ok);
} catch (e) {
  console.error(e);
} finally {
  chrome.kill();
  server.close();
  await rm(join(work, 'profile'), { recursive: true, force: true }).catch(() => {});
  await rm(camera, { force: true }).catch(() => {});
}
console.log(JSON.stringify(checks, null, 2));
for (const [k, v] of Object.entries(checks)) console.log(`${v.ok ? 'PASS' : 'FAIL'} ${k}`);
console.log(`out: ${OUT}`);
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
