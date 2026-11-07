#!/usr/bin/env node
// Records one dictation session through a REAL recording page in headless Chrome, with a session's y4m + wav (from
// tools/webcam_emulate.py) as the fake camera and mic. The page's own getUserMedia, record control, MediaRecorder and
// upload do all the work; this script only presses the control, waits, and reports what was sent.
//
//   node tools/emulate_webcam_session.mjs --video s.y4m --audio s.wav [--out rec.webm] [--server URL]
//        [--page /dictation.html] [--user practice:obama-weekly:obama] [--control toggle:#toggle | space:#hold]
//        [--duration S] [--chrome PATH]
//
// --server URL  open the page on URL (e.g. http://127.0.0.1:8765), so its upload goes to that server. Default: a stub on
//               127.0.0.1:<random port> that serves static/ and saves any POST body to --out, with the request line and
//               headers in <out>.request.json (no model calls).
// --page        page path, opened as <page>?user=<--user>. Default /dictation.html.
// --control     how the page records: toggle:<selector> clicks the button to start and again to stop (index.html);
//               space:<selector> holds the Space key (preheat.html); the selector's button must be enabled first.
//               Default by page: /preheat.html -> space:#hold, otherwise toggle:#toggle.
// --duration S  record for S seconds (default: the shorter file). Chrome loops the fake devices from the moment the
//               camera opens; the recording stops one frame before the further-ahead device reaches the end of its
//               file (from the tracks' frame counters), so it never wraps around to the start.
// Prints one JSON object: what was uploaded (bytes, Content-Type, headers, response), track settings, timing, and the
// wall-clock time of the whole run.
import { createServer } from 'node:http';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { extname, join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

const STARTED = Date.now();
const STATIC = new URL('../static/', import.meta.url);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.md': 'text/markdown' };
const FLAGS = ['video', 'audio', 'out', 'server', 'page', 'user', 'control', 'duration', 'chrome'];
const USAGE = 'usage: node tools/emulate_webcam_session.mjs --video s.y4m --audio s.wav [--out rec.webm] [--server URL] '
  + '[--page /dictation.html] [--user ID] [--control toggle:SEL|space:SEL] [--duration S] [--chrome PATH]';

function parseArgs(argv) {
  const opts = {
    chrome: process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    page: '/dictation.html', user: 'practice:obama-weekly:obama',
  };
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    if (!FLAGS.includes(key) || argv[i + 1] === undefined) throw new Error(`bad argument ${argv[i]}\n${USAGE}`);
    opts[key] = argv[i + 1];
  }
  if (!opts.video || !opts.audio) throw new Error(`--video and --audio are required\n${USAGE}`);
  opts.video = resolve(opts.video);
  opts.audio = resolve(opts.audio);
  opts.out = resolve(opts.out ?? opts.video.replace(/\.y4m$/, '') + '.recording.webm');
  opts.control ??= opts.page.endsWith('/preheat.html') ? 'space:#hold' : 'toggle:#toggle';
  const [mode, selector] = [opts.control.slice(0, opts.control.indexOf(':')), opts.control.slice(opts.control.indexOf(':') + 1)];
  if (!['toggle', 'space'].includes(mode) || !selector) throw new Error(`--control must be toggle:<selector> or space:<selector>`);
  opts.control = { mode, selector };
  return opts;
}

const probeDuration = (path) => Number(execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path]).toString().trim());

// The stub: static/, a placeholder phrase list (preheat.html waits for one), and any POST saved with its headers.
function startStub(out) {
  let upload = null;
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const json = (code, body) => { res.writeHead(code, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };
    if (req.method === 'POST') {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = Buffer.concat(chunks);
      await writeFile(out, body);
      upload = { method: req.method, path: url.pathname, query: Object.fromEntries(url.searchParams), headers: req.headers,
        bytes: body.length, savedTo: out };
      await writeFile(`${out}.request.json`, JSON.stringify(upload, null, 2));
      json(200, { transcript: `(stub) saved ${body.length} bytes`, stub: true });
      return;
    }
    if (url.pathname === '/phrases') return json(200, { phrases: ['(practice session)'] });
    try {
      const body = await readFile(new URL(`.${url.pathname === '/' ? '/index.html' : url.pathname}`, STATIC));
      res.writeHead(200, { 'Content-Type': TYPES[extname(url.pathname)] ?? 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  return new Promise((ok) => server.listen(0, '127.0.0.1', () => ok({ server, upload: () => upload })));
}

async function cdp(chrome) {
  const wsUrl = await new Promise((ok, fail) => {
    let log = '';
    chrome.stderr.on('data', (d) => { log += d; const m = log.match(/DevTools listening on (ws:\/\/\S+)/); if (m) ok(m[1]); });
    chrome.on('exit', (code) => fail(new Error(`chrome exited (${code}): ${log.slice(-500)}`)));
    setTimeout(() => fail(new Error('chrome did not start')), 20000);
  });
  const targets = await (await fetch(`${wsUrl.replace(/^ws:\/\/([^/]+).*/, 'http://$1')}/json/list`)).json();
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((ok) => ws.addEventListener('open', ok, { once: true }));
  let nextId = 0;
  const pending = new Map(), errors = [], posts = new Map();
  ws.addEventListener('message', ({ data }) => {
    const msg = JSON.parse(data);
    if (pending.has(msg.id)) {
      const { ok, fail } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? fail(new Error(msg.error.message)) : ok(msg.result);
      return;
    }
    const p = msg.params;
    if (msg.method === 'Runtime.exceptionThrown') errors.push(p.exceptionDetails.exception?.description ?? p.exceptionDetails.text);
    else if (msg.method === 'Runtime.consoleAPICalled' && p.type === 'error') errors.push(p.args.map((a) => a.value ?? a.description).join(' '));
    else if (msg.method === 'Network.requestWillBeSent' && p.request.method === 'POST') {
      posts.set(p.requestId, { url: p.request.url, headers: p.request.headers, sentAt: p.timestamp });
    } else if (msg.method === 'Network.responseReceived' && posts.has(p.requestId)) {
      posts.get(p.requestId).status = p.response.status;
    } else if ((msg.method === 'Network.loadingFinished' || msg.method === 'Network.loadingFailed') && posts.has(p.requestId)) {
      Object.assign(posts.get(p.requestId), { done: true, failed: p.errorText ?? null, finishedAt: p.timestamp });
    }
  });
  const send = (method, params = {}) => new Promise((ok, fail) => {
    const id = ++nextId;
    pending.set(id, { ok, fail });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(`page: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
    return r.result.value;
  };
  const waitFor = async (check, ms, every = 20) => {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(every)) if (await check()) return true;
    return false;
  };
  return { ws, send, evaluate, waitFor, errors, posts };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const duration = Number(opts.duration ?? Math.min(probeDuration(opts.video), probeDuration(opts.audio)));
  if (!(duration > 1)) throw new Error(`duration ${duration}s is too short`);

  const stub = opts.server ? null : await startStub(opts.out);
  const pageUrl = new URL(opts.page, opts.server ?? `http://127.0.0.1:${stub.server.address().port}`);
  pageUrl.searchParams.set('user', opts.user);
  const profile = await mkdtemp(join(tmpdir(), 'webcam-session-'));
  const chrome = spawn(opts.chrome, ['--headless=new', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    // The audio service's sandbox cannot open the wav ("Failed to read ... as input to the fake device").
    '--disable-features=AudioServiceSandbox',
    `--use-file-for-fake-video-capture=${opts.video}`, `--use-file-for-fake-audio-capture=${opts.audio}`,
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
    ...(process.env.CHROME_FLAGS ?? '').split(' ').filter(Boolean), 'about:blank'],
  { stdio: ['ignore', 'ignore', 'pipe'] });
  if (process.env.CHROME_LOG) chrome.stderr.on('data', (d) => process.stderr.write(d));
  try {
    const page = await cdp(chrome);
    await page.send('Runtime.enable');
    await page.send('Network.enable');
    await page.send('Page.enable');
    await page.send('Page.navigate', { url: pageUrl.href });
    const { mode, selector } = opts.control;
    const sel = JSON.stringify(selector);
    // The camera stream as the page previews it (or its global `stream`), and its enabled record control.
    const STREAM = `([...document.querySelectorAll('video')].map((v) => v.srcObject).find(Boolean) ?? (typeof stream !== 'undefined' ? stream : null))`;
    const ready = await page.waitFor(() => page.evaluate(`(() => { const b = document.querySelector(${sel});
      return document.readyState !== 'loading' && !!b && !b.disabled && !!${STREAM}?.getVideoTracks()[0]?.stats; })()`), 20000, 5);
    if (!ready) throw new Error(`not ready: no enabled ${selector} with a live camera preview on ${pageUrl.href}`);
    const press = mode === 'toggle' ? `document.querySelector(${sel}).click()` : `dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', key: ' ' }))`;
    const release = mode === 'toggle' ? press : `dispatchEvent(new KeyboardEvent('keyup', { code: 'Space', key: ' ' }))`;
    // Start and, in the same task, schedule the stop on the page's own clock. Each device's position in its file at the
    // start comes from the track frame counters (the camera's totalFrames equalled the first recorded frame's index in
    // a frame-numbered test file).
    const timing = await page.evaluate(`(() => {
      const s = ${STREAM}, v = s.getVideoTracks()[0], a = s.getAudioTracks()[0];
      const fps = v.getSettings().frameRate;
      const videoAtStartMs = v.stats.totalFrames / fps * 1000, audioAtStartMs = a.stats.totalFramesDuration;
      ${press};
      const startAt = performance.now();
      const recordFor = ${duration * 1000} - Math.max(videoAtStartMs, audioAtStartMs) - 1000 / fps;
      window.__emu = { startAt, recordFor, videoAtStartMs, audioAtStartMs };
      setTimeout(() => { window.__emu.stopAt = performance.now(); ${release}; }, recordFor);
      return window.__emu;
    })()`);
    const tracks = await page.evaluate(`(() => { const s = ${STREAM};
      return { video: s.getVideoTracks()[0].getSettings(), audio: s.getAudioTracks()[0].getSettings(), userAgent: navigator.userAgent }; })()`);
    for (const t of Object.values(tracks)) if (t && typeof t === 'object') { delete t.deviceId; delete t.groupId; }
    // The take's upload: the first finished POST that isn't the tracker's SAM re-anchor (pages POST /sam while live).
    const isTake = (p) => p.done && new URL(p.url).pathname !== '/sam';
    const uploaded = () => [...page.posts.values()].find(isTake);
    if (!(await page.waitFor(async () => uploaded() && (!stub || stub.upload()), timing.recordFor + 240000, 100))) {
      throw new Error('no upload finished');
    }
    const [requestId, post] = [...page.posts.entries()].find(([, p]) => isTake(p));
    const body = await page.send('Network.getResponseBody', { requestId }).then((r) => r.body.slice(0, 2000), () => null);
    const stopAt = await page.evaluate('window.__emu.stopAt');
    const recorder = await page.evaluate(`typeof recorder !== 'undefined' && recorder ? { mimeType: recorder.mimeType,
      videoBitsPerSecond: recorder.videoBitsPerSecond, audioBitsPerSecond: recorder.audioBitsPerSecond } : null`);
    page.ws.close();
    return {
      page: pageUrl.href, control: opts.control, files: { video: opts.video, audio: opts.audio }, sessionSeconds: duration,
      upload: { url: post.url, status: post.status, failed: post.failed, requestHeaders: post.headers, response: body,
        uploadSeconds: +(post.finishedAt - post.sentAt).toFixed(3), saved: stub ? stub.upload() : null },
      recorder, tracks,
      // The recording starts at file position videoAtStartMs (video) / audioAtStartMs (audio).
      timing: { ...timing, stopAt, recordedMs: stopAt - timing.startAt },
      consoleErrors: page.errors, wallClockSeconds: (Date.now() - STARTED) / 1000,
    };
  } finally {
    chrome.kill();
    stub?.server.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}

main().then((r) => { console.log(JSON.stringify(r, null, 2)); process.exit(r.upload.failed || r.upload.status >= 400 ? 1 : 0); },
  (e) => { console.error(e.message); process.exit(1); });
