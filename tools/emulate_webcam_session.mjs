#!/usr/bin/env node
// Records one dictation session through the REAL page (static/index.html) in headless Chrome, with a session's
// y4m + wav (from tools/webcam_emulate.py) as the fake camera and mic. The page's own getUserMedia, Start/Stop
// button, MediaRecorder and fetch('/transcribe') do all the work; this script only clicks and waits.
//
//   node tools/emulate_webcam_session.mjs --video s.y4m --audio s.wav [--out rec.webm] [--server URL]
//                                        [--duration S] [--chrome PATH]
//
// --server URL  serve the page from URL (e.g. http://127.0.0.1:8765), so the upload goes to URL/transcribe and
//               the page shows that server's transcript. Default: a stub on 127.0.0.1:<random port> that serves
//               static/ and saves the uploaded Blob to --out (no model calls).
// --duration S  record for S seconds (default: the shorter of the two files). Chrome loops the fake devices from
//               the moment the camera opens, so the click-to-stop time is shortened by however long the page
//               waited before the click; the recording therefore never wraps around to the file's start.
// Prints one JSON object: the recorder's mimeType/bitrates, the tracks' settings, the timing, and what came back.
import { createServer } from 'node:http';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { extname, join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

const STATIC = new URL('../static/', import.meta.url);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.md': 'text/markdown' };

function parseArgs(argv) {
  const opts = { chrome: process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' };
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    if (!['video', 'audio', 'out', 'server', 'duration', 'chrome'].includes(key) || argv[i + 1] === undefined) {
      throw new Error(`bad argument ${argv[i]}\nusage: node tools/emulate_webcam_session.mjs --video s.y4m --audio s.wav [--out rec.webm] [--server URL] [--duration S] [--chrome PATH]`);
    }
    opts[key] = argv[i + 1];
  }
  if (!opts.video || !opts.audio) throw new Error('--video and --audio are required');
  opts.video = resolve(opts.video);
  opts.audio = resolve(opts.audio);
  opts.out = resolve(opts.out ?? opts.video.replace(/\.y4m$/, '') + '.recording.webm');
  return opts;
}

const probeDuration = (path) => Number(execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path]).toString().trim());

// The stub: static/ plus POST /transcribe, which saves the upload and answers like server.py's shape.
function startStub(out) {
  let upload = null;
  const server = createServer(async (req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (req.method === 'POST' && path === '/transcribe') {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = Buffer.concat(chunks);
      await writeFile(out, body);
      upload = { bytes: body.length, contentType: req.headers['content-type'] ?? null, path: out };
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ transcript: `(stub) saved ${body.length} bytes` }));
      return;
    }
    try {
      const body = await readFile(new URL(`.${path === '/' ? '/index.html' : path}`, STATIC));
      res.writeHead(200, { 'Content-Type': TYPES[extname(path)] ?? 'application/octet-stream' });
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
  const pending = new Map(), errors = [];
  ws.addEventListener('message', ({ data }) => {
    const msg = JSON.parse(data);
    if (pending.has(msg.id)) {
      const { ok, fail } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? fail(new Error(msg.error.message)) : ok(msg.result);
    } else if (msg.method === 'Runtime.exceptionThrown') {
      errors.push(msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text);
    } else if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
      errors.push(msg.params.args.map((a) => a.value ?? a.description).join(' '));
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
  const waitFor = async (expression, ms, every = 20) => {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(every)) if (await evaluate(expression)) return true;
    return false;
  };
  return { ws, send, evaluate, waitFor, errors };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const duration = Number(opts.duration ?? Math.min(probeDuration(opts.video), probeDuration(opts.audio)));
  if (!(duration > 1)) throw new Error(`duration ${duration}s is too short`);

  const stub = opts.server ? null : await startStub(opts.out);
  const origin = opts.server ?? `http://127.0.0.1:${stub.server.address().port}`;
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
    await page.send('Page.enable');
    await page.send('Page.navigate', { url: new URL('/index.html', origin).href });
    // The page enables its Start button once getUserMedia resolves ("Camera ready.").
    if (!(await page.waitFor('document.readyState !== "loading" && !document.getElementById("toggle").disabled', 20000, 5))) {
      throw new Error(`camera never became ready: ${await page.evaluate('document.getElementById("status")?.textContent')}`);
    }
    // Click Start and, in the same task, schedule Stop on the page's own clock. Where each fake device is in its file
    // at the click comes from the tracks' frame counters (the camera's totalFrames equalled the first recorded
    // frame's index in a frame-numbered test file), so stopping one frame before the further-ahead device reaches
    // the end keeps the recording inside one pass of both files.
    const timing = await page.evaluate(`(() => {
      const toggle = document.getElementById('toggle');
      const v = stream.getVideoTracks()[0], a = stream.getAudioTracks()[0];
      const fps = v.getSettings().frameRate;
      const videoAtClickMs = v.stats.totalFrames / fps * 1000, audioAtClickMs = a.stats.totalFramesDuration;
      toggle.click();
      const clickAt = performance.now();
      const recordFor = ${duration * 1000} - Math.max(videoAtClickMs, audioAtClickMs) - 1000 / fps;
      window.__emu = { clickAt, recordFor, videoAtClickMs, audioAtClickMs };
      setTimeout(() => { window.__emu.stopAt = performance.now(); toggle.click(); }, recordFor);
      return window.__emu;
    })()`);
    const tracks = await page.evaluate(`({ video: stream.getVideoTracks()[0].getSettings(),
      audio: stream.getAudioTracks()[0].getSettings(), userAgent: navigator.userAgent })`);
    const finished = opts.server
      ? 'document.getElementById("status").textContent === "Done." || document.getElementById("status").textContent.startsWith("Transcription failed")'
      : 'window.__emu.stopAt !== undefined';
    if (!(await page.waitFor(finished, timing.recordFor + 180000, 200))) throw new Error('recording/upload did not finish');
    if (stub) for (const end = Date.now() + 30000; !stub.upload() && Date.now() < end;) await sleep(100);
    const after = await page.evaluate(`({ stopAt: window.__emu.stopAt, status: document.getElementById('status').textContent,
      transcript: document.getElementById('transcript').textContent, mimeType: recorder.mimeType,
      videoBitsPerSecond: recorder.videoBitsPerSecond, audioBitsPerSecond: recorder.audioBitsPerSecond })`);
    page.ws.close();
    const { stopAt, status, transcript, ...recorder } = after;
    return {
      page: new URL('/index.html', origin).href, files: { video: opts.video, audio: opts.audio }, sessionSeconds: duration,
      upload: stub ? stub.upload() : 'sent to --server (not captured)', recorder, tracks,
      // The recording starts at file position videoAtClickMs (video) / audioAtClickMs (audio); their difference is a
      // fake-device artifact that shifts audio against video in the recording.
      timing: { ...timing, stopAt, recordedMs: stopAt - timing.clickAt },
      status, transcript, consoleErrors: page.errors,
    };
  } finally {
    chrome.kill();
    stub?.server.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}

main().then((r) => { console.log(JSON.stringify(r, null, 2)); process.exit(r.upload === null ? 1 : 0); },
  (e) => { console.error(e.message); process.exit(1); });
