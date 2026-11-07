// Headless-Chrome check of static/track.html: the fake camera plays a synthetic talking face sitting exactly
// in the box that the /sam stub always returns. Passes when overlays render, the tracker holds the face
// confidently, a forced re-anchor increments the SAM counter (reason 'forced'), and the console stays clean.
//   node tests/track-page.e2e.mjs [screenshot.png]      (CHROME=/path/to/chrome to override the binary)
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { extname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { rng, makeFace, makeBackground, render } from './synth.mjs';

const OUT = process.argv[2] ?? join(tmpdir(), 'track-page.png');
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const STATIC = new URL('../static/', import.meta.url);
const FACE = [200, 100, 240, 280]; // the stub's face box, in the page's 640x480 frame
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2' };

let samHits = 0;
const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  if (req.method === 'POST' && path === '/sam') {
    samHits++;
    req.resume().on('end', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ objects: [{ id: 1, box: FACE }] }));
    });
    return;
  }
  try {
    const body = await readFile(new URL(`.${path}`, STATIC));
    res.writeHead(200, { 'Content-Type': TYPES[extname(path)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));

const profile = await mkdtemp(join(tmpdir(), 'track-e2e-'));
const camera = join(profile, 'face.y4m');
await new Promise((resolve, reject) => { // 3 s loop of a talking face at FACE, as a y4m fake camera
  const face = makeFace(FACE[2], FACE[3], 7), bg = makeBackground(640, 480, 3), rand = rng(1);
  const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', '640x480', '-r', '30', '-i', '-', '-pix_fmt', 'yuv420p', camera]);
  for (let i = 0; i < 90; i++) {
    const f = render(640, 480, bg, [{ face, x: FACE[0], y: FACE[1], mouthOpen: 0.5 + 0.5 * Math.sin(i / 1.6) }], { rand });
    ff.stdin.write(Buffer.from(f.data.buffer));
  }
  ff.stdin.end();
  ff.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`))));
});
const chrome = spawn(CHROME, ['--headless=new', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  `--use-file-for-fake-video-capture=${camera}`, '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--window-size=1100,1700', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });

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

  let nextId = 0;
  const pending = new Map(), errors = [];
  ws.addEventListener('message', ({ data }) => {
    const msg = JSON.parse(data);
    if (pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method === 'Runtime.consoleAPICalled' && ['error', 'assert'].includes(msg.params.type)) {
      errors.push(`console.${msg.params.type}: ${msg.params.args.map((a) => a.value ?? a.description).join(' ')}`);
    } else if (msg.method === 'Runtime.exceptionThrown') {
      errors.push(`exception: ${msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text}`);
    } else if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') {
      errors.push(`log: ${msg.params.entry.text} ${msg.params.entry.url ?? ''}`);
    }
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })).result.value;
  const waitFor = async (expression, ms) => {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(100)) if (await evaluate(expression)) return true;
    return false;
  };

  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/track.html` });
  const tracking = await waitFor('(window.__track?.frames ?? 0) > 30 && !!window.__track.last?.crop', 20000);
  const before = await evaluate('window.__track.samCalls');
  await evaluate('document.getElementById("reanchor").click()');
  const incremented = await waitFor(`window.__track.samCalls > ${before} && document.getElementById('sam-why').textContent.includes('forced')`, 5000);
  await sleep(1000);
  const check = await evaluate(`(() => {
    const ink = (id) => { const c = document.getElementById(id), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let n = 0; for (let k = 3; k < d.length; k += 4) if (d[k]) n++; return n; };
    const t = window.__track;
    return { samCalls: t.samCalls, counter: document.getElementById('sam').textContent, reasons: document.getElementById('sam-why').textContent,
      overlayPixels: ink('overlay'), mouthPixels: ink('mouth'), conf: t.last.conf, face: t.last.face, crop: t.last.crop,
      status: document.getElementById('status').textContent };
  })()`);
  const { cssContentSize: size } = await send('Page.getLayoutMetrics');
  const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width: size.width, height: size.height, scale: 1 } });
  await writeFile(OUT, Buffer.from(shot.data, 'base64'));
  ws.close();

  const result = { tracking, samCallsBefore: before, incremented, serverSamHits: samHits, ...check, consoleErrors: errors, screenshot: OUT };
  console.log(JSON.stringify(result, null, 2));
  return tracking && incremented && check.counter === String(check.samCalls) && samHits >= check.samCalls
    && check.conf >= 0.5 && check.overlayPixels > 0 && check.mouthPixels > 0 && errors.length === 0;
}

let ok = false;
try {
  ok = await main();
} catch (e) {
  console.error(e);
} finally {
  chrome.kill();
  server.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
