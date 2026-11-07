// Shared pieces of the headless-Chrome checks that drive a capture page (lipmode, dictation) with a REAL talking
// face: the public-domain talking head in /tmp/badge-sam cut to a 640x480 30 fps fake camera (plus its audio as a
// fake mic), SAM ground truth mapped onto it, a /sam stub that answers with the ground-truth face of whichever
// source frame the uploaded JPEG is (nearest 32x24 thumbnail), a CDP driver, and probes for the uploaded take.
import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

export const SOURCE = process.env.SOURCE ?? '/tmp/badge-sam/talking.mp4';
export const GT = process.env.GT ?? '/tmp/badge-sam/talking.gt.json';
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const STATIC = new URL('../static/', import.meta.url);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.md': 'text/markdown' };
const TW = 32, TH = 24; // thumbnail size for frame matching
const CROP_X = 160, SCALE = 2 / 3; // source 1280x720 -> center 960x720 -> 640x480

export function run(cmd, args, input) {
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
const gray = (fmt, extra = []) => ['-v', 'error', ...(fmt ? ['-f', fmt] : []), '-i', '-', '-vf', `scale=${TW}:${TH}`, ...extra, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'];

// The first 10 s of SOURCE as a 640x480 30 fps y4m (and, with audio, a mono 48 kHz 16-bit wav) in dir.
export async function fakeCamera(dir, { audio = false } = {}) {
  const camera = join(dir, 'talking-640x480.y4m'), wav = audio ? join(dir, 'talking.wav') : null;
  await run('ffmpeg', ['-v', 'error', '-y', '-t', '10', '-i', SOURCE, '-an', '-vf', 'crop=960:720:160:0,scale=640:480,fps=30', '-pix_fmt', 'yuv420p', camera]);
  if (wav) await run('ffmpeg', ['-v', 'error', '-y', '-t', '10', '-i', SOURCE, '-vn', '-ac', '1', '-ar', '48000', '-c:a', 'pcm_s16le', wav]);
  const camThumbs = thumbs(await run('ffmpeg', ['-v', 'error', '-i', camera, '-vf', `scale=${TW}:${TH}`, '-f', 'rawvideo', '-pix_fmt', 'gray', '-']));
  const gt = JSON.parse(await readFile(GT, 'utf8'));
  const toCam = ([x, y, w, h]) => [(x - CROP_X) * SCALE, y * SCALE, w * SCALE, h * SCALE];
  const gtAt = (j) => gt.frames[Math.min(gt.frames.length - 1, Math.round((j / 30) * gt.fps))]; // camera frame -> source frame
  const frameOf = async (jpeg) => nearest(await run('ffmpeg', gray('jpeg_pipe'), jpeg), camThumbs);
  return {
    camera, wav, toCam, gtAt,
    // what SAM would say for this JPEG of the camera: the ground-truth face of its source frame
    samBox: async (jpeg) => toCam(gtAt(await frameOf(jpeg)).face).map(Math.round),
    // per recorded frame of a webm, its camera frame index
    recordedFrames: async (file) => thumbs(await run('ffmpeg', ['-v', 'error', '-i', file, '-vf', `scale=${TW}:${TH}`, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']))
      .map((t) => nearest(t, camThumbs)),
  };
}

export async function serveStatic(path, res) {
  try {
    const file = await readFile(new URL(`.${path === '/' ? '/index.html' : path}`, STATIC));
    res.writeHead(200, { 'Content-Type': TYPES[extname(path)] ?? 'application/octet-stream' });
    res.end(file);
  } catch {
    res.writeHead(404);
    res.end();
  }
}
export const readBody = (req) => new Promise((resolve) => { const c = []; req.on('data', (d) => c.push(d)); req.on('end', () => resolve(Buffer.concat(c))); });
export const sendJSON = (res, obj) => { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(obj)); };

// Streams, per-frame pts and duration of an uploaded take; mean volume of its audio, if any.
export async function probeTake(file) {
  const streams = JSON.parse((await run('ffprobe', ['-v', 'error', '-show_entries', 'stream=codec_type,codec_name,width,height,sample_rate,channels', '-of', 'json', file])).toString()).streams;
  const pts = (await run('ffprobe', ['-v', 'error', '-select_streams', 'v', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', file])).toString().trim().split('\n').map(Number);
  const duration = pts.at(-1) + (pts.at(-1) - pts[0]) / Math.max(1, pts.length - 1);
  let meanVolumeDb = null;
  if (streams.some((s) => s.codec_type === 'audio')) {
    const log = await new Promise((resolve) => { const p = spawn('ffmpeg', ['-i', file, '-vn', '-af', 'volumedetect', '-f', 'null', '-']); let e = ''; p.stderr.on('data', (d) => { e += d; }); p.on('close', () => resolve(e)); });
    meanVolumeDb = Number(log.match(/mean_volume: (-?[\d.]+) dB/)?.[1] ?? NaN);
  }
  return { streams, video: streams.filter((s) => s.codec_type === 'video'), audio: streams.filter((s) => s.codec_type === 'audio'), pts, duration, meanVolumeDb };
}

// Five boxes across the take, each against SAM's ground truth for the source frame that was actually recorded at
// its time; draws the middle one (green) and the ground-truth lips (red) on that recorded frame into image.
export async function boxesOnMouth(cam, file, pts, boxes, image) {
  const recorded = await cam.recordedFrames(file), samples = [];
  const mid = Math.floor(boxes.length / 2);
  for (const i of [0, Math.floor(boxes.length / 4), mid, Math.floor((3 * boxes.length) / 4), boxes.length - 1]) {
    const [t, x, y, w, h] = boxes[i];
    let f = 0;
    pts.forEach((p, k) => { if (Math.abs(p * 1000 - t) < Math.abs(pts[f] * 1000 - t)) f = k; });
    const j = recorded[f], g = cam.gtAt(j), face = cam.toCam(g.face), mouth = g.mouth && cam.toCam(g.mouth);
    const inside = (px, py, b) => px >= b[0] && px <= b[0] + b[2] && py >= b[1] && py <= b[1] + b[3];
    const [cx, cy] = [x + w / 2, y + h / 2], mc = mouth && [mouth[0] + mouth[2] / 2, mouth[1] + mouth[3] / 2];
    samples.push({ i, t, box: [x, y, w, h], webmFrame: f, sourceFrame: j, gtFace: face.map(Math.round), gtMouth: mouth?.map(Math.round) ?? null,
      boxCenterInFace: inside(cx, cy, face), gtMouthCenterInBox: mc ? inside(mc[0], mc[1], [x, y, w, h]) : null,
      mouthOffsetPx: mc ? Math.round(Math.hypot(cx - mc[0], cy - mc[1])) : null });
    if (i === mid) {
      const lips = mouth ? `,drawbox=x=${Math.round(mouth[0])}:y=${Math.round(mouth[1])}:w=${Math.round(mouth[2])}:h=${Math.round(mouth[3])}:color=red:t=2` : '';
      await run('ffmpeg', ['-v', 'error', '-y', '-i', file, '-vf', `select=eq(n\\,${f}),drawbox=x=${x}:y=${y}:w=${w}:h=${h}:color=lime:t=3${lips}`, '-frames:v', '1', image]);
    }
  }
  return { ok: samples.every((s) => s.boxCenterInFace && s.gtMouthCenterInBox !== false), samples, image };
}

// Headless Chrome on the fake camera (and mic) with a CDP driver. errors collects console errors/warnings,
// exceptions and failed loads, each tagged with the page that was open.
export async function launch({ camera, wav, profile, base }) {
  const chrome = spawn(CHROME, ['--headless=new', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${camera}`, ...(wav ? [`--use-file-for-fake-audio-capture=${wav}`] : []), '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--window-size=1280,1000', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
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
  const key = (type) => cdp('Input.dispatchKeyEvent', { type, key: ' ', code: 'Space', windowsVirtualKeyCode: 32, nativeVirtualKeyCode: 32, ...(type === 'keyDown' ? { text: ' ' } : {}) });
  await cdp('Runtime.enable');
  await cdp('Log.enable');
  await cdp('Page.enable');

  return {
    cdp, evaluate, key, errors,
    async waitFor(expression, ms) {
      for (const end = Date.now() + ms; Date.now() < end; await sleep(100)) if (await evaluate(expression)) return true;
      return false;
    },
    async go(path) { page = path; await cdp('Page.navigate', { url: `${base}/${path}` }); await sleep(300); },
    async shot(file) {
      const { cssContentSize: size } = await cdp('Page.getLayoutMetrics');
      const s = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width: size.width, height: size.height, scale: 1 } });
      await writeFile(file, Buffer.from(s.data, 'base64'));
      return file;
    },
    async holdSpace(ms) { const t0 = Date.now(); await key('keyDown'); await sleep(ms); await key('keyUp'); return Date.now() - t0; },
    async holdButton(id, ms) {
      const [x, y] = await evaluate(`(() => { const r = document.getElementById(${JSON.stringify(id)}).getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()`);
      const t0 = Date.now();
      await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', buttons: 1, clickCount: 1 });
      await sleep(ms);
      await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', buttons: 0, clickCount: 1 });
      return Date.now() - t0;
    },
    close() { ws.close(); chrome.kill(); },
  };
}
