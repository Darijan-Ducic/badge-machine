// Synthetic grayscale scenes for the tracker tests and the replay demo clip:
// textured "faces" (smooth random blobs + eyes/nose + a mouth that can open) drawn at sub-pixel
// position and scale over a static background, with per-frame sensor noise and optional occluders.
// CLI: node tests/synth.mjs <out.mp4> <out.gt.json> [seconds]  → an encoded clip + ground truth.
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// Where the synthetic mouth is drawn, as fractions of the face box (matches the tracker's default ratios).
export const MOUTH = { cx: 0.5, cy: 0.78, w: 0.3, h: 0.1 };

function blob(lum, w, h, bx, by, rx, ry, amp) {
  const x0 = Math.max(0, Math.floor(bx - 3 * rx)), x1 = Math.min(w, Math.ceil(bx + 3 * rx));
  const y0 = Math.max(0, Math.floor(by - 3 * ry)), y1 = Math.min(h, Math.ceil(by + 3 * ry));
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const dx = (x + 0.5 - bx) / rx, dy = (y + 0.5 - by) / ry;
      lum[y * w + x] += amp * Math.exp(-0.5 * (dx * dx + dy * dy));
    }
  }
}

export function makeFace(w, h, seed = 1) {
  const rand = rng(seed);
  const lum = new Float32Array(w * h).fill(150);
  const alpha = new Float32Array(w * h);
  for (let k = 0; k < 90; k++) {
    const s = 3 + rand() * 10;
    blob(lum, w, h, rand() * w, rand() * h, s, s, (rand() - 0.5) * 90);
  }
  blob(lum, w, h, 0.3 * w, 0.38 * h, 0.08 * w, 0.035 * h, -80); // eyes
  blob(lum, w, h, 0.7 * w, 0.38 * h, 0.08 * w, 0.035 * h, -80);
  blob(lum, w, h, 0.5 * w, 0.6 * h, 0.06 * w, 0.03 * h, -60); // nose tip shadow
  blob(lum, w, h, 0.5 * w, 0.93 * h, 0.12 * w, 0.03 * h, -40); // chin shadow
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (x + 0.5 - w / 2) / (w / 2), dy = (y + 0.5 - h / 2) / (h / 2);
      const edge = (1 - Math.sqrt(dx * dx + dy * dy)) * Math.min(w, h) / 2; // px inside the ellipse
      alpha[y * w + x] = Math.max(0, Math.min(1, edge / 2 + 0.5));
    }
  }
  return { w, h, lum, alpha };
}

export function makeBackground(W, H, seed = 2) {
  const rand = rng(seed), g = 8, gw = Math.ceil(W / g) + 2, gh = Math.ceil(H / g) + 2;
  const grid = Float32Array.from({ length: gw * gh }, () => 50 + rand() * 150);
  const bg = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / g, v = y / g, i = Math.floor(u), j = Math.floor(v), fu = u - i, fv = v - j;
      const a = grid[j * gw + i], b = grid[j * gw + i + 1], c = grid[(j + 1) * gw + i], d = grid[(j + 1) * gw + i + 1];
      bg[y * W + x] = (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv + (rand() - 0.5) * 30;
    }
  }
  return bg;
}

// A face layer {face, x, y, s, mouthOpen} occupies the box [x, y, face.w*s, face.h*s].
export const faceBox = (L) => [L.x, L.y, L.face.w * (L.s ?? 1), L.face.h * (L.s ?? 1)];

export function mouthBox(L) {
  const [x, y, w, h] = faceBox(L), mh = MOUTH.h * (0.3 + (L.mouthOpen ?? 0.3)) * h;
  return [x + MOUTH.cx * w - (MOUTH.w * w) / 2, y + MOUTH.cy * h - mh / 2, MOUTH.w * w, mh];
}

function sample(arr, w, h, u, v) {
  u = Math.max(0, Math.min(w - 1, u));
  v = Math.max(0, Math.min(h - 1, v));
  const i = Math.min(w - 2, Math.floor(u)), j = Math.min(h - 2, Math.floor(v)), fu = u - i, fv = v - j, k = j * w + i;
  return (arr[k] * (1 - fu) + arr[k + 1] * fu) * (1 - fv) + (arr[k + w] * (1 - fu) + arr[k + w + 1] * fu) * fv;
}

function drawFace(out, W, H, L) {
  const { face } = L, s = L.s ?? 1, open = L.mouthOpen ?? 0.3;
  const [fx, fy, fw, fh] = faceBox(L), mh = MOUTH.h * (0.3 + open);
  for (let py = Math.max(0, Math.floor(fy)); py < Math.min(H, Math.ceil(fy + fh)); py++) {
    for (let px = Math.max(0, Math.floor(fx)); px < Math.min(W, Math.ceil(fx + fw)); px++) {
      const u = (px + 0.5 - fx) / s, v = (py + 0.5 - fy) / s; // continuous texture coords
      if (u < 0 || v < 0 || u > face.w || v > face.h) continue;
      const a = sample(face.alpha, face.w, face.h, u - 0.5, v - 0.5);
      if (a <= 0) continue;
      let l = sample(face.lum, face.w, face.h, u - 0.5, v - 0.5);
      const mx = (u / face.w - MOUTH.cx) / (MOUTH.w / 2), my = (v / face.h - MOUTH.cy) / (mh / 2);
      if (mx * mx + my * my < 1) l = 45;
      const k = py * W + px;
      out[k] = out[k] * (1 - a) + l * a;
    }
  }
}

// occluders: [{box: [x,y,w,h], seed}] filled with (static per seed) uniform noise.
export function render(W, H, bg, layers = [], { t = 0, noise = 2, rand = null, occluders = [] } = {}) {
  const out = Float32Array.from(bg);
  for (const L of layers) drawFace(out, W, H, L);
  for (const { box, seed } of occluders) {
    const r = rng(seed);
    const [x, y, w, h] = box.map(Math.round);
    for (let py = Math.max(0, y); py < Math.min(H, y + h); py++) {
      for (let px = Math.max(0, x); px < Math.min(W, x + w); px++) out[py * W + px] = 30 + r() * 200;
    }
  }
  if (noise && rand) for (let k = 0; k < out.length; k++) out[k] += noise * gauss(rand);
  return { data: Uint8ClampedArray.from(out), width: W, height: H, t };
}

// The replay demo scene: a talking head swaying, leaning in 20% and back, with a 0.4 s hand-over-mouth.
export function demoClip(seconds = 8, fps = 30, W = 640, H = 480) {
  const face = makeFace(180, 220, 5), bg = makeBackground(W, H, 9), rand = rng(17);
  const smooth = (a, b, t) => { const x = Math.max(0, Math.min(1, (t - a) / (b - a))); return x * x * (3 - 2 * x); };
  const frames = [], gt = [];
  for (let i = 0; i < Math.round(seconds * fps); i++) {
    const t = i / fps;
    const s = 1 + 0.2 * smooth(3, 4, t) - 0.2 * smooth(6, 7, t);
    const cx = 320 + 50 * Math.sin(2 * Math.PI * 0.25 * t) + 8 * Math.sin(2 * Math.PI * 0.9 * t);
    const cy = 240 + 15 * Math.sin(2 * Math.PI * 0.3 * t);
    const L = { face, s, x: cx - (face.w * s) / 2, y: cy - (face.h * s) / 2, mouthOpen: 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t) };
    const hidden = t >= 5 && t < 5.4;
    const [fx, fy, fw, fh] = faceBox(L);
    const occluders = hidden ? [{ box: [fx + 0.15 * fw, fy + 0.6 * fh, 0.7 * fw, 0.45 * fh], seed: 99 }] : [];
    frames.push(render(W, H, bg, [L], { t, rand, occluders }));
    gt.push({ i, face: faceBox(L).map(Math.round), mouth: hidden ? null : mouthBox(L).map(Math.round) });
  }
  return { frames, gt: { width: W, height: H, fps, frames: gt } };
}

if (import.meta.main) {
  const [out, gtPath, seconds = '8'] = process.argv.slice(2);
  if (!out || !gtPath) {
    console.error('usage: node tests/synth.mjs <out.mp4> <out.gt.json> [seconds]');
    process.exit(2);
  }
  const { frames, gt } = demoClip(Number(seconds));
  const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', `${gt.width}x${gt.height}`,
    '-r', String(gt.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', out],
  { stdio: ['pipe', 'inherit', 'inherit'] });
  for (const f of frames) ff.stdin.write(Buffer.from(f.data.buffer));
  ff.stdin.end();
  ff.on('close', (code) => {
    if (code !== 0) process.exit(code);
    writeFileSync(gtPath, JSON.stringify(gt));
    console.log(`wrote ${out} (${frames.length} frames) and ${gtPath}`);
  });
}
