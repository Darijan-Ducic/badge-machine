#!/usr/bin/env node
// Replays static/tracker.js on a recorded clip offline, with every "SAM call" answered by the
// ground-truth face box for that frame (free), and prints accuracy / cost / quality-gate metrics.
//
//   node tools/replay.mjs <frames source> <gt.json> [--config K=V ...] [--sweep] [--top N] [--tol F]
//
// <frames source>: anything ffmpeg reads (clip.mp4, frames/%04d.png). Frames are scaled to the GT size.
// <gt.json>: {width, height, fps, frames: [{i, face: [x,y,w,h]|null, mouth: [x,y,w,h]|null}]}.
// --config K=V   override a tracker CONFIG knob (repeatable); also the base for --sweep.
// --sweep        grid-search knobs; rank by mean error (bucketed to --tol, default 0.005 face widths),
//                then SAM calls/s, then crop jitter; print the top --top (default 10) and the best config as JSON.
// Crop jitter = RMS second difference of the crop center per frame / face width (what looks like lip motion).
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { CONFIG, withConfig, createTracker, createSmoother, mouthFromAnchor, anchorFromFace, center, qualityGate } from '../static/tracker.js';

// Sweep grid. Tracking axes need a full replay per combination; smoothing axes only re-smooth the raw
// crops of that replay. The mouth ratios are not swept: they are measured from SAM data, and on one clip
// the sweep would just fit the first keyframe's mouth position.
const TRACK_AXES = [
  [0.4, 0.55, 0.7].map((v) => ({ T_CONF: v })),
  [1, 2, 4].map((v) => ({ T_REFRESH: v })),
  [0, 0.08, 0.15].map((v) => ({ SNAP_TOL: v })),
  [[0.5, 0.4], [0.55, 0.35], [0.6, 0.3]].map(([y, h]) => ({ ANCHOR_Y: y, ANCHOR_H: h })), // patch bottom stays at 0.90
  [0, 1.2].map((v) => ({ ANCHOR_MASK: v })),
];
const POST_AXES = [
  [0.25, 0.5, 1, 2].map((v) => ({ SMOOTH_MIN_CUTOFF: v })),
  [1, 3, 10].map((v) => ({ SMOOTH_BETA: v })),
];

function parseArgs(argv) {
  const opts = { pos: [], config: {}, sweep: false, top: 10, tol: 0.005 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--sweep') opts.sweep = true;
    else if (a === '--top') opts.top = Number(argv[++i]);
    else if (a === '--tol') opts.tol = Number(argv[++i]);
    else if (a === '--config') {
      const [k, v] = argv[++i].split('=');
      opts.config[k] = v === 'Infinity' ? Infinity : Number(v);
      if (Number.isNaN(opts.config[k])) throw new Error(`--config ${k}: not a number: ${v}`);
    } else opts.pos.push(a);
  }
  if (opts.pos.length !== 2) throw new Error('usage: node tools/replay.mjs <frames source> <gt.json> [--config K=V ...] [--sweep] [--top N] [--tol F]');
  withConfig(opts.config); // rejects unknown keys
  return opts;
}

function decode(src, W, H) {
  return new Promise((resolve, reject) => {
    const ff = spawn('ffmpeg', ['-v', 'error', '-i', src, '-fps_mode', 'passthrough', '-vf', `scale=${W}:${H}`, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
      { stdio: ['ignore', 'pipe', 'inherit'] });
    const chunks = [];
    ff.stdout.on('data', (c) => chunks.push(c));
    ff.on('error', reject);
    ff.on('close', (code) => (code === 0 ? resolve(Buffer.concat(chunks)) : reject(new Error(`ffmpeg exited with ${code}`))));
  });
}

const quantile = (xs, q) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

async function track(frames, gtAt, cfg) {
  const tracker = createTracker({ samFace: async (f) => (gtAt.get(f.i)?.face ? [gtAt.get(f.i).face] : []), config: cfg });
  const results = [], ms = [];
  for (const f of frames) {
    const t0 = performance.now();
    results.push(await tracker.step({ ...f })); // fresh object: no pyramid cached by an earlier run
    ms.push(performance.now() - t0);
  }
  return { results, samCalls: tracker.samCalls, stepMs: quantile(ms.slice(1), 0.5) };
}

// Crop for each frame: as tracked, or the same raw crops re-smoothed with other smoothing knobs.
function crops(results, cfg, resmooth) {
  if (!resmooth) return results.map((r) => r.crop);
  const sm = createSmoother(cfg);
  return results.map((r) => (r.rawCrop ? sm.update(r.rawCrop, r.t) : null));
}

function metrics(results, cropList, gt, cfg, samCalls) {
  const errs = [], slow = [];
  let vsMouth = 0, vsFace = 0, noCrop = 0;
  results.forEach((r, i) => {
    const g = gt.byIndex.get(i);
    if (!g?.face) return;
    const ref = g.mouth ? center(g.mouth) : center(mouthFromAnchor(anchorFromFace(g.face, cfg), cfg));
    g.mouth ? vsMouth++ : vsFace++;
    const c = cropList[i];
    if (!c) { noCrop++; return; }
    const [cx, cy] = center(c);
    errs.push(Math.hypot(cx - ref[0], cy - ref[1]) / g.face[2]);
    const s = gt.slowMouth.get(i);
    if (s) slow.push(Math.hypot(cx - s[0], cy - s[1]) / g.face[2]);
  });
  const acc = [];
  for (let i = 1; i + 1 < cropList.length; i++) {
    const [a, b, c] = [i - 1, i, i + 1].map((k) => cropList[k] && center(cropList[k])), fw = gt.byIndex.get(i)?.face?.[2];
    if (a && b && c && fw) acc.push(Math.hypot(a[0] - 2 * b[0] + c[0], a[1] - 2 * b[1] + c[1]) / fw);
  }
  const gate = qualityGate(results.map((r) => r.flags), cfg);
  const seconds = results.length / gt.fps;
  return { mean: errs.length ? mean(errs) : NaN, p95: errs.length ? quantile(errs, 0.95) : NaN, vsMouth, vsFace, noCrop,
    slowMean: slow.length ? mean(slow) : NaN, slowP95: slow.length ? quantile(slow, 0.95) : NaN,
    jitter: acc.length ? Math.sqrt(mean(acc.map((x) => x * x))) : NaN, samCalls, samPerSec: samCalls / seconds, flagged: gate.ratio, ok: gate.ok };
}

// GT mouth center averaged over +-0.25 s: keeps head motion, drops syllable-rate lip motion and SAM mask noise,
// which a rigid crop should not follow (the raw metric rewards a crop that chases the lips).
function slowMouth(gt) {
  const k = Math.round(0.25 * gt.fps), out = new Map();
  for (const { i } of gt.frames) {
    const pts = [];
    for (let j = i - k; j <= i + k; j++) { const m = gt.byIndex.get(j)?.mouth; if (m) pts.push(center(m)); }
    if (pts.length) out.set(i, [mean(pts.map((p) => p[0])), mean(pts.map((p) => p[1]))]);
  }
  return out;
}

const count = (items) => Object.entries(items.reduce((m, k) => ((m[k] = (m[k] ?? 0) + 1), m), {})).map(([k, n]) => `${k} ${n}`).join(', ') || 'none';
const pct = (x) => `${(100 * x).toFixed(1)}%`;

function report(m, results, cfg, stepMs) {
  console.log(`mouth-crop center error / GT face width  (vs GT mouth on ${m.vsMouth} frames, vs mouthFromAnchor(GT face) on ${m.vsFace}${m.noCrop ? `, ${m.noCrop} frames had no crop` : ''})`);
  console.log(`  mean ${m.mean.toFixed(4)}   p95 ${m.p95.toFixed(4)}   (crop jitter ${m.jitter.toFixed(4)})`);
  console.log(`  vs GT mouth averaged over +-0.25 s (head motion, no lip motion): mean ${m.slowMean.toFixed(4)}   p95 ${m.slowP95.toFixed(4)}`);
  console.log(`SAM calls: ${m.samCalls} (${m.samPerSec.toFixed(2)}/s)   reasons: ${count(results.flatMap((r) => r.sam?.reasons ?? []))}`);
  console.log(`flagged: ${pct(m.flagged)} of frames   (${count(results.flatMap((r) => r.flags))})`);
  console.log(`quality gate: ${m.ok ? 'ACCEPT' : 'REJECT - ask the user to repeat'}   (flagged ${pct(m.flagged)} vs T_BAD ${pct(cfg.T_BAD)})`);
  console.log(`tracker step: median ${stepMs.toFixed(2)} ms/frame;  median conf ${quantile(results.slice(1).map((r) => r.conf), 0.5).toFixed(3)}`);
}

function* grid(axes) {
  if (!axes.length) { yield {}; return; }
  for (const head of axes[0]) for (const rest of grid(axes.slice(1))) yield { ...head, ...rest };
}

async function sweep(frames, gt, base, { top, tol }) {
  const rows = [];
  const tracks = [...grid(TRACK_AXES)];
  for (const [n, tk] of tracks.entries()) {
    const cfg = withConfig({ ...base, ...tk });
    const { results, samCalls } = await track(frames, gt.byIndex, cfg);
    for (const pk of grid(POST_AXES)) {
      const c2 = withConfig({ ...base, ...tk, ...pk });
      rows.push({ knobs: { ...tk, ...pk }, ...metrics(results, crops(results, c2, true), gt, c2, samCalls) });
    }
    process.stderr.write(`\rsweep: ${n + 1}/${tracks.length} tracking configs`);
  }
  process.stderr.write('\n');
  rows.sort((a, b) => Math.round(a.mean / tol) - Math.round(b.mean / tol) || a.samPerSec - b.samPerSec || a.jitter - b.jitter);
  const keys = [...new Set([...TRACK_AXES, ...POST_AXES].flatMap((ax) => Object.keys(ax[0])))];
  console.log(`\nsweep: ${rows.length} configs (${tracks.length} tracking x ${rows.length / tracks.length} smoothing), ranked by mean error (bucket ${tol}), then SAM/s, then jitter`);
  console.log(['rank', 'mean', 'p95', 'slowGT', 'jitter', 'SAM/s', 'flagged', 'gate', ...keys].join('\t'));
  rows.slice(0, top).forEach((r, i) =>
    console.log([i + 1, r.mean.toFixed(4), r.p95.toFixed(4), r.slowMean.toFixed(4), r.jitter.toFixed(4), r.samPerSec.toFixed(2), pct(r.flagged), r.ok ? 'accept' : 'REJECT', ...keys.map((k) => r.knobs[k])].join('\t')));
  return withConfig({ ...base, ...rows[0].knobs });
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const [src, gtPath] = opts.pos;
  const gt = JSON.parse(readFileSync(gtPath, 'utf8'));
  gt.byIndex = new Map(gt.frames.map((f) => [f.i, f]));
  gt.slowMouth = slowMouth(gt);
  const { width: W, height: H, fps } = gt;
  const buf = await decode(src, W, H);
  const n = Math.floor(buf.length / (W * H));
  const frames = Array.from({ length: n }, (_, i) => ({ data: buf.subarray(i * W * H, (i + 1) * W * H), width: W, height: H, t: i / fps, i }));
  console.log(`clip: ${src}  ${W}x${H} @ ${fps.toFixed(2)} fps, ${n} frames (${(n / fps).toFixed(2)} s); GT frames: ${gt.frames.length}`);
  if (n !== gt.frames.length) console.log(`  warning: decoded ${n} frames but GT lists ${gt.frames.length}`);

  const cfg = withConfig(opts.config);
  const changed = Object.entries(opts.config).map(([k, v]) => `${k}=${v}`).join(' ');
  console.log(`\nconfig: ${changed || 'defaults'}`);
  const run = await track(frames, gt.byIndex, cfg);
  report(metrics(run.results, crops(run.results, cfg, false), gt, cfg, run.samCalls), run.results, cfg, run.stepMs);

  if (opts.sweep) {
    const best = await sweep(frames, gt, opts.config, opts);
    const diff = Object.fromEntries(Object.entries(best).filter(([k, v]) => v !== CONFIG[k]));
    console.log(`\nbest config, replayed directly: ${Object.entries(diff).map(([k, v]) => `${k}=${v}`).join(' ')}`);
    const again = await track(frames, gt.byIndex, best);
    report(metrics(again.results, crops(again.results, best, false), gt, best, again.samCalls), again.results, best, again.stepMs);
    console.log('\nbest config JSON:');
    console.log(JSON.stringify(best));
  }
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
