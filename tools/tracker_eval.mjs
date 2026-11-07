#!/usr/bin/env node
// Tracker lab: replays browser trackers (v1 static/tracker.js, v2 static/tracker2.js, or variants of either) over the
// clip set in tools/tracker_clips.json and scores every frame's mouth box against SAM 3.1 "lips" ground truth.
// SAM calls the trackers make are answered from the cached ground truth (--sam gt, $0) or from real image-mode
// SAM through tools/sam_cache.py (--sam live; cache misses cost $0.0025 each, capped by --max-usd).
//
//   node tools/tracker_eval.mjs [--split heldout|dev|all] [--clips a,b] [--trackers v1,v2,"v2:KEY=V;KEY=V"]
//                               [--sam gt|live] [--max-usd 2] [--json out.json] [--dump DIR] [--quiet]
//
// Per clip and tracker: centre error and size error of the mouth box (GT mouth widths; mean and p95), the centre error
// on fast-head frames (GT head speed > FAST mouth widths/s) and the lag there, recovery time after jumps/occlusions,
// quality-gate precision/recall on labelled bad frames (stress clips) or the flagged share (natural clips), crop
// coverage of the GT lips: frames fully inside / mean area kept, for the per-frame crop and the server's median crop, SAM calls/s, $/min and ms/frame.
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { plan, applyFrame } from './tracker_augment.mjs';
import {
  centre, mouthFromCrop, frameErrors, headSpeed, lagMs, jumpEvents, recoveryMs, gatePR, inside, areaInside, medianBox, mean, quantile,
} from './tracker_metrics.mjs';
import * as V1 from '../static/tracker.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const FAST = 1.5; // mouth widths per second of GT head speed that counts as fast motion
const CAP = 3; // a frame's error is capped at this many mouth widths (a missing box counts as CAP)
const SAM_USD = 0.0025; // per image-mode SAM call (docs/pipeline.md)
const MAX_W = 640; // the browser tracks frames at most 640 wide (static/capture.js maxWidth)

// ---------- trackers ----------
async function loadTracker(spec) {
  const [name, knobs = ''] = spec.split(':');
  const config = Object.fromEntries(knobs.split(';').filter(Boolean).map((kv) => {
    const [k, v] = kv.split('=');
    return [k, v === 'Infinity' ? Infinity : v === 'true' ? true : v === 'false' ? false : Number(v)];
  }));
  if (name === 'v1') {
    return {
      label: spec, version: 'v1',
      create: (sam) => {
        const t = V1.createTracker({ samFace: (f) => sam(f, 'face'), config });
        const pad = t.config.CROP_PAD;
        return { step: (f) => t.step(f), mouth: (r) => r.crop && mouthFromCrop(r.crop, pad) };
      },
    };
  }
  if (name === 'v2') {
    const V2 = await import('../static/tracker2.js');
    return {
      label: spec, version: 'v2',
      create: (sam) => {
        const t = V2.createTracker({ sam, config });
        return { step: (f) => t.step(f), mouth: (r) => r.mouth ?? (r.crop && mouthFromCrop(r.crop, t.config.CROP_PAD)) };
      },
    };
  }
  throw new Error(`unknown tracker ${name}`);
}

// ---------- SAM answers ----------
function gtSam(p) {
  return async (frame, prompt) => {
    const f = p[frame.i];
    if (prompt === 'face') return f.samFace ? [f.samFace.map(Math.round)] : [];
    if (prompt === 'lips' || prompt === 'mouth') return f.samMouth ? [f.samMouth.map(Math.round)] : [];
    throw new Error(`no ground truth for SAM prompt ${prompt}`);
  };
}

// Real image-mode SAM, cached on disk by clip/frame/prompt (tools/sam_cache.py); misses are JPEG'd like the browser does.
function liveSam(clipName, cacheDir, maxUsd) {
  const file = join(cacheDir, 'sam-image-cache.jsonl');
  const cache = new Map();
  if (existsSync(file)) for (const line of readFileSync(file, 'utf8').split('\n').filter(Boolean)) { const e = JSON.parse(line); cache.set(e.key, e); }
  let proc = null, pending = null, buf = '';
  const start = () => {
    proc = spawn('uv', ['run', '--env-file', process.env.ENV_FILE ?? join(ROOT, '.env'), 'python', join(ROOT, 'tools/sam_cache.py'), cacheDir, String(maxUsd)],
      { cwd: ROOT, stdio: ['pipe', 'pipe', 'inherit'] });
    proc.stdout.on('data', (d) => {
      buf += d;
      let k;
      while ((k = buf.indexOf('\n')) >= 0) { const line = buf.slice(0, k); buf = buf.slice(k + 1); pending?.(JSON.parse(line)); }
    });
  };
  const fn = async (frame, prompt) => {
    const key = `${clipName}/${frame.i}/${prompt}/${frame.width}x${frame.height}`;
    if (cache.has(key)) return cache.get(key).boxes;
    if (!proc) start();
    const jpeg = join(cacheDir, 'jpeg', `${key.replaceAll('/', '_')}.jpg`);
    mkdirSync(dirname(jpeg), { recursive: true });
    await encodeJpeg(frame.rgb, frame.width, frame.height, jpeg);
    const answer = await new Promise((res) => { pending = res; proc.stdin.write(`${JSON.stringify({ key, prompt, jpeg })}\n`); });
    if (answer.error) throw new Error(answer.error);
    cache.set(key, answer);
    return answer.boxes;
  };
  fn.close = () => proc?.stdin.end();
  return fn;
}

function encodeJpeg(rgb, W, H, path) {
  return new Promise((res, rej) => {
    const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${H}`, '-i', '-', '-q:v', '3', path]);
    ff.on('close', (c) => (c === 0 ? res() : rej(new Error(`ffmpeg jpeg ${c}`))));
    ff.stdin.end(Buffer.from(rgb.buffer, rgb.byteOffset, rgb.byteLength));
  });
}

// ---------- frames ----------
async function* decode(src, W, H) {
  const ff = spawn('ffmpeg', ['-v', 'error', '-i', src, '-fps_mode', 'passthrough', '-vf', `scale=${W}:${H}`, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    { stdio: ['ignore', 'pipe', 'inherit'] });
  const size = W * H * 3;
  let acc = Buffer.alloc(0);
  for await (const chunk of ff.stdout) {
    acc = acc.length ? Buffer.concat([acc, chunk]) : chunk;
    while (acc.length >= size) {
      yield new Uint8ClampedArray(acc.subarray(0, size));
      acc = acc.subarray(size);
    }
  }
}

function gray(rgb, W, H) { // capture.js grab(): (77 R + 150 G + 29 B) >> 8
  const g = new Uint8ClampedArray(W * H);
  for (let i = 0, p = 0; i < g.length; i++, p += 3) g[i] = (rgb[p] * 77 + rgb[p + 1] * 150 + rgb[p + 2] * 29) >> 8;
  return g;
}

function rgba(rgb, W, H) {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0, p = 0, q = 0; i < W * H; i++, p += 3, q += 4) { out[q] = rgb[p]; out[q + 1] = rgb[p + 1]; out[q + 2] = rgb[p + 2]; out[q + 3] = 255; }
  return out;
}

// ---------- one clip ----------
async function runClip(clip, trackers, opts) {
  const gt = JSON.parse(readFileSync(clip.gt, 'utf8'));
  const k = Math.min(1, MAX_W / gt.width), W = Math.round(gt.width * k), H = Math.round(gt.height * k);
  const scale = (b) => b && b.map((v) => v * k);
  const mouths = gt.frames.map((f) => scale(f.mouth)), faces = gt.frames.map((f) => scale(f.face));
  const p = plan(clip.augment ?? {}, mouths, faces, gt.fps, W, H);
  const sam = opts.sam === 'live' ? liveSam(clip.name, opts.cacheDir, opts.maxUsd) : gtSam(p);
  const runs = trackers.map((t) => {
    const calls = {};
    let samMs = 0;
    const counted = async (f, prompt) => {
      calls[prompt] = (calls[prompt] ?? 0) + 1;
      const t0 = performance.now();
      try { return await sam(f, prompt); } finally { samMs += performance.now() - t0; }
    };
    const inst = t.create(counted);
    return { t, inst, calls, rows: [], get samMs() { return samMs; }, resetSam() { samMs = 0; } };
  });
  let i = 0;
  for await (const raw of decode(clip.video, W, H)) {
    if (i >= p.length) break;
    const rgb = applyFrame(raw, W, H, p[i]), g = gray(rgb, W, H), c = rgba(rgb, W, H);
    for (const run of runs) {
      const frame = { data: g, rgba: c, rgb, width: W, height: H, t: i / gt.fps, i };
      run.resetSam();
      const t0 = performance.now();
      const r = await run.inst.step(frame);
      const ms = performance.now() - t0 - run.samMs;
      run.rows.push({ crop: r.crop, mouth: run.inst.mouth(r), flagged: r.flags.length > 0, flags: r.flags, sam: !!r.sam, ms });
    }
    i++;
  }
  sam.close?.();
  const n = i, frames = p.slice(0, n);
  const truth = frames.map((f) => f.mouth);
  const speed = headSpeed(truth, gt.fps), fast = speed.map((v) => v != null && v > FAST);
  const occEnds = frames.flatMap((f, j) => (j && frames[j - 1].cover >= 0.5 && f.cover < 0.5 ? [j] : []));
  // an event in the clip's last second (e.g. the next camera shot starting on the final frame) cannot show a recovery
  const events = [...new Set([...jumpEvents(truth, 1), ...occEnds])].filter((e) => e + gt.fps <= n).sort((a, b) => a - b);
  const labelled = !!(clip.augment?.occlude || clip.augment?.blur);
  return {
    clip: clip.name, split: clip.split, address: clip.address, frames: n, seconds: n / gt.fps, gtFrames: truth.filter(Boolean).length,
    fastFrames: fast.filter(Boolean).length, events: events.length,
    results: runs.map((run) => score(run, frames, truth, fast, events, labelled, gt.fps, n)),
  };
}

function score(run, frames, truth, fast, events, labelled, fps, n) {
  const rows = run.rows, errs = frameErrors(rows.map((r) => r.mouth), truth);
  const ce = errs.map((e) => e && Math.min(CAP, e.centre));
  const valid = ce.filter((e) => e != null);
  const size = errs.filter(Boolean).map((e) => Math.min(CAP, (e.sizeW + e.sizeH) / 2));
  const fastErr = ce.filter((e, j) => e != null && fast[j]);
  const centres = rows.map((r) => r.mouth && centre(r.mouth)), gtc = truth.map((b) => b && centre(b));
  const rec = recoveryMs(ce, events, fps);
  const gate = gatePR(rows.map((r) => r.flagged), frames.map((f) => (labelled ? f.bad : f.mouth ? false : null)));
  const med = medianBox(rows.map((r) => r.crop));
  const withGt = truth.map((b, j) => [b, rows[j].crop]).filter(([b]) => b);
  const calls = Object.values(run.calls).reduce((a, b) => a + b, 0);
  const ms = rows.slice(1).map((r) => r.ms);
  return {
    tracker: run.t.label, version: run.t.version,
    centre: mean(valid), centreP95: quantile(valid, 0.95), size: mean(size),
    fast: fastErr.length ? mean(fastErr) : NaN, lagMs: fastErr.length >= 15 ? lagMs(centres, gtc, fast, fps) : NaN,
    recoveryMs: rec.length ? mean(rec.map((x) => Math.min(x, (n / fps) * 1000))) : NaN, unrecovered: rec.filter((x) => x === Infinity).length,
    gate: { labelled, ...gate, flagged: rows.filter((r) => r.flagged).length / n },
    coverage: withGt.filter(([b, c]) => c && inside(b, c)).length / withGt.length,
    medianCoverage: med ? withGt.filter(([b]) => inside(b, med)).length / withGt.length : 0,
    kept: mean(withGt.map(([b, c]) => (c ? areaInside(b, c) : 0))), medianKept: med ? mean(withGt.map(([b]) => areaInside(b, med))) : 0,
    samCalls: calls, samByPrompt: { ...run.calls }, samPerSec: calls / (n / fps), usdPerMin: (calls * SAM_USD) / (n / fps / 60),
    msMedian: quantile(ms, 0.5), msP95: quantile(ms, 0.95),
  };
}

// ---------- report ----------
const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : '–');
const f0 = (x) => (Number.isFinite(x) ? x.toFixed(0) : '–');
const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : '–');
const pct = (x) => (Number.isFinite(x) ? `${(100 * x).toFixed(0)}%` : '–');
const HEAD = ['clip', 'tracker', 'centre', 'p95', 'size', 'fast(n)', 'lag ms', 'recov ms', 'gate P/R | flagged', 'cover/kept', 'median crop', 'SAM/s', 'ms/frame'];
function row(c, r) {
  const gate = r.gate.labelled ? `${pct(r.gate.precision)}/${pct(r.gate.recall)}` : `${pct(r.gate.flagged)} flagged`;
  return [c.clip, r.tracker, f3(r.centre), f3(r.centreP95), f3(r.size), `${f3(r.fast)}(${c.fastFrames})`, f0(r.lagMs),
    c.events ? `${f0(r.recoveryMs)}${r.unrecovered ? ` (${r.unrecovered} never)` : ''}` : '–', gate, `${pct(r.coverage)}/${pct(r.kept)}`, `${pct(r.medianCoverage)}/${pct(r.medianKept)}`,
    f2(r.samPerSec), `${f2(r.msMedian)}/${f2(r.msP95)}`];
}

// Frame-weighted means over clips (time-weighted for rates); gate counts pooled over labelled clips.
function summarize(clips, label) {
  const out = [];
  const names = clips[0]?.results.map((r) => r.tracker) ?? [];
  for (const name of names) {
    const rs = clips.map((c) => ({ c, r: c.results.find((x) => x.tracker === name) }));
    const wmean = (key, w) => { let s = 0, ws = 0; for (const { c, r } of rs) if (Number.isFinite(r[key]) && w(c) > 0) { s += r[key] * w(c); ws += w(c); } return ws ? s / ws : NaN; };
    const lab = rs.filter(({ r }) => r.gate.labelled), tp = lab.reduce((a, { r }) => a + r.gate.tp, 0);
    const fp = lab.reduce((a, { r }) => a + r.gate.fp, 0), fn = lab.reduce((a, { r }) => a + r.gate.fn, 0);
    const nat = rs.filter(({ r }) => !r.gate.labelled);
    out.push({
      tracker: name, clips: rs.length, centre: wmean('centre', (c) => c.gtFrames), centreP95: wmean('centreP95', (c) => c.gtFrames),
      size: wmean('size', (c) => c.gtFrames), fast: wmean('fast', (c) => c.fastFrames), lagMs: wmean('lagMs', (c) => c.fastFrames),
      recoveryMs: wmean('recoveryMs', (c) => c.events), unrecovered: rs.reduce((a, { r }) => a + r.unrecovered, 0),
      gate: { precision: tp + fp ? tp / (tp + fp) : NaN, recall: tp + fn ? tp / (tp + fn) : NaN, tp, fp, fn,
        naturalFlagged: nat.length ? mean(nat.map(({ r }) => r.gate.flagged)) : NaN },
      coverage: wmean('coverage', (c) => c.gtFrames), medianCoverage: wmean('medianCoverage', (c) => c.gtFrames),
      kept: wmean('kept', (c) => c.gtFrames), medianKept: wmean('medianKept', (c) => c.gtFrames),
      samPerSec: wmean('samPerSec', (c) => c.seconds), usdPerMin: wmean('usdPerMin', (c) => c.seconds),
      msMedian: wmean('msMedian', (c) => c.frames), msP95: wmean('msP95', (c) => c.frames), label,
    });
  }
  // paired evidence: per clip, which tracker has the lower centre / fast error than the first tracker
  const base = names[0];
  for (const s of out.slice(1)) {
    const pairs = clips.map((c) => [c.results.find((x) => x.tracker === base), c.results.find((x) => x.tracker === s.tracker)]);
    s.vsBase = {
      base, centreWins: pairs.filter(([a, b]) => b.centre < a.centre).length, fastWins: pairs.filter(([a, b]) => b.fast < a.fast).length,
      fastClips: pairs.filter(([a, b]) => Number.isFinite(a.fast) && Number.isFinite(b.fast)).length, clips: pairs.length,
    };
  }
  return out;
}

function printTable(rows) {
  const w = HEAD.map((h, k) => Math.max(h.length, ...rows.map((r) => String(r[k]).length)));
  const line = (r) => r.map((v, k) => String(v).padEnd(w[k])).join('  ');
  console.log(line(HEAD));
  for (const r of rows) console.log(line(r));
}

function printSummary(sums) {
  for (const s of sums) {
    const gate = `gate P/R ${pct(s.gate.precision)}/${pct(s.gate.recall)} (tp ${s.gate.tp} fp ${s.gate.fp} fn ${s.gate.fn}), natural flagged ${pct(s.gate.naturalFlagged)}`;
    console.log(`  ${s.label.padEnd(10)} ${s.tracker.padEnd(14)} centre ${f3(s.centre)} p95 ${f3(s.centreP95)}  size ${f3(s.size)}  fast ${f3(s.fast)}  lag ${f0(s.lagMs)} ms  `
      + `recovery ${f0(s.recoveryMs)} ms (${s.unrecovered} never)  ${gate}  lips fully in crop ${pct(s.coverage)} (area kept ${pct(s.kept)}) / median crop ${pct(s.medianCoverage)} (${pct(s.medianKept)})  `
      + `SAM ${f2(s.samPerSec)}/s ($${f3(s.usdPerMin)}/min)  ${f2(s.msMedian)}/${f2(s.msP95)} ms`
      + (s.vsBase ? `  | vs ${s.vsBase.base}: lower centre on ${s.vsBase.centreWins}/${s.vsBase.clips} clips, lower fast on ${s.vsBase.fastWins}/${s.vsBase.fastClips}` : ''));
  }
}

function parseArgs(argv) {
  const o = { split: 'heldout', clips: null, trackers: ['v1', 'v2'], sam: 'gt', maxUsd: 2, json: null, manifest: join(ROOT, 'tools/tracker_clips.json'),
    cacheDir: '/tmp/badge-data/tracker-gt/sam-cache' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i], v = () => argv[++i];
    if (a === '--split') o.split = v();
    else if (a === '--clips') o.clips = v().split(',');
    else if (a === '--trackers') o.trackers = v().split(',');
    else if (a === '--sam') o.sam = v();
    else if (a === '--max-usd') o.maxUsd = Number(v());
    else if (a === '--json') o.json = v();
    else if (a === '--manifest') o.manifest = v();
    else if (a === '--cache') o.cacheDir = v();
    else throw new Error(`unknown argument ${a}`);
  }
  return o;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const manifest = JSON.parse(readFileSync(opts.manifest, 'utf8'));
  const byName = new Map(manifest.clips.map((c) => [c.name, c]));
  const clips = manifest.clips
    .filter((c) => (opts.clips ? opts.clips.includes(c.name) : opts.split === 'all' || c.split === opts.split))
    .map((c) => { const b = c.base ? byName.get(c.base) : c; return { ...c, video: b.video, gt: b.gt, address: b.address }; });
  const trackers = await Promise.all(opts.trackers.map(loadTracker));
  const results = [];
  for (const clip of clips) {
    process.stderr.write(`${clip.name} ...`);
    const t0 = performance.now();
    results.push(await runClip(clip, trackers, opts));
    process.stderr.write(` ${((performance.now() - t0) / 1000).toFixed(0)} s\n`);
  }
  console.log(`tracker lab: ${clips.length} clips (split ${opts.clips ? 'custom' : opts.split}), SAM answers: ${opts.sam === 'live' ? 'real image-mode SAM (cached)' : 'cached video-mode ground truth'}`);
  console.log(`units: errors in GT mouth widths; fast = GT head speed > ${FAST} mouth widths/s; recovery = until error < 0.5 for 3 frames\n`);
  printTable(results.flatMap((c) => c.results.map((r) => row(c, r))));
  const natural = results.filter((c) => !c.clip.includes('+')), stress = results.filter((c) => c.clip.includes('+'));
  const sums = [...(natural.length ? summarize(natural, 'natural') : []), ...(stress.length ? summarize(stress, 'stress') : []), ...summarize(results, 'all')];
  console.log('\nsummary (frame-weighted over clips):');
  printSummary(sums);
  if (opts.json) writeFileSync(opts.json, JSON.stringify({ opts: { ...opts, fast: FAST, cap: CAP }, clips: results, summary: sums }, null, 1));
}

main().catch((e) => { console.error(e); process.exit(1); });
