// Stress variants of real footage for the tracker lab, with exactly known ground truth: the whole frame moves on a
// seeded path (fast sway, lean in/out, sudden jumps), an occluder slides over the mouth, or motion blur hits.
// The true mouth/face boxes are warped with the frame, so SAM's ground truth for the source frame stays exact.
// Pure functions on RGB24 frames (Uint8ClampedArray, W*H*3); no I/O.
import { rng } from '../tests/synth.mjs';

export const BLUR_BAD = 8; // px of horizontal motion blur from which a frame counts as bad for lip reading
export const COVER_BAD = 0.5; // fraction of the mouth box covered from which a frame counts as bad
const SLIDE = 0.15; // s for an occluder to slide in (and out)

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// The frame warp: out(p) = src(c + (p - c - t) / s), c the frame centre; so a source point q lands at c + t + s(q - c).
export function warpBox([x, y, w, h], { s = 1, tx = 0, ty = 0 }, W, H) {
  const cx = W / 2, cy = H / 2;
  return [cx + tx + s * (x - cx), cy + ty + s * (y - cy), s * w, s * h];
}

export function warp(rgb, W, H, { s = 1, tx = 0, ty = 0 }) {
  const out = new Uint8ClampedArray(W * H * 3), cx = W / 2, cy = H / 2;
  for (let j = 0; j < H; j++) {
    const v = clamp(cy + (j + 0.5 - cy - ty) / s - 0.5, 0, H - 1), j0 = Math.min(H - 2, Math.floor(v)), fv = v - j0;
    for (let i = 0; i < W; i++) {
      const u = clamp(cx + (i + 0.5 - cx - tx) / s - 0.5, 0, W - 1), i0 = Math.min(W - 2, Math.floor(u)), fu = u - i0;
      const a = (j0 * W + i0) * 3, b = a + 3, c = a + W * 3, d = c + 3, o = (j * W + i) * 3;
      for (let k = 0; k < 3; k++) {
        out[o + k] = (rgb[a + k] * (1 - fu) + rgb[b + k] * fu) * (1 - fv) + (rgb[c + k] * (1 - fu) + rgb[d + k] * fu) * fv;
      }
    }
  }
  return out;
}

function overlap(a, b) {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  return ix * iy;
}

const within = (t, [t0, t1]) => t >= t0 && t < t1;
const median = (xs) => { const s = xs.filter((x) => x != null).sort((a, b) => a - b); return s.length ? s[s.length >> 1] : 0; };

// Per-frame plan for a clip whose source frames have true boxes mouths[i] / faces[i] (null = unknown).
// spec: {move: {seed, amp (face widths, default 0.45), zoom (default 0.15)}, jumps: [t s], occlude: [[t0, t1, 'skin'|'object']],
//        blur: [[t0, t1, px]]}. Frame i: {motion, mouth, face (true, warped), occ (occluder box) | null, cover, blur,
//        bad (covered >= COVER_BAD or blur >= BLUR_BAD), samMouth (what SAM can see: null when covered), samFace}.
export function plan(spec, mouths, faces, fps, W, H) {
  const fw = median(faces.map((f) => f?.[2])) || W / 3;
  const r = rng(spec.move?.seed ?? 1), ph = Array.from({ length: 6 }, () => 2 * Math.PI * r());
  const amp = (spec.move?.amp ?? 0.45) * fw, zoom = spec.move?.zoom ?? 0.15, jumps = spec.jumps ?? [];
  return mouths.map((m0, i) => {
    const t = i / fps;
    let tx = 0, ty = 0, s = 1;
    if (spec.move) {
      tx = amp * (0.7 * Math.sin(2 * Math.PI * 0.5 * t + ph[0]) + 0.3 * Math.sin(2 * Math.PI * 1.3 * t + ph[1]));
      ty = amp * (0.3 * Math.sin(2 * Math.PI * 0.4 * t + ph[2]) + 0.12 * Math.sin(2 * Math.PI * 1.1 * t + ph[3]));
      s = 1 + zoom * Math.sin(2 * Math.PI * 0.15 * t + ph[4]);
    }
    const passed = jumps.filter((tj) => t >= tj).length;
    if (passed % 2) tx += 0.6 * fw * (ph[5] < Math.PI ? 1 : -1); // jumps toggle a 0.6 face-width offset on and off
    const motion = { s, tx, ty };
    const mouth = m0 && warpBox(m0, motion, W, H), face = faces[i] && warpBox(faces[i], motion, W, H);
    let occ = null, cover = 0;
    const ev = (spec.occlude ?? []).find((e) => within(t, e));
    if (ev && mouth) {
      const [t0, t1, kind] = ev, mw = mouth[2], mh = mouth[3];
      const ow = Math.max(1.8 * mw, 0.5 * (face?.[2] ?? 2 * mw)), oh = Math.max(2.6 * mh, 0.35 * (face?.[3] ?? 2 * mw));
      const progress = Math.min(1, (t - t0) / SLIDE, (t1 - t) / SLIDE); // slide in from below, slide out
      const y = mouth[1] + mh / 2 - oh / 2 + (1 - Math.max(0, progress)) * oh;
      occ = { kind, box: [mouth[0] + mw / 2 - ow / 2, y, ow, oh] };
      cover = overlap(occ.box, mouth) / (mw * mh);
    }
    const bl = (spec.blur ?? []).find((e) => within(t, e)), blur = bl ? bl[2] : 0;
    const bad = mouth ? cover >= COVER_BAD || blur >= BLUR_BAD : null;
    return { i, motion, mouth, face, occ, cover, blur, bad, samMouth: cover >= COVER_BAD ? null : mouth, samFace: face };
  });
}

function drawOccluder(img, W, H, { kind, box }, face, seed) {
  const [ox, oy, ow, oh] = box, r = rng(seed);
  // skin: the person's own forehead, stretched over the occluder (a hand has the same skin); object: a dark mug/mic
  const src = kind === 'skin' && face ? [face[0] + 0.3 * face[2], face[1] + 0.1 * face[3], 0.4 * face[2], 0.2 * face[3]] : null;
  const copy = src ? Uint8ClampedArray.from(img) : null;
  for (let j = Math.max(0, Math.floor(oy)); j < Math.min(H, Math.ceil(oy + oh)); j++) {
    for (let i = Math.max(0, Math.floor(ox)); i < Math.min(W, Math.ceil(ox + ow)); i++) {
      const o = (j * W + i) * 3;
      if (src) {
        const u = clamp(Math.round(src[0] + ((i - ox) / ow) * src[2]), 0, W - 1), v = clamp(Math.round(src[1] + ((j - oy) / oh) * src[3]), 0, H - 1);
        const k = (v * W + u) * 3;
        for (let c = 0; c < 3; c++) img[o + c] = copy[k + c] * 0.92;
      } else {
        const g = 55 + 25 * r();
        img[o] = g; img[o + 1] = g; img[o + 2] = g + 8;
      }
    }
  }
}

function motionBlur(img, W, H, len) {
  const L = Math.max(1, Math.round(len)), half = L >> 1, row = new Float32Array(W * 3);
  for (let j = 0; j < H; j++) {
    const base = j * W * 3;
    for (let i = 0; i < W; i++) {
      let a = 0, b = 0, c = 0;
      for (let d = -half; d < L - half; d++) {
        const k = base + clamp(i + d, 0, W - 1) * 3;
        a += img[k]; b += img[k + 1]; c += img[k + 2];
      }
      row[i * 3] = a / L; row[i * 3 + 1] = b / L; row[i * 3 + 2] = c / L;
    }
    for (let k = 0; k < W * 3; k++) img[base + k] = row[k];
  }
}

// The stressed RGB frame for plan entry f (a new array; the input is not modified).
export function applyFrame(rgb, W, H, f) {
  const m = f.motion, moved = m && (m.s !== 1 || m.tx || m.ty);
  const img = moved ? warp(rgb, W, H, m) : Uint8ClampedArray.from(rgb);
  if (f.occ) drawOccluder(img, W, H, f.occ, f.face, 1000 + f.i);
  if (f.blur >= 1) motionBlur(img, W, H, f.blur);
  return img;
}
