// Camera capture for the pages that record clips for the lip readers (Lip mode; dictation can reuse it).
// Opens the camera (and the mic only with `audio: true`), runs the mouth tracker on every camera frame the way
// track.html does (SAM re-anchors via POST /sam?prompt=face), and while a take is open records the FULL camera
// stream with MediaRecorder plus, per recorded frame, the tracker's flags and smoothed mouth-crop box.
// The browser never makes the final crop: the server crops lip-mode and dictation clips with one shared function.
//
//   const cap = await openCapture({ video, audio: false, onFrame: (r) => draw(r) });
//   cap.start();  ...  const take = await cap.stop();
//   take: {blob, ms, width, height, flags: [[...flag]] per frame, boxes: [[t_ms, x, y, w, h]] per frame with a
//          box (recording pixels, top-left origin, t_ms from the take's start), gate: {ok, ratio}}
import { createTracker, qualityGate } from "./tracker.js";

export async function openCapture({ video, audio = false, maxWidth = 640, onFrame = () => {}, onError = () => {} }) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", frameRate: { ideal: 30 } }, audio });
  video.srcObject = stream;
  if (video.readyState < 1) await new Promise((resolve) => video.addEventListener("loadedmetadata", resolve, { once: true }));

  // the frame the tracker and SAM see (at most maxWidth wide); k maps its pixels to recording pixels
  const work = document.createElement("canvas");
  const s = Math.min(1, maxWidth / video.videoWidth);
  work.width = Math.round(video.videoWidth * s);
  work.height = Math.round(video.videoHeight * s);
  const k = video.videoWidth / work.width;
  const wctx = work.getContext("2d", { willReadFrequently: true });

  async function samFace() {
    const blob = await new Promise((resolve) => work.toBlob(resolve, "image/jpeg", 0.85));
    const res = await fetch("/sam?prompt=face", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return (data.objects || []).map((o) => o.box);
  }
  const tracker = createTracker({ samFace });

  function grayFrame(t) {
    const { width: w, height: h } = work, rgba = wctx.getImageData(0, 0, w, h).data, data = new Uint8ClampedArray(w * h);
    for (let i = 0, p = 0; i < data.length; i++, p += 4) data[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
    return { data, width: w, height: h, t };
  }

  let busy = null, last = null, take = null, frames = 0, presented = null;
  const fps = stream.getVideoTracks()[0].getSettings().frameRate ?? null, period = 1000 / (fps || 30);
  const held = (t) => ({ t, flags: last ? last.flags : ["no-face"], box: last?.crop ?? null });

  // One tracker step per camera frame. A step waiting on SAM blocks the next ones (the tracker is stateful), and a
  // busy main thread skips frame callbacks; such frames keep the last result, so every recorded frame still gets
  // flags and a box.
  async function step(t, slot) {
    wctx.drawImage(video, 0, 0, work.width, work.height);
    const r = await tracker.step(grayFrame(t / 1000));
    last = r;
    frames++;
    if (slot) Object.assign(slot, { flags: r.flags, box: r.crop });
    onFrame(r, api);
  }

  function onVideoFrame(_, meta) {
    const t = performance.now(), skipped = meta && presented != null ? meta.presentedFrames - presented - 1 : 0;
    if (meta) presented = meta.presentedFrames;
    if (take) {
      const now = t - take.t0, after = take.frames.at(-1)?.t ?? 0;
      for (let i = skipped; i >= 1; i--) if (now - i * period > after) take.frames.push(held(now - i * period));
    }
    if (!busy) {
      const slot = take && { t: t - take.t0 };
      if (slot) take.frames.push(slot);
      busy = step(t, slot).catch(onError).finally(() => { busy = null; });
    } else if (take) take.frames.push(held(t - take.t0));
    next();
  }
  const next = video.requestVideoFrameCallback
    ? () => video.requestVideoFrameCallback(onVideoFrame)
    : () => requestAnimationFrame(() => (video.readyState >= 2 ? onVideoFrame() : next()));
  next();

  const api = {
    stream, video, work, tracker,
    scale: k, // recording pixels per tracker pixel
    width: video.videoWidth, height: video.videoHeight,
    fps,
    get last() { return last; },
    get frames() { return frames; },
    get recording() { return !!take; },

    start() {
      if (take) return false;
      const opts = MediaRecorder.isTypeSupported("video/webm") ? { mimeType: "video/webm" } : {};
      const recorder = new MediaRecorder(stream, opts), chunks = [];
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      take = { recorder, chunks, frames: [], t0: performance.now() };
      recorder.start();
      return true;
    },

    async stop() {
      const tk = take;
      if (!tk) return null;
      take = null;
      const ms = performance.now() - tk.t0;
      const stopped = new Promise((resolve) => { tk.recorder.onstop = resolve; });
      tk.recorder.stop();
      await Promise.all([stopped, busy]); // the in-flight step fills its frame's slot
      const done = tk.frames.filter((f) => f.flags);
      const flags = done.map((f) => f.flags);
      const boxes = done.filter((f) => f.box).map((f) => [Math.round(f.t), ...f.box.map((v) => Math.round(v * k))]);
      return { blob: new Blob(tk.chunks, { type: tk.recorder.mimeType }), ms, width: api.width, height: api.height,
        flags, boxes, gate: qualityGate(flags, tracker.config) };
    },
  };
  return api;
}
