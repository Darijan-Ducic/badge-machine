// Camera capture for the pages that record clips for the lip readers (Lip mode, Dictation).
// Opens the camera (and the mic only with `audio: true`), runs the mouth tracker on every camera frame (SAM
// re-anchors via POST /sam?prompt=face, as in track.html), and while a take is open records the FULL camera
// stream with MediaRecorder plus, per recorded frame, the tracker's flags and smoothed mouth-crop box.
// The browser never makes the final crop: the server crops lip-mode and dictation clips with one shared function.
//
//   const cap = await openCapture({ video, audio: false, onFrame: (r) => draw(r) });
//   cap.start();  ...  const take = await cap.stop();
//   take: {blob, ms, width, height, flags: [[...flag]] per frame, boxes: [[t_ms, x, y, w, h]] per frame with a
//          box (recording pixels, top-left origin, t_ms from the take's start), gate: {ok, ratio}}
import { createTracker, qualityGate } from "./tracker.js";

export async function openCapture({ video, audio = false, maxWidth = 640, onFrame = () => {}, onError = () => {} }) {
  // Experiment flag, off by default: ?audio=raw opens the mic without Chrome's echo cancellation, noise suppression
  // and auto gain (the label-quality lab compares STT on both).
  if (audio === true && new URLSearchParams(location.search).get("audio") === "raw") {
    audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  }
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

  // SAM sees the color frame being tracked (which may be a queued one, older than what `work` shows now)
  const sam = document.createElement("canvas");
  sam.width = work.width;
  sam.height = work.height;
  async function samFace(frame) {
    sam.getContext("2d").putImageData(frame.image, 0, 0);
    const blob = await new Promise((resolve) => sam.toBlob(resolve, "image/jpeg", 0.85));
    const res = await fetch("/sam?prompt=face", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return (data.objects || []).map((o) => o.box);
  }
  const tracker = createTracker({ samFace });

  function grab(t) {
    wctx.drawImage(video, 0, 0, work.width, work.height);
    const { width: w, height: h } = work, image = wctx.getImageData(0, 0, w, h), rgba = image.data, data = new Uint8ClampedArray(w * h);
    for (let i = 0, p = 0; i < data.length; i++, p += 4) data[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
    return { data, width: w, height: h, t: t / 1000, image };
  }

  const MAX_QUEUE = 90; // ~3 s of frames waiting on SAM; past that a frame keeps the last result
  const fps = stream.getVideoTracks()[0].getSettings().frameRate ?? null, period = 1000 / (fps || 30);
  const queue = [];
  let draining = null, last = null, take = null, frames = 0, presented = null;
  const hold = (slot) => Object.assign(slot, { flags: last ? last.flags : ["no-face"], box: last?.crop ?? null });

  // The tracker sees every camera frame, in order: frames that arrive while a step waits on SAM queue up and are
  // tracked once it returns (skipping them makes the NCC search jump and drift). Frames whose callback a busy main
  // thread swallowed are gone; those keep the last result, so every recorded frame still gets flags and a box.
  async function drain() {
    while (queue.length) {
      const { frame, slot } = queue.shift();
      const r = await tracker.step(frame);
      last = r;
      frames++;
      if (slot) Object.assign(slot, { flags: r.flags, box: r.crop });
      onFrame(r, api);
    }
  }

  function onVideoFrame(_, meta) {
    const t = performance.now(), skipped = meta && presented != null ? meta.presentedFrames - presented - 1 : 0;
    if (meta) presented = meta.presentedFrames;
    let slot = null;
    if (take) {
      const now = t - take.t0, after = take.frames.at(-1)?.t ?? 0;
      for (let i = skipped; i >= 1; i--) if (now - i * period > after) take.frames.push(hold({ t: now - i * period }));
      take.frames.push((slot = { t: now }));
    }
    if (queue.length < MAX_QUEUE) queue.push({ frame: grab(t), slot });
    else if (slot) hold(slot);
    if (!draining) draining = drain().catch(onError).finally(() => { draining = null; });
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
      await Promise.all([stopped, draining]); // queued frames fill their slots
      const done = tk.frames.filter((f) => f.flags);
      const flags = done.map((f) => f.flags);
      const boxes = done.filter((f) => f.box).map((f) => [Math.round(f.t), ...f.box.map((v) => Math.round(v * k))]);
      return { blob: new Blob(tk.chunks, { type: tk.recorder.mimeType }), ms, width: api.width, height: api.height,
        flags, boxes, gate: qualityGate(flags, tracker.config) };
    },
  };
  return api;
}
