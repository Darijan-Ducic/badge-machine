"""Lip measurements over time from SAM 3.1 "lips" boxes plus classical pixel tests inside each box (lab-features).

A clip becomes a per-frame series at the sampler's FPS (frame i here = frame i stamped on Muse's tiles):
  open    inner-mouth opening: median height of dark-or-teeth pixels over the box's central columns / lip width
  width   lips box width / the clip's median lips box width (1.0 = this speaker's usual width)
  height  lips box height / the clip's median lips box width
  aspect  lips box height / width
  dark    fraction of the box's central region darker than half the clip's median box brightness (open cavity)
  teeth   fraction of the central region that is bright and unsaturated (visible teeth)
  d_open, d_width   frame-to-frame velocities of open and width
Everything is relative to the speaker's own lip size and brightness, so identity, scale and exposure drop out.
SAM's mask payload is an undocumented compressed encoding, so the lip shape inside the box is classical: redness
(R-G)/(R+G) thresholded by Otsu per clip (lips), dark pixels (cavity), bright unsaturated pixels (teeth).
"""

import io
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SERIES = ("open", "width", "height", "aspect", "dark", "teeth", "d_open", "d_width")
SIL_SIZE = (160, 96)  # silhouette canvas (w, h); the clip's median lip width maps to 0.75 of it


def decode(clip, fps: int) -> np.ndarray:
    """(n, H, W, 3) uint8 frames of `clip` at `fps` (the same fps filter muse_sampler decodes with)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(clip)], capture_output=True, text=True, check=True).stdout
    w, h = map(int, probe.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(clip), "-vf", f"fps={fps}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def track(frames: dict, n: int) -> np.ndarray:
    """(n, 4) float [x, y, w, h] lips box per frame from SAM's {frame: [[x, y, w, h], ...]}: on frames with several
    boxes the one closest to the previous pick (largest on the first), missing frames linearly interpolated and the
    ends held. Raises ValueError when no frame has a box."""
    picks, prev = {}, None
    for i in sorted(int(k) for k in frames):
        boxes = [np.asarray(b, float) for b in frames.get(i, frames.get(str(i), []))]
        if not boxes:
            continue
        if prev is None:
            box = max(boxes, key=lambda b: b[2] * b[3])
        else:
            box = min(boxes, key=lambda b: np.abs((b[:2] + b[2:] / 2) - (prev[:2] + prev[2:] / 2)).sum())
        picks[i], prev = box, box
    if not picks:
        raise ValueError("no frame has a lips box")
    known = np.array(sorted(picks))
    values = np.array([picks[i] for i in known])
    return np.stack([np.interp(np.arange(n), known, values[:, j]) for j in range(4)], axis=1)


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _patch(frame: np.ndarray, box) -> np.ndarray:
    x, y, w, h = (int(round(v)) for v in box)
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + max(w, 2)), min(H, y + max(h, 2))
    return frame[y0:y1, x0:x1].astype(np.float32)


class Reference:
    """Per-clip thresholds: brightness median, teeth brightness, lip redness (Otsu), all over every lips box."""

    def __init__(self, frames: np.ndarray, boxes: np.ndarray):
        luma, red, sat = [], [], []
        for frame, box in zip(frames, boxes):
            p = _patch(frame, box)
            if p.size == 0:
                continue
            luma.append(_luma(p).ravel())
            red.append(_red(p).ravel())
            sat.append(_sat(p).ravel())
        luma, red = np.concatenate(luma), np.concatenate(red)
        self.median = float(np.median(luma))
        self.dark = 0.5 * self.median
        self.bright = max(float(np.percentile(luma, 97)), 1.25 * self.median)
        self.red = _otsu(red)
        self.width = float(np.median(boxes[:, 2]))


def _luma(p):
    return 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]


def _red(p):
    return (p[..., 0] - p[..., 1]) / (p[..., 0] + p[..., 1] + 1.0)


def _sat(p):
    return (p.max(-1) - p.min(-1)) / (p.max(-1) + 1.0)


def classify(patch: np.ndarray, ref: Reference) -> np.ndarray:
    """Per-pixel class inside a lips box: 0 skin/other, 1 lip, 2 cavity (dark), 3 teeth (bright, unsaturated)."""
    luma, red, sat = _luma(patch), _red(patch), _sat(patch)
    out = np.zeros(luma.shape, np.uint8)
    out[red > ref.red] = 1
    out[luma < ref.dark] = 2
    out[(luma > 0.9 * ref.bright) & (sat < 0.35)] = 3
    return out


def measure(patch: np.ndarray, ref: Reference) -> dict:
    """Pixel measurements of one frame's lips box (see module doc)."""
    cls = classify(patch, ref)
    h, w = cls.shape
    core = cls[:, int(w * 0.3): max(int(w * 0.7), int(w * 0.3) + 1)]
    inner = (core == 2) | (core == 3)
    return {"open": float(np.median(inner.sum(0))) / ref.width if core.size else 0.0,
            "dark": float((core == 2).mean()) if core.size else 0.0,
            "teeth": float((core == 3).mean()) if core.size else 0.0}


def series(frames: np.ndarray, boxes: np.ndarray) -> dict:
    """{name: (n,) array} for SERIES, from decoded frames and their tracked lips boxes."""
    ref = Reference(frames, boxes)
    rows = [measure(_patch(f, b), ref) for f, b in zip(frames, boxes)]
    out = {k: np.array([r[k] for r in rows]) for k in ("open", "dark", "teeth")}
    out["width"] = boxes[:, 2] / ref.width
    out["height"] = boxes[:, 3] / ref.width
    out["aspect"] = boxes[:, 3] / np.maximum(boxes[:, 2], 1)
    out["d_open"] = np.diff(out["open"], prepend=out["open"][:1])
    out["d_width"] = np.diff(out["width"], prepend=out["width"][:1])
    return out


def silhouette(frame: np.ndarray, box, ref: Reference, size=SIL_SIZE) -> Image.Image:
    """The frame's mouth as a drawing on white: the lips box as a gray ellipse (outer lip extent), open cavity black,
    teeth white; box centre at the canvas centre, scaled so the clip's median lip width is 0.75 of the canvas width
    (no skin, no identity, no blur)."""
    patch = _patch(frame, box)
    cls = classify(patch, ref)
    scale = 0.75 * size[0] / ref.width
    w, h = max(1, round(cls.shape[1] * scale)), max(1, round(cls.shape[0] * scale))
    big = np.array(Image.fromarray(cls).resize((w, h), Image.NEAREST))
    x0, y0 = (size[0] - w) // 2, (size[1] - h) // 2
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).ellipse((x0, y0, x0 + w - 1, y0 + h - 1), fill=(170, 170, 170), outline=(90, 90, 90))
    canvas = np.array(image)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw, ch = min(w - sx0, size[0] - dx0), min(h - sy0, size[1] - dy0)
    sub = big[sy0:sy0 + ch, sx0:sx0 + cw]
    region = canvas[dy0:dy0 + ch, dx0:dx0 + cw]
    region[sub == 2] = (0, 0, 0)  # open cavity
    region[sub == 3] = (255, 255, 255)  # teeth
    return Image.fromarray(canvas)


def table(s: dict, names=("open", "width", "aspect", "dark", "teeth"), digits: int = 2) -> str:
    """One line per frame: `frame open width ...`, a header first."""
    lines = ["frame " + " ".join(names)]
    for i in range(len(s[names[0]])):
        lines.append(f"{i} " + " ".join(f"{s[k][i]:.{digits}f}" for k in names))
    return "\n".join(lines)


def chart(s: dict, names=("open", "width", "teeth"), size=(960, 360)) -> bytes:
    """A PNG line chart of the series against frame number (one colour per series, gridlines every 10 frames)."""
    colors = {"open": (220, 30, 30), "width": (30, 90, 220), "teeth": (30, 160, 60), "dark": (0, 0, 0),
              "aspect": (200, 120, 0), "height": (120, 0, 160)}
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    n = len(s[names[0]])
    left, top, right, bottom = 40, 20, size[0] - 10, size[1] - 30
    font = ImageFont.load_default(size=12)
    for f in range(0, n, 10):
        x = left + (right - left) * f / max(n - 1, 1)
        d.line((x, top, x, bottom), fill=(225, 225, 225))
        d.text((x - 6, bottom + 4), str(f), fill="black", font=font)
    for k, name in enumerate(names):
        v = np.asarray(s[name], float)
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo or 1.0
        pts = [(left + (right - left) * i / max(n - 1, 1), bottom - (bottom - top) * (x - lo) / span)
               for i, x in enumerate(v)]
        d.line(pts, fill=colors.get(name, (0, 0, 0)), width=2)
        d.text((left + 5 + 90 * k, 3), name, fill=colors.get(name, (0, 0, 0)), font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- template matching


def matrix(s: dict, names=("open", "width", "aspect", "teeth")) -> np.ndarray:
    """(n, d) feature matrix, each column z-scored within the clip (speaker-relative shapes, not levels)."""
    m = np.stack([np.asarray(s[k], float) for k in names], axis=1)
    return (m - m.mean(0)) / (m.std(0) + 1e-6)


def dtw(a: np.ndarray, b: np.ndarray, band: float | None = None) -> float:
    """Length-normalized dynamic-time-warping distance between (n, d) and (m, d) sequences (Euclidean frame cost);
    `band` limits |i/n - j/m| (Sakoe-Chiba, as a fraction) when given."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band is not None and abs(i / n - j / m) > band:
                continue
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def rank(query: np.ndarray, history: list[tuple[str, np.ndarray]]) -> list[tuple[str, float]]:
    """Labels ranked by their nearest template's DTW distance to `query` (closest first). `history` holds only
    clips recorded before the query (the caller's no-leak contract); one entry per earlier clip."""
    best = {}
    for label, template in history:
        d = dtw(query, template, band=0.3)
        best[label] = min(best.get(label, np.inf), d)
    return sorted(best.items(), key=lambda kv: kv[1])
