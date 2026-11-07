"""Pixel variants of the mouth crop, for experiments: what the frames look like before the lip readers see them.

`render` crops every source frame with its own box and writes a muted H.264 mp4 like crop.crop. Geometry: today's ONE
static box (the per-coordinate median of the tracker's per-frame boxes, clamped to the frame as crop.crop does) or a
STABILIZED crop (the median size, centred on each frame's Gaussian-smoothed box centre, shifted to stay inside the
frame), so a drifting head keeps the mouth at the middle of the crop. Each resized frame then gets optional gray,
local contrast and sharpening (PIL). `cropper(spec)` has crop.crop's signature, for clips.read_clip(crop_fn=...).
Frames keep their order and count; the output runs at the source's mean frame rate."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import crop

STABILIZED = "v1-tracker-stabilized"


@dataclass(frozen=True)
class Spec:
    stabilize: bool = False
    window: int = 5  # frames; the box centres are smoothed with a Gaussian of sigma window/2, cut at +-window
    width: int = crop.WIDTH  # output width in px
    gray: bool = False
    contrast: bool = False  # local contrast (wide unsharp mask) + a 1 % autocontrast stretch
    sharpen: float = 0.0  # unsharp-mask amount at radius 2 px of the output (1.5 = 150 %); 0 = off
    crf: int | None = None  # libx264 CRF; None = the encoder default (23), as crop.crop
    sam_lips: bool = False  # centre the (tracker-sized) box on SAM 3.1's "lips" boxes instead of the tracker's

    def tag(self) -> str:
        parts = [f"w{self.width}"] if self.width != crop.WIDTH else []
        parts += ["gray"] * self.gray + ["contrast"] * self.contrast
        parts += [f"sharp{self.sharpen:g}"] if self.sharpen else []
        parts += [f"crf{self.crf}"] if self.crf is not None else []
        return "-".join(parts)

    def crop_id(self) -> str:
        base = f"{STABILIZED}-s{self.window}" if self.stabilize else crop.TRACKER_MEDIAN
        base += "-sam-lips" if self.sam_lips else ""
        return f"{base}+{self.tag()}" if self.tag() else base


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Gaussian smoothing (sigma window/2, radius window) with point-reflected ends, so a straight line stays put."""
    if window < 1 or len(values) < 2:
        return values.astype(float)
    radius = min(window, len(values) - 1)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / (window / 2)) ** 2)
    left = 2 * values[0] - values[1:radius + 1][::-1]
    right = 2 * values[-1] - values[-radius - 1:-1][::-1]
    return np.convolve(np.concatenate([left, values, right]), kernel / kernel.sum(), mode="valid")


def stabilized_track(boxes: list[list[float]], window: int = 5) -> list[list[float]]:
    """[t_ms, x, y, w, h] boxes -> the same times with the median size, centred on the smoothed box centres."""
    b = np.asarray(boxes, float)
    w, h = np.median(b[:, 3]), np.median(b[:, 4])
    cx, cy = _smooth(b[:, 1] + b[:, 3] / 2, window), _smooth(b[:, 2] + b[:, 4] / 2, window)
    return [[float(t), float(x - w / 2), float(y - h / 2), float(w), float(h)] for t, x, y in zip(b[:, 0], cx, cy)]


def recentre(boxes: list[list[float]], lips: dict, times: list[float]) -> list[list[float]]:
    """Tracker-sized boxes centred on SAM lips: for each source frame i with lips (`lips` = sam.parse()["frames"]),
    [times[i], x, y, w, h] with w, h the median tracker size, centred on the lips box nearest the tracker's centre."""
    b = np.asarray(boxes, float)
    w, h = float(np.median(b[:, 3])), float(np.median(b[:, 4]))
    out = []
    for i, t in enumerate(times):
        if not lips.get(i):
            continue
        tx = np.interp(t, b[:, 0], b[:, 1] + b[:, 3] / 2)
        ty = np.interp(t, b[:, 0], b[:, 2] + b[:, 4] / 2)
        x, y, lw, lh = min((o["box"] for o in lips[i]),
                           key=lambda q: (q[0] + q[2] / 2 - tx) ** 2 + (q[1] + q[3] / 2 - ty) ** 2)
        out.append([float(t), x + lw / 2 - w / 2, y + lh / 2 - h / 2, w, h])
    return out


def enhance(image: Image.Image, spec: Spec) -> Image.Image:
    """One resized crop frame through the spec's gray -> contrast -> sharpen; unchanged with every option off."""
    if spec.gray:
        image = image.convert("L").convert("RGB")
    if spec.contrast:
        image = image.filter(ImageFilter.UnsharpMask(radius=max(4, image.width // 16), percent=40, threshold=0))
        image = ImageOps.autocontrast(image, cutoff=1, preserve_tone=True)  # one stretch for all bands: no hue shift
    if spec.sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=round(100 * spec.sharpen), threshold=0))
    return image


def _frames(source: Path) -> tuple[np.ndarray, np.ndarray]:
    """(frame times in ms from the first frame, frames as an (n, h, w, 3) uint8 array), every source frame."""
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:frame=best_effort_timestamp_time", "-of", "json", source],
        capture_output=True, text=True, check=True).stdout)
    width, height = probe["streams"][0]["width"], probe["streams"][0]["height"]
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", source, "-map", "0:v:0", "-fps_mode",
                          "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    times = [float(f["best_effort_timestamp_time"]) * 1000 for f in probe["frames"]
             if f.get("best_effort_timestamp_time") not in (None, "N/A")]
    if len(times) != len(frames):  # no usable timestamps: assume an even 30 fps
        times = [i * 1000 / 30 for i in range(len(frames))]
    times = np.asarray(times)
    return times - times[0], frames


def _box(track: np.ndarray, t: float) -> tuple[float, float, float, float]:
    return tuple(float(np.interp(t, track[:, 0], track[:, i])) for i in range(1, 5))


def render(source: Path, target: Path, boxes: list[list[float]], spec: Spec) -> None:
    """Write the spec's crop of `source` (boxes [[t_ms, x, y, w, h]], t from the clip start) to `target` (mp4)."""
    times, frames = _frames(Path(source))
    n, height, width = frames.shape[:3]
    if spec.stabilize:
        track = np.asarray(stabilized_track(boxes, spec.window))
    else:
        track = np.asarray([[0.0, *crop.tracker_box(boxes)]])
    w, h = min(width, round(track[0, 3])), min(height, round(track[0, 4]))
    out_w = spec.width
    out_h = max(2, round(h * out_w / w / 2) * 2)
    encoded = []
    for t, frame in zip(times, frames):
        x, y, bw, bh = _box(track, t)
        if spec.stabilize:  # fixed size, shifted to stay inside the frame
            x0 = int(np.clip(round(x), 0, width - w))
            y0 = int(np.clip(round(y), 0, height - h))
            x1, y1 = x0 + w, y0 + h
        else:  # crop.crop's clamp
            x0, y0 = max(0, round(x)), max(0, round(y))
            x1, y1 = min(width, round(x + bw)), min(height, round(y + bh))
            out_h = max(2, round((y1 - y0) * out_w / (x1 - x0) / 2) * 2)
        image = Image.fromarray(frame[y0:y1, x0:x1]).resize((out_w, out_h), Image.BICUBIC)
        encoded.append(np.asarray(enhance(image, spec)).tobytes())
    fps = (n - 1) * 1000 / times[-1] if n > 1 and times[-1] > 0 else 30.0
    quality = ["-crf", str(spec.crf)] if spec.crf is not None else []
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{out_w}x{out_h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    *quality, "-movflags", "+faststart", target], input=b"".join(encoded), check=True)


def cropper(spec: Spec, segment_video=None):
    """A crop.crop-compatible function rendering `spec`; without tracker boxes it falls back to crop.crop. With
    `spec.sam_lips` it runs `segment_video(source, "lips")` (default sam.segment_video) and recentres the boxes;
    without any lips found it keeps the tracker's boxes."""
    def crop_fn(source: Path, target: Path, faces=None, mouths=None) -> str:
        if not mouths:
            return crop.crop(source, target, faces=faces, mouths=mouths)
        if spec.sam_lips:
            import sam
            lips = (segment_video or sam.segment_video)(source, "lips")["frames"]
            mouths = recentre(mouths, lips, list(_frames(Path(source))[0])) or mouths
        render(source, target, mouths, spec)
        return spec.crop_id()
    return crop_fn
