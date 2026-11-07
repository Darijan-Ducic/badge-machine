"""Static per-clip mouth crop: one box for the whole clip, so the crop the lip readers see can't jitter.

Each frame's face box [x, y, w, h] gives a mouth box with the browser tracker's ratios (MOUTH_* and CROP_PAD, read
from the defaults in static/tracker.js so both crops stay in sync); the clip's box is the per-coordinate median of
those, written as a muted H.264 mp4 scaled to WIDTH px wide. Without any face box the clip passes through muted at
full frame. Every source frame is kept (no frame-rate change)."""

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from scorers import FPS

TRACKER = Path(__file__).parent / "static" / "tracker.js"
STATIC_MEDIAN = "v1-static-median"
FULL_FRAME = "full-frame-v0"
WIDTH = 256
_KEYS = ("MOUTH_CX", "MOUTH_CY", "MOUTH_W", "MOUTH_H", "CROP_PAD")


def tracker_geometry(path: Path = TRACKER) -> dict[str, float]:
    """The mouth ratios from the tracker's CONFIG defaults."""
    text = Path(path).read_text()
    return {key: float(re.search(rf"\b{key}:\s*([0-9.]+)", text).group(1)) for key in _KEYS}


GEOMETRY = tracker_geometry()


def mouth_box(face: list[float], g: dict[str, float] = GEOMETRY) -> list[float]:
    """The padded mouth box the tracker derives from a face box (tracker.js mouthBox)."""
    x, y, w, h = face
    mw, mh = g["MOUTH_W"] * w * g["CROP_PAD"], g["MOUTH_H"] * h * g["CROP_PAD"]
    return [x + g["MOUTH_CX"] * w - mw / 2, y + g["MOUTH_CY"] * h - mh / 2, mw, mh]


def median_box(faces: list[list[float]]) -> list[float] | None:
    if not faces:
        return None
    return [float(v) for v in np.median([mouth_box(face) for face in faces], axis=0)]


def _size(path: Path) -> tuple[int, int]:
    stream = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json",
         path], capture_output=True, text=True, check=True).stdout)["streams"][0]
    return stream["width"], stream["height"]


def crop(source: Path, target: Path, faces: list[list[float]] | None) -> str:
    """Write the muted crop of `source` to `target` (mp4); returns the crop id recorded with the clip."""
    box = median_box(faces or [])
    if box is None:
        crop_id, video_filter = FULL_FRAME, "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # H.264 4:2:0 needs even sides
    else:
        width, height = _size(source)
        x0, y0 = max(0, round(box[0])), max(0, round(box[1]))
        x1, y1 = min(width, round(box[0] + box[2])), min(height, round(box[1] + box[3]))
        crop_id, video_filter = STATIC_MEDIAN, f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={WIDTH}:-2,setsar=1"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", source, "-map", "0:v:0", "-an",
                    "-vf", video_filter, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    "-movflags", "+faststart", target], check=True)
    return crop_id


def count_frames(path: Path, fps: int = FPS) -> int:
    """Frames in the clip at `fps`, numbered as muse_sampler stamps them (its decode uses the same fps filter)."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:v:0", "-vf", f"fps={fps}",
                          "-f", "null", "-", "-progress", "pipe:1"], capture_output=True, text=True, check=True).stdout
    return int(re.findall(r"^frame=(\d+)", out, re.M)[-1])
