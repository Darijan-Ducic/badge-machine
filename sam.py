"""Meta's hosted SAM 3.1: text-prompted object boxes for one frame (image mode) or a whole clip (video mode)."""

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import httpx

from muse_video import BASE_URL

MODEL = "sam-3.1"
FRAME = re.compile(r"<(\d+)f>")
RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|>")
# The mask payload alphabet has no '|' (nor '<', '>', ','), so the first '|>' always closes the token.
MASK = re.compile(r"<\|mask;[^|]*\|>")


def parse(text: str) -> dict:
    """SAM output -> {"width", "height", "frames": {frame: [{"id", "box": [x, y, w, h]}]}}.

    Each line is `<Nf>` (source frame N; images are always frame 0) then comma-separated records
    `id<|box;x1;y1;x2;y2;w;h[;c]|><|mask;...|>`. The id is a stable object id within the response (sparse,
    not positional). Corners are top-left-origin source pixels with x2/y2 inclusive; w/h is the source size.
    Frames with no match emit no line, so they are absent from "frames".
    """
    result = {"width": None, "height": None, "frames": {}}
    text = MASK.sub("", text)
    markers = list(FRAME.finditer(text))
    for marker, end in zip(markers, [m.start() for m in markers[1:]] + [len(text)]):
        objects = []
        for object_id, fields in RECORD.findall(text, marker.end(), end):
            box = dict(field.split("=") for field in fields.split(";"))
            x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
            objects.append({"id": int(object_id), "box": [x1, y1, x2 - x1 + 1, y2 - y1 + 1]})
            result["width"], result["height"] = int(box["w"]), int(box["h"])
        result["frames"][int(marker.group(1))] = objects
    return result


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}


def respond(prompt: str, media: dict) -> str:
    """Stream one SAM request and return the concatenated `response.output_text.delta` text."""
    body = {
        "model": MODEL,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, media]}],
        "stream": True,
    }
    deltas = []
    with httpx.stream("POST", f"{BASE_URL}/responses", json=body, headers=_headers(), timeout=600) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(f"SAM HTTP {response.status_code}: {response.text[:500]}")
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: "):])
            if event.get("type") == "response.output_text.delta":
                deltas.append(event["delta"])
            elif event.get("type") in ("error", "response.failed", "response.incomplete"):
                raise RuntimeError(f"SAM {event['type']}: {json.dumps(event)[:500]}")
    return "".join(deltas)


def image_media(image: bytes) -> dict:
    mime = "image/png" if image.startswith(b"\x89PNG") else "image/jpeg"
    return {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(image).decode()}"}


def segment_image(image: bytes, prompt: str) -> list[dict]:
    """JPEG/PNG bytes -> [{"id", "box": [x, y, w, h]}] in pixels of that image."""
    return parse(respond(prompt, image_media(image)))["frames"].get(0, [])


def to_h264_mp4(source: str | Path, target: str | Path) -> None:
    """Write any ffmpeg-readable video as a faststart H.264 MP4 with every source frame kept (SAM rejects VP8/VP9/WebM)."""
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", source],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    video = ["-c:v", "copy"] if codec == "h264" else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-fps_mode", "passthrough"]
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", source, "-map", "0:v:0", "-map", "0:a:0?", *video,
         "-c:a", "aac", "-movflags", "+faststart", target],
        check=True,
    )


def upload(path: str | Path) -> str:
    with open(path, "rb") as f:
        response = httpx.post(
            f"{BASE_URL}/files", headers=_headers(), data={"purpose": "user_data"},
            files={"file": (Path(path).name, f, "video/mp4")}, timeout=600,
        )
    response.raise_for_status()
    return response.json()["id"]


def segment_video(video_path: str | Path, prompt: str) -> dict:
    """Video file -> parse() result: {"width", "height", "frames": {frame_index: [{"id", "box"}]}}; ids track objects."""
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp, "clip.mp4")
        to_h264_mp4(video_path, mp4)
        file_id = upload(mp4)
    return parse(respond(prompt, {"type": "input_video", "file_id": file_id}))
