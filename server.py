"""Local server: serves static/ plus the API for transcripts, SAM boxes, preheat clips, lip mode and the clip
explorer."""

import json
import os
import re
import subprocess
import tempfile
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

from clips import DEFAULT_USER, backend, clip_detail, clip_list, label_clip, load, process_clip, stats, vtt_clip
from muse_video import BASE_URL
from sam import segment_image
from scorers import load_phrases

PORT = int(os.environ.get("PORT", 8765))
STORE = None  # clip store: None = MongoDB when MONGODB_URI is set, else data/clips (clips.backend)
CLIP_ROUTE = re.compile(r"^/clips/([^/]+)(/video)?$")
LABEL_ROUTE = re.compile(r"^/vtt/([^/]+)/label$")
STATIC = Path(__file__).parent / "static"
# Dedicated ASR model: muse-spark-1.3 claimed "no audio attached" on ~half of identical input_audio requests.
ASR_REQUEST = {"mode": "ENDPOINTING", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}


def transcribe(recording: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        source, wav = Path(tmp, "recording"), Path(tmp, "speech.wav")
        source.write_bytes(recording)
        # The ASR endpoint takes mono 16-bit PCM WAV at 16 or 24 kHz.
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", source, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
            check=True,
        )
        response = httpx.post(
            f"{BASE_URL}/asr/transcribe",
            params={"sessionId": str(uuid.uuid4())},
            headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"},
            files={
                "request": (None, json.dumps(ASR_REQUEST), "application/json"),
                "audio": ("speech.wav", wav.read_bytes(), "audio/wav"),
            },
            timeout=120,
        )
    response.raise_for_status()
    return response.json()["transcript"]


def tracker_headers(headers) -> tuple[list | None, list[str]]:
    """The browser tracker's crop boxes and flags sent with a recording: `X-Tracker-Boxes` a JSON list of
    [t_ms, x, y, w, h] (already padded; absent = crop the full frame), `X-Tracker-Flags` a JSON list of strings or
    a comma-separated list."""
    boxes, flags = headers.get("X-Tracker-Boxes"), (headers.get("X-Tracker-Flags") or "").strip()
    if boxes is not None:
        boxes = json.loads(boxes)
        if not all(isinstance(b, list) and len(b) == 5 for b in boxes):
            raise ValueError("X-Tracker-Boxes must be a JSON list of [t_ms, x, y, w, h]")
    if flags.startswith("["):
        return boxes, [str(f) for f in json.loads(flags)]
    return boxes, [f.strip() for f in flags.split(",") if f.strip()]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC, **kwargs)

    def send_json(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_bytes(self, content_type: str, payload: bytes):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path)
        clip = CLIP_ROUTE.match(url.path)
        if url.path not in ("/phrases", "/stats", "/clips") and not clip:
            super().do_GET()
            return
        user = parse_qs(url.query).get("user", [""])[0] or None  # none: the user of the latest clip
        try:
            if url.path == "/phrases":
                self.send_json(200, {"phrases": load_phrases()})
            elif url.path == "/stats":
                self.send_json(200, stats(load(STORE, user), load_phrases(), user=user))
            elif url.path == "/clips":
                self.send_json(200, clip_list(STORE, load_phrases(), user))
            elif clip.group(2):
                video = backend(STORE).video(clip.group(1))
                self.send_bytes("video/mp4", video) if video is not None else self.send_json(404, {"error": "no clip"})
            else:
                detail = clip_detail(STORE, clip.group(1))
                self.send_json(200, detail) if detail is not None else self.send_json(404, {"error": "no clip"})
        except Exception as e:
            self.send_json(500, {"error": str(e)})

    def post_label(self, clip_id: str, data: bytes):
        """The user's label for a lip-mode clip, {"text", "source"}: 400 for a bad body, 404 for an unknown clip."""
        try:
            request = json.loads(data)
            text, source = (request.get("text"), request.get("source")) if isinstance(request, dict) else (None, None)
            found = label_clip(STORE, clip_id, text, source)
        except ValueError as e:  # not JSON, no text, or a source outside clips.LABEL_SOURCES
            self.send_json(400, {"error": str(e)})
            return
        self.send_json(200, {"clip": clip_id, "label": text}) if found else self.send_json(404, {"error": "no clip"})

    def do_POST(self):
        url = urlparse(self.path)
        label = LABEL_ROUTE.match(url.path)
        if url.path not in ("/transcribe", "/sam", "/preheat/clip", "/vtt") and not label:
            self.send_error(404)
            return
        query = parse_qs(url.query)
        try:
            data = self.rfile.read(int(self.headers["Content-Length"]))
            if label:
                self.post_label(label.group(1), data)
                return
            if url.path == "/transcribe":
                body = {"transcript": transcribe(data)}
            elif url.path == "/preheat/clip":
                fps, prompted = query.get("fps"), query.get("prompted")
                mouths, flags = tracker_headers(self.headers)
                body = process_clip(data, float(fps[0]) if fps else None, int(prompted[0]) if prompted else None,
                                    transcribe=transcribe, phrases=load_phrases(), store=STORE,
                                    user_id=query.get("user", [""])[0] or DEFAULT_USER, mouths=mouths,
                                    tracker_flags=flags)
            elif url.path == "/vtt":
                # Lip mode: one muted recording -> the VTT pick; read like /preheat/clip, but with K=3 and no STT.
                mouths, flags = tracker_headers(self.headers)
                body = vtt_clip(data, load_phrases(), STORE, user_id=query.get("user", [""])[0] or DEFAULT_USER,
                                mouths=mouths, tracker_flags=flags)
            else:
                # One JPEG/PNG frame -> SAM boxes [x, y, w, h] in that frame's pixels; ids are per-request ordinals.
                prompt = query.get("prompt", [""])[0].strip()
                if not prompt:
                    raise ValueError("missing ?prompt=<noun phrase>")
                body = {"objects": segment_image(data, prompt)}
            code = 200
        except Exception as e:
            body, code = {"error": str(e)}, 500
        self.send_json(code, body)


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
