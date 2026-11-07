"""Local server for the webcam page: serves static/ and turns a recorded clip into a Muse Spark transcript."""

import json
import os
import subprocess
import tempfile
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from muse_video import BASE_URL

PORT = 8765
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


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC, **kwargs)

    def do_POST(self):
        if self.path != "/transcribe":
            self.send_error(404)
            return
        try:
            body = {"transcript": transcribe(self.rfile.read(int(self.headers["Content-Length"])))}
            code = 200
        except Exception as e:
            body, code = {"error": str(e)}, 500
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
