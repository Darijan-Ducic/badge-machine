"""STT: Muse Voice Transcribe on a recording's audio — the whole text, and the turns its endpointing found."""

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx

from muse_video import BASE_URL

# Dedicated ASR model: muse-spark-1.3 claimed "no audio attached" on ~half of identical input_audio requests.
ASR_REQUEST = {"mode": "ENDPOINTING", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}


def transcribe_turns(recording: bytes, request: dict | None = None) -> dict:
    """Any ffmpeg-readable media -> {"text": transcript, "turns": [{"start_ms", "end_ms", "text"}]}. `request` fields
    override ASR_REQUEST's (experiments only)."""
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
                "request": (None, json.dumps({**ASR_REQUEST, **(request or {})}), "application/json"),
                "audio": ("speech.wav", wav.read_bytes(), "audio/wav"),
            },
            timeout=120,
        )
    response.raise_for_status()
    body = response.json()
    return {
        "text": body["transcript"],
        "turns": [{"start_ms": t["startMs"], "end_ms": t["endMs"], "text": t["transcript"]} for t in body.get("turns", [])],
    }


def transcribe(recording: bytes) -> str:
    return transcribe_turns(recording)["text"]
