"""Probe the ASR request options of muse-voice-transcribe on one short clip: which `mode` values the API takes, and
what each returns. One call per value, sequential.

  uv run --env-file .env python tools/exp_labels_modes.py AUDIO [MODE ...]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr import ASR_REQUEST, transcribe_turns  # noqa: E402

audio = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as tmp:
    clip = Path(tmp, "clip.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", audio, "-t", "6", "-ac", "1", "-ar", "16000", clip], check=True)
    data = clip.read_bytes()
for mode in sys.argv[2:] or ["ENDPOINTING", "BATCH", "OFFLINE", "FULL", "NONE", "bogus"]:
    try:
        r = transcribe_turns(data, request={"mode": mode})
        print(json.dumps({"mode": mode, "text": r["text"], "turns": len(r["turns"])}))
    except httpx.HTTPStatusError as e:
        print(json.dumps({"mode": mode, "status": e.response.status_code, "body": e.response.text[:400]}))
print("base request", ASR_REQUEST)
