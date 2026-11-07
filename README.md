# badge-machine

## Muse Spark video understanding

1. `cp .env.example .env` and set `MODEL_API_KEY` (from https://dev.meta.ai).
2. `uv run --env-file .env muse_video.py <clip.mp4 | https://…/clip.mp4> "your question"`

Local files ≤50 MB are sent inline; larger ones are uploaded via `/v1/files` (≤1 GiB). mp4 only.

## Webcam transcript

`uv run --env-file .env server.py`, then open http://localhost:8765. Record from the front webcam; on stop, the clip's audio is transcribed by `muse-voice-transcribe-1.0`.
