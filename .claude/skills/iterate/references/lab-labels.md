# Lab: STT label quality (Obama Weekly Address vs the official transcripts, 2026-11-07)

Read this before trusting any VTT accuracy number: every VTT metric is measured against these STT labels.
- **Numbers (official press-office text):** per-turn label WER is 4.3% raw on 26 Atlas sessions / 90 turns (3.5–5.5% between repeat calls). About half of that is STT repeats: 2.0 repeated words per 100, in 26% of turns, and many are phrases ("road to road to"). The phrase collapse brings WER to 2.2%. Turn boundaries lose 0.3–0.8% of words.
- **Cause:** muse-voice-transcribe repeats words and phrases at random. It also does it on the source wav with no Chrome in the path, and 3 calls on the same audio differ on 54 of 58 turns.
- **Lever: phrase collapse. ACTIVE** in `preheat._normalize` (n ≤ 4), commit 23b551b. It removes 3 of 6,916 official words that were really spoken. Reopen when a speaker really repeats phrases (check with `collapse_repeats` on their reference text).
- **Lever: Chrome audio processing off. REJECTED.** Arms (default / raw / source): repeated words 2.27 / 2.63 / 2.14 per 100, boundary loss 0.8% / 2.7% / 2.6%. Noise suppression makes the −30 dB pauses real, so the cuts land between words. `?audio=raw` is a capture flag, off by default. Reopen on a real laptop mic in a noisy room.
- **Lever: best-of-3 STT. REJECTED.** Same 2.4% WER as the collapse, at 3× the STT calls.
- **Lever: 600 ms of digital silence around each turn's STT audio. PARKED.** Boundary loss goes from 0.78% to 0.49%; doubling does not change. Reopen if boundary loss rises above 1%.
- **Lever: ASR mode. CLOSED.** Only `ENDPOINTING` is accepted; 11 other values return 400. Reopen when the API docs list other modes.
- **Commands** (official texts fetched to `/tmp/lab-labels/official`):
  `uv run python -I tools/exp_labels_fetch.py /tmp/lab-labels/official` ·
  `uv run --env-file .env python tools/exp_labels_measure.py atlas` ·
  `uv run python tools/exp_labels_record.py --arm raw --out DIR @list` (stub server, no Atlas) ·
  `uv run --env-file .env python tools/exp_labels_stt.py OUT.jsonl FILES [--repeat 3 | --silence-ms 600]` ·
  `uv run python tools/exp_labels_measure.py file OUT.jsonl`
