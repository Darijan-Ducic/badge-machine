"""Save the label lab's experiments as run records (Training page, Runs tab). No model calls.

  uv run --env-file .env python tools/exp_labels_save.py
"""

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runs  # noqa: E402

BASE = {"author": "agent:lab-labels", "loop": "framework", "lab": "lab-labels", "dataset": "practice:obama-weekly:obama",
        "reference": "obamawhitehouse.archives.gov press-office transcripts (Commons captions as a second reference)"}
RECORDS = [
    {"name": "lab-labels-e1 label quality baseline", "lever": "stt-label:measure",
     "user_id": "practice:obama-weekly:obama",
     "hypothesis": "Per-turn STT labels match the official transcript to within a few percent",
     "arms": ["atlas sessions (Chrome default audio)"], "n": {"sessions": 26, "turns": 90},
     "metrics": {"turn_wer": 0.0426, "turn_wer_normalized_old": 0.0326, "turn_wer_phrase_collapsed": 0.0219,
                 "session_wer": 0.0485, "turn_repeat_words_per100": 2.03, "turns_with_repeat": 0.256,
                 "boundary_lost_rate": 0.003, "boundaries_with_loss": 0.078, "cost_usd": 0},
     "verdict": "pass", "reopen": "re-measure on a second speaker or after any STT model change"},
    {"name": "lab-labels-e2 chrome audio processing off", "lever": "capture:audio-processing",
     "user_id": "exp:lab-labels-raw:practice:obama-weekly:obama",
     "hypothesis": "Chrome echo cancellation / noise suppression / auto gain cause the STT doubled words",
     "arms": ["default", "raw (?audio=raw)", "source wav (no Chrome)"], "n": {"sessions": 16, "turns": [58, 54, 54]},
     "metrics": {"turn_wer": [0.0497, 0.0522, 0.053], "turn_repeat_words_per100": [2.27, 2.63, 2.14],
                 "boundary_lost_rate": [0.0078, 0.027, 0.026], "boundaries_with_loss": [0.167, 0.368, 0.368],
                 "stt_calls": 210},
     "verdict": "fail", "reopen": "a real (non-emulated) mic with room noise; keep default ON: it makes -30 dB pauses real"},
    {"name": "lab-labels-e3 repeat fixes", "lever": "stt-label:repeat-collapse",
     "user_id": "exp:lab-labels-raw:practice:obama-weekly:obama",
     "hypothesis": "Collapsing immediate phrase repeats (n<=4) removes the doubling error at no cost",
     "arms": ["1 call", "best-of-3 (fewest repeats)", "medoid-of-3", "1 call + phrase collapse", "1 call + 600 ms silence pad"],
     "n": {"sessions": 16, "turns": 58},
     "metrics": {"turn_wer": [0.037, 0.0244, 0.0282, 0.0244, 0.0544], "turn_repeat_words_per100": [1.15, 0.10, 0.48, 0.0, 2.73],
                 "boundary_lost_rate": [0.0078, 0.0078, 0.0078, 0.0078, 0.0049], "official_words_wrongly_collapsed": "3/6916",
                 "stt_calls": 292},
     "verdict": "pass", "reopen": "a speaker who really repeats phrases; landed as preheat._normalize (commit 23b551b)"},
]

for r in RECORDS:
    runs.save({"id": uuid.uuid4().hex[:12], "t": time.time(), **BASE, **r})
    print("saved", r["name"])
