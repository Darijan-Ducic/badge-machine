"""Clip store for preheat: keep each STT-labeled clip with both lip readers' scores, and report training stats."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from preheat import STAGES, match_phrase, predict, prequential, stage_for, train
from scorers import gemini_scores, muse_scores

DATA = Path("data/clips")
_lock = threading.Lock()  # clips are processed concurrently; appends and retraining must not interleave


def vocab_id(phrases: list[str]) -> str:
    # Score vectors are only comparable under the same phrase list, so records are tagged with it.
    return hashlib.sha1("\n".join(phrases).encode()).hexdigest()[:8]


def load(store: Path = DATA) -> list[dict]:
    index = store / "index.jsonl"
    return [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []


def _arrays(labeled: list[dict], n_phrases: int):
    gemini = np.array([r["gemini"] for r in labeled]).reshape(-1, n_phrases)
    muse = np.array([r["muse"] for r in labeled]).reshape(-1, n_phrases)
    return gemini, muse, np.array([r["label"] for r in labeled], dtype=int)


def stats(records: list[dict], phrases: list[str], window: int = 20) -> dict:
    records = [r for r in records if r["vocab"] == vocab_id(phrases)]
    labeled = [r for r in records if r["label"] is not None]
    gemini, muse, labels = _arrays(labeled, len(phrases))
    history = [{"i": i, **h} for i, h in enumerate(prequential(gemini, muse, labels))]

    def accuracy(entries):
        return {k: float(np.mean([e[k] for e in entries])) if entries else None for k in ("fused", "gemini", "muse")}

    upcoming = sorted((n, name) for n, name in STAGES if n > len(labeled))
    threshold = train(gemini, muse, labels).threshold
    return {
        "labeled": len(labeled),
        "clips": len(records),
        "stage": stage_for(len(labeled)),
        "next_stage": {"name": upcoming[0][1], "at": upcoming[0][0]} if upcoming else None,
        "threshold": None if np.isinf(threshold) else float(threshold),
        "camera_fps": records[-1]["fps"] if records else None,
        "window": window,
        "accuracy": {"recent": accuracy(history[-window:]), "all": accuracy(history)},
        "history": history,
    }


def process_clip(recording: bytes, fps, prompted, transcribe, phrases, store: Path = DATA,
                 gemini=gemini_scores, muse=muse_scores) -> dict:
    """STT-label one recording, read its lips with both models, score them against the label, and store it."""
    clip_id = str(time.time_ns())
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{clip_id}.webm"
    path.write_bytes(recording)
    with ThreadPoolExecutor(3) as pool:
        heard = pool.submit(transcribe, recording)
        gemini_probs, muse_probs = pool.submit(gemini, path, phrases), pool.submit(muse, path, phrases)
        heard, gemini_probs, muse_probs = heard.result(), gemini_probs.result(), muse_probs.result()
    label = match_phrase(heard, phrases)

    with _lock:
        records = [r for r in load(store) if r["vocab"] == vocab_id(phrases)]
        # Predict, then train: this clip is read by a model that has never seen its label.
        model = train(*_arrays([r for r in records if r["label"] is not None], len(phrases)), gate=False)
        fused = predict(model, gemini_probs[None], muse_probs[None])[0]
        record = {"id": clip_id, "t": time.time(), "fps": fps, "prompted": prompted, "stt": heard, "label": label,
                  "vocab": vocab_id(phrases), "gemini": gemini_probs.tolist(), "muse": muse_probs.tolist()}
        with (store / "index.jsonl").open("a") as index:
            index.write(json.dumps(record) + "\n")
        summary = stats(records + [record], phrases)

    reads = {"fused": fused, "gemini": gemini_probs, "muse": muse_probs}
    return {
        "clip": clip_id,
        "stt": heard,
        "label": label,
        "label_phrase": None if label is None else phrases[label],
        "read": {k: {"phrase": phrases[int(p.argmax())], "p": float(p.max())} for k, p in reads.items()},
        "correct": None if label is None else {k: int(p.argmax()) == label for k, p in reads.items()},
        "stats": summary,
    }
