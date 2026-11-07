"""Dictation sessions for natural preheat: a recording is cut at its pauses into turns of at most 8 s, each turn is
labeled by STT on its own audio, and kept turns go to the clip pipeline in the background.

Store: MongoDB (collection `sessions`, recordings in GridFS bucket `sessions`) when MONGODB_URI is set, else
data/sessions. Session status: recorded -> transcribed -> processing -> done | failed.
Turn status: pending -> processed | skipped | failed, with a reason when not processed.
"""

import functools
import io
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pymongo
from gridfs import GridFSBucket
from gridfs.errors import NoFile

from asr import transcribe_turns

# System knobs — the same for every user.
NOISE_DB = -30  # silencedetect: audio under this level...
MIN_SILENCE_S = 0.25  # ...for at least this long is a pause between speech spans
LEVEL_HOP_MS = 50  # loudness resolution for finding a dip inside a long span
DIP_DB = 6  # a dip is at least this far under the span's median loudness
MIN_PIECE_MS = 2000  # a dip cut leaves both pieces at least this long
MIN_TURN_MS = 500  # shorter turns are skipped: "too short"
MAX_TURN_MS = 8000  # turns are merged up to / cut down to this; longer ones are skipped: "over 8 s"
DAILY_CAP = 50  # processed turns per user per local day; the rest are skipped: "daily cap"

DATA = Path("data/sessions")
SAMPLE_RATE = 16000
# Turn segments are re-encoded (stream copy can only cut at keyframes) in the recording's own container family.
CONTAINERS = {
    "webm": ("video/webm", ["-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8", "-crf", "10", "-b:v", "4M",
                            "-c:a", "libopus"]),
    "mp4": ("video/mp4", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac"]),
}
TURN_RESULT = ("clip_id", "vtt", "vtt_gemini", "vtt_muse", "correct")
_lock = threading.RLock()  # file store: turns finish concurrently; read-modify-append must not interleave
_cap_lock = threading.Lock()  # concurrent sessions of one user must not both spend the same daily-cap slots


# ---- segmentation ---------------------------------------------------------------------------------------------------

def parse_silences(log: str, duration_ms: int) -> list[tuple[int, int]]:
    """silencedetect's log -> [(start_ms, end_ms)]; a silence still open at the end runs to duration_ms."""
    silences, start = [], None
    for kind, seconds in re.findall(r"silence_(start|end): (-?[\d.]+)", log):
        ms = max(0, round(float(seconds) * 1000))
        if kind == "start":
            start = ms
        elif start is not None:
            silences.append((start, ms))
            start = None
    if start is not None:
        silences.append((start, duration_ms))
    return silences


def _speech(duration_ms: int, silences) -> list[tuple[int, int]]:
    spans, at = [], 0
    for start, end in sorted(silences):
        if start > at:
            spans.append((at, start))
        at = max(at, end)
    if at < duration_ms:
        spans.append((at, duration_ms))
    return spans


def _split(start: int, end: int, levels, hop_ms: int) -> list[tuple[int, int]]:
    """A span over MAX_TURN_MS is cut at its deepest dip leaving both pieces >= MIN_PIECE_MS, else hard at MAX_TURN_MS."""
    if end - start <= MAX_TURN_MS:
        return [(start, end)]
    middle = lambda k: k * hop_ms + hop_ms // 2  # a level covers [k*hop, (k+1)*hop); a cut there falls at its middle
    inside = [k for k in range(len(levels)) if start <= middle(k) < end]
    allowed = [k for k in inside if middle(k) - start >= MIN_PIECE_MS and end - middle(k) >= MIN_PIECE_MS]
    cut = start + MAX_TURN_MS
    if allowed:
        deepest = min(allowed, key=lambda k: levels[k])  # the earliest, on a tie
        if levels[deepest] <= float(np.median([levels[k] for k in inside])) - DIP_DB:
            cut = middle(deepest)
    return _split(start, cut, levels, hop_ms) + _split(cut, end, levels, hop_ms)


def plan_segments(duration_ms: int, silences, levels, hop_ms: int = LEVEL_HOP_MS) -> list[tuple[int, int]]:
    """Speech spans between silences, merged greedily while a turn stays within MAX_TURN_MS, long ones split."""
    merged = []
    for start, end in _speech(duration_ms, silences):
        if merged and end - merged[-1][0] <= MAX_TURN_MS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [piece for start, end in merged for piece in _split(start, end, levels, hop_ms)]


# ---- turn policy, boxes ---------------------------------------------------------------------------------------------

def _length_reason(start_ms: int, end_ms: int) -> str | None:
    length = end_ms - start_ms
    return "too short" if length < MIN_TURN_MS else "over 8 s" if length > MAX_TURN_MS else None


def policy(turns, *, used_today: int, cap: int | None = None) -> list[tuple[str, str | None]]:
    """[{"start_ms", "end_ms", "text"}] -> [("pending", None) | ("skipped", reason)]; only kept turns spend the cap."""
    cap = DAILY_CAP if cap is None else cap
    decisions = []
    for turn in turns:
        reason = _length_reason(turn["start_ms"], turn["end_ms"]) or (None if (turn["text"] or "").strip() else "no speech")
        if reason is None and used_today >= cap:
            reason = "daily cap"
        if reason is None:
            used_today += 1
        decisions.append(("pending", None) if reason is None else ("skipped", reason))
    return decisions


def slice_boxes(boxes, start_ms: int, end_ms: int) -> list[list]:
    """Per-frame [t_ms, x, y, w, h] boxes inside [start_ms, end_ms), with t re-based to start_ms."""
    return [[t - start_ms, *rest] for t, *rest in boxes or [] if start_ms <= t < end_ms]


def used_today(sessions, user_id: str, now: float) -> int:
    """Turns of this user's sessions from now's local day that are pending or processed: the daily cap spent so far.
    Dataset sessions (practice / emulated footage) don't spend it."""
    day = time.localtime(now)[:3]
    return sum(turn["status"] in ("pending", "processed") for s in sessions
               if s["user_id"] == user_id and not s.get("dataset") and time.localtime(s["t"])[:3] == day
               for turn in s["turns"])


# ---- stores ---------------------------------------------------------------------------------------------------------

class _FileStore:
    """Records as JSON lines in <root>/index.jsonl, each line a record's full latest state (the last line per id wins);
    recordings as <root>/<id>.<webm|mp4>."""

    def __init__(self, root: Path):
        self.root = root

    def all(self) -> list[dict]:
        index, latest = self.root / "index.jsonl", {}
        with _lock:
            lines = index.read_text().splitlines() if index.exists() else []
        for line in lines:
            record = json.loads(line)
            latest[record["id"]] = record  # a dict keeps first-seen order, so records stay in insertion order
        return list(latest.values())

    def get(self, session_id: str) -> dict | None:
        return next((r for r in self.all() if r["id"] == session_id), None)

    def _append(self, record: dict) -> None:
        with _lock, (self.root / "index.jsonl").open("a") as index:
            index.write(json.dumps(record) + "\n")

    def insert(self, record: dict, recording: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{record['id']}.{record['container']}").write_bytes(recording)
        self._append(record)

    def update(self, session_id: str, fields: dict, turn: int | None = None) -> None:
        with _lock:
            record = self.get(session_id)
            (record if turn is None else record["turns"][turn]).update(fields)
            self._append(record)

    def recording(self, session_id: str) -> bytes | None:
        record = self.get(session_id)
        return None if record is None else (self.root / f"{session_id}.{record['container']}").read_bytes()


class MongoStore:
    """Records in the `sessions` collection, recordings in GridFS bucket `sessions` with the session id as file id."""

    def __init__(self, db):
        self.db = db
        self.records = db["sessions"]
        self.recordings = GridFSBucket(db, bucket_name="sessions")

    def all(self) -> list[dict]:
        return list(self.records.find({}, {"_id": False}))

    def get(self, session_id: str) -> dict | None:
        return self.records.find_one({"id": session_id}, {"_id": False})

    def insert(self, record: dict, recording: bytes) -> None:
        self.recordings.upload_from_stream_with_id(record["id"], f"{record['id']}.{record['container']}", recording)
        self.records.insert_one(dict(record))  # insert_one writes _id into the dict it gets; keep the caller's clean

    def update(self, session_id: str, fields: dict, turn: int | None = None) -> None:
        prefix = "" if turn is None else f"turns.{turn}."
        self.records.update_one({"id": session_id}, {"$set": {prefix + k: v for k, v in fields.items()}})

    def recording(self, session_id: str) -> bytes | None:
        try:
            return self.recordings.open_download_stream(session_id).read()
        except NoFile:
            return None


@functools.cache
def _mongo_client(uri: str):
    return pymongo.MongoClient(uri)  # one client, and so one connection pool, per process


def _backend(store):
    if store is None:  # default: MongoDB when MONGODB_URI is set, else the local data dir
        uri = os.environ.get("MONGODB_URI")
        store = MongoStore(_mongo_client(uri)[os.environ.get("MONGODB_DB") or "badge_machine"]) if uri else DATA
    return _FileStore(store) if isinstance(store, Path) else store


# ---- media ----------------------------------------------------------------------------------------------------------

def _container(recording: bytes) -> str:
    return "mp4" if recording[4:8] == b"ftyp" else "webm"  # browsers record WebM (Chrome, Firefox) or MP4 (Safari)


def _decode(source: Path, wav: Path) -> np.ndarray:
    """The recording's audio as 16 kHz mono int16 samples (also left in `wav`)."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", source, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le", wav], check=True)
    with wave.open(str(wav)) as audio:
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)


def _silences(wav: Path, duration_ms: int) -> list[tuple[int, int]]:
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af",
                          f"silencedetect=noise={NOISE_DB}dB:d={MIN_SILENCE_S}", "-f", "null", "-"],
                         check=True, capture_output=True, text=True).stderr
    return parse_silences(log, duration_ms)


def _levels(pcm: np.ndarray, hop_ms: int = LEVEL_HOP_MS) -> list[float]:
    hop = SAMPLE_RATE * hop_ms // 1000
    frames = pcm[: len(pcm) // hop * hop].astype(np.float64).reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return (20 * np.log10(np.maximum(rms, 1.0) / 32768)).tolist()  # dBFS


def _wav_bytes(pcm: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _cut(source: Path, start_ms: int, end_ms: int, container: str, out: Path) -> None:
    # -ss/-t after -i: decode from the start and keep exactly the frames in [start, end) — frame-accurate.
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", source, "-ss", f"{start_ms / 1000:.3f}",
                    "-t", f"{(end_ms - start_ms) / 1000:.3f}", "-fps_mode", "passthrough", *CONTAINERS[container][1],
                    out], check=True)


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


# ---- sessions -------------------------------------------------------------------------------------------------------

def start_session(recording: bytes, *, user_id: str, boxes, flags, mode: str = "stt_on", dataset: str | None = None,
                  source: str | None = None, start: float | None = None, process_turn, executor,
                  transcribe=transcribe_turns, store=None) -> dict:
    """Store the recording, cut it into turns, STT the whole session and each turn, and return the session's STT text;
    kept turns are submitted to `executor` for `process_turn`. `start` is the recording's offset in `source`, in s."""
    store = _backend(store)
    session_id, container = uuid.uuid4().hex[:12], _container(recording)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp, "speech.wav")
        Path(tmp, f"recording.{container}").write_bytes(recording)
        pcm = _decode(Path(tmp, f"recording.{container}"), wav)
        duration_ms = round(len(pcm) * 1000 / SAMPLE_RATE)
        store.insert({"id": session_id, "t": time.time(), "user_id": user_id, "dataset": dataset, "source": source,
                      "start": 0.0 if start is None else float(start), "duration_ms": duration_ms,
                      "container": container, "mode": mode, "flags": flags, "status": "recorded", "reason": None,
                      "stt_text": None, "turns": []}, recording)
        try:
            segments = plan_segments(duration_ms, _silences(wav, duration_ms), _levels(pcm))
            # The reply is one extra STT call on the whole session (full context, one round trip of latency);
            # each turn long enough to keep is labeled by STT on its own audio, all in parallel.
            per_ms = SAMPLE_RATE // 1000
            with ThreadPoolExecutor(8) as pool:
                whole = pool.submit(transcribe, wav.read_bytes())
                heard = {i: pool.submit(transcribe, _wav_bytes(pcm[a * per_ms:b * per_ms]))
                         for i, (a, b) in enumerate(segments) if _length_reason(a, b) is None}
                text = whole.result()["text"]
        except Exception as e:
            store.update(session_id, {"status": "failed", "reason": _reason(e)})
            raise

    turns = []
    for i, (a, b) in enumerate(segments):
        stt, error = None, None
        if i in heard:
            try:
                stt = heard[i].result()["text"]
            except Exception as e:
                error = _reason(e)
        turns.append({"i": i, "start_ms": a, "end_ms": b, "stt": stt, "status": "failed" if error else None,
                      "reason": error, **dict.fromkeys(TURN_RESULT)})
    with _cap_lock:
        labeled = [t for t in turns if t["status"] is None]
        # Practice / emulated footage (a dataset session) is not held to the daily cap of a real user.
        spent, cap = (0, math.inf) if dataset else (used_today(store.all(), user_id, time.time()), None)
        decisions = policy([{"start_ms": t["start_ms"], "end_ms": t["end_ms"], "text": t["stt"]} for t in labeled],
                           used_today=spent, cap=cap)
        for t, (status, reason) in zip(labeled, decisions):
            t["status"], t["reason"] = status, reason
        store.update(session_id, {"status": "transcribed", "stt_text": text, "turns": turns})

    pending = [t for t in turns if t["status"] == "pending"]
    if not pending:
        store.update(session_id, {"status": "done"})
    for turn in pending:
        executor.submit(_process_turn, store, session_id, recording, container, turn, boxes, user_id, mode,
                        process_turn)
    return {"session": session_id, "stt": text, "turns": len(turns)}


def _process_turn(store, session_id, recording, container, turn, boxes, user_id, mode, process_turn) -> None:
    store.update(session_id, {"status": "processing"})
    try:
        with tempfile.TemporaryDirectory() as tmp:
            source, segment = Path(tmp, f"recording.{container}"), Path(tmp, f"turn-{turn['i']}.{container}")
            source.write_bytes(recording)
            _cut(source, turn["start_ms"], turn["end_ms"], container, segment)
            result = process_turn(segment, transcript=turn["stt"],
                                  boxes=slice_boxes(boxes, turn["start_ms"], turn["end_ms"]), user_id=user_id,
                                  session_id=session_id, turn=turn["i"], mode=mode)
        fields = {"status": "processed", "reason": None, **{k: result.get(k) for k in TURN_RESULT}}
    except Exception as e:
        fields = {"status": "failed", "reason": _reason(e)}
    store.update(session_id, fields, turn=turn["i"])
    if all(t["status"] in ("processed", "skipped", "failed") for t in store.get(session_id)["turns"]):
        store.update(session_id, {"status": "done"})


# ---- the /sessions contract -----------------------------------------------------------------------------------------

def _summary(record: dict) -> dict:
    turns, duration = record["turns"], record["duration_ms"] / 1000
    return {"id": record["id"], "t": record["t"], "user_id": record["user_id"], "dataset": record["dataset"],
            "source": record["source"], "start": record["start"], "end": round(record["start"] + duration, 3),
            "duration_s": round(duration, 2), "status": record["status"], "turns": len(turns),
            "processed": sum(t["status"] == "processed" for t in turns),
            "skipped": sum(t["status"] == "skipped" for t in turns),
            "vtt_correct": sum(bool(t["correct"] and t["correct"].get("vtt")) for t in turns)}


def list_sessions(user: str | None = None, store=None) -> dict:
    """GET /sessions?user=: every user with a session count, and one user's sessions newest first (default: the user of
    the most recent session)."""
    records = _backend(store).all()
    if user is None and records:
        user = max(records, key=lambda r: r["t"])["user_id"]
    mine = sorted((r for r in records if r["user_id"] == user), key=lambda r: r["t"], reverse=True)
    return {"users": [{"id": u, "sessions": n} for u, n in sorted(Counter(r["user_id"] for r in records).items())],
            "sessions": [_summary(r) for r in mine]}


def get_session(session_id: str, store=None) -> dict | None:
    """GET /sessions/<id>, or None when there is no such session."""
    record = _backend(store).get(session_id)
    if record is None:
        return None
    summary = _summary(record)
    return {**{k: summary[k] for k in ("id", "t", "user_id", "dataset", "source", "start", "end", "status")},
            "video": f"/sessions/{session_id}/video", "stt_text": record["stt_text"],
            "turns": [{"i": t["i"], "start": t["start_ms"] / 1000, "end": t["end_ms"] / 1000, "stt": t["stt"],
                       "status": t["status"], "reason": t["reason"], **{k: t[k] for k in TURN_RESULT}}
                      for t in record["turns"]]}


def session_video(session_id: str, store=None) -> tuple[bytes, str] | None:
    """GET /sessions/<id>/video: (the recording as stored, its content type), or None when there is no such session."""
    store = _backend(store)
    record = store.get(session_id)
    data = None if record is None else store.recording(session_id)
    return None if data is None else (data, CONTAINERS[record["container"]][0])
