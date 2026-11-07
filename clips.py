"""Clip store for Training v1: each clip keeps its raw model outputs; features, models and stats are recomputed.

Per clip (docs/uncertainty-design.md, docs/training-v1.md): crop the recording (crop.py), then in parallel STT (unless
a transcript is given), K Muse open samples, the Muse closed call and Gemini on the crop. The record stores only raw
outputs + the full config + the label; `candidates.featurize` rebuilds pools and features from them at read time,
with each clip's bank counting only texts said BEFORE it (else `prior` would leak the label). A user's replay is
their v1 clips under one phrase list, in store order; every clip is scored by a model trained on earlier clips only.
"""

import functools
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pymongo
from gridfs import GridFSBucket
from gridfs.errors import NoFile

import crop
import muse_sampler
import scorers
import visemes
from candidates import FEATURES, NONE, featurize, match
from readiness import readiness
from scorer import STAGES, auto_send, predict, prequential, stage_for, train

DATA = Path("data/clips")
COSTS = Path(__file__).parent / "data" / "viseme_costs.json"
DEFAULT_USER = "local"  # who preheat clips belong to when the page sends no ?user=
PREHEAT_K = 5
RUNTIME_K = 3  # lip mode answers while the user waits; K=5 is for dictation and practice
LABEL_SOURCES = ("undo", "confirm", "edit")  # how the user gave a lip-mode clip its label
NONE_TEXT = "∅ none of these"
V1 = "v1"  # record format marker; older records (fixed-vocabulary scores) are kept but not replayed
_lock = threading.Lock()  # clips finish concurrently; store order is training order, so saves must not interleave


def vocab_id(phrases: list[str]) -> str:
    return hashlib.sha1("\n".join(phrases).encode()).hexdigest()[:8]


def costs_hash(path: Path = COSTS) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- stores


class FileStore:
    """Records as JSON lines in <root>/index.jsonl, clip bytes as <root>/<id>.mp4."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, user_id: str | None = None) -> list[dict]:
        index = self.root / "index.jsonl"
        records = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
        return records if user_id is None else [r for r in records if r.get("user_id") == user_id]

    def save(self, record: dict, video: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{record['id']}.mp4").write_bytes(video)
        with (self.root / "index.jsonl").open("a") as index:
            index.write(json.dumps(record) + "\n")

    def get(self, clip_id: str) -> dict | None:
        return next((r for r in self.load() if r.get("id") == clip_id), None)

    def video(self, clip_id: str) -> bytes | None:
        if self.get(clip_id) is None:  # only ids we stored, never a path built from arbitrary input
            return None
        path = self.root / f"{clip_id}.mp4"
        return path.read_bytes() if path.exists() else None

    def set_label(self, clip_id: str, text: str, source: str) -> bool:
        with _lock:
            records = self.load()
            hit = next((r for r in records if r.get("record") == V1 and r.get("id") == clip_id), None)
            if hit is None:
                return False
            hit["label"] = {**hit["label"], "user_edit": text, "source": source}
            (self.root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        return True

    def users(self) -> list[dict]:
        counts = Counter(r["user_id"] for r in self.load() if r.get("record") == V1)
        return [{"id": user, "clips": n} for user, n in sorted(counts.items())]

    def delete_user(self, user_id: str) -> int:
        _require_user(user_id)
        with _lock:
            records = self.load()
            gone = {r["id"] for r in records if r.get("record") == V1 and r.get("user_id") == user_id}
            kept = "".join(json.dumps(r) + "\n" for r in records if r.get("id") not in gone)
            if records:
                (self.root / "index.jsonl").write_text(kept)
            for clip_id in gone:
                (self.root / f"{clip_id}.mp4").unlink(missing_ok=True)
        return len(gone)


class MongoStore:
    """Records in the `clip_records` collection, clip bytes in GridFS bucket `clips` with the clip id as file id."""

    def __init__(self, db):
        self.db = db
        self.records = db["clip_records"]
        self.clips = GridFSBucket(db, bucket_name="clips")

    def load(self, user_id: str | None = None) -> list[dict]:
        # _id is an ObjectId minted per insert, increasing within a process, so it replays insertion order.
        query = {} if user_id is None else {"user_id": user_id}
        return list(self.records.find(query, {"_id": False}).sort("_id", pymongo.ASCENDING))

    def save(self, record: dict, video: bytes) -> None:
        self.clips.upload_from_stream_with_id(record["id"], f"{record['id']}.mp4", video)
        self.records.insert_one(dict(record))  # insert_one writes _id into the dict it gets; keep the caller's clean

    def get(self, clip_id: str) -> dict | None:
        return self.records.find_one({"id": clip_id}, {"_id": False})

    def video(self, clip_id: str) -> bytes | None:
        try:
            return self.clips.open_download_stream(clip_id).read()
        except NoFile:
            return None

    def set_label(self, clip_id: str, text: str, source: str) -> bool:
        update = {"$set": {"label.user_edit": text, "label.source": source}}
        return self.records.update_one({"record": V1, "id": clip_id}, update).matched_count == 1

    def users(self) -> list[dict]:
        groups = self.records.aggregate([{"$match": {"record": V1}},
                                         {"$group": {"_id": "$user_id", "clips": {"$sum": 1}}}])
        return sorted(({"id": g["_id"], "clips": g["clips"]} for g in groups), key=lambda u: u["id"])

    def delete_user(self, user_id: str) -> int:
        _require_user(user_id)
        query = {"record": V1, "user_id": user_id}  # never {"user_id": None}: that matches every legacy record
        ids = [r["id"] for r in self.records.find(query, {"id": True})]
        for clip_id in ids:
            try:
                self.clips.delete(clip_id)
            except NoFile:
                pass
        self.records.delete_many(query)
        return len(ids)


def _require_user(user_id) -> None:
    if not isinstance(user_id, str) or not user_id:
        raise ValueError(f"delete_user needs a user id, got {user_id!r}")


@functools.cache
def _mongo_client(uri: str):
    return pymongo.MongoClient(uri)  # one client, and so one connection pool, per process


def backend(store=None):
    if store is None:  # default: MongoDB when MONGODB_URI is set, else the local data dir
        uri = os.environ.get("MONGODB_URI")
        store = MongoStore(_mongo_client(uri)[os.environ.get("MONGODB_DB") or "badge_machine"]) if uri else DATA
    return FileStore(store) if isinstance(store, Path) else store


def load(store=None, user_id: str | None = None) -> list[dict]:
    return backend(store).load(user_id)


# ---------------------------------------------------------------- recording a clip


def _gemini_call(gemini, clip: Path, phrases: list[str]) -> dict:
    """Gemini as a stored call: `output` {phrase: p}, or `error` instead of raising."""
    usage, start = {}, time.perf_counter()
    try:
        output, error = {p: float(x) for p, x in zip(phrases, gemini(clip, phrases, usage=usage))}, None
    except Exception as e:  # one failed call must not sink the clip
        output, error = None, f"{type(e).__name__}: {e}"
    item = {"call": "gemini", "latency_ms": round((time.perf_counter() - start) * 1000), "output": output,
            "error": error}
    return {**item, "usage": usage} if usage else item


def read_clip(recording: bytes, *, phrases: list[str], user_id: str, mode: str, k: int, transcribe=None,
              transcript: str | None = None, faces: list | None = None, mouths: list | None = None,
              tracker_flags=(), fps: float | None = None, prompted: int | None = None, extra: dict | None = None,
              sample_open=None, score_closed=None, gemini=None, crop_width: int | None = None) -> tuple[dict, bytes]:
    """Crop one recording and run every call on it in parallel; returns (record, crop bytes), not stored yet.

    Crop boxes: browser-tracker `mouths` [[t_ms, x, y, w, h]] or SAM `faces` [[x, y, w, h]] (see crop.crop). The
    label is `transcript` when given, else STT of the original recording (with audio) when `transcribe` is given,
    else none. Readers default to the real calls, looked up at call time. `crop_width` overrides the crop's output
    width (crop.WIDTH) for experiments."""
    sample_open = sample_open or muse_sampler.sample_open
    score_closed = score_closed or muse_sampler.score_closed
    gemini = gemini or scorers.gemini_scores
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        source, clip = Path(tmp, "recording"), Path(tmp, "clip.mp4")
        source.write_bytes(recording)
        crop_id = crop.crop(source, clip, faces=faces, mouths=mouths, width=crop_width or crop.WIDTH)
        frames = crop.count_frames(clip)
        with ThreadPoolExecutor(4) as pool:
            heard = pool.submit(transcribe, recording) if transcript is None and transcribe else None
            samples = pool.submit(sample_open, clip, k)
            closed = pool.submit(score_closed, clip, phrases)
            gemini_item = pool.submit(_gemini_call, gemini, clip, phrases)
            stt = heard.result() if heard else transcript
            samples, closed, gemini_item = samples.result(), closed.result(), gemini_item.result()
        video = clip.read_bytes()
    record = {
        "record": V1, "id": str(time.time_ns()), "t": started, "user_id": user_id, "mode": mode, "fps": fps,
        "prompted": prompted, "phrases": list(phrases), "vocab": vocab_id(phrases),
        "config": {**muse_sampler.CONFIG, "crop": crop_id, "k": k, "gemini_model": scorers.GEMINI_MODEL,
                   "viseme_costs": costs_hash()},
        "tracker_flags": list(tracker_flags), "frames": frames, "label": {"stt_text": stt, "user_edit": None},
        "samples": samples, "closed": closed, "gemini": gemini_item, **(extra or {}),
    }
    return record, video


def save(record: dict, video: bytes, store=None) -> None:
    with _lock:
        backend(store).save(record, video)


def record_clip(recording: bytes, *, store=None, **options) -> dict:
    """read_clip, then store it; returns the record."""
    record, video = read_clip(recording, **options)
    save(record, video, store)
    return record


# ---------------------------------------------------------------- replay: features recomputed from raw outputs


def _label_text(record: dict) -> str | None:
    label = record["label"]
    return label["user_edit"] if label.get("user_edit") is not None else label.get("stt_text")


def _replayed(records: list[dict], user: str | None, vocab: str) -> list[dict]:
    return [r for r in records if r.get("record") == V1 and r["user_id"] == user and r["vocab"] == vocab]


def _featurized(records: list[dict]) -> list:
    """Each record as a candidates.Clip; its bank counts only the labels of the records before it."""
    distance = functools.partial(visemes.viseme_distance, costs=visemes.load_costs(COSTS))
    counts, clips = Counter(), []
    for r in records:
        text = _label_text(r)
        bank = {"phrases": r["phrases"], "counts": dict(counts)}
        clips.append(featurize(r["samples"], r["closed"]["output"], r["gemini"]["output"], bank, text,
                               viseme_distance=distance))
        if text is not None:
            counts[text] += 1
    return clips


def _trained_before(clips: list, i: int):
    return train([c for c in clips[:i] if c.label is not None])


def _shown(clip, j: int) -> str:
    return NONE_TEXT if j == len(clip.pool) - 1 else clip.pool[j]


def _action(model, probs: np.ndarray) -> str:
    if int(np.argmax(probs)) == len(probs) - 1:
        return "none"
    return "auto" if auto_send(model, probs) else "confirm"


def _threshold(model) -> float | None:
    return None if np.isinf(model.threshold) else float(model.threshold)


def _latest_user(records: list[dict]) -> str | None:
    return next((r["user_id"] for r in reversed(records) if r.get("record") == V1), None)


# ---------------------------------------------------------------- stats, list, detail


def stats(records: list[dict], phrases: list[str], window: int = 20, user: str | None = None) -> dict:
    """Training stats of one user's clips under `phrases` (default user: the one of the latest clip)."""
    user = user or _latest_user(records)
    records = _replayed(records, user, vocab_id(phrases))
    labeled = [c for c in _featurized(records) if c.label is not None]
    history = [{"i": i, **h} for i, h in enumerate(prequential(labeled))]

    def accuracy(entries):
        return {k: float(np.mean([e[k] for e in entries])) if entries else None for k in ("fused", "gemini", "muse")}

    upcoming = sorted((n, name) for n, name in STAGES if n > len(labeled))
    return {
        "labeled": len(labeled),
        "clips": len(records),
        "stage": stage_for(len(labeled)),
        "next_stage": {"name": upcoming[0][1], "at": upcoming[0][0]} if upcoming else None,
        "threshold": _threshold(train(labeled)),
        "camera_fps": records[-1]["fps"] if records else None,
        "window": window,
        "accuracy": {"recent": accuracy(history[-window:]), "all": accuracy(history)},
        "history": history,
        "user": user,
        "pool_recall": float(np.mean([h["pool_hit"] for h in history])) if history else None,
        "stages": [{"name": name, "at": n} for n, name in sorted(STAGES)],
    }


def phrase_readiness(store, phrases: list[str], user: str | None = None) -> dict:
    """GET /readiness: per phrase one user labeled (by STT or their own label), its clip count, recent VTT accuracy
    and whether lip mode is ready for it (readiness.readiness), from one predict-then-train pass over that user's
    clips in store order (default user: the one of the latest clip)."""
    records = load(store, user) if user else load(store)
    user = user or _latest_user(records)
    clips = _featurized(_replayed(records, user, vocab_id(phrases)))
    return readiness(prequential(clips), [c.stt for c in clips])


def clip_list(store, phrases: list[str], user: str | None = None) -> dict:
    """GET /clips: every user with their clip count, and one user's clips (default: the latest clip's), newest first;
    each scored by a model trained on that user's earlier clips."""
    store = backend(store)
    users = store.users()
    records = store.load(user) if user else store.load()
    user = user or _latest_user(records)
    records = _replayed(records, user, vocab_id(phrases))
    clips = _featurized(records)
    labeled = [i for i, c in enumerate(clips) if c.label is not None]
    history = dict(zip(labeled, prequential([clips[i] for i in labeled])))
    rows, model, trained_on = [], None, None
    for i, (record, clip) in enumerate(zip(records, clips)):
        n_before = sum(c.label is not None for c in clips[:i])
        if n_before != trained_on:  # unlabeled clips add nothing to train on: reuse the model
            model, trained_on = _trained_before(clips, i), n_before
        probs = predict(model, [clip])[0]
        top, h = int(np.argmax(probs)), history.get(i)
        rows.append({
            "id": record["id"], "t": record["t"], "user_id": record["user_id"], "mode": record["mode"],
            "stt": record["label"]["stt_text"],
            "top": {"text": _shown(clip, top), "p": float(probs[top])},
            "action": _action(model, probs),
            "correct": None if h is None else {k: bool(h[k]) for k in ("fused", "gemini", "muse")},
            "pool_hit": None if h is None else bool(h["pool_hit"]),
            "stage": model.stage,
        })
    return {"users": users, "clips": rows[::-1]}


def clip_detail(store, clip_id: str) -> dict | None:
    """GET /clips/<id>: the clip's raw outputs and its pool, scored by a model trained on the user's earlier clips."""
    store = backend(store)
    record = store.get(clip_id)
    if record is None or record.get("record") != V1:
        return None
    records = _replayed(store.load(record["user_id"]), record["user_id"], record["vocab"])
    i = [r["id"] for r in records].index(clip_id)
    clips = _featurized(records[:i + 1])  # later clips can't reach this clip's bank or model
    clip, model = clips[i], _trained_before(clips, i)
    probs = predict(model, [clip])[0]
    pool = [{"text": _shown(clip, j), "p": float(probs[j]), "is_label": j == clip.label,
             "features": dict(zip(FEATURES, map(float, clip.features[j])))} for j in range(len(clip.pool))]
    gemini, muse = record["gemini"]["output"] or {}, _muse_read(clip, record["samples"])
    return {
        "id": record["id"], "t": record["t"], "user_id": record["user_id"], "mode": record["mode"],
        "stt": record["label"]["stt_text"], "config": record["config"], "video": f"/clips/{clip_id}/video",
        "frames": record["frames"], "stage": model.stage, "action": _action(model, probs),
        "threshold": _threshold(model),
        "pool": sorted(pool, key=lambda c: -c["p"]),
        "samples": [{key: s[key] for key in ("sample_idx", "latency_ms", "error", "output")}
                    for s in record["samples"]],
        "closed": {key: record["closed"][key] for key in ("latency_ms", "error", "output")},
        "gemini": {"top": [{"phrase": p, "p": v} for p, v in sorted(gemini.items(), key=lambda kv: -kv[1])[:5]]},
        "vtt_muse": {"text": muse["phrase"], "p": muse["p"]},
    }


# ---------------------------------------------------------------- preheat


def _muse_read(clip, samples: list[dict]) -> dict:
    """The most common top hypothesis across the valid samples, with its vote share among them."""
    if clip.muse_pick is None:
        return {"phrase": None, "p": 0.0}
    outputs = [s["output"] for s in samples if s.get("output") is not None]
    tops = [max(o["hypotheses"], key=lambda h: h["s"])["text"] for o in outputs if o.get("hypotheses")]
    return {"phrase": clip.muse_pick, "p": sum(match(t, [clip.muse_pick]) is not None for t in tops) / len(outputs)}


def _read_back(record: dict, store) -> tuple[list[dict], dict, dict | None]:
    """Read a just-stored clip back like the runtime would, with a model trained on this user's earlier clips only.

    Returns (the user's records, each reader's pick {"fused" | "gemini" | "muse": {"phrase", "p"}}, whether each pick
    matches the STT label, or None without one)."""
    records = load(store, record["user_id"])
    replay = _replayed(records, record["user_id"], record["vocab"])
    i = [r["id"] for r in replay].index(record["id"])
    clips = _featurized(replay[:i + 1])
    clip = clips[i]
    probs = predict(train([c for c in clips[:i] if c.label is not None], gate=False), [clip])[0]
    top, heard = int(np.argmax(probs)), record["label"]["stt_text"]
    gemini_output = record["gemini"]["output"] or {}

    def said(text):
        return heard is not None and text is not None and match(heard, [text]) is not None

    read = {
        "fused": {"phrase": _shown(clip, top), "p": float(probs[top])},
        "gemini": {"phrase": clip.gemini_pick, "p": float(gemini_output.get(clip.gemini_pick, 0.0))},
        "muse": _muse_read(clip, record["samples"]),
    }
    correct = None if heard is None else {
        "fused": top != len(clip.pool) - 1 and said(clip.pool[top]),
        "gemini": said(clip.gemini_pick),
        "muse": said(clip.muse_pick),
    }
    return records, read, correct


def process_clip(recording: bytes, fps, prompted, transcribe, phrases, store=None, *, user_id: str = DEFAULT_USER,
                 k: int = PREHEAT_K, mouths: list | None = None, tracker_flags=(), sample_open=None,
                 score_closed=None, gemini=None) -> dict:
    """Preheat: STT-label one recording, crop it with the browser tracker's boxes (`mouths`; full frame without),
    read it with every model, store it, and read it back like the runtime would (a model trained on this user's
    earlier clips only)."""
    record = record_clip(recording, phrases=phrases, user_id=user_id, mode="stt_on", k=k, transcribe=transcribe,
                         mouths=mouths, tracker_flags=tracker_flags, fps=fps, prompted=prompted, store=store,
                         sample_open=sample_open, score_closed=score_closed, gemini=gemini)
    records, read, correct = _read_back(record, store)
    heard = record["label"]["stt_text"]
    label = match(heard, phrases) if heard is not None else None
    return {
        "clip": record["id"],
        "stt": heard,
        "label": label,
        "label_phrase": None if label is None else phrases[label],
        "read": read,
        "correct": correct,
        "stats": stats(records, phrases, user=user_id),
    }


def process_turn(segment: Path, *, transcript: str | None, boxes: list | None, user_id: str, session_id: str,
                 turn: int, mode: str, phrases: list[str], store=None, sample_open=None, score_closed=None,
                 gemini=None) -> dict:
    """Dictation: one session turn (sessions.start_session's `process_turn`) as a clip labeled by the turn's own STT,
    cropped with the turn's boxes and read back like process_clip; returns the fields sessions.TURN_RESULT keeps.
    A ∅ VTT pick is "∅" (candidates.NONE), which static/vtt.js shows as none of these."""
    record = record_clip(Path(segment).read_bytes(), phrases=phrases, user_id=user_id, mode=mode, k=PREHEAT_K,
                         transcript=transcript, mouths=boxes, extra={"session_id": session_id, "turn": turn},
                         store=store, sample_open=sample_open, score_closed=score_closed, gemini=gemini)
    _, read, correct = _read_back(record, store)
    vtt, gemini_read, muse = read["fused"], read["gemini"], read["muse"]
    return {
        "clip_id": record["id"],
        "vtt": {"text": NONE if vtt["phrase"] == NONE_TEXT else vtt["phrase"], "p": vtt["p"]},
        "vtt_gemini": {"text": gemini_read["phrase"], "p": gemini_read["p"]},
        "vtt_muse": {"text": muse["phrase"], "p": muse["p"]},
        "correct": None if correct is None else {"vtt": correct["fused"], "gemini": correct["gemini"],
                                                 "muse": correct["muse"]},
    }


# ---------------------------------------------------------------- lip mode


def vtt_clip(recording: bytes, phrases: list[str], store=None, *, user_id: str = DEFAULT_USER,
             mouths: list | None = None, tracker_flags=(), sample_open=None, score_closed=None, gemini=None) -> dict:
    """Lip mode: crop one muted recording with the browser tracker's boxes, read it with every model (K=3), store it
    unlabeled (the user labels it later, `label_clip`) and score it with a model trained on this user's earlier
    labeled clips only. `vtt` and `alternatives` are the most probable real candidates: ∅ is never offered, and a ∅
    top pick makes the action "none"."""
    start = time.perf_counter()
    record = record_clip(recording, phrases=phrases, user_id=user_id, mode="lip", k=RUNTIME_K, mouths=mouths,
                         tracker_flags=tracker_flags, store=store, sample_open=sample_open,
                         score_closed=score_closed, gemini=gemini)
    replay = _replayed(load(store, user_id), user_id, record["vocab"])
    i = [r["id"] for r in replay].index(record["id"])
    clips = _featurized(replay[:i + 1])  # clips stored after this one can't reach its bank or model
    clip, model = clips[i], _trained_before(clips, i)
    probs = predict(model, [clip])[0]
    best = sorted(range(len(clip.pool) - 1), key=lambda j: -probs[j])[:3]  # ∅ is last
    alternatives = [{"text": clip.pool[j], "p": float(probs[j])} for j in best]
    return {"clip": record["id"], "vtt": alternatives[0], "action": _action(model, probs),
            "alternatives": alternatives, "latency_ms": round((time.perf_counter() - start) * 1000)}


def label_clip(store, clip_id: str, text, source) -> bool:
    """The user's label for a stored clip: `label.user_edit`, which training prefers over STT, and how they gave it.
    False if there is no such clip; ValueError for a missing text or a source outside LABEL_SOURCES."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing label text")
    if source not in LABEL_SOURCES:
        raise ValueError(f"label source must be one of {', '.join(LABEL_SOURCES)}, got {source!r}")
    return backend(store).set_label(clip_id, text, source)
