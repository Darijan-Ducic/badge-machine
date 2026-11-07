"""Storage contract both clip-store backends must honor: what is kept, in what order, and where it lands."""

import threading
import uuid
from pathlib import Path

import gridfs
import numpy as np
import pymongo
import pytest

from clips import load, process_clip, vocab_id

PHRASES = ["yes", "no", "turn on the lights", "turn off the lights"]
RECORD_KEYS = {"id", "t", "fps", "prompted", "stt", "label", "vocab", "gemini", "muse"}


def peaked(index, n=len(PHRASES)):
    probs = np.full(n, 0.1 / (n - 1))
    probs[index] = 0.9
    return probs


def record_clip(said, recording=b"webm-bytes", fps=30.0, prompted=0, gemini_pick=0, muse_pick=0, gemini=None, muse=None,
                **where):
    return process_clip(recording, fps, prompted, transcribe=lambda _: said, phrases=PHRASES,
                        gemini=gemini or (lambda clip, ps: peaked(gemini_pick, len(ps))),
                        muse=muse or (lambda clip, ps: peaked(muse_pick, len(ps))), **where)


def saved_clip(store, clip_id) -> bytes:
    """Read a clip back from where each backend's layout puts it: <dir>/<id>.webm, or GridFS bucket `clips` by id."""
    if isinstance(store, Path):
        return (store / f"{clip_id}.webm").read_bytes()
    return gridfs.GridFSBucket(store.db, bucket_name="clips").open_download_stream(clip_id).read()


def test_a_stored_clip_loads_back_as_exactly_its_record(store):
    result = record_clip("no", fps=24.0, prompted=3, gemini_pick=1, muse_pick=2, store=store)

    (record,) = load(store)
    assert set(record) == RECORD_KEYS
    assert (record["id"], record["fps"], record["prompted"], record["stt"], record["label"]) == (result["clip"], 24.0, 3, "no", 1)
    assert record["vocab"] == vocab_id(PHRASES)
    assert (record["gemini"], record["muse"]) == (peaked(1).tolist(), peaked(2).tolist())


def test_concurrent_clips_load_in_the_order_they_were_stored(store):
    # Prequential accuracy replays clips in load order, so it must be the order clips were predicted and stored,
    # not the order their recordings arrived.
    scoring, release = threading.Event(), threading.Event()

    def slow_gemini(clip, ps):
        scoring.set()
        release.wait(5)
        return peaked(0, len(ps))

    first = threading.Thread(target=lambda: record_clip("arrived first", gemini=slow_gemini, store=store))
    first.start()
    assert scoring.wait(5)
    record_clip("arrived second", store=store)
    release.set()
    first.join(5)

    assert [r["stt"] for r in load(store)] == ["arrived second", "arrived first"]


def test_the_recording_bytes_are_kept_under_the_clip_id(store):
    recording = b"\x1aE\xdf\xa3" + bytes(range(256)) * 1200  # webm magic + ~300 KB, more than one GridFS chunk

    result = record_clip("yes", recording=recording, store=store)

    assert saved_clip(store, result["clip"]) == recording


def test_both_lip_readers_get_a_local_file_holding_the_recording(store):
    # The scorers run ffmpeg on the clip, so they need a real path on disk with the recording's bytes in it.
    recording, seen = b"\x1aE\xdf\xa3 not really a webm", {}

    def reader(name):
        def score(clip, ps):
            seen[name] = Path(clip).read_bytes()
            return peaked(0, len(ps))
        return score

    record_clip("yes", recording=recording, gemini=reader("gemini"), muse=reader("muse"), store=store)

    assert seen == {"gemini": recording, "muse": recording}


def test_without_mongodb_uri_clips_go_to_the_local_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.chdir(tmp_path)

    result = record_clip("yes")

    assert [r["id"] for r in load()] == [result["clip"]]
    assert [r["id"] for r in load(tmp_path / "data" / "clips")] == [result["clip"]]


@pytest.mark.parametrize("db_env, db_name", [("team_db", "team_db"), (None, "badge_machine")])
def test_with_mongodb_uri_clips_go_to_that_mongo_database(monkeypatch, tmp_path, mongo_client, db_env, db_name):
    clients = []
    monkeypatch.setattr(pymongo, "MongoClient", lambda *a, **kw: clients.append(mongo_client(*a, **kw)) or clients[-1])
    monkeypatch.setenv("MONGODB_URI", f"mongodb://{uuid.uuid4().hex}.test")
    if db_env is None:
        monkeypatch.delenv("MONGODB_DB", raising=False)
    else:
        monkeypatch.setenv("MONGODB_DB", db_env)
    monkeypatch.chdir(tmp_path)

    result = record_clip("yes")

    assert [r["id"] for r in load()] == [result["clip"]]
    assert [c.list_database_names() for c in clients] == [[db_name]]
    assert not (tmp_path / "data").exists()
