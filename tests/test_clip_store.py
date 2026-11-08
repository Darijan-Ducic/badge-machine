"""Storage contract both clip-store backends must honor: what is kept, in what order, where it lands, what is deleted."""

import json
import subprocess
import threading
import uuid
from pathlib import Path

import gridfs
import pymongo
import pytest

from clips import MongoStore, backend, load, process_clip, record_clip, vocab_id
from tests.fakes import PHRASES, Gemini, readers

RECORD_KEYS = {"record", "id", "t", "user_id", "mode", "fps", "prompted", "phrases", "vocab", "config",
               "tracker_flags", "frames", "crop_rect", "label", "samples", "closed", "gemini", "gemini_open"}


def record(recording, said, store=None, user="local", **kw):
    return process_clip(recording, 30.0, 0, transcribe=lambda _: said, phrases=PHRASES, store=store, user_id=user,
                        **{**readers("yes"), **kw})


def saved_clip(store, clip_id) -> bytes:
    """Read a clip back from where each backend's layout puts it: <dir>/<id>.mp4, or GridFS bucket `clips` by id."""
    if isinstance(store, Path):
        return (store / f"{clip_id}.mp4").read_bytes()
    return gridfs.GridFSBucket(store.db, bucket_name="clips").open_download_stream(clip_id).read()


def streams(data: bytes, tmp_path: Path) -> list[str]:
    path = tmp_path / "probe.mp4"
    path.write_bytes(data)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "json",
                          path], capture_output=True, text=True, check=True).stdout
    return [f"{s['codec_type']}:{s['codec_name']}" for s in json.loads(out)["streams"]]


def test_a_stored_clip_loads_back_as_exactly_its_record(store, recording):
    result = record(recording, "no", store=store, user="ann")

    (stored,) = load(store)
    assert set(stored) == RECORD_KEYS
    assert (stored["record"], stored["id"], stored["user_id"], stored["label"]["stt_text"]) == (
        "v1", result["clip"], "ann", "no")
    assert stored["vocab"] == vocab_id(PHRASES)
    assert backend(store).get(result["clip"]) == stored


def test_concurrent_clips_load_in_the_order_they_were_stored(store, recording):
    # Prequential accuracy replays clips in load order, so it must be the order clips were stored,
    # not the order their recordings arrived.
    scoring, release = threading.Event(), threading.Event()

    def slow_gemini(clip, ps, usage=None):
        scoring.set()
        release.wait(5)
        return Gemini("yes")(clip, ps, usage)

    first = threading.Thread(target=lambda: record(recording, "arrived first", store=store, gemini=slow_gemini))
    first.start()
    assert scoring.wait(5)
    record(recording, "arrived second", store=store)
    release.set()
    first.join(5)

    assert [r["label"]["stt_text"] for r in load(store)] == ["arrived second", "arrived first"]


def test_the_clip_the_models_read_is_kept_under_the_clip_id(store, recording, tmp_path):
    fakes = readers("yes")

    result = record(recording, "yes", store=store, **fakes)

    kept = saved_clip(store, result["clip"])
    assert backend(store).video(result["clip"]) == kept
    assert [call[0] for call in fakes["sample_open"].calls + fakes["score_closed"].calls
            + fakes["gemini"].calls] == [kept] * 3, "every lip reader gets a local file holding the stored clip"
    assert streams(kept, tmp_path) == ["video:h264"], "the stored clip is the muted H.264 crop"


def test_stt_hears_the_original_recording(store, recording):
    heard = []

    process_clip(recording, 30.0, 0, transcribe=lambda data: heard.append(data) or "yes", phrases=PHRASES,
                 store=store, **readers("yes"))

    assert heard == [recording]


def test_a_given_transcript_is_the_label_and_no_stt_runs(store, recording):
    record_clip(recording, phrases=PHRASES, user_id="practice:grid:s1", mode="practice", k=2,
                transcript="bin blue at f two now", transcribe=lambda _: pytest.fail("STT must not run"),
                extra={"key": "abc123", "seed": 7, "order": 0}, store=store, **readers("yes"))

    (stored,) = load(store)
    assert stored["label"] == {"stt_text": "bin blue at f two now", "user_edit": None}
    assert (stored["key"], stored["seed"], stored["order"], stored["mode"]) == ("abc123", 7, 0, "practice")
    assert len(stored["samples"]) == 2


def test_delete_user_removes_only_that_users_records_and_clips(store, recording):
    gone = [record(recording, "yes", store=store, user="practice:smoke:obama")["clip"] for _ in range(2)]
    kept = record(recording, "no", store=store, user="ann")["clip"]

    removed = backend(store).delete_user("practice:smoke:obama")

    assert removed == 2
    assert [r["id"] for r in load(store)] == [kept]
    assert [backend(store).video(i) for i in gone] == [None, None]
    assert backend(store).video(kept) == saved_clip(store, kept)


def test_delete_user_keeps_legacy_records_without_a_user(store, recording):
    legacy = {"id": "1", "t": 0.0, "fps": 30.0, "prompted": 0, "stt": "yes", "label": 0, "vocab": vocab_id(PHRASES),
              "gemini": [0.25] * 4, "muse": [0.25] * 4}
    if isinstance(store, Path):
        (store / "index.jsonl").write_text(json.dumps(legacy) + "\n")
    else:
        store.records.insert_one(dict(legacy))
    record(recording, "yes", store=store, user="ann")

    backend(store).delete_user("ann")

    assert load(store) == [legacy]
    with pytest.raises(ValueError):
        backend(store).delete_user("")


def test_without_mongodb_uri_clips_go_to_the_local_data_dir(monkeypatch, tmp_path, recording):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.chdir(tmp_path)

    result = record(recording, "yes")

    assert [r["id"] for r in load()] == [result["clip"]]
    assert [r["id"] for r in load(tmp_path / "data" / "clips")] == [result["clip"]]


@pytest.mark.parametrize("db_env, db_name", [("team_db", "team_db"), (None, "badge_machine")])
def test_with_mongodb_uri_clips_go_to_that_mongo_database(monkeypatch, tmp_path, mongo_client, db_env, db_name,
                                                          recording):
    clients = []
    monkeypatch.setattr(pymongo, "MongoClient", lambda *a, **kw: clients.append(mongo_client(*a, **kw)) or clients[-1])
    monkeypatch.setenv("MONGODB_URI", f"mongodb://{uuid.uuid4().hex}.test")
    if db_env is None:
        monkeypatch.delenv("MONGODB_DB", raising=False)
    else:
        monkeypatch.setenv("MONGODB_DB", db_env)
    monkeypatch.chdir(tmp_path)

    result = record(recording, "yes")

    assert [r["id"] for r in load()] == [result["clip"]]
    assert [c.list_database_names() for c in clients] == [[db_name]]
    assert not (tmp_path / "data").exists()
