"""Run-record store contract, on both backends: JSON files under a directory, and the MongoDB `runs` collection
(mongomock in-process; the hosted Atlas check is the integrator's live smoke test)."""

import json

import mongomock
import pytest

import runs


def run_record(run_id="r1", t=1.0, user_id="practice:sim", gate=None, precision=None):
    """A record in the `GET /runs` shape, with the nullable fields exercised."""
    return {
        "id": run_id, "t": t, "name": f"run {run_id}", "dataset": "sim:seed=4:n=140", "user_id": user_id,
        "config": {"drop": ["viseme_sim"], "pool_cap": 24, "stages": {"defaults": 0, "full": 30},
                   "costs": {"alphabet": "visemes-v1", "default_sub": 1.0, "ins": 1.0, "del": 1.0, "sub": {}},
                   "order": "abc", "hash": "0123456789abcdef"},
        "baseline": None if gate is None else "r0",
        "metrics": {"clips": 140, "vtt_acc": 0.75, "vtt_acc_heldout": 0.9285714285714286, "gemini_acc": 0.2,
                    "muse_acc": 0.3, "pool_recall": 1.0, "autosend_precision": precision, "autosend_coverage": 0.0},
        "gate": gate,
        "curve": [{"i": 0, "vtt": 0.0, "gemini": 1.0, "muse": 0.0}, {"i": 1, "vtt": 0.5, "gemini": 0.5, "muse": 0.0}],
    }


@pytest.fixture(params=["file", "mongo"])
def store(request, tmp_path):
    return tmp_path / "runs" if request.param == "file" else runs.MongoRuns(mongomock.MongoClient()["test"])


def test_a_saved_run_loads_back_exactly(store):
    record = run_record(gate="pass", precision=0.95)

    runs.save(record, store)

    assert runs.load("r1", store) == record


def test_saving_does_not_change_the_callers_record(store):
    record = run_record()
    before = json.loads(json.dumps(record))

    runs.save(record, store)

    assert record == before  # e.g. pymongo's insert_one writes an _id into the dict it is given


def test_an_unknown_run_loads_as_none(store):
    runs.save(run_record(), store)

    assert runs.load("nope", store) is None


def test_runs_list_oldest_first(store):
    for run_id, t in [("b", 2.0), ("c", 3.0), ("a", 1.0)]:
        runs.save(run_record(run_id, t), store)

    assert [r["id"] for r in runs.list_runs(store)] == ["a", "b", "c"]


def test_runs_list_filters_by_user(store):
    runs.save(run_record("mine", 1.0, user_id="u1"), store)
    runs.save(run_record("theirs", 2.0, user_id="u2"), store)

    assert [r["id"] for r in runs.list_runs(store, user_id="u1")] == ["mine"]
    assert [r["id"] for r in runs.list_runs(store)] == ["mine", "theirs"]


def test_empty_store_lists_nothing(store):
    assert runs.list_runs(store) == []


def test_a_run_id_cannot_reach_outside_the_runs_dir(tmp_path):
    root = tmp_path / "runs"
    runs.save(run_record(), root)
    (tmp_path / "secret.json").write_text(json.dumps({"id": "secret"}))

    assert runs.load("../secret", root) is None


def test_default_store_is_the_data_dir_without_mongodb_uri(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")

    runs.save(run_record())

    assert json.loads((tmp_path / "runs" / "r1.json").read_text()) == run_record()
    assert runs.load("r1") == run_record()


def test_default_store_is_the_mongodb_runs_collection_with_mongodb_uri(tmp_path, monkeypatch):
    client = mongomock.MongoClient()
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    monkeypatch.delenv("MONGODB_DB", raising=False)
    monkeypatch.setattr(runs.pymongo, "MongoClient", lambda uri: client)
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    runs._mongo_client.cache_clear()

    runs.save(run_record())

    assert client["badge_machine"]["runs"].find_one({"id": "r1"}, {"_id": False}) == run_record()
    assert not (tmp_path / "runs").exists()
    runs._mongo_client.cache_clear()
