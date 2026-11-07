"""Experiment run records (the `GET /runs` shape, built by `experiment.run`): save, load one, list.

Backends follow the clip store (clips.py): the MongoDB collection `runs` when `MONGODB_URI` is set (database
`MONGODB_DB`, default `badge_machine`), else one JSON file per run under `data/runs/<id>.json`. `store` may also be
passed explicitly: a directory Path, or a `MongoRuns`.
"""

import functools
import json
import os
import re
from pathlib import Path

import pymongo

RUNS = Path("data/runs")
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")  # run ids become file names; nothing that can leave the directory


class _FileRuns:
    def __init__(self, root: Path):
        self.root = root

    def save(self, record: dict) -> None:
        if not _SAFE_ID.fullmatch(record["id"]):
            raise ValueError(f"run id {record['id']!r} must match {_SAFE_ID.pattern}")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{record['id']}.json").write_text(json.dumps(record))

    def load(self, run_id: str) -> dict | None:
        path = self.root / f"{run_id}.json"
        return json.loads(path.read_text()) if _SAFE_ID.fullmatch(run_id) and path.exists() else None

    def all(self, query: dict) -> list[dict]:
        records = [json.loads(p.read_text()) for p in self.root.glob("*.json")] if self.root.exists() else []
        return sorted((r for r in records if all(r.get(k) == v for k, v in query.items())), key=lambda r: r["t"])


class MongoRuns:
    """Run records in the `runs` collection, one document per run."""

    def __init__(self, db):
        self.runs = db["runs"]

    def save(self, record: dict) -> None:
        self.runs.insert_one(dict(record))  # insert_one writes _id into the dict it gets; keep the caller's clean

    def load(self, run_id: str) -> dict | None:
        return self.runs.find_one({"id": run_id}, {"_id": False})

    def all(self, query: dict) -> list[dict]:
        return list(self.runs.find(query, {"_id": False}).sort("t", pymongo.ASCENDING))


@functools.cache
def _mongo_client(uri: str):
    return pymongo.MongoClient(uri)  # one client, and so one connection pool, per process


def _backend(store):
    if store is None:
        uri = os.environ.get("MONGODB_URI")
        store = MongoRuns(_mongo_client(uri)[os.environ.get("MONGODB_DB") or "badge_machine"]) if uri else RUNS
    return _FileRuns(store) if isinstance(store, Path) else store


def save(record: dict, store: Path | MongoRuns | None = None) -> None:
    _backend(store).save(record)


def load(run_id: str, store: Path | MongoRuns | None = None) -> dict | None:
    return _backend(store).load(run_id)


def list_runs(store: Path | MongoRuns | None = None, user_id: str | None = None) -> list[dict]:
    """Oldest first; only `user_id`'s runs when given."""
    return _backend(store).all({} if user_id is None else {"user_id": user_id})
