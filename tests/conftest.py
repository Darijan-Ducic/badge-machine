"""Clip-store fixtures: every store test runs once on the file backend and once on the MongoDB backend (mongomock)."""

import types

import mongomock
import pytest
from mongomock.gridfs import enable_gridfs_integration

from clips import MongoStore

enable_gridfs_integration()  # pymongo's GridFS type-checks its database; let it accept mongomock's


def in_memory_mongo(*_args, **_kwargs):
    client = mongomock.MongoClient()
    client.options = types.SimpleNamespace(timeout=None)  # GridFSBucket reads client.options.timeout; mongomock lacks it
    return client


@pytest.fixture
def mongo_client():
    """Factory for in-memory MongoClients that pymongo's GridFS accepts; same call shape as pymongo.MongoClient."""
    return in_memory_mongo


@pytest.fixture(params=["file", "mongo"])
def store(request, tmp_path):
    return tmp_path if request.param == "file" else MongoStore(in_memory_mongo()["badge_machine_test"])


class OfflineGemini:
    """The google.genai client the Gemini open reader gets in tests: records each request and fails like a dropped
    connection, so a clip read without a Gemini open fake makes no paid call (its failed samples are skipped)."""

    def __init__(self):
        self.requests = []
        self.models = types.SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **request):
        self.requests.append(request)
        raise ConnectionError("no network in tests")


@pytest.fixture(autouse=True)
def offline_gemini(monkeypatch):
    import muse_sampler
    fake = OfflineGemini()
    monkeypatch.setattr(muse_sampler, "gemini_client", lambda: fake, raising=False)
    return fake


@pytest.fixture(scope="session")
def recording(tmp_path_factory) -> bytes:
    """A real browser-like webm (VP8 + Opus tone, 0.5 s at 30 fps), so the crop step runs real ffmpeg."""
    from tests.fakes import make_recording
    return make_recording(tmp_path_factory.mktemp("recording") / "clip.webm")


@pytest.fixture(autouse=True)
def _every_reader(monkeypatch):
    """The suite describes the full-reader pipeline; a test of the fast path sets clips.LIP_READERS itself."""
    import clips
    monkeypatch.setattr(clips, "LIP_READERS", "all")
