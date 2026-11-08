"""tools/ingest_practice.py: practice footage from a manifest -> cut, SAM face boxes, static crop, v1 clip records.
SAM and the lip readers are fakes with the real return shapes — no network."""

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

import muse_sampler
import sam
import scorers
from clips import backend, load
from tests.fakes import readers
from tools import ingest_practice

FACE = [40, 20, 80, 80]


@pytest.fixture(scope="module")
def footage(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("footage") / "speech.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", path], check=True)
    return path


def duration(path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True, check=True).stdout)


class FakeSam:
    """sam.segment_video(path, prompt): one face per frame, in sam.parse's shape. Records each call."""

    def __init__(self, boxes=True, delay_first=0.0):
        self.calls, self.boxes, self.delay_first, self.lock = [], boxes, delay_first, threading.Lock()

    def __call__(self, path, prompt):
        with self.lock:
            self.calls.append({"prompt": prompt, "seconds": duration(path)})
            first = len(self.calls) == 1
        if first:
            time.sleep(self.delay_first)
        frames = {i: [{"id": 0, "box": FACE}] for i in range(30)} if self.boxes else {}
        return {"width": 160, "height": 120, "frames": frames}


def forbid(*_a, **_kw):
    pytest.fail("no paid call may run")


@pytest.fixture
def fakes(monkeypatch):
    def install(sam_fake=None, **kw):
        fake = readers("yes", **kw)
        monkeypatch.setattr(sam, "segment_video", sam_fake or FakeSam())
        monkeypatch.setattr(muse_sampler, "sample_open", fake["sample_open"])
        monkeypatch.setattr(muse_sampler, "score_closed", fake["score_closed"])
        monkeypatch.setattr(scorers, "gemini_scores", fake["gemini"])
        return fake
    return install


def manifest(tmp_path, footage, n=2, **extra) -> Path:
    path = tmp_path / "manifest.jsonl"
    lines = [{"clip": str(footage), "text": f"sentence number {i}", "speaker": "spk1", "start": 0.1 * i,
              "end": 0.1 * i + 1.0, **extra} for i in range(n)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def ingest(path, store, *flags):
    return ingest_practice.main([str(path), "--dataset", "test", *flags], store=store)


def test_dry_run_prints_count_and_cost_and_makes_no_calls(tmp_path, footage, monkeypatch, capsys):
    for module, name in [(sam, "segment_video"), (muse_sampler, "sample_open"), (muse_sampler, "score_closed"),
                         (scorers, "gemini_scores")]:
        monkeypatch.setattr(module, name, forbid)

    class NoStore:
        def __getattr__(self, name):
            pytest.fail(f"dry run touched the store ({name})")

    assert ingest(manifest(tmp_path, footage, n=3), NoStore(), "--dry-run", "--limit", "2") == 0

    out = capsys.readouterr().out
    assert "2 clips" in out and "$0.30" in out


def test_practice_clips_are_cut_cropped_and_stored_with_the_transcript_as_label(tmp_path, footage, store, fakes):
    sam_fake = FakeSam()
    fakes(sam_fake)
    path = manifest(tmp_path, footage, n=2)

    assert ingest(path, store, "--seed", "3", "--k", "2") == 0

    records = load(store)
    assert len(records) == 2
    for r in records:
        line = json.loads(path.read_text().splitlines()[int(r["label"]["stt_text"].split()[-1])])
        assert (r["user_id"], r["mode"], r["seed"]) == ("practice:test:spk1", "practice", 3)
        assert r["label"] == {"stt_text": line["text"], "user_edit": None}
        assert r["key"] == ingest_practice.clip_key(line["clip"], line["start"], line["end"])
        assert (r["config"]["crop"], r["config"]["k"], len(r["samples"])) == ("v1-static-median", 2, 2)
        assert (r["source"], r["start"], r["end"]) == (line["clip"], line["start"], line["end"])
    assert [r["order"] for r in records] == [0, 1]
    assert [c["prompt"] for c in sam_fake.calls] == ["face", "face"]
    assert all(c["seconds"] == pytest.approx(1.0, abs=0.07) for c in sam_fake.calls), "SAM sees the 1 s segment"


def test_a_clip_without_a_face_passes_through_at_full_frame_and_is_flagged(tmp_path, footage, store, fakes):
    fakes(FakeSam(boxes=False))

    ingest(manifest(tmp_path, footage, n=1), store)

    (record,) = load(store)
    assert (record["config"]["crop"], record["tracker_flags"]) == ("full-frame-v0", ["no_face"])


def test_already_stored_clips_are_skipped(tmp_path, footage, store, fakes):
    path = manifest(tmp_path, footage, n=2)
    fakes()
    ingest(path, store, "--limit", "1")

    second = FakeSam()
    fake = fakes(second)
    ingest(path, store)
    third = FakeSam()
    fakes(third)
    ingest(path, store)

    assert len(second.calls) == 1, "only the clip not stored yet is processed"
    assert len(fake["sample_open"].calls) == 1
    assert third.calls == [], "nothing left to do"
    keys = [r["key"] for r in load(store)]
    assert len(keys) == len(set(keys)) == 2


def test_clips_are_stored_in_the_seeded_order_even_when_they_finish_out_of_order(tmp_path, footage, store, fakes):
    path = manifest(tmp_path, footage, n=4)
    fakes(FakeSam(delay_first=1.0))  # the first clip in the order finishes last

    ingest(path, store, "--seed", "11", "--concurrency", "2")

    records = load(store)
    assert [r["order"] for r in records] == [0, 1, 2, 3]
    fresh = backend(tmp_path / "again")
    fakes()
    ingest(path, fresh, "--seed", "11")
    assert [r["key"] for r in load(fresh)] == [r["key"] for r in records], "same seed, same order"
    other = backend(tmp_path / "other")
    fakes()
    ingest(path, other, "--seed", "12")
    assert [r["key"] for r in load(other)] != [r["key"] for r in records], "the order is shuffled by the seed"


def test_a_manifest_line_with_its_own_phrase_list_is_read_and_stored_under_it(tmp_path, footage, store, fakes):
    fake = fakes()
    own = ["it's eleven o'clock", "don't forget a jacket"]

    ingest(manifest(tmp_path, footage, n=1, phrases=own), store)

    (record,) = load(store)
    assert record["phrases"] == own
    assert fake["score_closed"].calls[0][1] == own, "the closed call scores the line's phrases"
