"""A dictation turn becomes a clip: sessions hands each kept turn's segment to `clips.process_turn`, which stores it
as a K=5 clip labeled by the turn's STT and reads it back with a model trained on the user's earlier clips only.
Every store test runs on both backends; the readers are fakes, no network."""

import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from clips import backend, clip_detail, load, process_turn, read_clip, record_clip
from sessions import TURN_RESULT, start_session
from tests.fakes import PHRASES, Gemini, SampleOpen, ScoreClosed, readers, v1_output
from tests.test_sessions import media
from tests.test_vtt import LandsAfter

NONE_WIN = dict(sample_open=SampleOpen(v1_output(present=False)), score_closed=ScoreClosed(none=0.95, default=0.01),
                gemini=Gemini(probs=dict.fromkeys(PHRASES, 0.25)))
BOXES = [[0, 8, 24, 32, 16], [33, 10, 24, 30, 16], [66, 40, 0, 10, 10]]  # median [10, 24, 30, 16] in the 64x48 frame


@pytest.fixture
def segment(recording, tmp_path):
    path = tmp_path / "turn-2.webm"
    path.write_bytes(recording)
    return path


def turn(store, segment, transcript="yes", fakes=None, user="ann", **kw):
    options = {"boxes": BOXES, "user_id": user, "session_id": "s1", "turn": 2, "mode": "stt_on", **kw}
    return process_turn(segment, transcript=transcript, phrases=PHRASES, store=store,
                        **options, **(fakes or readers("yes", gemini="yes")))


def add_labeled(store, recording, n: int, user="ann"):
    for i in range(n):
        said = PHRASES[i % 2]
        record_clip(recording, phrases=PHRASES, user_id=user, mode="stt_on", k=3, transcript=said, store=store,
                    **readers(said, gemini=said))


def height(data: bytes, tmp_path) -> int:
    path = tmp_path / "h.mp4"
    path.write_bytes(data)
    return int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                               "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout)


# ---------------------------------------------------------------- the stored clip


def test_a_turn_is_stored_as_a_k5_clip_labeled_by_its_stt_with_its_session_and_turn(store, segment):
    fakes = readers("yes", gemini="yes")

    result = turn(store, segment, transcript="turn on the lights", fakes=fakes)

    assert [call[1] for call in fakes["sample_open"].calls] == [5]
    (stored,) = load(store)
    assert stored["id"] == result["clip_id"]
    assert (stored["user_id"], stored["mode"], stored["session_id"], stored["turn"]) == ("ann", "stt_on", "s1", 2)
    assert stored["label"] == {"stt_text": "turn on the lights", "user_edit": None}
    assert stored["config"]["crop"] == "v1-tracker-median"


# ---------------------------------------------------------------- the read-back


def test_a_turn_returns_each_readers_pick_and_whether_it_matches_stt(store, segment):
    # Muse tops: off, off, on, off, on -> "turn off the lights" with a 3/5 vote share. Defaults score
    # "turn on the lights" .4 + .4 + .24 + .8 (closed) + .9 (Gemini) = 2.74, above "turn off" (1.64) and "hello" (1.2).
    fakes = readers("turn off the lights", "turn off the lights", "turn on the lights", "turn off the lights",
                    "turn on the lights", closed={"turn on the lights": 0.8}, gemini="turn on the lights")

    result = turn(store, segment, transcript="Turn on the lights.", fakes=fakes)

    assert set(result) == set(TURN_RESULT)
    assert result["vtt"]["text"] == "turn on the lights" and 0 < result["vtt"]["p"] < 1
    assert result["vtt_gemini"] == {"text": "turn on the lights", "p": pytest.approx(0.9)}
    assert result["vtt_muse"] == {"text": "turn off the lights", "p": pytest.approx(0.6)}
    assert result["correct"] == {"vtt": True, "gemini": True, "muse": False}


def test_a_none_win_turn_reads_as_none_and_is_not_correct(store, segment):
    result = turn(store, segment, fakes=NONE_WIN)

    assert result["vtt"]["text"] == "∅", "static/vtt.js VTT.NONE"
    assert result["correct"]["vtt"] is False


def test_a_turn_is_read_by_a_model_trained_on_the_users_earlier_clips(store, segment, recording):
    untrained = turn(store, segment, user="bob")["vtt"]["p"]
    add_labeled(store, recording, 10)

    result = turn(store, segment)

    assert result["vtt"]["p"] == pytest.approx(clip_detail(store, result["clip_id"])["pool"][0]["p"])
    assert result["vtt"]["p"] > untrained, "10 earlier labeled clips sharpen the temperature"


def test_clips_stored_after_a_turn_do_not_train_its_read(store, segment, recording):
    later = [read_clip(recording, phrases=PHRASES, user_id="ann", mode="stt_on", k=3, transcript=PHRASES[i % 2],
                       **readers(PHRASES[i % 2], gemini=PHRASES[i % 2])) for i in range(10)]

    result = turn(LandsAfter(store, later), segment)

    assert len(load(store, "ann")) == 11, "the later clips did land"
    assert result["vtt"]["p"] == pytest.approx(clip_detail(store, result["clip_id"])["pool"][0]["p"])


# ---------------------------------------------------------------- inside a session


def test_each_session_turn_is_cropped_with_the_boxes_of_its_own_span(store, tmp_path):
    # Speech at 0.5-2.5 s and 9.0-11.0 s: 10.5 s apart is over 8 s, so two turns. Boxes before 6 s are 4:3, after 2:1;
    # the crop is scaled to 256 wide, so a turn cropped with its own boxes is 192 or 128 high.
    recording = media(tmp_path / "session.webm", 11.5, [(0.5, 2.5), (9.0, 11.0)])
    boxes = [[t, 40, 30, 64, 48 if t < 6000 else 32] for t in range(0, 11500, 33)]

    def adapter(segment, **turn_fields):
        return process_turn(segment, phrases=PHRASES, store=store, **turn_fields, **readers("yes", gemini="yes"))

    with ThreadPoolExecutor(2) as pool:
        start_session(recording, user_id="ann", boxes=boxes, flags=[], process_turn=adapter, executor=pool,
                      transcribe=lambda wav: {"text": "yes", "turns": []}, store=tmp_path / "sessions")

    heights = {r["turn"]: height(backend(store).video(r["id"]), tmp_path) for r in load(store, "ann")}
    assert heights == {0: 192, 1: 128}
