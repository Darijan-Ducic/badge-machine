"""Dictation sessions: a recording is cut at its pauses into turns of at most 8 s, each turn is labeled by STT on its
own audio, stored with its status, and handed to the clip pipeline in the background; /sessions reads it back."""

import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gridfs
import pymongo
import pytest

import sessions
from sessions import (MongoStore, get_session, list_sessions, parse_silences, plan_segments, policy, session_video,
                      slice_boxes, start_session, used_today)

FPS = 30
LIST_KEYS = {"users", "sessions"}
SUMMARY_KEYS = {"id", "t", "user_id", "dataset", "source", "start", "end", "duration_s", "status", "turns", "processed",
                "skipped", "vtt_correct"}
DETAIL_KEYS = {"id", "t", "user_id", "dataset", "source", "start", "end", "status", "video", "stt_text", "turns"}
TURN_KEYS = {"i", "start", "end", "stt", "status", "reason", "clip_id", "vtt", "vtt_gemini", "vtt_muse", "correct"}


# ---- pure helpers: segmentation -------------------------------------------------------------------------------------

def test_speech_spans_merge_greedily_while_the_turn_stays_within_8_s():
    silences = [(0, 1000), (2500, 3000), (4000, 6000), (9000, 9600), (12000, 20000)]
    # Speech: 1000-2500, 3000-4000, 6000-9000, 9600-12000. 1000-4000 takes in 6000-9000 because 1000-9000 is exactly
    # 8 s (the limit is inclusive); 9600-12000 would stretch it to 11 s, so it starts the next turn.
    assert plan_segments(20000, silences, [-20.0] * 400) == [(1000, 9000), (9600, 12000)]


def test_a_span_over_8_s_is_cut_at_its_deepest_dip_that_leaves_both_pieces_at_least_2_s():
    levels = [-20.0] * 240  # 12 s of speech, one level per 50 ms; a cut at level k falls at its midpoint k*50+25 ms
    levels[30] = -60.0  # deepest overall, but a cut at 1525 ms would leave a 1.5 s first piece
    levels[100] = -40.0  # 5000-5050 ms: the deepest allowed dip, cut at 5025
    levels[180] = -35.0
    assert plan_segments(12000, [], levels) == [(0, 5025), (5025, 12000)]


def test_a_span_over_8_s_with_no_dip_is_hard_cut_every_8_s():
    levels = [-20.0] * 360  # 18 s of level speech
    levels[200] = -24.0  # 4 dB down is a wobble, not a dip (a dip is at least 6 dB under the span's median)
    assert plan_segments(18000, [], levels) == [(0, 8000), (8000, 16000), (16000, 18000)]


def test_silencedetect_output_becomes_ms_intervals_and_a_silence_still_open_at_the_end_runs_to_it():
    log = """[silencedetect @ 0x6000021d8000] silence_start: 0
[silencedetect @ 0x6000021d8000] silence_end: 0.500062 | silence_duration: 0.500062
[silencedetect @ 0x6000021d8000] silence_start: 3.3
[silencedetect @ 0x6000021d8000] silence_end: 3.800063 | silence_duration: 0.500063
[silencedetect @ 0x6000021d8000] silence_start: 18.3
"""
    assert parse_silences(log, 19500) == [(0, 500), (3300, 3800), (18300, 19500)]


# ---- pure helpers: turn policy, boxes, daily count ------------------------------------------------------------------

def turn(start_ms, end_ms, text="turn on the lights"):
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text}


def test_policy_keeps_turns_from_half_a_second_to_8_s_inclusive():
    assert policy([turn(0, 499), turn(0, 500), turn(0, 8000), turn(0, 8001)], used_today=0) == [
        ("skipped", "too short"), ("pending", None), ("pending", None), ("skipped", "over 8 s")]


def test_policy_skips_turns_whose_stt_heard_nothing():
    assert policy([turn(0, 3000, ""), turn(0, 3000, "  "), turn(0, 3000, "yes")], used_today=0) == [
        ("skipped", "no speech"), ("skipped", "no speech"), ("pending", None)]


def test_the_daily_cap_is_spent_only_by_turns_that_will_be_processed():
    turns = [turn(0, 3000), turn(0, 300), turn(0, 3000, ""), turn(0, 3000), turn(0, 3000)]
    assert policy(turns, used_today=48, cap=50) == [
        ("pending", None), ("skipped", "too short"), ("skipped", "no speech"), ("pending", None), ("skipped", "daily cap")]


def test_the_default_daily_cap_is_50_turns():
    assert policy([turn(0, 3000), turn(0, 3000)], used_today=49) == [("pending", None), ("skipped", "daily cap")]


def test_boxes_are_sliced_to_the_turn_start_inclusive_end_exclusive_and_rebased_to_its_start():
    # 30 fps frames at 0, 33, 67, 100, 133, 167, 200 ms; the turn [67, 167) holds the frames at 67, 100 and 133.
    boxes = [[t, 10 + t, 20, 30, 40] for t in (0, 33, 67, 100, 133, 167, 200)]
    assert slice_boxes(boxes, 67, 167) == [[0, 77, 20, 30, 40], [33, 110, 20, 30, 40], [66, 143, 20, 30, 40]]


def test_a_recording_without_boxes_gives_each_turn_none():
    assert slice_boxes(None, 0, 1000) == [] and slice_boxes([], 0, 1000) == []


def test_used_today_counts_this_users_pending_and_processed_turns_since_local_midnight():
    def at(*ymdhms):
        return time.mktime((*ymdhms, 0, 0, -1))

    def session(user, t, *statuses):
        return {"user_id": user, "t": t, "turns": [{"status": s} for s in statuses]}

    now = at(2026, 11, 7, 15, 0, 0)
    records = [
        session("u1", at(2026, 11, 7, 14, 0, 0), "processed", "pending", "skipped", "failed"),  # 2 count
        session("u1", at(2026, 11, 7, 0, 0, 1), "processed"),  # just after midnight: counts
        session("u1", at(2026, 11, 6, 23, 59, 59), "processed", "processed"),  # yesterday
        session("u2", now, "processed"),  # someone else
    ]
    assert used_today(records, "u1", now) == 3


# ---- recordings, fakes, and stores for start_session ----------------------------------------------------------------

def media(path: Path, seconds: float, bursts, fmt="webm", live=False) -> bytes:
    """A synthetic recording: a 30 fps test pattern, and a 440 Hz tone during each (start_s, end_s) burst."""
    gate = "+".join(f"between(t,{a},{b})" for a, b in bursts)
    codecs = {"webm": ["-c:v", "libvpx", "-c:a", "libopus"],
              "mp4": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]}[fmt]
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate={FPS}:d={seconds}",
               "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*440*t)*({gate})':s=48000:d={seconds}", *codecs]
    if live:  # streamed like MediaRecorder: no duration and no cues in the container
        return subprocess.run([*command, "-live", "1", "-f", "webm", "pipe:1"], check=True, capture_output=True).stdout
    subprocess.run([*command, path], check=True)
    return path.read_bytes()


BURSTS = [(0.5, 3.3), (3.8, 7.8), (8.5, 8.8), (9.3, 16.8), (17.3, 18.3)]
# silencedetect (-30 dB, >= 0.25 s) finds every gap; merging while a turn stays within 8 s gives:
#   0.5-7.8 (two bursts, 7.3 s) | 8.5-8.8 (0.3 s: 8.5-16.8 would be 8.3 s) | 9.3-16.8 (7.5 s) | 17.3-18.3 (1 s)
TURNS_S = [(0.5, 7.8), (8.5, 8.8), (9.3, 16.8), (17.3, 18.3)]
STT = {19.5: "turn on the lights yes", 7.3: "turn on the lights", 7.5: "", 1.0: "yes"}  # by audio length in s


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> bytes:
    return media(tmp_path_factory.mktemp("rec") / "session.webm", 19.5, BURSTS)


@pytest.fixture(scope="module")
def mp4_recording(tmp_path_factory) -> bytes:
    return media(tmp_path_factory.mktemp("rec") / "session.mp4", 3.5, [(0.5, 2.5)], fmt="mp4")


@pytest.fixture(scope="module")
def live_recording(tmp_path_factory) -> bytes:
    return media(tmp_path_factory.mktemp("rec") / "unused", 6.0, [(0.5, 3.0)], live=True)


LIVE_STT = {6.0: "yes please", 2.5: "yes please"}


@pytest.fixture(params=["file", "mongo"])
def store(request, tmp_path, mongo_client):
    return tmp_path / "sessions" if request.param == "file" else MongoStore(mongo_client()["badge_machine_test"])


def audio_seconds(recording: bytes) -> float:
    with tempfile.NamedTemporaryFile() as source:
        source.write(recording)
        source.flush()
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", source.name, "-f", "s16le", "-ac", "1", "-ar", "16000",
                              "pipe:1"], check=True, capture_output=True).stdout
    return len(pcm) / 32000


class FakeSTT:
    """transcribe_turns stand-in: answers by the length of the audio it gets (the whole session vs each turn)."""

    def __init__(self, by_seconds):
        self.by_seconds, self.calls = by_seconds, []

    def __call__(self, recording):
        seconds = audio_seconds(recording)
        self.calls.append(seconds)
        text = self.by_seconds[min(self.by_seconds, key=lambda s: abs(s - seconds))]
        if isinstance(text, Exception):
            raise text
        return {"text": text, "turns": [{"start_ms": 0, "end_ms": round(seconds * 1000), "text": text}] if text else []}


def probe(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "format=format_name:stream=nb_read_frames", "-of", "default=nw=1", path],
                         check=True, capture_output=True, text=True).stdout
    fields = dict(line.split("=", 1) for line in out.split())
    return {"format": fields["format_name"], "frames": int(fields["nb_read_frames"])}


class FakeClips:
    """process_turn stand-in: records each call (and probes its segment), returns the real result shape."""

    def __init__(self, fail_turn=None, gate=None, during=None, probe_segments=False):
        self.fail_turn, self.gate, self.during, self.probe_segments = fail_turn, gate, during, probe_segments
        self.calls = {}

    def __call__(self, segment_path, *, transcript, boxes, user_id, session_id, turn, mode):
        if self.gate is not None:
            self.gate.wait(3)
        if self.during is not None:
            self.during(session_id)
        self.calls[turn] = {"path": segment_path, "transcript": transcript, "boxes": boxes, "user_id": user_id,
                            "session_id": session_id, "mode": mode,
                            **(probe(segment_path) if self.probe_segments else {})}
        if turn == self.fail_turn:
            raise RuntimeError("muse 503")
        # Verdicts differ per reader (VTT right on even turns, Gemini never, Muse always) so counts can't mix them up.
        return {"clip_id": f"clip-{session_id}-{turn}",
                "vtt": {"text": transcript, "p": 0.81}, "vtt_gemini": {"text": transcript, "p": 0.64},
                "vtt_muse": {"text": "turn of the lights", "p": 0.42},
                "correct": {"vtt": turn % 2 == 0, "gemini": False, "muse": True}}


class Later:
    """Executor that holds submitted work until run(), so a test can look at the session in between."""

    def __init__(self):
        self.queue = []

    def submit(self, fn, *args, **kwargs):
        self.queue.append(lambda: fn(*args, **kwargs))

    def run(self):
        while self.queue:
            self.queue.pop(0)()


def run(recording, store, stt=None, clips=None, user="u1", boxes=None, **options):
    with ThreadPoolExecutor(4) as pool:  # leaving the block waits for the background turns
        return start_session(recording, user_id=user, boxes=boxes or [], flags={}, process_turn=clips or FakeClips(),
                             executor=pool, transcribe=stt or FakeSTT(STT), store=store, **options)


def statuses(session_id, store):
    return [(t["status"], t["reason"]) for t in get_session(session_id, store=store)["turns"]]


# ---- start_session --------------------------------------------------------------------------------------------------

def test_turns_are_cut_at_the_pauses_with_their_times(store, recording):
    result = run(recording, store)

    turns = get_session(result["session"], store=store)["turns"]
    assert [t["i"] for t in turns] == [0, 1, 2, 3]
    assert [(t["start"], t["end"]) for t in turns] == [
        (pytest.approx(a, abs=0.03), pytest.approx(b, abs=0.03)) for a, b in TURNS_S]
    assert result["turns"] == 4


def test_each_turn_is_labeled_by_stt_on_its_own_audio_and_empty_stt_is_no_speech(store, recording):
    stt = FakeSTT(STT)

    result = run(recording, store, stt)

    detail = get_session(result["session"], store=store)
    assert [(t["stt"], t["status"], t["reason"]) for t in detail["turns"]] == [
        ("turn on the lights", "processed", None), (None, "skipped", "too short"), ("", "skipped", "no speech"),
        ("yes", "processed", None)]
    assert sorted(round(s, 1) for s in stt.calls) == [1.0, 7.3, 7.5, 19.5]  # the 0.3 s turn never goes to STT
    # The dictation reply is the whole session's own STT, not a join of the turns.
    assert result["stt"] == detail["stt_text"] == "turn on the lights yes"


def test_start_session_returns_before_any_turn_is_processed(store, recording):
    release, pool = threading.Event(), ThreadPoolExecutor(4)
    try:
        result = start_session(recording, user_id="u1", boxes=[], flags={}, process_turn=FakeClips(gate=release),
                               executor=pool, transcribe=FakeSTT(STT), store=store)
        returned = get_session(result["session"], store=store)
        release.set()
    finally:
        release.set()
        pool.shutdown(wait=True)

    assert [t["status"] for t in returned["turns"]] == ["pending", "skipped", "skipped", "pending"]
    assert returned["status"] in ("transcribed", "processing")
    assert get_session(result["session"], store=store)["status"] == "done"


def test_session_status_moves_recorded_transcribed_processing_done(store, recording):
    seen, later, stt = {}, Later(), FakeSTT(STT)

    def watching_stt(audio):
        seen.setdefault("during STT", list_sessions("u1", store=store)["sessions"][0]["status"])
        return stt(audio)

    clips = FakeClips(during=lambda sid: seen.setdefault("during clips", get_session(sid, store=store)["status"]))
    result = start_session(recording, user_id="u1", boxes=[], flags={}, process_turn=clips, executor=later,
                           transcribe=watching_stt, store=store)
    seen["returned"] = get_session(result["session"], store=store)["status"]
    later.run()
    seen["finished"] = get_session(result["session"], store=store)["status"]

    assert seen == {"during STT": "recorded", "returned": "transcribed", "during clips": "processing",
                    "finished": "done"}


def test_a_turn_whose_clip_processing_raises_fails_with_the_reason_while_the_others_complete(store, recording):
    result = run(recording, store, clips=FakeClips(fail_turn=0))

    detail = get_session(result["session"], store=store)
    assert [(t["status"], t["clip_id"]) for t in detail["turns"]] == [
        ("failed", None), ("skipped", None), ("skipped", None), ("processed", f"clip-{result['session']}-3")]
    assert "muse 503" in detail["turns"][0]["reason"]
    assert detail["status"] == "done"


def test_a_turn_whose_stt_raises_fails_with_the_reason_while_the_others_complete(store, recording):
    result = run(recording, store, FakeSTT({**STT, 1.0: RuntimeError("asr 503")}))

    detail = get_session(result["session"], store=store)
    assert [t["status"] for t in detail["turns"]] == ["processed", "skipped", "skipped", "failed"]
    assert "asr 503" in detail["turns"][3]["reason"]
    assert detail["status"] == "done"


def test_when_the_session_stt_fails_the_session_is_kept_as_failed_and_the_error_raised(store, recording):
    with pytest.raises(RuntimeError, match="asr down"):
        run(recording, store, FakeSTT({**STT, 19.5: RuntimeError("asr down")}))

    (session,) = list_sessions("u1", store=store)["sessions"]
    assert session["status"] == "failed"


def test_the_daily_cap_spans_the_users_sessions_and_not_other_users(monkeypatch, store, recording):
    monkeypatch.setattr(sessions, "DAILY_CAP", 1)

    first, second, other = run(recording, store), run(recording, store), run(recording, store, user="u2")

    assert statuses(first["session"], store)[3] == ("skipped", "daily cap")
    assert statuses(first["session"], store)[0] == ("processed", None)
    assert [statuses(second["session"], store)[i] for i in (0, 3)] == [("skipped", "daily cap")] * 2
    assert statuses(other["session"], store)[0] == ("processed", None)


@pytest.mark.parametrize("fmt", ["webm", "mp4"])
def test_each_kept_turn_gets_a_frame_accurate_segment_in_the_recordings_container(store, recording, mp4_recording, fmt):
    source, stt, seconds = {"webm": (recording, STT, 19.5), "mp4": (mp4_recording, {3.5: "yes", 2.0: "yes"}, 3.5)}[fmt]
    frame_ms = [round(k * 1000 / FPS) for k in range(round(seconds * FPS))]
    boxes = [[t, k, 0, 10, 10] for k, t in enumerate(frame_ms)]  # x = the frame number, to see which frames came along
    clips = FakeClips(probe_segments=True)

    result = run(source, store, FakeSTT(stt), clips=clips, boxes=boxes)

    first = get_session(result["session"], store=store)["turns"][0]
    start_ms, end_ms = round(first["start"] * 1000), round(first["end"] * 1000)
    window = [k for k, t in enumerate(frame_ms) if start_ms <= t < end_ms]
    call = clips.calls[0]
    assert fmt in call["format"]  # ffprobe's format_name: "matroska,webm" / "mov,mp4,m4a,3gp,3g2,mj2"
    assert call["frames"] == len(window)  # exactly the recording's frames inside the turn: frame-accurate
    assert [b[1] for b in call["boxes"]] == window
    assert call["boxes"][0][0] == frame_ms[window[0]] - start_ms
    assert (call["transcript"], call["user_id"], call["session_id"], call["mode"]) == (
        first["stt"], "u1", result["session"], "stt_on")
    assert not Path(call["path"]).exists()  # the temp segment is cleaned up after the clip pipeline ran


def test_a_streamed_recording_without_a_container_duration_still_gets_its_length(store, live_recording):
    result = run(live_recording, store, FakeSTT(LIVE_STT))

    (summary,) = list_sessions("u1", store=store)["sessions"]
    assert summary["duration_s"] == pytest.approx(6.0, abs=0.05)
    assert statuses(result["session"], store) == [("processed", None)]


# ---- the /sessions contract -----------------------------------------------------------------------------------------

def test_list_sessions_matches_the_contract(store, recording):
    practice = run(recording, store, dataset="obama-weekly", source="https://example.org/address.webm", start=120.0)
    live = run(recording, store)  # a live dictation: no dataset, no source
    run(recording, store, user="u2")

    listing = list_sessions("u1", store=store)

    assert set(listing) == LIST_KEYS
    assert sorted(listing["users"], key=lambda u: u["id"]) == [{"id": "u1", "sessions": 2}, {"id": "u2", "sessions": 1}]
    newest, oldest = listing["sessions"]
    assert set(newest) == set(oldest) == SUMMARY_KEYS
    assert (newest["id"], oldest["id"]) == (live["session"], practice["session"])  # newest first
    assert (oldest["user_id"], oldest["dataset"], oldest["source"], oldest["status"]) == (
        "u1", "obama-weekly", "https://example.org/address.webm", "done")
    assert (oldest["start"], oldest["end"], oldest["duration_s"]) == (
        120.0, pytest.approx(139.5, abs=0.05), pytest.approx(19.5, abs=0.05))
    # 4 turns: 0 and 3 processed, 1 too short, 2 no speech; FakeClips reads turn 0 right and turn 3 wrong.
    assert (oldest["turns"], oldest["processed"], oldest["skipped"], oldest["vtt_correct"]) == (4, 2, 2, 1)
    assert (newest["dataset"], newest["source"], newest["start"], newest["end"]) == (
        None, None, 0.0, pytest.approx(19.5, abs=0.05))


def test_get_session_matches_the_contract(store, recording):
    result = run(recording, store, dataset="obama-weekly", source="2016-03-12_weekly_address.webm", start=30.0)
    sid = result["session"]

    detail = get_session(sid, store=store)

    assert set(detail) == DETAIL_KEYS
    assert (detail["id"], detail["user_id"], detail["dataset"], detail["source"], detail["status"]) == (
        sid, "u1", "obama-weekly", "2016-03-12_weekly_address.webm", "done")
    assert (detail["video"], detail["stt_text"]) == (f"/sessions/{sid}/video", "turn on the lights yes")
    assert (detail["start"], detail["end"]) == (30.0, pytest.approx(49.5, abs=0.05))
    assert abs(detail["t"] - time.time()) < 60
    assert all(set(t) == TURN_KEYS for t in detail["turns"])
    processed, skipped = detail["turns"][0], detail["turns"][1]
    assert (processed["clip_id"], processed["vtt"], processed["vtt_gemini"], processed["vtt_muse"], processed["correct"]) == (
        f"clip-{sid}-0", {"text": "turn on the lights", "p": 0.81}, {"text": "turn on the lights", "p": 0.64},
        {"text": "turn of the lights", "p": 0.42}, {"vtt": True, "gemini": False, "muse": True})
    assert (skipped["clip_id"], skipped["vtt"], skipped["vtt_gemini"], skipped["vtt_muse"], skipped["correct"]) == (
        None, None, None, None, None)


def test_without_a_user_the_listing_is_the_most_recent_sessions_user(store, live_recording):
    stt = FakeSTT(LIVE_STT)
    run(live_recording, store, stt, user="u1")
    run(live_recording, store, stt, user="u2")
    assert [s["user_id"] for s in list_sessions(store=store)["sessions"]] == ["u2"]

    run(live_recording, store, stt, user="u1")
    assert [s["user_id"] for s in list_sessions(store=store)["sessions"]] == ["u1", "u1"]


def test_an_empty_store_lists_nothing(store):
    assert list_sessions(store=store) == {"users": [], "sessions": []}


@pytest.mark.parametrize("fmt, content_type", [("webm", "video/webm"), ("mp4", "video/mp4")])
def test_the_session_video_is_the_recording_as_stored(store, recording, mp4_recording, fmt, content_type):
    source, stt = {"webm": (recording, STT), "mp4": (mp4_recording, {3.5: "yes", 2.0: "yes"})}[fmt]

    result = run(source, store, FakeSTT(stt))

    assert session_video(result["session"], store=store) == (source, content_type)


def test_an_unknown_session_is_none(store):
    assert get_session("no-such-session", store=store) is None
    assert session_video("no-such-session", store=store) is None


# ---- backend selection, as in the clip store ------------------------------------------------------------------------

def test_without_mongodb_uri_sessions_go_to_the_local_data_dir(monkeypatch, tmp_path, live_recording):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.chdir(tmp_path)

    result = run(live_recording, None, FakeSTT(LIVE_STT))

    assert [s["id"] for s in list_sessions(store=tmp_path / "data" / "sessions")["sessions"]] == [result["session"]]


def test_with_mongodb_uri_sessions_go_to_the_sessions_collection_and_bucket(monkeypatch, tmp_path, mongo_client,
                                                                            live_recording):
    clients = []
    monkeypatch.setattr(pymongo, "MongoClient", lambda *a, **kw: clients.append(mongo_client(*a, **kw)) or clients[-1])
    monkeypatch.setenv("MONGODB_URI", f"mongodb://{uuid.uuid4().hex}.test")
    monkeypatch.setenv("MONGODB_DB", "team_db")
    monkeypatch.chdir(tmp_path)

    result = run(live_recording, None, FakeSTT(LIVE_STT))

    (client,) = clients
    db = client["team_db"]
    assert [d["id"] for d in db["sessions"].find()] == [result["session"]]
    assert gridfs.GridFSBucket(db, bucket_name="sessions").open_download_stream(result["session"]).read() == live_recording
    assert list_sessions()["sessions"][0]["id"] == result["session"]
    assert not (tmp_path / "data").exists()
