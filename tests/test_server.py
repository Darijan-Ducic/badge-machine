"""HTTP contract of /stats, /clips, /clips/<id>, /clips/<id>/video, /preheat/clip, /vtt, /vtt/<id>/label, /dictation
and /sessions, served from a real ThreadingHTTPServer over each store backend — model calls are fakes, no network."""

import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

import clips
import muse_sampler
import scorers
import server
import sessions
from candidates import FEATURES
from clips import backend, load, record_clip
from tests.fakes import PHRASES, Gemini, SampleOpen, ScoreClosed, readers, v1_output
from tests.test_sessions import media

OLD_STATS_KEYS = {"labeled", "clips", "stage", "next_stage", "threshold", "camera_fps", "window", "accuracy", "history"}
LIST_KEYS = {"id", "t", "user_id", "mode", "stt", "top", "action", "correct", "pool_hit", "stage"}
DETAIL_KEYS = {"id", "t", "user_id", "mode", "stt", "config", "video", "frames", "stage", "action", "threshold", "pool",
               "samples", "closed", "gemini", "vtt_muse"}
CLIP_KEYS = {"clip", "stt", "label", "label_phrase", "read", "correct", "stats"}
VTT_KEYS = {"clip", "vtt", "action", "alternatives", "latency_ms"}
NONE_WIN = dict(sample_open=SampleOpen(v1_output(present=False)), score_closed=ScoreClosed(none=0.95, default=0.01),
                gemini=Gemini(probs=dict.fromkeys(PHRASES, 0.25)))


@pytest.fixture
def api(store, monkeypatch):
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "load_phrases", lambda: PHRASES)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout=60) as client:
        yield client
    httpd.shutdown()
    httpd.server_close()


def add(store, recording, said, *tops, user="ann", mode="stt_on", k=3, fakes=None, **kw):
    record = record_clip(recording, phrases=PHRASES, user_id=user, mode=mode, k=k, transcript=said, store=store,
                         **(fakes or readers(*tops, **kw)))
    return record["id"]


def get(api, path):
    response = api.get(path)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------- /stats


def test_stats_keep_every_old_key_and_add_user_pool_recall_and_stages(api, store, recording):
    add(store, recording, "yes", "yes")
    add(store, recording, "what a lovely afternoon", "yes")
    add(store, recording, "no", "no", user="bob")

    s = get(api, "/stats?user=ann")

    assert set(s) == OLD_STATS_KEYS | {"user", "pool_recall", "stages"}
    assert (s["user"], s["clips"], s["labeled"], s["pool_recall"]) == ("ann", 2, 2, 0.5)
    assert s["stages"] == [{"name": "defaults", "at": 0}, {"name": "temperature", "at": 10},
                           {"name": "core", "at": 30}, {"name": "full", "at": 60}]


def test_an_empty_store_lists_nothing_and_has_empty_stats(api):
    assert get(api, "/clips") == {"users": [], "clips": []}
    s = get(api, "/stats")
    assert (s["user"], s["clips"], s["labeled"], s["history"]) == (None, 0, 0, [])


def test_without_a_user_clips_and_stats_follow_the_latest_clip(api, store, recording):
    add(store, recording, "yes", "yes", user="ann")
    add(store, recording, "no", "no", user="ann")
    latest = add(store, recording, "yes", "yes", user="bob")

    listing = get(api, "/clips")

    assert [c["id"] for c in listing["clips"]] == [latest]
    assert sorted(listing["users"], key=lambda u: u["id"]) == [{"id": "ann", "clips": 2}, {"id": "bob", "clips": 1}]
    assert (get(api, "/stats")["user"], get(api, "/stats")["clips"]) == ("bob", 1)


# ---------------------------------------------------------------- /clips


def test_clip_list_matches_the_contract_newest_first(api, store, recording):
    hit = add(store, recording, "yes", "yes", gemini="yes")
    unlabeled = add(store, recording, None, "no", mode="runtime")
    none_win = add(store, recording, "what a lovely afternoon", fakes=NONE_WIN)
    add(store, recording, "no", "no", user="bob")

    listing = get(api, "/clips?user=ann")

    assert sorted(listing["users"], key=lambda u: u["id"]) == [{"id": "ann", "clips": 3}, {"id": "bob", "clips": 1}]
    rows = listing["clips"]
    assert [r["id"] for r in rows] == [none_win, unlabeled, hit]
    assert all(set(r) == LIST_KEYS for r in rows)
    by_id = {r["id"]: r for r in rows}
    assert by_id[hit]["top"]["text"] == "yes" and 0 < by_id[hit]["top"]["p"] < 1
    assert (by_id[hit]["correct"], by_id[hit]["pool_hit"]) == ({"fused": True, "gemini": True, "muse": True}, True)
    assert (by_id[hit]["action"], by_id[hit]["stage"]) == ("confirm", "defaults")  # defaults never auto-send
    assert (by_id[hit]["user_id"], by_id[hit]["mode"], by_id[hit]["stt"]) == ("ann", "stt_on", "yes")
    assert (by_id[unlabeled]["correct"], by_id[unlabeled]["pool_hit"], by_id[unlabeled]["stt"]) == (None, None, None)
    assert by_id[none_win]["top"]["text"] == "∅ none of these"
    assert (by_id[none_win]["action"], by_id[none_win]["pool_hit"]) == ("none", False)
    assert by_id[none_win]["correct"] == {"fused": False, "gemini": False, "muse": False}


def test_clip_detail_matches_the_contract_with_a_failed_sample(api, store, recording):
    fakes = readers("yes", fail={1}, gemini="yes")
    clip = add(store, recording, "yes", fakes=fakes)

    d = get(api, f"/clips/{clip}")

    assert set(d) == DETAIL_KEYS
    assert (d["id"], d["user_id"], d["mode"], d["stt"]) == (clip, "ann", "stt_on", "yes")
    assert (d["video"], d["frames"]) == (f"/clips/{clip}/video", 12)
    assert d["config"] == backend(store).get(clip)["config"] and d["config"]["k"] == 3
    assert (d["stage"], d["threshold"], d["action"]) == ("defaults", None, "confirm")
    assert [set(s) for s in d["samples"]] == [{"sample_idx", "latency_ms", "error", "output"}] * 3
    failed = d["samples"][1]
    assert (failed["sample_idx"], failed["output"]) == (1, None) and "Connection error" in failed["error"]
    assert d["samples"][0]["output"] == v1_output(("yes", 0.6), ("hello", 0.2))
    assert set(d["closed"]) == {"latency_ms", "error", "output"} and d["closed"]["output"]["none"] == 0.1
    assert d["gemini"]["top"][0] == {"phrase": "yes", "p": pytest.approx(0.9)}
    assert all(set(g) == {"phrase", "p"} for g in d["gemini"]["top"])
    assert [g["p"] for g in d["gemini"]["top"]] == sorted((g["p"] for g in d["gemini"]["top"]), reverse=True)
    assert all(set(c) == {"text", "p", "is_label", "features"} and list(c["features"]) == FEATURES for c in d["pool"])
    assert sum(c["p"] for c in d["pool"]) == pytest.approx(1)
    assert [c["text"] for c in d["pool"] if c["is_label"]] == ["yes"]
    assert d["pool"][0]["text"] == "yes", "pool is listed most probable first"


def test_clip_detail_of_a_none_win(api, store, recording):
    clip = add(store, recording, "what a lovely afternoon", fakes=NONE_WIN)

    d = get(api, f"/clips/{clip}")

    assert d["pool"][0]["text"] == "∅ none of these" and d["pool"][0]["is_label"] is True
    assert d["pool"][0]["features"]["is_none"] == 1.0
    assert d["action"] == "none"


def test_detail_prior_counts_only_texts_said_before_the_clip(api, store, recording):
    add(store, recording, "yes", "yes")
    second = add(store, recording, "no", "no")
    third = add(store, recording, "yes", "yes")

    def prior(clip, text):
        [candidate] = [c for c in get(api, f"/clips/{clip}")["pool"] if c["text"] == text]
        return candidate["features"]["prior"]

    # bank = 4 listed phrases; prior = log((count + 1) / (total + bank size + 1))
    assert prior(second, "no") == pytest.approx(math.log(1 / 6))  # earlier: {"yes": 1}
    assert prior(third, "yes") == pytest.approx(math.log(2 / 7))  # earlier: {"yes": 1, "no": 1}


def test_detail_pool_is_scored_by_a_model_trained_only_on_earlier_clips(api, store, recording):
    ids = []
    for i in range(11):
        said = PHRASES[i % 4]
        ids.append(add(store, recording, said, PHRASES[(i + (i % 3 == 0)) % 4], gemini=said))

    ninth, tenth = get(api, f"/clips/{ids[9]}"), get(api, f"/clips/{ids[10]}")
    assert (ninth["stage"], ninth["threshold"]) == ("defaults", None), "9 earlier labeled clips, not counting itself"
    assert tenth["stage"] == "temperature", "10 earlier labeled clips"

    for i in range(20):  # later clips that would change any model trained on them (and push it to "core")
        add(store, recording, "turn off the lights", "yes", gemini="no")

    assert get(api, f"/clips/{ids[10]}") == tenth
    assert {r["id"]: r["stage"] for r in get(api, "/clips?user=ann")["clips"]}[ids[10]] == "temperature"


def test_video_route_serves_the_stored_clip_as_mp4(api, store, recording):
    clip = add(store, recording, "yes", "yes")

    response = api.get(f"/clips/{clip}/video")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == backend(store).video(clip) and len(response.content) > 0


def test_unknown_clips_are_404(api):
    assert api.get("/clips/does-not-exist").status_code == 404
    assert api.get("/clips/does-not-exist/video").status_code == 404


# ---------------------------------------------------------------- /preheat/clip


def test_preheat_route_keeps_its_response_keys_and_reads_with_k5(api, store, recording, monkeypatch):
    fakes = readers("yes", gemini="yes")
    monkeypatch.setattr(muse_sampler, "sample_open", fakes["sample_open"])
    monkeypatch.setattr(muse_sampler, "score_closed", fakes["score_closed"])
    monkeypatch.setattr(scorers, "gemini_scores", fakes["gemini"])
    monkeypatch.setattr(server, "transcribe", lambda data: "yes")

    response = api.post("/preheat/clip?fps=30&prompted=0&user=ann", content=recording)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == CLIP_KEYS
    assert (body["label"], body["label_phrase"], body["stats"]["user"]) == (0, "yes", "ann")
    assert fakes["sample_open"].calls[0][1] == 5
    assert [r["user_id"] for r in load(store)] == ["ann"]


@pytest.fixture
def fake_calls(monkeypatch):
    fakes = readers("yes", gemini="yes")
    monkeypatch.setattr(muse_sampler, "sample_open", fakes["sample_open"])
    monkeypatch.setattr(muse_sampler, "score_closed", fakes["score_closed"])
    monkeypatch.setattr(scorers, "gemini_scores", fakes["gemini"])
    monkeypatch.setattr(server, "transcribe", lambda data: "yes")
    return fakes


def width(data: bytes, tmp_path) -> int:
    path = tmp_path / "w.mp4"
    path.write_bytes(data)
    return int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width",
                               "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout)


def test_tracker_headers_parse_boxes_and_flags():
    headers = {"X-Tracker-Boxes": "[[0, 8, 24, 32, 16], [33, 10, 24, 30, 16]]", "X-Tracker-Flags": '["occluded?"]'}

    assert server.tracker_headers(headers) == ([[0, 8, 24, 32, 16], [33, 10, 24, 30, 16]], ["occluded?"])
    assert server.tracker_headers({"X-Tracker-Flags": "occluded?, lost"}) == (None, ["occluded?", "lost"])
    assert server.tracker_headers({}) == (None, [])


def test_preheat_crops_with_the_tracker_boxes_from_the_header(api, store, recording, fake_calls, tmp_path):
    # 64x48 recording; boxes' median [10, 24, 30, 16] (x {8, 10, 40}, y {24, 24, 0}, w {32, 30, 10}, h {16, 16, 10})
    boxes = [[0, 8, 24, 32, 16], [33, 10, 24, 30, 16], [66, 40, 0, 10, 10]]

    response = api.post("/preheat/clip?fps=30&prompted=0", content=recording,
                        headers={"X-Tracker-Boxes": json.dumps(boxes), "X-Tracker-Flags": '["occluded?"]'})

    assert response.status_code == 200, response.text
    (stored,) = load(store)
    assert (stored["config"]["crop"], stored["tracker_flags"]) == ("v1-tracker-median", ["occluded?"])
    assert width(backend(store).video(stored["id"]), tmp_path) == 256


def test_preheat_without_tracker_boxes_reads_the_full_frame(api, store, recording, fake_calls, tmp_path):
    assert api.post("/preheat/clip?fps=30&prompted=0", content=recording).status_code == 200

    (stored,) = load(store)
    assert (stored["config"]["crop"], stored["tracker_flags"]) == ("full-frame-v0", [])
    assert width(backend(store).video(stored["id"]), tmp_path) == 64


def test_preheat_without_a_user_records_the_default_user(api, store, recording, monkeypatch):
    fakes = readers("yes", gemini="yes")
    monkeypatch.setattr(muse_sampler, "sample_open", fakes["sample_open"])
    monkeypatch.setattr(muse_sampler, "score_closed", fakes["score_closed"])
    monkeypatch.setattr(scorers, "gemini_scores", fakes["gemini"])
    monkeypatch.setattr(server, "transcribe", lambda data: "yes")

    assert api.post("/preheat/clip?fps=30&prompted=0", content=recording).status_code == 200

    assert [r["user_id"] for r in load(store)] == [clips.DEFAULT_USER]


# ---------------------------------------------------------------- /clips/<id> vtt_muse


def test_clip_detail_carries_the_muse_vote_as_vtt_muse(api, store, recording):
    # K=3 open samples whose tops are no, yes, no -> "no" with 2 of 3 votes
    clip = add(store, recording, None, "no", "yes", "no", mode="lip")

    assert get(api, f"/clips/{clip}")["vtt_muse"] == {"text": "no", "p": pytest.approx(2 / 3)}


def test_vtt_muse_is_null_when_every_open_sample_failed(api, store, recording):
    clip = add(store, recording, None, "no", mode="lip", fail={0, 1, 2})

    assert get(api, f"/clips/{clip}")["vtt_muse"] == {"text": None, "p": 0.0}


# ---------------------------------------------------------------- /vtt and /vtt/<id>/label (lip mode)


@pytest.fixture
def muted(recording, tmp_path) -> bytes:
    """The browser-like recording with its audio dropped, as lip mode sends it."""
    source, target = tmp_path / "with-audio.webm", tmp_path / "muted.webm"
    source.write_bytes(recording)
    subprocess.run(["ffmpeg", "-v", "error", "-i", source, "-an", "-c:v", "copy", target], check=True)
    return target.read_bytes()


@pytest.fixture
def lip_calls(monkeypatch):
    fakes = {**readers("yes", gemini="yes"), "heard": []}
    monkeypatch.setattr(muse_sampler, "sample_open", fakes["sample_open"])
    monkeypatch.setattr(muse_sampler, "score_closed", fakes["score_closed"])
    monkeypatch.setattr(scorers, "gemini_scores", fakes["gemini"])
    monkeypatch.setattr(server, "transcribe", lambda data: fakes["heard"].append(data) or "yes")
    return fakes


def post_vtt(api, video, user="ann", **headers) -> dict:
    response = api.post(f"/vtt?user={user}", content=video, headers={"Content-Type": "video/webm", **headers})
    assert response.status_code == 200, response.text
    return response.json()


def test_vtt_route_reads_a_muted_clip_with_k3_and_answers_the_page_contract(api, store, muted, lip_calls):
    boxes = [[0, 8, 24, 32, 16], [33, 10, 24, 30, 16], [66, 40, 0, 10, 10]]

    body = post_vtt(api, muted, **{"X-Tracker-Boxes": json.dumps(boxes), "X-Tracker-Flags": '["occluded?"]'})

    assert set(body) == VTT_KEYS
    assert (body["vtt"]["text"], body["action"]) == ("yes", "confirm")  # defaults never auto-send
    assert set(body["vtt"]) == {"text", "p"} and 0 < body["vtt"]["p"] < 1
    assert body["alternatives"][0] == body["vtt"] and len(body["alternatives"]) == 3
    assert all(set(a) == {"text", "p"} for a in body["alternatives"])
    assert isinstance(body["latency_ms"], int)
    (stored,) = load(store)
    assert (stored["id"], stored["user_id"], stored["mode"]) == (body["clip"], "ann", "lip")
    assert (stored["config"]["crop"], stored["tracker_flags"]) == ("v1-tracker-median", ["occluded?"])
    assert stored["label"] == {"stt_text": None, "user_edit": None}
    assert [call[1] for call in lip_calls["sample_open"].calls] == [3]
    assert lip_calls["heard"] == [], "lip mode runs no STT"


def test_label_route_stores_the_users_label_on_that_clip(api, store, muted, lip_calls):
    clip, other = post_vtt(api, muted)["clip"], post_vtt(api, muted)["clip"]

    response = api.post(f"/vtt/{clip}/label", json={"text": "turn on the lights", "source": "confirm"})

    assert (response.status_code, response.json()) == (200, {"clip": clip, "label": "turn on the lights"})
    assert backend(store).get(clip)["label"] == {"stt_text": None, "user_edit": "turn on the lights",
                                                 "source": "confirm"}
    assert backend(store).get(other)["label"] == {"stt_text": None, "user_edit": None}


@pytest.mark.parametrize("request_body", [{"text": "yes", "source": "typed"}, {"text": "yes", "source": None},
                                          {"text": "yes"}, {"source": "edit"}, {"text": "", "source": "edit"}])
def test_label_route_rejects_a_bad_source_or_missing_text_with_400(api, store, muted, lip_calls, request_body):
    clip = post_vtt(api, muted)["clip"]

    response = api.post(f"/vtt/{clip}/label", json=request_body)

    assert response.status_code == 400, response.text
    assert backend(store).get(clip)["label"] == {"stt_text": None, "user_edit": None}


def test_label_route_rejects_a_body_that_is_not_json_with_400(api, store, muted, lip_calls):
    clip = post_vtt(api, muted)["clip"]

    response = api.post(f"/vtt/{clip}/label", content=b"yes please")

    assert response.status_code == 400, response.text
    assert backend(store).get(clip)["label"] == {"stt_text": None, "user_edit": None}


def test_label_route_is_404_for_an_unknown_clip(api):
    response = api.post("/vtt/no-such-clip/label", json={"text": "yes", "source": "edit"})

    assert response.status_code == 404


# ---------------------------------------------------------------- /dictation and /sessions


@pytest.fixture(scope="module")
def session_recording(tmp_path_factory) -> bytes:
    """Speech at 0.5-2.5 s and 9.0-11.0 s: two turns (together they would be over 8 s)."""
    return media(tmp_path_factory.mktemp("session") / "session.webm", 11.5, [(0.5, 2.5), (9.0, 11.0)])


@pytest.fixture
def dictation(api, store, fake_calls, monkeypatch, tmp_path):
    """/dictation over fake STT and readers; turns run on a pool that is drained before the fakes are undone."""
    turns = ThreadPoolExecutor(2)
    monkeypatch.setattr(server, "TURNS", turns)
    monkeypatch.setattr(server, "SESSIONS", tmp_path / "sessions" if isinstance(store, Path)
                        else sessions.MongoStore(store.db))
    monkeypatch.setattr(server, "transcribe_turns", lambda wav: {"text": "yes", "turns": []})
    yield
    turns.shutdown(wait=True)


def post_dictation(api, video, **headers) -> dict:
    response = api.post("/dictation?user=ann", content=video, headers={"Content-Type": "video/webm", **headers})
    assert response.status_code == 200, response.text
    return response.json()


def finished(api, session_id: str) -> dict:
    for _ in range(300):
        detail = get(api, f"/sessions/{session_id}")
        if detail["status"] in ("done", "failed"):
            return detail
        time.sleep(0.1)
    raise AssertionError(f"session {session_id} still {detail['status']}")


def test_dictation_route_answers_the_session_its_stt_and_its_turn_count(api, dictation, session_recording):
    body = post_dictation(api, session_recording)

    assert set(body) == {"session", "stt", "turns"}
    assert (body["stt"], body["turns"]) == ("yes", 2)
    assert get(api, f"/sessions/{body['session']}")["user_id"] == "ann"


def test_dictation_turns_become_clips_of_that_session_read_by_every_model(api, store, dictation, fake_calls,
                                                                         session_recording):
    session = post_dictation(api, session_recording)["session"]

    turns = finished(api, session)["turns"]

    assert [(t["i"], t["status"], t["stt"]) for t in turns] == [(0, "processed", "yes"), (1, "processed", "yes")]
    assert [(backend(store).get(t["clip_id"])["session_id"], backend(store).get(t["clip_id"])["turn"])
            for t in turns] == [(session, 0), (session, 1)]
    assert all(t["vtt"]["text"] == "yes" and t["correct"] == {"vtt": True, "gemini": True, "muse": True}
               for t in turns)
    assert [call[1] for call in fake_calls["sample_open"].calls] == [5, 5]


def test_dictation_turns_are_cropped_with_the_tracker_boxes_from_the_header(api, store, dictation, session_recording):
    boxes = [[t, 40, 30, 64, 48] for t in range(0, 11500, 33)]

    session = post_dictation(api, session_recording, **{"X-Tracker-Boxes": json.dumps(boxes),
                                                        "X-Tracker-Flags": '["occluded?"]'})["session"]

    turns = finished(api, session)["turns"]
    assert [backend(store).get(t["clip_id"])["config"]["crop"] for t in turns] == ["v1-tracker-median"] * 2


def test_sessions_route_lists_the_users_sessions_and_serves_the_recording(api, dictation, session_recording):
    session = post_dictation(api, session_recording)["session"]
    finished(api, session)

    listing = get(api, "/sessions?user=ann")
    video = api.get(f"/sessions/{session}/video")

    assert [s["id"] for s in listing["sessions"]] == [session] and listing["users"] == [{"id": "ann", "sessions": 1}]
    assert (video.status_code, video.headers["content-type"]) == (200, "video/webm")
    assert video.content == session_recording


def test_unknown_sessions_are_an_api_404(api, dictation):
    for path in ("/sessions/no-such-session", "/sessions/no-such-session/video"):
        response = api.get(path)
        # the API's JSON error, not static/'s "no such file" page
        assert (response.status_code, response.headers["content-type"]) == (404, "application/json"), path


def test_a_29_s_session_of_tracker_boxes_fits_in_one_header(api, store, dictation, session_recording):
    # 29 s at 30 fps with every number at its widest for a 1280x960 camera; timed after this 11.5 s recording ends,
    # so no turn crops with them
    boxes = json.dumps([[12000 + round(i * 1000 / 30), 1279, 959, 1280, 960] for i in range(870)])
    assert len(boxes) > 20_000

    session = post_dictation(api, session_recording, **{"X-Tracker-Boxes": boxes})["session"]

    assert [t["status"] for t in finished(api, session)["turns"]] == ["processed", "processed"]
