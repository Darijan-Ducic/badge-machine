"""HTTP contract of /stats, /clips, /clips/<id>, /clips/<id>/video and /preheat/clip, served from a real
ThreadingHTTPServer over each store backend — model calls are fakes, no network."""

import math
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

import clips
import muse_sampler
import scorers
import server
from candidates import FEATURES
from clips import backend, load, record_clip
from tests.fakes import PHRASES, Gemini, SampleOpen, ScoreClosed, readers, v1_output

OLD_STATS_KEYS = {"labeled", "clips", "stage", "next_stage", "threshold", "camera_fps", "window", "accuracy", "history"}
LIST_KEYS = {"id", "t", "user_id", "mode", "stt", "top", "action", "correct", "pool_hit", "stage"}
DETAIL_KEYS = {"id", "t", "user_id", "mode", "stt", "config", "video", "frames", "stage", "action", "threshold", "pool",
               "samples", "closed", "gemini"}
CLIP_KEYS = {"clip", "stt", "label", "label_phrase", "read", "correct", "stats"}
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


def test_preheat_without_a_user_records_the_default_user(api, store, recording, monkeypatch):
    fakes = readers("yes", gemini="yes")
    monkeypatch.setattr(muse_sampler, "sample_open", fakes["sample_open"])
    monkeypatch.setattr(muse_sampler, "score_closed", fakes["score_closed"])
    monkeypatch.setattr(scorers, "gemini_scores", fakes["gemini"])
    monkeypatch.setattr(server, "transcribe", lambda data: "yes")

    assert api.post("/preheat/clip?fps=30&prompted=0", content=recording).status_code == 200

    assert [r["user_id"] for r in load(store)] == [clips.DEFAULT_USER]
