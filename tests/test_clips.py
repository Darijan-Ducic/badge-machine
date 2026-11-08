"""The v1 clip path: raw outputs stored per clip, features recomputed at stats time, preheat reads and stats.

Default-stage scores below are hand-derived from scorer.DEFAULT_WEIGHTS: a real candidate scores
vote + top_vote + mean_score + closed + gemini, ∅ scores none_closed + no_speech.
"""

import hashlib
import json
from pathlib import Path

import pytest

import muse_sampler
import scorers
from clips import load, process_clip, record_clip, stats
from tests.fakes import (CLOSED_USAGE, GEMINI_USAGE, OPEN_USAGE, PHRASES, Gemini, SampleOpen, ScoreClosed, readers,
                         v1_output)

OLD_STATS_KEYS = {"labeled", "clips", "stage", "next_stage", "threshold", "camera_fps", "window", "accuracy", "history"}
STATS_KEYS = OLD_STATS_KEYS | {"user", "pool_recall", "stages"}
CLIP_KEYS = {"clip", "stt", "label", "label_phrase", "read", "correct", "stats"}
COSTS = Path(__file__).resolve().parent.parent / "data" / "viseme_costs.json"


def run(store, recording, said, *tops, phrases=PHRASES, fps=30.0, user="local", **kw):
    return process_clip(recording, fps, 0, transcribe=lambda _: said, phrases=phrases, store=store, user_id=user,
                        **readers(*tops, **kw))


def test_empty_stats_follow_the_frontend_contract():
    s = stats([], PHRASES)

    assert set(s) == STATS_KEYS
    assert (s["labeled"], s["clips"], s["stage"], s["threshold"], s["camera_fps"], s["history"]) == (0, 0, "defaults", None, None, [])
    assert s["next_stage"] == {"name": "temperature", "at": 10}
    assert s["accuracy"]["recent"] == {"fused": None, "gemini": None, "muse": None}
    assert (s["user"], s["pool_recall"]) == (None, None)
    # scorer.STAGES, lowest first
    assert s["stages"] == [{"name": "defaults", "at": 0}, {"name": "temperature", "at": 10},
                           {"name": "core", "at": 30}, {"name": "full", "at": 60}]


def test_a_clip_stores_raw_outputs_config_and_label_but_no_features(store, recording):
    fakes = readers("no", gemini="no")

    process_clip(recording, 24.0, 3, transcribe=lambda _: "no", phrases=PHRASES, store=store, **fakes)

    (record,) = load(store)
    assert fakes["sample_open"].calls[0][1] == 5, "K=5 for preheat"
    assert [s["usage"] for s in record["samples"]] == [OPEN_USAGE] * 5, "every sample stored with its usage"
    assert all(set(s) == {"call", "sample_idx", "latency_ms", "output", "error", "usage"} for s in record["samples"])
    assert record["samples"][0]["output"] == v1_output(("no", 0.6), ("hello", 0.2))
    assert record["closed"]["output"] == {"scores": {p: 0.05 for p in PHRASES}, "none": 0.1}
    assert record["closed"]["usage"] == CLOSED_USAGE
    assert record["gemini"]["output"] == pytest.approx({"yes": 0.1 / 3, "no": 0.9, "turn on the lights": 0.1 / 3,
                                                        "turn off the lights": 0.1 / 3})
    assert (record["gemini"]["error"], record["gemini"]["usage"]) == (None, GEMINI_USAGE)
    assert isinstance(record["gemini"]["latency_ms"], int)
    assert record["config"] == {**muse_sampler.CONFIG, "crop": "full-frame-v0", "k": 5,
                                "gemini_model": scorers.GEMINI_MODEL,
                                "viseme_costs": hashlib.sha256(COSTS.read_bytes()).hexdigest()[:12]}
    assert record["label"] == {"stt_text": "no", "user_edit": None}
    assert (record["user_id"], record["mode"], record["tracker_flags"]) == ("local", "stt_on", [])
    assert (record["fps"], record["prompted"], record["phrases"]) == (24.0, 3, PHRASES)
    assert record["frames"] == 12  # 0.5 s at 24 fps
    assert "features" not in json.dumps(record) and "pool" not in record


def test_a_failed_gemini_call_is_stored_as_an_error_not_raised(store, recording):
    def broken(clip, phrases, usage=None):
        raise TimeoutError("deadline exceeded")

    process_clip(recording, 30.0, 0, transcribe=lambda _: "yes", phrases=PHRASES, store=store,
                 sample_open=SampleOpen(v1_output(("yes", 0.6))), score_closed=ScoreClosed(), gemini=broken)

    (record,) = load(store)
    assert record["gemini"]["output"] is None
    assert record["gemini"]["error"] == "TimeoutError: deadline exceeded"


def test_preheat_reads_follow_the_spec(store, recording):
    # Muse tops: off, off, on, off, on -> "turn off the lights" with a 3/5 vote share. Defaults score
    # "turn on the lights" .4 + .4 + .24 + .8 (closed) + .9 (Gemini) = 2.74, above "turn off" (1.64) and "hello" (1.2).
    result = run(store, recording, "Turn on the lights.", "turn off the lights", "turn off the lights",
                 "turn on the lights", "turn off the lights", "turn on the lights",
                 closed={"turn on the lights": 0.8}, gemini="turn on the lights")

    assert set(result) == CLIP_KEYS
    assert (result["stt"], result["label"], result["label_phrase"]) == ("Turn on the lights.", 2, "turn on the lights")
    assert result["read"]["fused"]["phrase"] == "turn on the lights" and 0 < result["read"]["fused"]["p"] < 1
    assert result["read"]["gemini"] == {"phrase": "turn on the lights", "p": pytest.approx(0.9)}
    assert result["read"]["muse"] == {"phrase": "turn off the lights", "p": pytest.approx(0.6)}
    assert result["correct"] == {"fused": True, "gemini": True, "muse": False}
    assert (result["stats"]["labeled"], result["stats"]["camera_fps"]) == (1, 30.0)


def test_muse_vote_share_counts_only_the_samples_that_returned(store, recording):
    # K=5, samples 3 and 4 fail: valid tops yes, yes, no -> "yes" with 2 of 3 votes
    result = run(store, recording, "yes", "yes", "yes", "no", "no", "no", fail={3, 4})

    assert result["read"]["muse"] == {"phrase": "yes", "p": pytest.approx(2 / 3)}


def test_a_none_win_reads_as_none_of_these(store, recording):
    # No speech in any sample, closed "none" .95: ∅ scores .95 + 1 (no_speech) against at most .01 + .25 for a phrase.
    result = process_clip(recording, 30.0, 0, transcribe=lambda _: "yes", phrases=PHRASES, store=store,
                          sample_open=SampleOpen(v1_output(present=False)),
                          score_closed=ScoreClosed(none=0.95, default=0.01),
                          gemini=Gemini(probs=dict.fromkeys(PHRASES, 0.25)))

    assert result["read"]["fused"]["phrase"] == "∅ none of these"
    assert result["correct"]["fused"] is False


def test_off_list_speech_is_a_labeled_pool_miss(store, recording):
    result = run(store, recording, "what a lovely afternoon", "yes")

    assert (result["label"], result["label_phrase"]) == (None, None)
    assert result["correct"] == {"fused": False, "gemini": False, "muse": False}
    assert (result["stats"]["clips"], result["stats"]["labeled"], result["stats"]["pool_recall"]) == (1, 1, 0.0)


def test_accuracy_counts_every_labeled_clip(store, recording):
    run(store, recording, "yes", "no", gemini="yes")
    result = run(store, recording, "no", "no", gemini="yes")

    assert result["stats"]["accuracy"]["all"]["gemini"] == 0.5
    assert result["stats"]["accuracy"]["all"]["muse"] == 0.5
    assert [h["i"] for h in result["stats"]["history"]] == [0, 1]


def test_clips_from_a_different_phrase_list_are_ignored(store, recording):
    run(store, recording, "yes", "yes")

    assert stats(load(store), ["hello", "goodbye"])["clips"] == 0


def test_stats_are_per_user(store, recording):
    run(store, recording, "yes", "yes", user="ann")
    run(store, recording, "no", "no", user="ann")
    run(store, recording, "yes", "yes", user="bob")

    assert (stats(load(store), PHRASES, user="ann")["clips"], stats(load(store), PHRASES, user="bob")["clips"]) == (2, 1)
    assert stats(load(store), PHRASES, user="bob")["user"] == "bob"


def test_without_a_user_stats_are_for_the_user_of_the_latest_clip(store, recording):
    run(store, recording, "yes", "yes", user="ann")
    run(store, recording, "no", "no", user="bob")
    run(store, recording, "no", "no", user="ann")

    s = stats(load(store), PHRASES)

    assert (s["user"], s["clips"]) == ("ann", 2)


def test_the_bank_holds_only_texts_said_before_the_clip(store, recording):
    # Phrase list of 2, so the 3 bank entries nearest in visemes are the whole bank: a text in the bank is in the
    # pool. No model ever proposes "open sesame", so only the bank can: the first clip must be a pool miss (its own
    # text and the later clip's text are not in its bank), the second a hit (the first clip said it).
    for _ in range(2):
        record_clip(recording, phrases=["yes", "no"], user_id="local", mode="stt_on", k=3,
                    transcript="open sesame", store=store, **readers("yes"))

    s = stats(load(store), ["yes", "no"])

    assert [h["pool_hit"] for h in s["history"]] == [False, True]
    assert s["pool_recall"] == 0.5


def test_a_runtime_clip_without_stt_is_stored_unlabeled(store, recording):
    record_clip(recording, phrases=PHRASES, user_id="local", mode="runtime", k=3, store=store, **readers("yes"))

    (record,) = load(store)
    assert (record["mode"], record["label"]) == ("runtime", {"stt_text": None, "user_edit": None})
    assert (stats(load(store), PHRASES)["clips"], stats(load(store), PHRASES)["labeled"]) == (1, 0)


def test_a_config_override_is_logged_over_the_sampler_defaults(store, recording):
    record = record_clip(recording, phrases=PHRASES, user_id="local", mode="experiment", k=3, store=store,
                         config={"grid": "3x3", "fps": 12}, **readers("yes"))

    assert (record["config"]["grid"], record["config"]["fps"]) == ("3x3", 12)
    assert record["config"]["tile_width"] == muse_sampler.CONFIG["tile_width"]
    assert load(store)[0]["config"]["grid"] == "3x3"


# ---------------------------------------------------------------- prompted preheat labels


import clips  # noqa: E402


def _prompted_record(stt, prompted=0, phrases=("this is lip reading", "we built this today"), user_edit=None):
    return {"label": {"stt_text": stt, "user_edit": user_edit}, "prompted": prompted, "phrases": list(phrases)}


def test_a_misheard_prompted_take_is_labeled_with_the_line_on_screen():
    assert clips._label_text(_prompted_record("This is sleep reading")) == "this is lip reading"
    assert clips._label_text(_prompted_record("They built this today", prompted=1)) == "we built this today"


def test_a_prompted_take_that_says_something_else_keeps_its_stt_and_a_user_edit_still_wins():
    assert clips._label_text(_prompted_record("like the future of AI is here.")) == "like the future of AI is here."
    assert clips._label_text(_prompted_record("", prompted=1)) == ""
    assert clips._label_text(_prompted_record("This is sleep reading", user_edit="no sound at all")) == "no sound at all"
