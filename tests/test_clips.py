import numpy as np

from clips import load, process_clip, stats

PHRASES = ["yes", "no", "turn on the lights", "turn off the lights"]
STATS_KEYS = {"labeled", "clips", "stage", "next_stage", "threshold", "camera_fps", "window", "accuracy", "history"}
CLIP_KEYS = {"clip", "stt", "label", "label_phrase", "read", "correct", "stats"}


def peaked(index, n=len(PHRASES)):
    """Same shape the real scorers return: one probability per phrase, summing to 1."""
    probs = np.full(n, 0.1 / (n - 1))
    probs[index] = 0.9
    return probs


def run(tmp_path, said, gemini_pick, muse_pick, phrases=PHRASES, fps=30.0):
    return process_clip(b"webm-bytes", fps, 0, transcribe=lambda _: said, phrases=phrases, store=tmp_path,
                        gemini=lambda clip, ps: peaked(gemini_pick, len(ps)), muse=lambda clip, ps: peaked(muse_pick, len(ps)))


def test_empty_stats_follow_the_frontend_contract():
    s = stats([], PHRASES)

    assert set(s) == STATS_KEYS
    assert (s["labeled"], s["clips"], s["stage"], s["threshold"], s["camera_fps"], s["history"]) == (0, 0, "defaults", None, None, [])
    assert s["next_stage"] == {"name": "temperature", "at": 10}
    assert s["accuracy"]["recent"] == {"fused": None, "gemini": None, "muse": None}


def test_clip_is_labeled_by_stt_and_each_reader_is_scored_against_it(tmp_path):
    result = run(tmp_path, "Turn on the lights.", gemini_pick=2, muse_pick=3)

    assert set(result) == CLIP_KEYS
    assert (result["label"], result["label_phrase"]) == (2, "turn on the lights")
    assert result["read"]["gemini"]["phrase"] == "turn on the lights"
    assert result["read"]["muse"]["phrase"] == "turn off the lights"
    assert result["correct"]["gemini"] is True and result["correct"]["muse"] is False
    assert (result["stats"]["labeled"], result["stats"]["camera_fps"]) == (1, 30.0)
    assert len(load(tmp_path)) == 1


def test_off_list_speech_is_stored_but_not_scored(tmp_path):
    result = run(tmp_path, "what a lovely afternoon", gemini_pick=0, muse_pick=0)

    assert (result["label"], result["correct"]) == (None, None)
    assert (result["stats"]["clips"], result["stats"]["labeled"]) == (1, 0)


def test_accuracy_counts_every_labeled_clip(tmp_path):
    run(tmp_path, "yes", gemini_pick=0, muse_pick=1)
    result = run(tmp_path, "no", gemini_pick=0, muse_pick=1)

    assert result["stats"]["accuracy"]["all"]["gemini"] == 0.5
    assert result["stats"]["accuracy"]["all"]["muse"] == 0.5
    assert [h["i"] for h in result["stats"]["history"]] == [0, 1]


def test_clips_from_a_different_phrase_list_are_ignored(tmp_path):
    run(tmp_path, "yes", gemini_pick=0, muse_pick=0)

    assert stats(load(tmp_path), ["hello", "goodbye"])["clips"] == 0
