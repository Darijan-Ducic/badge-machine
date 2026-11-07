"""Per-phrase readiness from a predict-then-train history paired with each clip's STT text (GET /readiness payload).

History entries are `scorer.prequential` entries; only `fused` (VTT correct vs STT) matters here. Expected values
below are worked out by hand from the hand-built histories.
"""

import pytest

from readiness import ACC_MIN, N_MIN, WINDOW, readiness


def history_of(*said_and_correct):
    """[(stt, vtt correct), ...] -> (history entries, stt texts), in clip order."""
    return [{"fused": ok} for _, ok in said_and_correct], [stt for stt, _ in said_and_correct]


def by_text(payload):
    return {p["text"]: p for p in payload["phrases"]}


def test_defaults_follow_liplearner():
    assert (N_MIN, WINDOW, ACC_MIN) == (5, 5, 0.8)


def test_counts_clips_per_phrase():
    history, stts = history_of(("call mom", True), ("stop", False), ("call mom", True))

    phrases = by_text(readiness(history, stts))

    assert {text: p["n"] for text, p in phrases.items()} == {"call mom": 2, "stop": 1}


def test_accuracy_is_over_the_most_recent_window_only():
    # 6 clips: the first is right, the 2nd wrong, the last 4 right. Window 5 -> 4/5 = 0.8 exactly
    # (a window of 6 would give 5/6, a window of 4 would give 4/4).
    history, stts = history_of(*[("call mom", ok) for ok in (True, False, True, True, True, True)])

    (phrase,) = readiness(history, stts)["phrases"]

    assert phrase["n"] == 6
    assert phrase["vtt_acc"] == pytest.approx(0.8)


def test_ready_at_exactly_n_min_clips_and_acc_min_accuracy():
    history, stts = history_of(*[("call mom", ok) for ok in (True, False, True, True, True)])

    (phrase,) = readiness(history, stts)["phrases"]

    assert (phrase["n"], phrase["vtt_acc"], phrase["ready"]) == (5, pytest.approx(0.8), True)


def test_not_ready_below_n_min_even_when_always_right():
    history, stts = history_of(*[("stop", True)] * 4)

    (phrase,) = readiness(history, stts)["phrases"]

    assert (phrase["n"], phrase["vtt_acc"], phrase["ready"]) == (4, 1.0, False)


def test_not_ready_below_acc_min():
    history, stts = history_of(*[("stop", ok) for ok in (True, True, True, False, False)])

    (phrase,) = readiness(history, stts)["phrases"]

    assert (phrase["vtt_acc"], phrase["ready"]) == (pytest.approx(0.6), False)


def test_a_phrase_recovers_once_old_mistakes_leave_the_window():
    early_misses = [("stop", False)] * 5 + [("stop", True)] * 5

    (phrase,) = readiness(*history_of(*early_misses))["phrases"]

    assert (phrase["n"], phrase["vtt_acc"], phrase["ready"]) == (10, 1.0, True)


def test_stt_texts_are_matched_after_normalizing():
    history, stts = history_of(("Call mom.", True), ("call  MOM", False), ("call mom mom", True))

    (phrase,) = readiness(history, stts)["phrases"]

    assert (phrase["text"], phrase["n"]) == ("call mom", 3)


def test_clips_without_speech_text_are_not_a_phrase():
    history, stts = history_of((None, False), ("", False), ("∅", False), ("  ...  ", False), ("stop", True))

    assert [p["text"] for p in readiness(history, stts)["phrases"]] == ["stop"]


def test_knobs_override_the_defaults():
    history, stts = history_of(*[("stop", ok) for ok in (False, True, True)])

    strict = readiness(history, stts)["phrases"][0]
    relaxed = readiness(history, stts, n_min=3, window=2, acc_min=1.0)["phrases"][0]

    assert (strict["vtt_acc"], strict["ready"]) == (pytest.approx(2 / 3), False)
    assert (relaxed["vtt_acc"], relaxed["ready"]) == (1.0, True)


def test_unscored_clips_count_but_carry_no_accuracy():
    history = [{"fused": None}, {"fused": None}]

    (phrase,) = readiness(history, ["stop", "stop"])["phrases"]

    assert (phrase["n"], phrase["vtt_acc"], phrase["ready"]) == (2, None, False)


def test_payload_shape_most_clips_first():
    history, stts = history_of(("stop", True), ("call mom", True), ("call mom", True), ("yes please", False))

    payload = readiness(history, stts)

    assert set(payload) == {"phrases"}
    assert [p["text"] for p in payload["phrases"]] == ["call mom", "stop", "yes please"]
    for phrase in payload["phrases"]:
        assert set(phrase) == {"text", "n", "vtt_acc", "ready"}
        assert isinstance(phrase["n"], int) and isinstance(phrase["ready"], bool)
        assert isinstance(phrase["vtt_acc"], float)


def test_history_and_texts_must_pair_up():
    with pytest.raises(ValueError):
        readiness([{"fused": True}], ["stop", "stop"])
