"""Text-level metrics, chronological splits and per-clip cost of the experiment harness, checked against values
worked out by hand.

Viseme codes used below (visemes-v1, CMUdict): "pat" = P AE T -> P A T; "bat" = B AE T -> P A T (a homophene: lips
can't show p vs b); "cat" = K AE T -> K A T.
"""

import pytest

import visemes
from experiment import PRICES, clip_cost, evaluate, splits, viseme_acc, word_acc
from tests.simulator import simulate


def test_the_hand_derivations_use_the_real_viseme_codes():
    assert visemes.to_visemes("pat") == visemes.to_visemes("bat") == ["P", "A", "T"]
    assert visemes.to_visemes("cat") == ["K", "A", "T"]


# ---- word accuracy = 1 - WER over normalized words, summed over clips ----

def test_word_acc_sums_word_edits_over_all_stt_words():
    # clip 1: no edits (2 words); clip 2: "some" deleted (3 words) -> 1 - 1/5
    assert word_acc(["call mom", "play music"], ["call mom", "play some music"]) == pytest.approx(0.8)


def test_word_acc_normalizes_case_and_punctuation():
    assert word_acc(["Call Mom!"], ["call mom"]) == 1.0


def test_no_vtt_text_deletes_every_stt_word():
    assert word_acc([""], ["where are my keys"]) == 0.0


def test_word_acc_is_one_minus_wer_even_below_zero():
    # 2 inserted words against a 1-word reference: WER 2
    assert word_acc(["stop it now"], ["stop"]) == pytest.approx(-1.0)


def test_word_acc_without_stt_words_is_none():
    assert word_acc(["stop"], [""]) is None


# ---- viseme accuracy = 1 - viseme edits / STT visemes, unit costs ----

def test_a_homophene_is_right_on_the_lips_but_wrong_as_a_word():
    assert viseme_acc(["pat"], ["bat"]) == 1.0
    assert word_acc(["pat"], ["bat"]) == 0.0


def test_viseme_acc_sums_viseme_edits_over_all_stt_visemes():
    # "pat" vs "bat": 0 edits of 3; "cat" vs "pat": K -> P, 1 edit of 3 -> 1 - 1/6
    assert viseme_acc(["pat", "cat"], ["bat", "pat"]) == pytest.approx(1 - 1 / 6)


def test_viseme_acc_uses_unit_costs_not_the_tuned_ones(monkeypatch):
    # Whatever the offline-tuned substitution costs say, the metric is a plain edit count.
    tuned = {"alphabet": "visemes-v1", "default_sub": 0.1, "ins": 0.1, "del": 0.1, "sub": {}}
    monkeypatch.setattr(visemes, "_default_costs", lambda: tuned)

    assert viseme_acc(["cat"], ["pat"]) == pytest.approx(1 - 1 / 3)


def test_no_vtt_text_misses_every_stt_viseme():
    assert viseme_acc([""], ["pat"]) == 0.0


# ---- chronological splits: 60 / 20 / 20 of the clips that have STT ----

def test_splits_are_chronological_60_20_20():
    records = [{"id": i, "stt": "stop"} for i in range(20)]

    parts = splits(records)

    assert {name: [r["id"] for r in part] for name, part in parts.items()} == {
        "train": list(range(12)), "heldout": [12, 13, 14, 15], "test": [16, 17, 18, 19]}


def test_splits_skip_clips_without_stt_and_round_down():
    records = [{"id": i, "stt": None if i in (2, 5) else "stop"} for i in range(9)]  # 7 labeled

    parts = splits(records)

    assert {name: [r["id"] for r in part] for name, part in parts.items()} == {
        "train": [0, 1, 3, 4], "heldout": [6], "test": [7, 8]}


# ---- cost per clip from the usage stored with each call ----

MUSE_IN, MUSE_OUT = PRICES["muse"]
GEMINI_IN, GEMINI_OUT = PRICES["gemini"]


def test_prices_per_million_tokens():
    assert PRICES == {"muse": (1.25, 4.25), "gemini": (2.0, 10.0)}


def test_clip_cost_adds_every_call_at_its_models_price():
    record = {
        "samples": [{"call": "open", "output": {}, "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 200_000}},
                    {"call": "open", "output": {}, "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 200_000}},
                    {"call": "open", "output": None, "error": "timeout"}],  # failed: no usage, no cost
        "closed": {"call": "closed", "output": {"scores": {}, "none": 0.1},
                   "usage": {"prompt_tokens": 2_000_000, "completion_tokens": 0}},
        "gemini": {"call": "gemini", "output": {"stop": 1.0},
                   "usage": {"prompt_token_count": 500_000, "candidates_token_count": 100_000,
                             "thoughts_token_count": 100_000}},
    }
    # Muse: 4M in, 0.4M out; Gemini: 0.5M in, 0.2M out (thinking is billed as output)
    expected = 4 * MUSE_IN + 0.4 * MUSE_OUT + 0.5 * GEMINI_IN + 0.2 * GEMINI_OUT

    assert clip_cost(record) == pytest.approx(expected)


def test_clip_cost_without_any_usage_is_none():
    (record,) = simulate(0, 1)

    assert clip_cost(record) is None


def test_cost_per_clip_is_the_mean_over_clips_with_usage():
    records = simulate(5, 10, to_visemes=visemes.to_visemes)
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    for i, record in enumerate(records):
        if i in (6, 7):  # the heldout split
            record["samples"] = [{**s, "usage": usage} for s in record["samples"][:i - 5]]  # 1 and 2 calls

    assert evaluate(records, {}, "heldout")["cost_per_clip"] == pytest.approx(1.5 * MUSE_IN)
    assert evaluate(records, {}, "train")["cost_per_clip"] is None


def test_wrapped_closed_and_gemini_calls_featurize_like_bare_outputs():
    # Stored clips keep closed/Gemini as call items {"call", "output", "error", "usage"}; the design-doc shape has
    # the bare outputs. Both must replay the same.
    bare = simulate(6, 10, to_visemes=visemes.to_visemes)
    wrapped = [{**r, "closed": {"call": "closed", "output": r["closed"], "error": None},
                "gemini": {"call": "gemini", "output": r["gemini"], "error": None}} for r in bare]

    for split in ("train", "heldout"):
        assert evaluate(wrapped, {}, split) == evaluate(bare, {}, split)
