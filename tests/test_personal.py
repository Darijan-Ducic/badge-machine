"""personal.py: personal layers built from ONE user's EARLIER labeled clips only (predict-then-train).

The no-leak tests poison what a clip must not see (its own label, every later clip, other users' clips) and assert
the clip's reads don't move; the control test changes an earlier label and asserts they do, so the history is used.
"""

import copy
import json

import pytest

import personal
import visemes
from tests.fakes import v1_output

USER, OTHER = "practice:u", "practice:other"


def _sample(output, i=0):
    return {"call": "open", "sample_idx": i, "latency_ms": 1, "output": output, "error": None}


def _slots_output(slots, hypotheses):
    """A v1 open output with these slots ([(word, s), ...] per slot) and hypotheses [(text, s), ...]."""
    out = v1_output(*hypotheses)
    out["slots"] = [{"f": [i, i], "alts": [{"w": w, "s": s} for w, s in alts]} for i, alts in enumerate(slots)]
    return out


def _record(i, stt, *, user=USER, hypotheses=(("we need to see", 0.5),), slots=None, frames=60):
    output = _slots_output(slots, hypotheses) if slots else v1_output(*hypotheses)
    return {"record": "v1", "id": f"c{i}", "user_id": user, "mode": "stt_on", "vocab": "v", "frames": frames,
            "phrases": ["yes", "no"], "label": {"stt_text": stt, "user_edit": None},
            "samples": [_sample(output, k) for k in range(3)],
            "closed": {"call": "closed", "output": {"scores": {"yes": 0.1, "no": 0.1}, "none": 0.5}, "error": None},
            "gemini": {"call": "gemini", "output": {"yes": 0.5, "no": 0.5}, "error": None}}


# ---------------------------------------------------------------- history


def test_earlier_is_the_users_labeled_clips_strictly_before_the_clip():
    records = [_record(0, "a b"), _record(1, "c d", user=OTHER), _record(2, None), _record(3, "e f"),
               _record(4, "g h"), _record(5, "i j")]
    assert [r["id"] for r in personal.earlier(records, 4)] == ["c0", "c3"]


def test_earlier_with_last_keeps_only_the_most_recent_n():
    records = [_record(i, f"w{i}") for i in range(6)]
    assert [r["id"] for r in personal.earlier(records, 5, last=2)] == ["c3", "c4"]


def test_the_label_is_the_users_edit_when_there_is_one():
    record = _record(0, "heard text")
    record["label"]["user_edit"] = "edited text"
    assert personal.label(record) == "edited text"


# ---------------------------------------------------------------- lift: score vs own STT minus vs the others'


def test_a_constant_read_has_zero_mean_lift():
    stts = ["we need to make sure", "the middle class", "hi everybody this week", "to the people of"]
    lifts = personal.lift(["the to of we"] * 4, stts, "word")
    assert sum(lifts) / len(lifts) == pytest.approx(0.0, abs=1e-12)


def test_reading_the_own_turn_right_lifts_above_reading_another_turn():
    stts = ["we need to make sure", "the middle class families", "hi everybody this week"]
    right = personal.lift(stts, stts, "viseme")
    shifted = personal.lift(stts[1:] + stts[:1], stts, "viseme")
    assert min(right) > 0 > max(shifted)


def test_length_matched_lift_gives_a_filler_of_the_right_length_no_credit():
    stts = ["a b", "c d", "e f g h i j", "k l m n o p"]
    filler = [" ".join(f"q{c}" for c in "uvwxyz"[:len(s.split())]) for s in stts]  # distinct: no doubling collapse
    assert max(personal.lift(filler, stts, "word")) > 1  # plain lift pays for matching the turn's length
    assert personal.lift(filler, stts, "word", nearest=1) == [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------- personal language prior


def test_lm_prefers_a_word_order_the_user_said_over_an_unseen_order():
    lm = personal.LM(["we need to make sure", "we need to act now"])
    assert lm.logprob("we need to make sure") > lm.logprob("sure make to need we")


def test_lm_gives_unknown_words_a_finite_logprob_below_known_ones():
    lm = personal.LM(["middle class families"])
    unknown, known = lm.logprob("zebra"), lm.logprob("class")
    assert unknown > float("-inf") and unknown < known


def test_empty_lm_scores_every_text_alike_per_word():
    lm = personal.LM([])
    assert lm.per_word("we need") == pytest.approx(lm.per_word("middle class"))


def test_medoid_is_the_text_with_the_best_mean_word_accuracy_against_the_others():
    # "we need to act": 1.0 vs its twin + 0.75 vs "go" = 1.75 beats "we need to go": 0.75 + 0.75
    assert personal.medoid(["we need to act", "we need to act", "we need to go"]) == "we need to act"


def test_medoid_of_nothing_is_empty():
    assert personal.medoid([]) == ""


def test_slot_decode_without_the_prior_takes_each_slots_best_alt():
    samples = [_sample(_slots_output([[("we", 0.6), ("he", 0.3)], [("seed", 0.5), ("need", 0.4)]],
                                     [("we seed", 0.5)]))]
    assert personal.decode_slots(samples, personal.LM(["we need"]), weight=0.0) == "we seed"


def test_slot_decode_with_the_prior_picks_the_lookalike_the_user_says():
    samples = [_sample(_slots_output([[("we", 0.6), ("he", 0.3)], [("seed", 0.5), ("need", 0.4)]],
                                     [("we seed", 0.5)]))]
    assert personal.decode_slots(samples, personal.LM(["we need", "we need more"]), weight=1.0) == "we need"


def test_rescore_with_the_prior_picks_the_hypothesis_in_the_users_words():
    samples = [_sample(v1_output(("i love you more", 0.5), ("middle class families", 0.4)))]
    lm = personal.LM(["the middle class families", "middle class jobs"])
    assert personal.rescore(samples, lm, weight=0.0) == "i love you more"
    assert personal.rescore(samples, lm, weight=1.0) == "middle class families"


# ---------------------------------------------------------------- personal viseme costs


def test_confusion_costs_lower_a_pair_the_user_confuses_and_stay_loadable(tmp_path):
    pairs = [(["P", "A", "T"], ["F", "A", "T"])] * 6
    costs = personal.confusion_costs(pairs, min_count=5)
    assert costs["sub"]["F|P"] < 1
    assert "K|T" not in costs["sub"]
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(costs))
    assert visemes.load_costs(path)["alphabet"] == visemes.ALPHABET_VERSION


def test_confusion_costs_ignore_a_pair_seen_fewer_than_min_count():
    costs = personal.confusion_costs([(["P"], ["F"])] * 2, min_count=5)
    assert costs["sub"] == {}


# ---------------------------------------------------------------- personal scorer target


def test_scorer_target_is_the_pool_candidate_with_the_best_word_accuracy():
    pool = ["i love you", "we need to make sure", "we need", "∅"]
    assert personal.best_candidate(pool, "we need to make sure everyone") == 1


def test_scorer_target_is_none_when_every_candidate_ties():
    assert personal.best_candidate(["zebra", "lights", "∅"], "we need to act") is None


# ---------------------------------------------------------------- no leak through the arms


def _stream():
    phrases = ["we need to make sure", "middle class families", "make sure everyone", "hi everybody"]
    records = []
    for i in range(14):
        said = phrases[i % len(phrases)]
        records.append(_record(i, said, hypotheses=((said if i % 3 else "i love you", 0.5), ("we need to", 0.3)),
                               slots=[[("we", 0.5), ("he", 0.4)], [("need", 0.5), ("seed", 0.4)]],
                               frames=40 + i))
    return records


def test_a_clips_reads_ignore_its_own_label_and_every_later_clip():
    records = _stream()
    i = 10
    before = personal.Personal(records).reads(i)
    poisoned = copy.deepcopy(records)
    poisoned[i]["label"]["stt_text"] = "middle class families"
    for r in poisoned[i + 1:]:
        r["label"]["stt_text"] = "middle class families"
        r["samples"] = [_sample(v1_output(("middle class families", 0.9)), k) for k in range(3)]
    poisoned += [_record(20 + k, "middle class families") for k in range(5)]
    assert personal.Personal(poisoned).reads(i) == before


def test_a_clips_reads_ignore_other_users_clips():
    records = _stream()
    i = 10
    before = personal.Personal(records).reads(i)
    # "he seed" is a label that does move the reads when it is this user's (the next test)
    mixed = records[:3] + [_record(50 + k, "he seed", user=OTHER) for k in range(6)] + records[3:]
    assert personal.Personal(mixed).reads(i + 6) == before


def test_a_clips_reads_use_its_earlier_labels():
    records = _stream()
    i = 10
    before = personal.Personal(records).reads(i)
    changed = copy.deepcopy(records)
    for r in changed[:i]:
        r["label"]["stt_text"] = "he seed"
    assert personal.Personal(changed).reads(i) != before


def test_reads_with_last_n_ignore_clips_older_than_the_window():
    records = _stream()
    i = 12
    before = personal.Personal(records, last=4).reads(i)
    changed = copy.deepcopy(records)
    for r in changed[:i - 4]:  # older than the 4-clip window: may differ freely
        r["label"]["stt_text"] = "zebra crossing lights"
    # features of clip i's own pool may count older labels in the bank prior; the personal layers must not
    assert {k: v for k, v in personal.Personal(changed, last=4).reads(i).items() if k in personal.WINDOWED} == \
        {k: v for k, v in before.items() if k in personal.WINDOWED}
