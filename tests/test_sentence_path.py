"""The sentence path: a dictation turn (its samples read sentences longer than any phrase) gets the ROVER sentence of
its samples' top hypotheses as VTT, and its pool miss does not teach the phrase scorer that ∅ wins.

Found on real turns (practice:obama-weekly:obama, 46 turns): every label was ∅ because a 10-25-word turn is never
an exact pool candidate, so the temperature stage fitted a negative scale and VTT picked zero-score candidates.
"""

import numpy as np

import experiment
from candidates import FEATURES, featurize
from scorer import DEFAULT_WEIGHTS, predict, train
from tests.fakes import PHRASES, readers, v1_output
from tests.test_scorer import hand_clip
from tests.test_turns import segment, turn  # noqa: F401 (segment is a fixture)

# ROVER over these tops (s order: s1, s2, s3) is "we need to do more today", which no single hypothesis says.
S1, S2, S3 = "we need to do more tonight", "we need to go more today", "we must to do more today"
ROVER = "we need to do more today"


def sentence_samples(*tops: tuple[str, float]) -> list[dict]:
    return [{"call": "open", "sample_idx": i, "output": v1_output((text, s), ("hello", 0.1)), "error": None}
            for i, (text, s) in enumerate(tops)]


def featurized(samples, stt=None, phrases=PHRASES):
    return featurize(samples, {"scores": dict.fromkeys(phrases, 0.05), "none": 0.1}, dict.fromkeys(phrases, 0.25),
                     {"phrases": phrases, "counts": {}}, stt)


# ---------------------------------------------------------------- (a) the temperature scale stays positive


def test_temperature_scale_stays_positive_when_every_label_is_none():
    # default scores: "a" 2.5, "b" 0, ∅ 0.3. Fitting P(∅) unconstrained wants a negative scale, which ranks "b" first.
    clips = [hand_clip([{"vote": 1, "top_vote": 1, "mean_score": 0.5}, {}], {"none_closed": 0.3}, label=2,
                       pool=["a", "b"]) for _ in range(12)]

    model = train(clips)

    assert model.stage == "temperature"
    scale = model.weights[FEATURES.index("vote")] / DEFAULT_WEIGHTS[FEATURES.index("vote")]
    assert scale > 0
    np.testing.assert_allclose(model.weights, scale * DEFAULT_WEIGHTS)
    assert int(np.argmax(predict(model, clips[:1])[0])) == 0


# ---------------------------------------------------------------- (b) sentence pool misses don't train the scorer


def _with_sentence(clip, text="we need to do more today"):
    clip.sentence = text
    return clip


def test_a_sentence_turn_that_misses_the_pool_does_not_train_the_scorer():
    phrase = [hand_clip([{"vote": 1}], {"none_closed": 0.1}, label=0) for _ in range(10)]
    misses = [_with_sentence(hand_clip([{"vote": 1}], {"none_closed": 0.1}, label=1)) for _ in range(20)]

    assert train(misses).stage == "defaults"
    assert train(phrase + misses).stage == "temperature"  # 10 trainable clips, not 30


def test_a_sentence_turn_that_hits_the_pool_still_trains():
    hits = [_with_sentence(hand_clip([{"vote": 1}], {"none_closed": 0.1}, label=0)) for _ in range(30)]

    assert train(hits).stage == "core"


# ---------------------------------------------------------------- (c) which clips take the sentence path


def test_a_reading_longer_than_every_phrase_gets_the_rover_sentence_of_the_top_hypotheses():
    clip = featurized(sentence_samples((S1, 0.6), (S2, 0.5), (S3, 0.4)))

    assert clip.sentence == ROVER


def test_a_reading_no_longer_than_the_longest_phrase_stays_on_the_phrase_path():
    four = featurized(sentence_samples(("turn on the lights", 0.6), ("turn off the lights", 0.5)))
    five = featurized(sentence_samples(("turn on the lights now", 0.6), ("turn off the lights now", 0.5)))

    assert four.sentence is None  # the longest phrase has 4 words
    assert five.sentence is not None


def test_the_length_gate_is_the_median_over_samples():
    short = featurized(sentence_samples(("yes", 0.6), ("no", 0.5), (S1, 0.4)))

    assert short.sentence is None


def test_a_clip_without_valid_samples_stays_on_the_phrase_path():
    clip = featurized([{"call": "open", "sample_idx": 0, "output": None, "error": "boom"}])

    assert clip.sentence is None


# ---------------------------------------------------------------- (c) the sentence is the VTT


def test_evaluate_scores_the_sentence_as_vtt_on_sentence_turns():
    record = {"samples": sentence_samples((S1, 0.6), (S2, 0.5), (S3, 0.4)), "closed": None, "gemini": None,
              "bank": {"phrases": PHRASES, "counts": {}}, "stt": ROVER}

    heldout = experiment.evaluate([record] * 5, None, "heldout")

    assert heldout["word_acc"] == 1.0
    assert heldout["sentence_acc"] == 1.0


def test_a_dictation_turn_reads_back_the_sentence_as_its_vtt(store, segment):
    fakes = readers()
    fakes["sample_open"].outputs = tuple(v1_output((text, s), ("hello", 0.1)) for text, s in
                                         ((S1, 0.6), (S2, 0.5), (S3, 0.4), (S1, 0.6), (S2, 0.5)))

    result = turn(store, segment, transcript=ROVER, fakes=fakes)

    assert result["vtt"]["text"] == ROVER
    assert result["correct"]["vtt"] is True


def test_a_phrase_turn_still_reads_back_the_pool_pick(store, segment):
    result = turn(store, segment, transcript="yes", fakes=readers("yes", gemini="yes"))

    assert result["vtt"]["text"] == "yes"
