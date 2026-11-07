"""Staged candidate scorer, trained on simulated raw outputs run through the real featurizer.

Reference points measured on the simulator alone (no scorer involved), which the thresholds below lean on:
  "visemes carry the signal" world (muse/closed/gemini 0.5, visemes 0.8): the documented default weights pick the
  STT phrase 34% of the time, Gemini 20%; argmax of `viseme_sim` alone 100%.
  all-miss world with bad-quality reports (signals 1.5, miss_bad_quality 0.9): default weights pick ∅ on 1%.
"""

import numpy as np
import pytest

from candidates import FEATURES, NONE, Clip
from scorer import CORE, DEFAULT_WEIGHTS, Model, auto_send, predict, prequential, stage_for, train
from tests.simulator import featurized, simulate


def rate(history, start, stop, key="fused"):
    return np.mean([h[key] for h in history[start:stop]])


def hand_clip(real_rows, none_row, label, stt="go on", pool=None, gemini_pick=None, muse_pick=None):
    """A featurized clip built by hand: rows are {feature: value} dicts, missing features are 0; ∅ goes last."""
    rows = [*real_rows, {"is_none": 1.0, **none_row}]
    features = np.array([[row.get(name, 0.0) for name in FEATURES] for row in rows])
    pool = (pool or [f"phrase {i}" for i in range(len(real_rows))]) + [NONE]
    return Clip(pool, features, label, stt, gemini_pick, muse_pick)


def test_stage_grows_with_labeled_clip_count():
    assert [stage_for(n) for n in (0, 9, 10, 29, 30, 59, 60)] == [
        "defaults", "defaults", "temperature", "temperature", "core", "core", "full"]


def test_defaults_stage_uses_the_hand_set_weights():
    model = train(featurized(simulate(0, 5)))

    assert model.stage == "defaults"
    assert (model.weights == DEFAULT_WEIGHTS).all()


def test_temperature_stage_only_rescales_the_default_score():
    model = train(featurized(simulate(1, 20, muse=2, closed=2, gemini=2)))

    assert model.stage == "temperature"
    scale = model.weights[FEATURES.index("vote")] / DEFAULT_WEIGHTS[FEATURES.index("vote")]
    np.testing.assert_allclose(model.weights, scale * DEFAULT_WEIGHTS)
    assert scale != pytest.approx(1.0)


def test_core_stage_fits_only_the_core_features():
    model = train(featurized(simulate(2, 40, muse=2, closed=2, gemini=2, visemes=0.5, miss_rate=0.2)))

    assert model.stage == "core"
    assert CORE == ["vote", "top_vote", "mean_score", "closed", "gemini", "is_none", "none_closed", "bad_quality",
                    "no_speech"]
    outside = [i for i, name in enumerate(FEATURES) if name not in CORE]
    assert (model.weights[outside] == 0).all()


def test_predicts_a_distribution_over_each_clips_own_pool():
    small = hand_clip([{"vote": 1.0}], {}, label=0)
    big = hand_clip([{"vote": 1.0}, {"closed": 0.5}, {}, {}], {"none_closed": 0.2}, label=0)

    probs = predict(train([]), [small, big])

    assert [len(p) for p in probs] == [2, 5]
    np.testing.assert_allclose([p.sum() for p in probs], [1.0, 1.0])
    # defaults: vote 1 vs ∅'s none_closed weight on 0 -> scores [1, 0] -> softmax e/(e+1)
    assert probs[0][0] == pytest.approx(np.e / (np.e + 1))


def test_each_clip_is_scored_by_a_model_trained_only_on_earlier_clips():
    history = prequential(featurized(simulate(3, 12, muse=2, closed=2, gemini=2)))

    assert history[9]["stage"] == "defaults"  # 9 earlier clips
    assert history[10]["stage"] == "temperature"  # 10 earlier clips


def test_history_reports_the_spec_accuracy_definitions():
    # ∅ is the right answer here and wins, but `fused` only credits a correct transcription.
    miss = hand_clip([{"vote": 0.1}], {"none_closed": 3.0}, label=1, stt="go on", gemini_pick="Go on.",
                     muse_pick="no")
    hit = hand_clip([{"vote": 1.0}, {"vote": 0.2}], {}, label=0, stt="Go on!", pool=["go on", "no"],
                    gemini_pick="no", muse_pick="go on")

    first, second = prequential([miss, hit])

    assert first == {"stage": "defaults", "fused": False, "gemini": True, "muse": False,
                     "confidence": pytest.approx(np.exp(3) / (np.exp(3) + np.exp(0.1))), "pool_hit": False}
    assert (second["fused"], second["gemini"], second["muse"], second["pool_hit"]) == (True, False, True, True)


def test_accuracy_climbs_once_a_feature_the_defaults_ignore_carries_the_signal():
    clips = featurized(simulate(4, 140, muse=0.5, closed=0.5, gemini=0.5, visemes=0.8))

    history = prequential(clips)

    assert {h["stage"] for h in history[80:]} == {"full"}
    assert rate(history, 0, 60) < 0.5  # defaults/temperature/core cannot use viseme_sim
    assert rate(history, 80, 140) > rate(history, 0, 60) + 0.3
    assert rate(history, 80, 140) > rate(history, 80, 140, key="gemini") + 0.3


def test_pure_noise_feature_ends_near_zero_weight():
    # no_speech reports are coin flips, independent of whether the phrase is in the pool.
    clips = featurized(simulate(5, 200, muse=1, closed=1, gemini=1, visemes=0.5, miss_rate=0.3, no_speech=0.5))

    model = train(clips)

    assert model.stage == "full"
    weights = dict(zip(FEATURES, model.weights))
    assert abs(weights["no_speech"]) < 0.1 * abs(weights["bad_quality"])
    assert abs(weights["no_speech"]) < 0.1 * abs(weights["viseme_sim"])


def test_none_wins_more_on_bad_quality_pool_misses_after_training():
    knobs = dict(muse=1.5, closed=1.5, gemini=1.5, visemes=0.3)
    training = featurized(simulate(6, 80, miss_rate=0.3, **knobs))
    misses = featurized(simulate(7, 40, miss_rate=1.0, **knobs))
    assert all(c.pool[c.label] == NONE for c in misses)  # the fixture really is all pool misses

    def none_wins(model):
        return np.mean([p.argmax() == len(p) - 1 for p in predict(model, misses)])

    before, after = none_wins(train([])), none_wins(train(training))

    assert before < 0.2
    assert after > before + 0.5


def test_never_auto_sends_before_training():
    assert train([]).threshold == np.inf
    assert train(featurized(simulate(8, 9, muse=3, closed=3, gemini=3))).threshold == np.inf


def test_learns_an_auto_send_threshold_once_trained():
    threshold = train(featurized(simulate(9, 40, muse=3, closed=3, gemini=3))).threshold

    assert 0 < threshold <= 1


def test_none_top_pick_never_auto_sends():
    model = Model("full", DEFAULT_WEIGHTS, threshold=0.5)

    assert auto_send(model, np.array([0.9, 0.05, 0.05]))
    assert not auto_send(model, np.array([0.4, 0.3, 0.3]))  # below threshold
    assert not auto_send(model, np.array([0.05, 0.05, 0.9]))  # ∅ is last


def test_threshold_ignores_confident_none_picks():
    # Pool-hit clips: two indistinguishable candidates, the label is always the second -> every real pick is wrong.
    # Miss clips: ∅ backed by the closed call's "none" -> ∅ picked confidently and correctly, but never sendable.
    hits = [hand_clip([{"vote": 1.0}, {"vote": 1.0}], {}, label=1) for _ in range(10)]
    misses = [hand_clip([{}, {}], {"none_closed": 1.0}, label=2) for _ in range(10)]
    clips = [c for pair in zip(hits, misses) for c in pair]

    model = train(clips)

    assert model.stage == "temperature"
    assert model.threshold == np.inf
