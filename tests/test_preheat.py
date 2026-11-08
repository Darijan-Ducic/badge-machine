import numpy as np
from scipy.special import softmax

from preheat import _normalize, match_phrase, prequential, stage_for, train

PHRASES = ["yes", "no", "turn on the lights", "turn off the lights", "call mom", "play music"]
N_PHRASES = 24


def simulated_probs(labels, signal, sharpness, rng):
    """Phrase probabilities from a fake lip reader: true phrase's logit gets `signal` above unit Gaussian noise."""
    logits = rng.normal(size=(len(labels), N_PHRASES))
    logits[np.arange(len(labels)), labels] += signal
    return softmax(sharpness * logits, axis=1)


def fused_accuracy(history, start, stop, key="fused"):
    return np.mean([h[key] for h in history[start:stop]])


def test_match_ignores_case_and_punctuation():
    assert match_phrase("Turn on the lights.", PHRASES) == 2


def test_match_tolerates_a_word_doubled_by_stt():
    assert match_phrase("call call mom", PHRASES) == 4  # 0.76 similarity: fuzzy matching alone rejects it


def test_match_picks_the_closest_near_miss():
    assert match_phrase("turn of the lights", PHRASES) == 3


def test_match_rejects_speech_outside_the_vocabulary():
    assert match_phrase("what a lovely afternoon", PHRASES) is None


def test_stage_grows_with_labeled_clip_count():
    assert [stage_for(n) for n in (0, 9, 10, 29, 30, 59, 60)] == [
        "defaults", "defaults", "temperature", "temperature", "per-model", "per-model", "with-prior"]


def test_each_clip_is_scored_by_a_model_trained_only_on_earlier_clips():
    rng = np.random.default_rng(0)
    y = rng.integers(0, N_PHRASES, 12)
    history = prequential(simulated_probs(y, 2, 1, rng), simulated_probs(y, 2, 1, rng), y)

    assert history[9]["stage"] == "defaults"  # 9 earlier clips
    assert history[10]["stage"] == "temperature"  # 10 earlier clips


def test_accuracy_climbs_once_a_confidently_wrong_model_can_be_down_weighted():
    rng = np.random.default_rng(1)
    y = rng.integers(0, N_PHRASES, 160)
    gemini = simulated_probs(y, 2.5, 1, rng)
    muse = simulated_probs(y, 0.0, 5, rng)  # pure guessing, but loud about it

    history = prequential(gemini, muse, y)

    assert {h["stage"] for h in history[40:60]} == {"per-model"}
    assert fused_accuracy(history, 40, 60) > fused_accuracy(history, 0, 30) + 0.2
    assert fused_accuracy(history, 60, 160) >= fused_accuracy(history, 60, 160, key="gemini") - 0.05


def test_phrase_prior_helps_when_the_user_mostly_says_one_thing():
    rng = np.random.default_rng(2)
    y = np.where(rng.random(200) < 0.8, 0, rng.integers(1, N_PHRASES, 200))
    history = prequential(simulated_probs(y, 0.5, 1, rng), simulated_probs(y, 0.5, 1, rng), y)

    assert fused_accuracy(history, 100, 200, key="gemini") < 0.5
    assert fused_accuracy(history, 100, 200) >= 0.75


def test_never_auto_sends_before_training():
    rng = np.random.default_rng(3)
    y = rng.integers(0, N_PHRASES, 5)

    assert train(simulated_probs(y, 3, 1, rng), simulated_probs(y, 3, 1, rng), y).threshold == np.inf


def test_learns_an_auto_send_threshold_once_trained():
    rng = np.random.default_rng(4)
    y = rng.integers(0, N_PHRASES, 40)

    threshold = train(simulated_probs(y, 4, 1, rng), simulated_probs(y, 4, 1, rng), y).threshold

    assert 0 < threshold <= 1


def test_normalize_collapses_a_phrase_repeated_by_stt():
    # Measured on Obama Weekly Address labels: half of the STT label errors are repeated phrases ("road to road to").
    assert _normalize("It was a road to road to socialism") == "it was a road to socialism"
