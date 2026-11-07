import numpy as np
from scipy.special import log_softmax

from fusion import fit, gate_threshold, predict

N_PHRASES = 20


def simulated_model(labels, signal, sharpness, rng):
    """Log-probs from a fake lip-reading model: the true phrase's logit gets `signal` above unit Gaussian noise;
    `sharpness` multiplies the logits, so >1 makes the model overconfident without changing its top pick."""
    logits = rng.normal(size=(len(labels), N_PHRASES))
    logits[np.arange(len(labels)), labels] += signal
    return log_softmax(sharpness * logits, axis=1)


def accuracy(scores, labels):
    return (scores.argmax(1) == labels).mean()


def split(rng, n=2000):
    return rng.integers(0, N_PHRASES, n), rng.integers(0, N_PHRASES, n)


def test_fused_models_beat_the_best_single_model_on_held_out_clips():
    rng = np.random.default_rng(0)
    train_y, test_y = split(rng)
    train = np.stack([simulated_model(train_y, 2.0, 1, rng), simulated_model(train_y, 1.5, 1, rng)], -1)
    test = np.stack([simulated_model(test_y, 2.0, 1, rng), simulated_model(test_y, 1.5, 1, rng)], -1)

    fused = predict(test, fit(train, train_y))

    best_single = max(accuracy(test[..., 0], test_y), accuracy(test[..., 1], test_y))
    assert accuracy(fused, test_y) > best_single + 0.05


def test_uninformative_model_gets_near_zero_weight():
    rng = np.random.default_rng(1)
    y, _ = split(rng)
    features = np.stack([simulated_model(y, 2.0, 1, rng), simulated_model(y, 0.0, 1, rng)], -1)

    informative, noise = fit(features, y)

    assert informative > 0.5
    assert abs(noise) < 0.1 * informative


def test_overconfident_model_is_calibrated_on_held_out_clips():
    rng = np.random.default_rng(2)
    train_y, test_y = split(rng)
    train = simulated_model(train_y, 2.0, 10, rng)[..., None]
    test = simulated_model(test_y, 2.0, 10, rng)[..., None]
    raw_confidence = np.exp(test[..., 0]).max(1).mean()
    assert raw_confidence - accuracy(test[..., 0], test_y) > 0.3  # the fixture really is overconfident

    fused = predict(test, fit(train, train_y))

    assert accuracy(fused, test_y) == accuracy(test[..., 0], test_y)  # calibration rescales, never reranks
    assert abs(fused.max(1).mean() - accuracy(fused, test_y)) < 0.05


def test_gate_picks_lowest_confidence_that_keeps_precision():
    # Sorted by confidence: right, right, wrong, right, wrong -> running precision 1, 1, .67, .75, .6.
    probs = np.array([[.9, .05, .05], [.8, .1, .1], [.7, .2, .1], [.6, .3, .1], [.5, .3, .2]])
    labels = np.array([0, 0, 1, 0, 2])

    assert gate_threshold(probs, labels, precision=0.75) == 0.6


def test_gate_never_auto_sends_when_no_confidence_level_is_precise_enough():
    probs = np.array([[.9, .1], [.8, .2]])
    labels = np.array([1, 1])

    assert gate_threshold(probs, labels, precision=0.5) == np.inf
