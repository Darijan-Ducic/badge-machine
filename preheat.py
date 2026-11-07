"""Preheat: turn STT-labeled clips into training data and grow the fusion model in stages.

The model only gets more free parameters as labeled clips accumulate, so it can never out-fit its data:
  0+  defaults     product of the two models' probabilities; never auto-send (uncalibrated)
  10+ temperature  one shared weight on the summed log-probs
  30+ per-model    one weight per model, so a confidently wrong model can be switched off
  60+ with-prior   plus the user's own phrase frequencies
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from fusion import fit, gate_threshold
from fusion import predict as fused_probs

STAGES = [(60, "with-prior"), (30, "per-model"), (10, "temperature"), (0, "defaults")]
FOLDS = 5


@dataclass
class Model:
    stage: str
    weights: np.ndarray
    log_prior: np.ndarray | None
    threshold: float


def stage_for(n_labeled: int) -> str:
    return next(name for minimum, name in STAGES if n_labeled >= minimum)


def _normalize(text: str) -> str:
    words = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()
    # STT occasionally doubles a word ("seven seven"); collapse immediate repeats on both sides of the match.
    return " ".join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])


def match_phrase(transcript: str, phrases: list[str], min_ratio: float = 0.8) -> int | None:
    said = _normalize(transcript)
    candidates = [_normalize(p) for p in phrases]
    if said in candidates:
        return candidates.index(said)
    ratios = [SequenceMatcher(None, said, c).ratio() for c in candidates]
    best = int(np.argmax(ratios))
    return best if ratios[best] >= min_ratio else None


def _features(stage: str, gemini: np.ndarray, muse: np.ndarray, log_prior: np.ndarray | None) -> np.ndarray:
    g, m = np.log(np.clip(gemini, 1e-6, 1)), np.log(np.clip(muse, 1e-6, 1))
    if stage in ("defaults", "temperature"):
        return (g + m)[..., None]
    if stage == "per-model":
        return np.stack([g, m], -1)
    return np.stack([g, m, np.broadcast_to(log_prior, g.shape)], -1)


def _fit(stage: str, gemini: np.ndarray, muse: np.ndarray, labels: np.ndarray) -> Model:
    if stage == "defaults":
        return Model(stage, np.ones(1), None, np.inf)
    n_phrases = gemini.shape[1]
    log_prior = None
    if stage == "with-prior":  # Laplace-smoothed so unseen phrases stay possible
        log_prior = np.log((np.bincount(labels, minlength=n_phrases) + 1) / (len(labels) + n_phrases))
    weights = fit(_features(stage, gemini, muse, log_prior), labels)
    return Model(stage, weights, log_prior, np.inf)


def predict(model: Model, gemini: np.ndarray, muse: np.ndarray) -> np.ndarray:
    return fused_probs(_features(model.stage, gemini, muse, model.log_prior), model.weights)


def train(gemini: np.ndarray, muse: np.ndarray, labels: np.ndarray, precision: float = 0.9, gate: bool = True) -> Model:
    stage = stage_for(len(labels))
    model = _fit(stage, gemini, muse, labels)
    if gate and stage != "defaults":
        # Pick the auto-send threshold on out-of-fold predictions so it isn't fooled by memorized clips.
        folds = np.arange(len(labels)) % FOLDS
        held_out = np.zeros_like(gemini)
        for k in range(FOLDS):
            rest = folds != k
            held_out[~rest] = predict(_fit(stage, gemini[rest], muse[rest], labels[rest]), gemini[~rest], muse[~rest])
        model.threshold = gate_threshold(held_out, labels, precision)
    return model


def prequential(gemini: np.ndarray, muse: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Score each clip with a model trained only on earlier clips: the honest, climbing accuracy for the UI."""
    history = []
    for i, label in enumerate(labels):
        model = train(gemini[:i], muse[:i], labels[:i], gate=False)
        probs = predict(model, gemini[i:i + 1], muse[i:i + 1])[0]
        history.append({
            "stage": model.stage,
            "fused": bool(probs.argmax() == label),
            "gemini": bool(gemini[i].argmax() == label),
            "muse": bool(muse[i].argmax() == label),
            "confidence": float(probs.max()),
        })
    return history
