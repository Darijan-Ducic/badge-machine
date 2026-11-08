"""Staged scorer over featurized clips (`candidates.Clip`): a conditional logit over each clip's own candidate pool.

Pools differ in size, so clips are padded into one tensor and masked (`fusion.fit` / `fusion.predict`). Free
weights grow with the number of STT-labeled clips, so the model can never out-fit its data:
  0+  defaults     DEFAULT_WEIGHTS, never auto-send
  10+ temperature  one fitted scale on the default score
  30+ core         fitted weights on CORE, every other feature 0
  60+ full         fitted weights on every feature

Default weights: each direct "this is the phrase" signal counts once — open-sample agreement (`vote`, `top_vote`,
`mean_score`), the closed call (`closed`) and Gemini (`gemini`) — and ∅ is backed only by the explicit reports that
nothing was said or nothing fits (`none_closed`, `no_speech`). Indirect or self-assessed signals (`slot_support`,
`viseme_sim`, `prior`, `in_bank`, `is_none`, `bad_quality`) start at 0 and must earn weight from labels.
`pick_history` (what the user's earlier takes with the same Gemini top pick turned out to be) starts at a tie-breaker
weight, PICK_HISTORY_WEIGHT: at most 0.1, it can't overturn a confident Gemini pick (1.0 vs ~0), but it orders the
candidates Gemini leaves near 0, where Gemini's own order is noise (replay: docs/iteration-log.md, 2026-11-07).

Auto-send: P(top) >= threshold and the top pick is not ∅. The threshold comes from out-of-fold predictions (5 folds),
counting only clips whose held-out top pick is a real candidate, since a ∅ pick is never sent.
"""

from dataclasses import dataclass

import numpy as np

from candidates import FEATURES, Clip, match
from fusion import fit, gate_threshold
from fusion import predict as fused_probs

STAGES = [(60, "full"), (30, "core"), (10, "temperature"), (0, "defaults")]
FOLDS = 5
CORE = ["vote", "top_vote", "mean_score", "closed", "gemini", "pick_history", "is_none", "none_closed", "bad_quality",
        "no_speech"]
PICK_HISTORY_WEIGHT = 0.1
DEFAULT_WEIGHTS = np.array([{"vote": 1.0, "top_vote": 1.0, "mean_score": 1.0, "closed": 1.0, "gemini": 1.0,
                             "pick_history": PICK_HISTORY_WEIGHT, "none_closed": 1.0, "no_speech": 1.0}.get(name, 0.0)
                            for name in FEATURES])
_CORE_COLUMNS = [FEATURES.index(name) for name in CORE]
TEMPERATURE_MIN = 1e-2  # the temperature scale stays positive, so it never reverses the default ranking


@dataclass
class Model:
    stage: str
    weights: np.ndarray  # one per feature; a candidate's score is features @ weights at every stage
    threshold: float


def stage_for(n_labeled: int, stages: list[tuple[int, str]] = STAGES) -> str:
    """`stages`: (minimum labeled clips, stage name) pairs in any order; one must start at 0."""
    return next(name for minimum, name in sorted(stages, reverse=True) if n_labeled >= minimum)


def _pad(clips: list[Clip]) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(c.pool) for c in clips)
    features = np.zeros((len(clips), width, len(FEATURES)))
    mask = np.zeros((len(clips), width), bool)
    for i, clip in enumerate(clips):
        features[i, :len(clip.pool)] = clip.features
        mask[i, :len(clip.pool)] = True
    return features, mask


def _fit(stage: str, features: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if stage == "defaults":
        return DEFAULT_WEIGHTS.copy()
    if stage == "temperature":
        scale = fit((features @ DEFAULT_WEIGHTS)[..., None], labels, mask=mask, bounds=[(TEMPERATURE_MIN, None)])[0]
        return scale * DEFAULT_WEIGHTS
    columns = _CORE_COLUMNS if stage == "core" else list(range(len(FEATURES)))
    weights = np.zeros(len(FEATURES))
    weights[columns] = fit(features[..., columns], labels, mask=mask)
    return weights


def predict(model: Model, clips: list[Clip]) -> list[np.ndarray]:
    """Per clip, a probability for each candidate in its pool (∅ last)."""
    if not clips:
        return []
    features, mask = _pad(clips)
    probs = fused_probs(features, model.weights, mask=mask)
    return [p[:len(c.pool)] for p, c in zip(probs, clips)]


def auto_send(model: Model, probs: np.ndarray) -> bool:
    return int(np.argmax(probs)) != len(probs) - 1 and float(np.max(probs)) >= model.threshold


def trains(clip: Clip) -> bool:
    """A labeled clip trains the phrase scorer, except a sentence turn whose STT missed the pool: a dictation sentence
    is almost never an exact candidate, so its ∅ label would only teach that ∅ wins."""
    return clip.label is not None and not (clip.sentence is not None and clip.label == len(clip.pool) - 1)


def train(clips: list[Clip], precision: float = 0.9, gate: bool = True, stages: list[tuple[int, str]] = STAGES
          ) -> Model:
    labeled = [c for c in clips if trains(c)]
    stage = stage_for(len(labeled), stages)
    if stage == "defaults":
        return Model(stage, DEFAULT_WEIGHTS.copy(), np.inf)
    features, mask = _pad(labeled)
    labels = np.array([c.label for c in labeled])
    model = Model(stage, _fit(stage, features, mask, labels), np.inf)
    if gate:
        # Pick the auto-send threshold on out-of-fold predictions so it isn't fooled by memorized clips.
        folds = np.arange(len(labels)) % FOLDS
        held_out = np.zeros(mask.shape)
        for k in range(FOLDS):
            rest = folds != k
            weights = _fit(stage, features[rest], mask[rest], labels[rest])
            held_out[~rest] = fused_probs(features[~rest], weights, mask=mask[~rest])
        sendable = held_out.argmax(1) != mask.sum(1) - 1
        model.threshold = float(gate_threshold(held_out[sendable], labels[sendable], precision))
    return model


def _matches(stt: str | None, text: str | None) -> bool:
    return stt is not None and text is not None and match(stt, [text]) is not None


def prequential(clips: list[Clip], stages: list[tuple[int, str]] = STAGES) -> list[dict]:
    """Score each labeled clip with a model trained only on earlier clips: the honest, climbing accuracy for the UI.

    fused: top non-∅ candidate matches STT and ∅ didn't win; gemini: Gemini's argmax phrase matches STT;
    muse: the most common top hypothesis across samples matches STT; pool_hit: STT text is in the pool.
    An unlabeled clip (a lip-mode clip before the user labels it) is scored, but all four are None.
    """
    history = []
    for i, clip in enumerate(clips):
        model = train(clips[:i], gate=False, stages=stages)
        probs = predict(model, [clip])[0]
        top, none = int(probs.argmax()), len(clip.pool) - 1
        labeled = clip.label is not None
        fused = _matches(clip.stt, clip.sentence) if clip.sentence is not None else (
            top != none and _matches(clip.stt, clip.pool[top]))
        history.append({
            "stage": model.stage,
            "fused": fused if labeled else None,
            "gemini": _matches(clip.stt, clip.gemini_pick) if labeled else None,
            "muse": _matches(clip.stt, clip.muse_pick) if labeled else None,
            "confidence": float(probs.max()),
            "pool_hit": clip.label != none if labeled else None,
        })
    return history
