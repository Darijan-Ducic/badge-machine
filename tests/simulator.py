"""Test helper: fake viseme functions and a simulator of raw per-clip model outputs.

The real `visemes.py` (CMUdict + `visemes-v1` codes) is built elsewhere; tests inject these FAKES instead:
every letter is its own "viseme" code and distance is a plain normalized Levenshtein. They keep the contract
(`to_visemes(text) -> list[str]`, `viseme_distance(a, b, costs=None) -> float` in [0, 1]) and are easy to do by hand.
"""

import numpy as np
from scipy.special import softmax

from preheat import _normalize


def fake_to_visemes(text: str) -> list[str]:
    """FAKE `visemes.to_visemes`: letters as codes, spaces and punctuation dropped, consecutive repeats collapsed."""
    letters = [ch for ch in _normalize(text) if ch.isalpha()]
    return [ch for i, ch in enumerate(letters) if i == 0 or ch != letters[i - 1]]


def fake_viseme_distance(a: list[str], b: list[str], costs=None) -> float:
    """FAKE `visemes.viseme_distance`: unit-cost Levenshtein divided by the longer length (0 for two empties)."""
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / max(len(a), len(b))


# ---- simulator of raw per-clip outputs (schema v1 open samples, closed call, Gemini) ----

PHRASES = ["call mom", "play some music", "what time is it", "open the door", "set a timer", "stop", "yes please",
           "no thanks", "where are my keys", "turn on the lights", "good morning", "send a text"]
OFF_BANK = ["feed the cat", "how is the weather", "bring me water", "i need help", "wake me at seven",
            "read my email", "find my phone", "order a pizza"]  # said on pool-miss clips; never offered to any model
BANK = {"phrases": PHRASES, "counts": {p: 2 for p in PHRASES}}
BAD_QUALITY = [{"mouth_visible": "partial", "blur": "medium"}, {"mouth_visible": "full", "blur": "high"},
               {"mouth_visible": "none", "blur": "low"}]


def _open_output(rng, truth, idx, muse, visemes, bad_quality, no_speech, to_visemes, alphabet):
    logits = rng.normal(size=len(PHRASES))
    if idx is not None:
        logits[idx] += muse
    probs = softmax(logits)
    first, second = np.argsort(-probs)[:2]
    hypotheses = [{"text": PHRASES[i], "s": round(float(probs[i]), 3)} for i in (first, second)]
    top_words, other_words = PHRASES[first].split(), PHRASES[second].split()
    slots = []
    for i, word in enumerate(top_words):
        alts = [{"w": word, "s": hypotheses[0]["s"]}]
        if i < len(other_words) and other_words[i] != word:
            alts.append({"w": other_words[i], "s": hypotheses[1]["s"]})
        slots.append({"f": [4 + 10 * i, 13 + 10 * i], "alts": alts})
    if rng.random() < visemes:
        codes = to_visemes(truth)
    else:
        codes = list(rng.choice(alphabet, size=rng.integers(4, 15)))
    quality = (BAD_QUALITY[rng.integers(len(BAD_QUALITY))] if rng.random() < bad_quality
               else {"mouth_visible": "full", "blur": str(rng.choice(["low", "medium"]))})
    return {
        "schema_version": "v1",
        "speech": {"present": bool(rng.random() >= no_speech), "frames": [4, 4 + 10 * len(top_words)]},
        "quality": {**quality, "occluded_frames": []},
        "visemes": [{"v": str(v), "f": [4 + 3 * i, 6 + 3 * i]} for i, v in enumerate(["REST", *codes, "REST"])],
        "slots": slots,
        "hypotheses": hypotheses,
    }


def simulate(seed, n, *, k=5, muse=1.0, closed=1.0, gemini=1.0, visemes=0.0, miss_rate=0.0, bad_quality=0.05,
             miss_bad_quality=0.9, no_speech=0.05, fail=0.0, to_visemes=fake_to_visemes) -> list[dict]:
    """`n` STT-labeled clips of raw outputs, each `{"samples", "closed", "gemini", "bank", "stt"}`.

    Signal knobs (logit boost of the true phrase over unit Gaussian noise): `muse` for each open sample's
    hypotheses and slots, `closed` for the closed call (whose true answer is "none" on a pool miss), `gemini`.
    `visemes`: probability an open sample's viseme string is the truth's (else random codes).
    `miss_rate`: fraction of clips whose phrase is off-bank, so no model can propose it (label ∅).
    `bad_quality` / `miss_bad_quality`: per-sample probability of a bad-quality report on normal / miss clips.
    `no_speech`: per-sample probability of `speech.present` false (independent noise). `fail`: per-sample failure.
    """
    rng = np.random.default_rng(seed)
    alphabet = sorted({c for p in PHRASES + OFF_BANK for c in to_visemes(p)})
    clips = []
    for _ in range(n):
        miss = rng.random() < miss_rate
        truth = str(rng.choice(OFF_BANK if miss else PHRASES))
        idx = None if miss else PHRASES.index(truth)
        samples = []
        for j in range(k):
            output = None if rng.random() < fail else _open_output(
                rng, truth, idx, muse, visemes, miss_bad_quality if miss else bad_quality, no_speech, to_visemes,
                alphabet)
            samples.append({"call": "open", "sample_idx": j, "latency_ms": int(rng.integers(1500, 4000)),
                            "output": output, "error": None if output else "timeout"})
        closed_logits = rng.normal(size=len(PHRASES) + 1)  # last entry is "none of these"
        closed_logits[len(PHRASES) if miss else idx] += closed
        closed_scores = softmax(closed_logits)
        gemini_logits = rng.normal(size=len(PHRASES))
        if idx is not None:
            gemini_logits[idx] += gemini
        clips.append({
            "samples": samples,
            "closed": {"scores": {p: round(float(s), 4) for p, s in zip(PHRASES, closed_scores)},
                       "none": round(float(closed_scores[-1]), 4)},
            "gemini": {p: round(float(s), 4) for p, s in zip(PHRASES, softmax(gemini_logits))},
            "bank": BANK,
            "stt": truth,
        })
    return clips


def featurized(raw: list[dict]) -> list:
    """Run each raw clip through `candidates.featurize` with the fake viseme functions."""
    from candidates import featurize
    return [featurize(**clip, to_visemes=fake_to_visemes, viseme_distance=fake_viseme_distance) for clip in raw]
