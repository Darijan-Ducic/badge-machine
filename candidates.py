"""Candidate pool + features + label for one clip, recomputed from its stored raw model outputs (replay is free).

Implements "Post-processor spec (exact)" in docs/uncertainty-design.md. Inputs per clip:
  samples  muse_sampler open-call records `{"call", "sample_idx", "latency_ms", "output", "error"}`; `output` is a
           schema-v1 dict, or None for a failed call (skipped)
  closed   the closed call's output `{"scores": {phrase: s}, "none": s}`, or None if it failed
  gemini   `{phrase: p}` over the phrase list, or None / {} if unavailable
  bank     `{"phrases": [phrase-list entries], "counts": {past STT text: count}}`
  stt      the STT transcript when STT is on (gives the label), else None

Readings chosen where the spec leaves room (simplest one each time):
  - The pool cap of 24 includes ∅, so at most 23 real candidates, kept in source order: hypotheses, slot
    recombinations, closed top 3, Gemini top 3, bank nearest 3. The phrase-list ones (closed, Gemini, bank) are
    reserved: they always get a slot, and open candidates fill only the slots left. Empty texts are not candidates.
  - `sentence` / `sentence_p`: the sentence path (sentence.read) when the samples read a sentence longer than every
    phrase, else None / 0.
  - "Matches" is exact normalized text equality (no fuzzy fallback; see `match`) for votes, the closed/Gemini
    lookups, phrase-list membership, the user count and the label. For `mean_score` the matched hypothesis's `s` is
    used (the first exact match when a sample repeats a phrase). `slot_support` compares single words exactly.
  - Bank entries are the phrase list plus past STT texts, deduped by normalized text, phrase list first; bank size
    = their number; total = sum of counts (counts whose texts normalize alike are merged).
  - Bank-nearest ties keep bank order; with no valid samples every distance ties, so the first 3 entries are taken.
  - Fractions and means over zero valid samples are 0 (including `viseme_sim`).
  - A sample with no `mouth_visible` counts as bad quality (it is not "full"); only `speech.present is False`
    counts as no speech.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import log, prod

import numpy as np

from preheat import _normalize
from sentence import read as read_sentence

FEATURES = ["vote", "top_vote", "mean_score", "slot_support", "viseme_sim", "closed", "gemini", "prior", "in_bank",
            "is_none", "none_closed", "bad_quality", "no_speech"]
NONE = "∅"
POOL_CAP = 24


@dataclass
class Clip:
    pool: list[str]  # ∅ last
    features: np.ndarray  # (len(pool), len(FEATURES))
    label: int | None  # index into pool; ∅'s index on a pool miss; None without STT
    stt: str | None
    gemini_pick: str | None  # Gemini's argmax phrase
    muse_pick: str | None  # most common top hypothesis across valid samples
    sentence: str | None = None  # the sentence path's VTT (sentence.read) on a sentence turn, else None
    sentence_p: float = 0.0  # its confidence


def match(text: str, options: list[str]) -> int | None:
    """Index of the option `text` equals after normalizing, or None.

    Exact only: a fuzzy fallback (ratio >= .85) merged opposite commands like "turn on/off the lights" (.92), sharing
    their votes and mislabeling clips. A near miss is a pool miss, which the ∅ candidate is there to absorb.
    """
    said = _normalize(text)
    return next((i for i, option in enumerate(options) if _normalize(option) == said), None)


def _top(scores: dict, n: int) -> list[str]:
    return [text for text, _ in sorted(scores.items(), key=lambda item: -item[1])[:n]]


def _recombinations(output: dict) -> list[str]:
    slots = [sorted(s["alts"], key=lambda a: -a["s"])[:2] for s in output.get("slots") or [] if s.get("alts")]
    if not slots:
        return []
    combos = sorted(product(*slots), key=lambda combo: -prod(a["s"] for a in combo))[:3]
    return [" ".join(a["w"] for a in combo) for combo in combos]


def _observed(output: dict) -> list[str]:
    codes = [v["v"] for v in output.get("visemes") or [] if v["v"] != "REST"]
    return [c for i, c in enumerate(codes) if i == 0 or c != codes[i - 1]]


def _top_hypothesis(output: dict) -> str | None:
    hypotheses = output.get("hypotheses") or []
    return max(hypotheses, key=lambda h: h["s"])["text"] if hypotheses else None


def _bad_quality(output: dict) -> bool:
    if _dropped(output, "quality"):
        return False  # unknown, not bad
    quality = output.get("quality") or {}
    return quality.get("mouth_visible") != "full" or quality.get("blur") == "high"


def _dropped(output: dict, field: str) -> bool:
    """True when a schema variant (muse_sampler.SCHEMAS, schema_version other than v1) has no `field` by design; its
    features then use the replacement noted at each use. v1 outputs never take this path."""
    return output.get("schema_version") not in (None, "v1") and field not in output


def _no_speech(output: dict) -> bool:
    if _dropped(output, "speech"):
        return not output.get("hypotheses")  # variant prompts: no visible speech -> empty hypotheses
    return (output.get("speech") or {}).get("present") is False


def _observed_or_read(output: dict, to_visemes) -> list[str]:
    if _dropped(output, "visemes"):  # the top hypothesis's own visemes stand in for the observed ones
        top = _top_hypothesis(output)
        return to_visemes(_normalize(top)) if top else []
    return _observed(output)


def _alt_words(output: dict) -> set[str]:
    if _dropped(output, "slots"):  # every word of every hypothesis stands in for the slot alternatives
        return {w for h in output.get("hypotheses") or [] for w in _normalize(h["text"]).split()}
    return {w for s in output.get("slots") or [] for a in s.get("alts") or [] for w in _normalize(a["w"]).split()}


def featurize(samples: list[dict], closed: dict | None, gemini: dict | None, bank: dict, stt: str | None = None, *,
              to_visemes=None, viseme_distance=None) -> Clip:
    if to_visemes is None or viseme_distance is None:
        import visemes
        to_visemes = to_visemes or visemes.to_visemes
        viseme_distance = viseme_distance or visemes.viseme_distance

    outputs = [s["output"] for s in samples if s.get("output") is not None]
    closed_scores = (closed or {}).get("scores") or {}
    gemini = gemini or {}
    phrases = list(bank.get("phrases") or [])
    counts: dict[str, int] = {}
    for text, n in (bank.get("counts") or {}).items():
        counts[_normalize(text)] = counts.get(_normalize(text), 0) + n
    entries: dict[str, str] = {}  # normalized -> first-seen text; phrase list first, then past STT texts
    for text in phrases + list((bank.get("counts") or {})):
        entries.setdefault(_normalize(text), text)
    entries.pop("", None)
    observed = [_observed_or_read(o, to_visemes) for o in outputs]
    hypotheses = [[h["text"] for h in o.get("hypotheses") or []] for o in outputs]
    alt_words = [_alt_words(o) for o in outputs]

    def nearness(entry: str) -> float:
        codes = to_visemes(entry)
        return min((viseme_distance(codes, seen) for seen in observed), default=np.inf)  # stable sort keeps ties

    pool: dict[str, str] = {}  # normalized -> first-seen text
    opened = [h for hs in hypotheses for h in hs] + [r for o in outputs for r in _recombinations(o)]
    listed = _top(closed_scores, 3) + _top(gemini, 3) + sorted(entries.values(), key=nearness)[:3]
    reserved = {_normalize(text) for text in listed} - {""}  # phrase-list candidates always get a slot
    room = POOL_CAP - 1 - len(reserved)  # slots left for open candidates that are not reserved
    for text in opened + listed:
        key = _normalize(text)
        if key and key not in pool and (key in reserved or room > 0):
            pool[key] = text.strip()
            room -= key not in reserved

    k = len(outputs)
    tops = [_top_hypothesis(o) for o in outputs]
    log_denominator = log(sum(counts.values()) + len(entries) + 1)

    def mean(values) -> float:
        return sum(values) / k if k else 0.0

    def lookup(text: str, scores: dict) -> float:
        at = match(text, list(scores))
        return float(list(scores.values())[at]) if at is not None else 0.0

    def real(text: str) -> list[float]:
        words, codes = _normalize(text).split(), to_visemes(_normalize(text))
        found = [match(text, hs) for hs in hypotheses]
        return [
            mean(i is not None for i in found),
            mean(top is not None and match(text, [top]) is not None for top in tops),
            mean(o["hypotheses"][i]["s"] if i is not None else 0 for o, i in zip(outputs, found)),
            mean(sum(w in seen for w in words) / len(words) for seen in alt_words),
            1 - mean(viseme_distance(codes, seen) for seen in observed) if k else 0.0,
            lookup(text, closed_scores),
            lookup(text, gemini),
            log(lookup(text, counts) + 1) - log_denominator,
            float(match(text, phrases) is not None),
            0.0, 0.0, 0.0, 0.0,
        ]

    none_row = [0.0] * 9 + [
        1.0,
        float((closed or {}).get("none") or 0.0),
        mean(_bad_quality(o) for o in outputs),
        mean(_no_speech(o) for o in outputs),
    ]
    texts = list(pool.values())
    features = np.array([real(t) for t in texts] + [none_row], dtype=float)
    texts.append(NONE)

    label = None
    if stt is not None:
        hit = match(stt, texts[:-1])
        label = hit if hit is not None else len(texts) - 1
    named = [t for t in tops if t is not None]
    votes = Counter(map(_normalize, named))
    muse_pick = max(named, key=lambda t: votes[_normalize(t)]) if named else None  # ties: first seen
    gemini_pick = _top(gemini, 1)[0] if gemini else None
    read = read_sentence(outputs, phrases)
    return Clip(texts, features, label, stt, gemini_pick, muse_pick, *(read or (None, 0.0)))
