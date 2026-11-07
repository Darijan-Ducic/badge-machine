"""Test helper: a simulated clip stream and a FAKE evaluator with the experiment.evaluate contract.

`SimEvaluator.evaluate(records, config, split, limit=None) -> M` replays `evaluator.replay` (simulator raw outputs →
routing/context/tools knobs → candidates.featurize) and scores with scorer.train/predict, like experiment.py does:
  train    predict-then-train over the train split (each clip read by a model trained only on earlier clips)
  heldout  one model trained on every clip before the split (gated, so it has an auto-send threshold)
  test     same, trained on train + heldout
Metrics are fractions. Prices per call follow docs/training-v1.md §8 ($0.02 per Muse open sample, $0.01 closed,
$0.03 Gemini, $0.01 crop). A prompt hint (skills) cannot change stored outputs, so here it has no effect: the fake
only charges for it, the way a real re-call would.
"""

import numpy as np

import evaluator
import scorer
import visemes
from candidates import match
from preheat import _normalize
from tests.simulator import OFF_BANK, PHRASES, simulate
from versions import config_hash, default_config

PRICES = {"open": 0.02, "closed": 0.01, "gemini": 0.03, "crop": 0.01}


def world(n=400, seed=7, **sim) -> list[dict]:
    """`n` chronological STT-labeled clips with ids, real visemes-v1 codes, and some off-bank (pool-miss) phrases."""
    knobs = {"muse": 0.3, "closed": 1.0, "gemini": 0.3, "visemes": 0.4, "miss_rate": 0.2, **sim}
    raw = simulate(seed, n, to_visemes=visemes.to_visemes, **knobs)
    return [{"id": f"clip-{i:04d}", "t": 1_700_000_000 + 60 * i, **r} for i, r in enumerate(raw)]


def weak_baseline() -> dict:
    """A deliberately under-tuned start: one Muse sample of the five stored, and no closed call."""
    config = default_config()
    config["k"], config["sources"] = 1, ["open", "gemini"]
    return config


def _edits(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _cost(record: dict, config: dict) -> float:
    routed = evaluator.route(record, config)
    return (PRICES["crop"] + PRICES["open"] * len(routed["samples"]) + PRICES["closed"] * (routed["closed"] is not None)
            + PRICES["gemini"] * (routed["gemini"] is not None))


class SimEvaluator:
    def __init__(self):
        self.calls = []  # (split, config hash, limit), in call order
        self.failure_calls = []  # ids of the records each failures() call was given
        self._clip = {}  # (config hash, record id) -> Clip: replay is per record, so splits share it
        self._preq = {}  # (config hash, record ids) -> predict-then-train probabilities

    def _clips(self, records, config):
        h = config_hash(config)
        missing = [r for r in records if (h, r["id"]) not in self._clip]
        for record, clip in zip(missing, evaluator.replay(missing, config)):
            self._clip[(h, record["id"])] = clip
        return [self._clip[(h, r["id"])] for r in records]

    def _prequential(self, records, config):
        key = (config_hash(config), tuple(r["id"] for r in records))
        if key not in self._preq:
            self._preq[key] = evaluator.prequential_probs(self._clips(records, config), config["stages"])
        return self._preq[key]

    def failures(self, train_records, config):
        self.failure_calls.append([r.get("id") for r in train_records])
        labeled = [r for r in train_records if r.get("stt") is not None]
        return evaluator.failures_from(labeled, self._clips(labeled, config), self._prequential(labeled, config))

    def evaluate(self, records, config, split, limit=None):
        self.calls.append((split, config_hash(config), limit))
        labeled = [r for r in records if r.get("stt") is not None]
        clips = self._clips(labeled, config)
        lo, hi = evaluator.bounds(len(labeled), split)
        if limit is not None:
            hi = min(hi, lo + limit)
        if split == "train":
            probs = self._prequential(labeled[:hi], config)[lo:]
            model = None
        else:
            model = evaluator.fit(clips[:lo], config["stages"], gate=True)
            probs = scorer.predict(model, clips[lo:hi])
        scored = clips[lo:hi]
        word_err = word_ref = vis_err = vis_ref = 0
        exact, top3, sent, sent_ok = [], [], [], []
        for clip, p in zip(scored, probs):
            none = len(clip.pool) - 1
            top = int(np.argmax(p))
            vtt = "" if top == none else clip.pool[top]
            ref = _normalize(clip.stt).split()
            word_err += _edits(_normalize(vtt).split(), ref)
            word_ref += len(ref)
            truth_codes = visemes.to_visemes(clip.stt)
            vis_err += _edits(visemes.to_visemes(vtt), truth_codes)
            vis_ref += len(truth_codes)
            ok = top != none and match(clip.stt, [clip.pool[top]]) is not None
            exact.append(ok)
            ranked = [i for i in np.argsort(-p) if i != none][:3]
            top3.append(match(clip.stt, [clip.pool[i] for i in ranked]) is not None)
            if model is not None:
                send = scorer.auto_send(model, p)
                sent.append(send)
                if send:
                    sent_ok.append(ok)
        costs = [_cost(r, config) for r in labeled[lo:hi]]
        return {
            "clips": len(scored),
            "word_acc": 1 - word_err / word_ref,
            "sentence_acc": float(np.mean(exact)),
            "viseme_acc": 1 - vis_err / vis_ref,
            "top3_acc": float(np.mean(top3)),
            "autosend_precision": float(np.mean(sent_ok)) if sent_ok else None,
            "autosend_coverage": float(np.mean(sent)) if model is not None else None,
            "cost_per_clip": float(np.mean(costs)),
        }


def _pool_miss_texts(context, n=8):
    """Off-bank STT texts the failure summary shows as pool misses (what a reader of the summary would add)."""
    texts = []
    for cluster in context["failures"]["clusters"]:
        if cluster["type"] == "pool_miss":
            texts += [c["stt"] for c in cluster["top_confusions"]]
    texts += [e["stt"] for e in context["failures"]["examples"] if not e["stt_in_pool"]]
    seen = []
    for t in texts:
        if t not in seen and t not in PHRASES:
            seen.append(t)
    return seen[:n] or None


def _add_misses(context):
    texts = _pool_miss_texts(context)
    return None if texts is None else {"change": {"op": "curate_bank", "add": texts, "drop": []},
                                       "hypothesis": "Pool misses are phrases the bank lacks; adding them lets "
                                                     "bank-nearest candidates reach them.",
                                       "failures_cited": [e["id"] for e in context["failures"]["examples"]
                                                          if not e["stt_in_pool"]][:5]}


def _p(change, hypothesis):
    return {"change": change, "hypothesis": hypothesis, "failures_cited": []}


# A long scripted session: some changes truly help on this world, many don't. Order matters for the staircase.
DEMO_SCRIPT = [
    _p({"op": "set_feature", "feature": "vote", "on": False}, "vote duplicates top_vote; dropping it cuts noise"),
    _p({"op": "set_sources", "sources": ["open", "closed", "gemini"]}, "ranked-wrong clips: add the closed call"),
    _p({"op": "set_pool_cap", "pool_cap": 3}, "a small pool gives the scorer fewer distractors"),
    _p({"op": "set_k", "k": 3}, "one sample is noisy; votes across 3 samples should agree on the truth"),
    _p({"op": "set_viseme_cost", "pair": "P|F", "cost": 0.5}, "p/b/m and f/v are confused on the lips"),
    _add_misses,
    _p({"op": "set_prior_strength", "prior_strength": 2.0}, "lean on the user's phrase history"),
    _p({"op": "set_k", "k": 5}, "more samples, steadier votes"),
    _p({"op": "set_prompt_hint", "hint": "Lips pressed together mean p, b or m; list all three."},
       "slot alternatives miss bilabial look-alikes"),
    _p({"op": "set_gemini_gate", "below": 0.6}, "Gemini only helps when Muse samples disagree"),
    _p({"op": "curate_bank", "add": [], "drop": [PHRASES[0]]}, "a rarely said phrase steals votes"),
    _p({"op": "set_stages", "temperature": 5, "core": 15, "full": 40}, "unlock fitted weights sooner"),
    _p({"op": "set_feature", "feature": "bad_quality", "on": False}, "quality reports are unreliable"),
    _p({"op": "set_viseme_cost", "pair": "T|S", "cost": 0.4}, "t/d/n/l and s/z look close"),
    _p({"op": "set_pool_cap", "pool_cap": 8}, "trim the long tail of the pool"),
    _p({"op": "set_sources", "sources": ["open", "closed"]}, "Gemini is near noise on this user"),
    _p({"op": "set_k", "k": 4}, "four samples may be enough"),
]

__all__ = ["DEMO_SCRIPT", "OFF_BANK", "PHRASES", "SimEvaluator", "weak_baseline", "world"]


def staircase_ok(rows: list[dict]) -> bool:
    values = [r["current_word_acc"] for r in rows]
    return all(b >= a for a, b in zip(values, values[1:]))

