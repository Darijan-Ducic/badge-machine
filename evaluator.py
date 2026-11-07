"""Adapter between the self-improvement loop (iterate.py) and the experiment harness (experiment.py).

The loop codes against one small interface, and tests inject a fake with the same shape:
  evaluate(records, config, split, limit=None) -> M   M = the 8 contract metrics, fractions (see METRICS)
  failures(train_records, config) -> [failure]        clips the config reads wrong, with their candidate pools
  unsupported(config) -> str | None                   optional: why this evaluator cannot run the config

ExperimentEvaluator is the production one. experiment.evaluate(records, config, split) knows only the
post-processing knobs costs, drop, stages and pool_cap (it raises on any other key), so the routing and context knobs
are applied HERE, to the raw records, before the call (`route`): k keeps the first K stored open samples, sources
removes the calls left out, gemini_below removes Gemini when the K samples' top hypotheses agree at least that much,
bank adds/drops phrases. prior_strength scales a feature column and prompt_hint needs paid re-calls; experiment can
do neither, so `unsupported` names them and the loop refuses those changes before evaluating anything. A paid pilot
needs `limit` (the first N clips of the split); experiment.evaluate has none, so pilots raise PilotUnsupported.

Splits: chronological over the records with STT: train = first n*3//5, heldout = up to n*4//5, test = the rest — the
same cut experiment.py makes. `replay` / `failures` are this adapter's own free replay (candidates.featurize +
scorer) for the failure summary the proposer reads: predict-then-train over only the TRAIN records it is given, so a
failure is a clip that a model not trained on it got wrong. They never decide adoption; experiment.evaluate does.
"""

import inspect
from collections import Counter
from dataclasses import replace
from functools import partial

import numpy as np

import scorer
import visemes
from candidates import FEATURES, NONE, Clip, featurize, match
from preheat import _normalize

SPLITS = ("train", "heldout", "test")
METRICS = ("clips", "word_acc", "sentence_acc", "viseme_acc", "top3_acc", "autosend_precision", "autosend_coverage",
           "cost_per_clip")
EXPERIMENT_KNOBS = ("costs", "drop", "stages", "pool_cap")
SHOWN_CANDIDATES = 5


class PilotUnsupported(RuntimeError):
    """The evaluator cannot score just the first N clips of a split, so a paid change cannot be piloted."""


# ---------------------------------------------------------------- splits + metrics


def bounds(n: int, split: str) -> tuple[int, int]:
    """[start, end) of `split` among `n` chronological labeled clips."""
    cuts = {"train": (0, n * 3 // 5), "heldout": (n * 3 // 5, n * 4 // 5), "test": (n * 4 // 5, n)}
    if split not in cuts:
        raise ValueError(f"split {split!r} must be one of {SPLITS}")
    return cuts[split]


def split(records: list[dict], name: str) -> list[dict]:
    labeled = [r for r in records if r.get("stt") is not None]
    lo, hi = bounds(len(labeled), name)
    return labeled[lo:hi]


def _fraction(metrics: dict, key: str, *, low: float | None = 0.0, nullable: bool = False):
    value = metrics[key]
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value > 1 or (low is not None and value < low):
        raise ValueError(f"metric {key} = {value!r} must be a fraction (<= 1); percent scale is refused")
    return float(value)


def check_metrics(metrics: dict) -> dict:
    """The 8 contract metrics of an evaluator result, range-checked. 1 - WER may go below 0 (inserted words)."""
    if missing := [k for k in METRICS if k not in metrics]:
        raise ValueError(f"evaluator result lacks {missing}")
    clips, cost = metrics["clips"], metrics["cost_per_clip"]
    if isinstance(clips, bool) or not isinstance(clips, int) or clips < 0:
        raise ValueError(f"clips = {clips!r} must be a count")
    if cost is not None and (isinstance(cost, bool) or not isinstance(cost, int | float) or cost < 0):
        raise ValueError(f"cost_per_clip = {cost!r} must be dollars >= 0 or None")
    return {
        "clips": clips,
        "word_acc": _fraction(metrics, "word_acc", low=None),
        "sentence_acc": _fraction(metrics, "sentence_acc"),
        "viseme_acc": _fraction(metrics, "viseme_acc", low=None),
        "top3_acc": _fraction(metrics, "top3_acc"),
        "autosend_precision": _fraction(metrics, "autosend_precision", nullable=True),
        "autosend_coverage": _fraction(metrics, "autosend_coverage", nullable=True),
        "cost_per_clip": None if cost is None else float(cost),
    }


# ---------------------------------------------------------------- routing + context, on raw records


def _bare(item):
    """A stored call item {call, output, error, usage} -> its output; a bare output passes through."""
    return item.get("output") if isinstance(item, dict) and "call" in item and "output" in item else item


def agreement(samples: list[dict]) -> float:
    """Share of the valid open samples whose top hypothesis is the most common one (0 with none)."""
    tops = []
    for sample in samples:
        hypotheses = (sample.get("output") or {}).get("hypotheses") or []
        if hypotheses:
            tops.append(_normalize(max(hypotheses, key=lambda h: h["s"])["text"]))
    return Counter(tops).most_common(1)[0][1] / len(tops) if tops else 0.0


def _curated(bank: dict, edit: dict) -> dict:
    dropped = {_normalize(p) for p in edit["drop"]}
    phrases = [p for p in bank.get("phrases") or [] if _normalize(p) not in dropped]
    have = {_normalize(p) for p in phrases}
    phrases += [p for p in edit["add"] if _normalize(p) not in have]
    counts = {text: n for text, n in (bank.get("counts") or {}).items() if _normalize(text) not in dropped}
    return {**bank, "phrases": phrases, "counts": counts}


def route(record: dict, config: dict) -> dict:
    """`record` as the config's routing and context would have produced it; the stored record is not edited."""
    sources = config["sources"]
    opens = [s for s in record.get("samples") or [] if s.get("call", "open") == "open"]
    samples = opens[:config["k"]] if "open" in sources else []
    gemini = record.get("gemini") if "gemini" in sources else None
    if gemini is not None and config["gemini_below"] is not None and agreement(samples) >= config["gemini_below"]:
        gemini = None
    return {**record, "samples": samples, "closed": record.get("closed") if "closed" in sources else None,
            "gemini": gemini, "bank": _curated(record.get("bank") or {}, config["bank"])}


def _cap(clip: Clip, cap: int) -> Clip:
    # Candidate rows don't depend on each other, so keeping the first cap-1 equals featurizing with that cap.
    if len(clip.pool) <= cap:
        return clip
    keep = cap - 1
    label = clip.label if clip.label is None or clip.label < keep else keep  # a cut-off answer is a pool miss
    return replace(clip, pool=clip.pool[:keep] + clip.pool[-1:], label=label,
                   features=np.vstack([clip.features[:keep], clip.features[-1:]]))


def replay(records: list[dict], config: dict, *, to_visemes=None, viseme_distance=None) -> list[Clip]:
    """Featurize every record under `config` (routing, context, viseme costs, pool cap, dropped features, prior)."""
    to_visemes = to_visemes or visemes.to_visemes
    distance = partial(viseme_distance or visemes.viseme_distance, costs=config["costs"])
    dropped = [FEATURES.index(name) for name in config["drop"]]
    prior = FEATURES.index("prior")
    clips = []
    for record in records:
        r = route(record, config)
        clip = featurize(r["samples"], _bare(r["closed"]), _bare(r["gemini"]), r["bank"], r.get("stt"),
                         to_visemes=to_visemes, viseme_distance=distance)
        clip = _cap(clip, config["pool_cap"])
        clip.features[:, prior] *= config["prior_strength"]
        clip.features[:, dropped] = 0.0
        clips.append(clip)
    return clips


# ---------------------------------------------------------------- scoring under a stage plan


def fit(clips: list[Clip], stages: dict, gate: bool = False) -> scorer.Model:
    """scorer.train under the config's stage plan (passed through if scorer takes `stages`, else swapped in)."""
    plan = sorted(((minimum, name) for name, minimum in stages.items()), reverse=True)
    if "stages" in inspect.signature(scorer.train).parameters:
        return scorer.train(clips, gate=gate, stages=plan)
    saved, scorer.STAGES = scorer.STAGES, plan
    try:
        return scorer.train(clips, gate=gate)
    finally:
        scorer.STAGES = saved


def prequential_probs(clips: list[Clip], stages: dict) -> list[np.ndarray]:
    """Each clip's candidate probabilities from a model trained only on the clips before it."""
    return [scorer.predict(fit(clips[:i], stages), [clip])[0] for i, clip in enumerate(clips)]


def failures(train_records: list[dict], config: dict, **viseme_fns) -> list[dict]:
    """Clips among `train_records` that the config reads wrong (VTT != STT), with their top candidates."""
    labeled = [r for r in train_records if r.get("stt") is not None]
    clips = replay(labeled, config, **viseme_fns)
    return failures_from(labeled, clips, prequential_probs(clips, config["stages"]))


def failures_from(records: list[dict], clips: list[Clip], probs: list[np.ndarray]) -> list[dict]:
    """The failures among already replayed and scored clips (records, clips and probs aligned)."""
    found = []
    for i, (record, clip, probs) in enumerate(zip(records, clips, probs)):
        none, top = len(clip.pool) - 1, int(np.argmax(probs))
        if top != none and match(clip.stt, [clip.pool[top]]) is not None:
            continue
        shown = np.argsort(-probs)[:SHOWN_CANDIDATES]
        found.append({"id": str(record.get("id", i)), "stt": clip.stt, "vtt": NONE if top == none else clip.pool[top],
                      "stt_in_pool": clip.label != none,
                      "candidates": [{"text": clip.pool[j], "p": round(float(probs[j]), 3)} for j in shown]})
    return found


# ---------------------------------------------------------------- production: experiment.evaluate


def to_experiment(config: dict) -> dict:
    """The knobs experiment.evaluate accepts; routing/context knobs are already applied to the records by `route`."""
    return {key: config[key] for key in EXPERIMENT_KNOBS}


class ExperimentEvaluator:
    def unsupported(self, config: dict) -> str | None:
        reasons = []
        if config["prior_strength"] != 1.0:
            reasons.append("experiment.evaluate has no prior_strength knob")
        if config["prompt_hint"]:
            reasons.append("a prompt hint needs paid re-calls, and experiment.evaluate cannot pilot (no limit=)")
        return "; ".join(reasons) or None

    def evaluate(self, records: list[dict], config: dict, split: str, limit: int | None = None) -> dict:
        import experiment  # built in parallel; imported on first use so this module loads without it

        extra = {}
        if limit is not None:
            if "limit" not in inspect.signature(experiment.evaluate).parameters:
                raise PilotUnsupported("experiment.evaluate has no limit=, so a 20-clip pilot is impossible")
            extra["limit"] = limit
        routed = [route(r, config) for r in records]
        return check_metrics(experiment.evaluate(routed, to_experiment(config), split, **extra))

    def failures(self, train_records: list[dict], config: dict) -> list[dict]:
        return failures(train_records, config)
