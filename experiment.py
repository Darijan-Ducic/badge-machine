"""Offline experiment harness: replay stored raw clip outputs under a baseline and a variant post-processing config,
on the same clips in the same order, and gate the variant on held-out VTT accuracy (docs/self-improvement.md).

Input is an ordered list of raw clip records in the design-doc shape (docs/uncertainty-design.md, "Post-processor
spec"): `{"samples", "closed", "gemini", "bank", "stt"}` plus any store fields (`id`, ...). `closed` and `gemini`
may be the bare outputs or the stored call items `{"call", "output", "error", "usage"}`. Features are recomputed from
the raw outputs for every evaluation, so each knob below costs nothing to replay. Clips without STT are not scored.

Knobs (a spec dict; missing knobs take DEFAULTS, unknown ones are refused; a run's stored `config` is a valid spec):
  costs      viseme substitution costs: a dict in the data/viseme_costs.json shape, a path to such a file, or None
             for data/viseme_costs.json
  drop       feature names zeroed in every clip; a zeroed column is a dropped feature at every stage
  stages     {stage name: minimum labeled clips}, names from scorer (defaults, temperature, core, full); one at 0
  pool_cap   candidates kept per clip, ∅ included (2 .. candidates.POOL_CAP): the first ones in pool order
  readiness  {"n_min", "window", "acc_min"} for the run's per-phrase readiness (readiness.py)

Splits, chronological over the clips with STT: train = first 60%, heldout = next 20%, test = last 20%. Train is
scored predict-then-train (each clip by a model fit on the train clips before it, no auto-send threshold); heldout
and test are each scored by ONE model, with its auto-send threshold, fit on every clip before the split.

`evaluate(records, config, split)` metrics:
  sentence_acc  VTT = STT exactly (top non-∅ candidate matches the normalized STT text; ∅ winning is a miss)
  word_acc      1 - WER: word edits of VTT text vs STT text (normalized; ∅ = no words), summed over clips, over the
                total STT words; below 0 when VTT inserts more words than STT has
  viseme_acc    1 - viseme edits (visemes.to_visemes, unit costs) summed over clips, over the total STT visemes
  top3_acc      STT text among the 3 most probable non-∅ candidates
  gemini_acc / muse_acc   VTT · Gemini / VTT · Muse picks = STT;  pool_recall  STT text in the pool
  autosend_precision / autosend_coverage   on heldout/test (None on train); precision None when nothing is sent
  cost_per_clip mean over clips with stored token `usage` of their calls priced by PRICES, else None
A run record reports both train and heldout; its `metrics` (the `GET /runs` contract) are: vtt_acc, gemini_acc,
muse_acc, pool_recall from the train stream; vtt_acc_heldout and auto-send from heldout. The test split is evaluated
only on request, at the end of a loop.
Gate: a run against a baseline run is "pass" only if vtt_acc_heldout is strictly higher; the baseline must have run
on the same dataset id and clip order, else ValueError.

Config hash: every knob (costs resolved to their values), the dataset id and the clip order (clip ids, or a content
hash of the raw outputs when a record has no id). Run notes (name, hypothesis, author, user, code commit, system,
candidate and parent version) are outside it.
"""

import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

import visemes
from candidates import FEATURES, POOL_CAP, Clip, featurize, match
from preheat import _normalize
from readiness import ACC_MIN, N_MIN, WINDOW, readiness
from scorer import STAGES, auto_send, predict, train

CURVE_WINDOW = 20
AUTHORS = ("agent", "human")
SPLITS = ("train", "heldout", "test")
PRICES = {"muse": (1.25, 4.25), "gemini": (2.0, 10.0)}  # $ per 1M input / output tokens
DEFAULTS = {
    "costs": None,
    "drop": [],
    "stages": {name: minimum for minimum, name in STAGES},
    "pool_cap": POOL_CAP,
    "readiness": {"n_min": N_MIN, "window": WINDOW, "acc_min": ACC_MIN},
}
_STAGE_NAMES = {name for _, name in STAGES}
_RAW = ("samples", "closed", "gemini", "bank", "stt")
_DERIVED = ("order", "hash")  # in a run's stored config, not knobs


# ---------------------------------------------------------------- config


def _costs(costs) -> dict:
    if costs is None:
        costs = visemes.load_costs()
    elif isinstance(costs, str | Path):
        costs = visemes.load_costs(Path(costs))
    return {"alphabet": costs["alphabet"], "default_sub": float(costs["default_sub"]), "ins": float(costs["ins"]),
            "del": float(costs["del"]), "sub": {pair: float(cost) for pair, cost in sorted(costs["sub"].items())}}


def _int(value, name: str, low: int, high: float = np.inf) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} = {value!r} must be an integer in [{low}, {high}]")
    return value


def resolve(spec: dict | None) -> dict:
    """The full knob set for `spec`, defaults filled in and normalized so equal configs compare (and hash) equal."""
    spec = {k: v for k, v in (spec or {}).items() if k not in _DERIVED}
    if unknown := set(spec) - set(DEFAULTS):
        raise ValueError(f"unknown knobs {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    drop = sorted(set(spec.get("drop", [])))
    if unknown := set(drop) - set(FEATURES):
        raise ValueError(f"cannot drop {sorted(unknown)}: not in candidates.FEATURES")
    stages = {name: _int(minimum, f"stages[{name}]", 0) for name, minimum in spec.get("stages", DEFAULTS["stages"]).items()}
    if not set(stages) <= _STAGE_NAMES or 0 not in stages.values() or len(set(stages.values())) < len(stages):
        raise ValueError(f"stages {stages}: names from {sorted(_STAGE_NAMES)}, distinct minimums, one at 0")
    readiness_spec = spec.get("readiness", {})
    if unknown := set(readiness_spec) - set(DEFAULTS["readiness"]):
        raise ValueError(f"unknown readiness knobs {sorted(unknown)}")
    knobs = {**DEFAULTS["readiness"], **readiness_spec}
    return {
        "costs": _costs(spec.get("costs")),
        "drop": drop,
        "stages": dict(sorted(stages.items(), key=lambda item: item[1])),
        "pool_cap": _int(spec.get("pool_cap", POOL_CAP), "pool_cap", 2, POOL_CAP),
        "readiness": {"n_min": _int(knobs["n_min"], "n_min", 0), "window": _int(knobs["window"], "window", 1),
                      "acc_min": float(knobs["acc_min"])},
    }


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def order_of(records: list[dict]) -> str:
    """Digest of the clip sequence: each clip's id, or a hash of its raw outputs when it has none."""
    return _digest([str(r["id"]) if r.get("id") is not None else _digest({k: r.get(k) for k in _RAW})
                    for r in records])


def config_for(records: list[dict], spec: dict | None, dataset: str) -> dict:
    """The run's `config`: resolved knobs + the clip-order digest + the hash over knobs, dataset id and order."""
    knobs = resolve(spec)
    order = order_of(records)
    return {**knobs, "order": order, "hash": _digest({"knobs": knobs, "dataset": dataset, "order": order})}


# ---------------------------------------------------------------- metrics on texts and calls


def _edits(a: list, b: list) -> int:
    """Unit-cost Levenshtein distance between two sequences."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _accuracy(pairs: list[tuple[list, list]]) -> float | None:
    total = sum(len(said) for _, said in pairs)
    return 1 - sum(_edits(read, said) for read, said in pairs) / total if total else None


def word_acc(vtts: list[str], stts: list[str]) -> float | None:
    """1 - WER over normalized words, summed over clips; None without STT words."""
    return _accuracy([(_normalize(v).split(), _normalize(s).split()) for v, s in zip(vtts, stts)])


def viseme_acc(vtts: list[str], stts: list[str]) -> float | None:
    """1 - viseme edit count (unit costs) over the STT viseme count, summed over clips; None without STT visemes."""
    return _accuracy([(visemes.to_visemes(v), visemes.to_visemes(s)) for v, s in zip(vtts, stts)])


def _call_cost(usage: dict, model: str) -> float:
    price_in, price_out = PRICES[model]
    if model == "muse":  # OpenAI-style usage; completion tokens include reasoning
        tokens_in, tokens_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    else:  # Gemini usage_metadata; thinking is billed as output
        tokens_in = usage.get("prompt_token_count", 0)
        tokens_out = usage.get("candidates_token_count", 0) + usage.get("thoughts_token_count", 0)
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def clip_cost(record: dict) -> float | None:
    """Dollars for the clip's stored calls (open samples, closed, Gemini) that carry `usage`; None if none do."""
    calls = [(s, "muse") for s in record.get("samples") or []]
    calls += [(record.get("closed"), "muse"), (record.get("gemini"), "gemini")]
    costs = [_call_cost(call["usage"], model) for call, model in calls
             if isinstance(call, dict) and "call" in call and call.get("usage")]
    return sum(costs) if costs else None


# ---------------------------------------------------------------- replay


def splits(records: list[dict]) -> dict[str, list[dict]]:
    """Chronological train / heldout / test (60 / 20 / 20, rounded down) of the records that have STT."""
    labeled = [r for r in records if r.get("stt") is not None]
    return {name: labeled[lo:hi] for name, (lo, hi) in _bounds(len(labeled)).items()}


def _bounds(n: int) -> dict[str, tuple[int, int]]:
    train_end, heldout_end = n * 3 // 5, n * 4 // 5
    return {"train": (0, train_end), "heldout": (train_end, heldout_end), "test": (heldout_end, n)}


def _output(call):
    return call.get("output") if isinstance(call, dict) and "call" in call else call


def _cap(clip: Clip, cap: int) -> Clip:
    # Candidate rows don't depend on each other, so keeping the first cap-1 equals featurizing with that cap.
    if len(clip.pool) <= cap:
        return clip
    keep = cap - 1
    label = clip.label if clip.label is None or clip.label < keep else keep  # a cut-off answer is a pool miss
    return replace(clip, pool=clip.pool[:keep] + clip.pool[-1:], label=label,
                   features=np.vstack([clip.features[:keep], clip.features[-1:]]))


def _featurized(records: list[dict], knobs: dict) -> tuple[list[dict], list[Clip]]:
    """The records with STT, in order, and each one as a Clip under the knobs."""
    distance = partial(visemes.viseme_distance, costs=knobs["costs"])
    dropped = [FEATURES.index(name) for name in knobs["drop"]]
    labeled = [r for r in records if r.get("stt") is not None]
    clips = []
    for record in labeled:
        clip = featurize(record["samples"], _output(record.get("closed")), _output(record.get("gemini")),
                         record["bank"], record["stt"], to_visemes=visemes.to_visemes, viseme_distance=distance)
        clip = _cap(clip, knobs["pool_cap"])
        clip.features[:, dropped] = 0.0
        clips.append(clip)
    return labeled, clips


def _plan(knobs: dict) -> list[tuple[int, str]]:
    return [(minimum, stage) for stage, minimum in knobs["stages"].items()]


def _mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _matches(stt: str, text: str | None) -> bool:
    return text is not None and match(stt, [text]) is not None


def _score(records: list[dict], clips: list[Clip], split: str, plan) -> tuple[dict, list[dict]]:
    """Metrics for one split, and its per-clip history ({"fused", "gemini", "muse"}, scorer.prequential keys)."""
    lo, hi = _bounds(len(clips))[split]
    scored = clips[lo:hi]
    if split == "train":  # predict, then train
        probs = [predict(train(clips[:i], gate=False, stages=plan), [clips[i]])[0] for i in range(lo, hi)]
        sent = None
    else:
        model = train(clips[:lo], stages=plan)  # nothing from this split or later reaches training
        probs = predict(model, scored)
        sent = [auto_send(model, p) for p in probs]

    vtts, top3 = [], []
    for clip, p in zip(scored, probs):
        none = len(clip.pool) - 1
        ranked = [int(j) for j in np.argsort(-p, kind="stable") if j != none]
        if clip.sentence is not None:  # a sentence turn: the sentence path is its VTT
            vtts.append(clip.sentence)
        else:
            vtts.append("" if int(p.argmax()) == none else clip.pool[int(p.argmax())])
        top3.append(any(_matches(clip.stt, clip.pool[j]) for j in ranked[:3]))
    history = [{"fused": bool(vtt) and _matches(c.stt, vtt),
                "gemini": _matches(c.stt, c.gemini_pick), "muse": _matches(c.stt, c.muse_pick)}
               for c, vtt in zip(scored, vtts)]
    stts = [c.stt for c in scored]
    sent_correct = [h["fused"] for h, send in zip(history, sent or []) if send]
    metrics = {
        "clips": len(scored),
        "word_acc": word_acc(vtts, stts),
        "sentence_acc": _mean(h["fused"] for h in history),
        "viseme_acc": viseme_acc(vtts, stts),
        "top3_acc": _mean(top3),
        "gemini_acc": _mean(h["gemini"] for h in history),
        "muse_acc": _mean(h["muse"] for h in history),
        "pool_recall": _mean(c.label != len(c.pool) - 1 for c in scored),
        "autosend_precision": _mean(sent_correct) if sent_correct else None,
        "autosend_coverage": None if sent is None else _mean(sent),
        "cost_per_clip": _mean(cost for cost in map(clip_cost, records[lo:hi]) if cost is not None),
    }
    return metrics, history


def evaluate(records: list[dict], config: dict | None, split: str) -> dict:
    """Metrics of one split ("train", "heldout", "test") of `records` under `config` (a spec or a run's config)."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r} must be one of {SPLITS}")
    knobs = resolve(config)
    labeled, clips = _featurized(records, knobs)
    return _score(labeled, clips, split, _plan(knobs))[0]


def _head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(records: list[dict], spec: dict | None = None, *, dataset: str, name: str, baseline: dict | None = None,
        user_id: str | None = None, hypothesis: str = "", author: str = "human", system_version: str | None = None,
        code_commit: str | None = None, candidate_version: str | None = None, parent_version: str | None = None
        ) -> dict:
    """Replay `records` in order under `spec`; return the run record (`GET /runs` shape + splits + run notes).

    `baseline`: an earlier run record on the same dataset id and clip order; sets `baseline` and the gate.
    """
    if author not in AUTHORS:
        raise ValueError(f"author {author!r} must be one of {AUTHORS}")
    config = config_for(records, spec, dataset)
    if baseline is not None and (baseline["dataset"], baseline["config"]["order"]) != (dataset, config["order"]):
        raise ValueError("baseline ran on a different dataset or clip order; compare runs on the same clips only")
    labeled, clips = _featurized(records, config)
    if len(clips) < 3:
        raise ValueError(f"{len(clips)} clips with STT; need at least 3 for train, heldout and test splits")
    plan = _plan(config)
    train_metrics, history = _score(labeled, clips, "train", plan)
    heldout, _ = _score(labeled, clips, "heldout", plan)
    metrics = {
        "clips": len(clips),
        "vtt_acc": train_metrics["sentence_acc"],
        "vtt_acc_heldout": heldout["sentence_acc"],
        "gemini_acc": train_metrics["gemini_acc"],
        "muse_acc": train_metrics["muse_acc"],
        "pool_recall": train_metrics["pool_recall"],
        "autosend_precision": heldout["autosend_precision"],
        "autosend_coverage": heldout["autosend_coverage"],
    }
    curve = []
    for i in range(len(history)):
        window = history[max(0, i - CURVE_WINDOW + 1):i + 1]
        curve.append({"i": i, "vtt": _mean(h["fused"] for h in window), "gemini": _mean(h["gemini"] for h in window),
                      "muse": _mean(h["muse"] for h in window)})
    gate = None
    if baseline is not None:
        gate = "pass" if metrics["vtt_acc_heldout"] > baseline["metrics"]["vtt_acc_heldout"] else "fail"
    return {
        "id": uuid.uuid4().hex[:12],
        "t": time.time(),
        "name": name,
        "dataset": dataset,
        "user_id": user_id,
        "config": config,
        "baseline": None if baseline is None else baseline["id"],
        "metrics": metrics,
        "gate": gate,
        "curve": curve,
        "splits": {"train": train_metrics, "heldout": heldout},
        "readiness": readiness(history, [c.stt for c in clips[:len(history)]], **config["readiness"]),
        "hypothesis": hypothesis,
        "author": author,
        "code_commit": code_commit if code_commit is not None else _head(),
        "system_version": system_version,
        "candidate_version": candidate_version,
        "parent_version": parent_version,
    }


def compare(records: list[dict], variants: dict[str, dict], *, dataset: str, baseline: dict | None = None,
            variant_notes: dict[str, dict] | None = None, **notes) -> list[dict]:
    """Run the baseline spec, then each named variant spec against it, all on `records` as given.

    Returns [baseline record, *variant records]. `notes` (user_id, author, system_version, code_commit) go on every
    run; `variant_notes` {variant name: {hypothesis, candidate_version, parent_version}} only on that variant's.
    """
    base = run(records, baseline, dataset=dataset, name="baseline", **notes)
    return [base] + [run(records, spec, dataset=dataset, name=name, baseline=base,
                         **{**notes, **(variant_notes or {}).get(name, {})}) for name, spec in variants.items()]
