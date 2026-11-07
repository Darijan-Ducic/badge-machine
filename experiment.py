"""Offline experiment harness: replay stored raw clip outputs under a baseline and a variant post-processing config,
on the same clips in the same order, and gate the variant on held-out VTT accuracy.

Input is an ordered list of raw clip records in the design-doc shape (docs/uncertainty-design.md, "Post-processor
spec"): `{"samples", "closed", "gemini", "bank", "stt"}` plus any store fields (`id`, ...). Features are recomputed
from the raw outputs for every run, so each knob below costs nothing to replay. Clips without STT are not scored.

Knobs (a spec dict; missing knobs take DEFAULTS, unknown ones are refused):
  costs      viseme substitution costs: a dict in the data/viseme_costs.json shape, a path to such a file, or None
             for data/viseme_costs.json
  drop       feature names zeroed in every clip; a zeroed column is a dropped feature at every stage
  stages     {stage name: minimum labeled clips}, names from scorer (defaults, temperature, core, full); one at 0
  pool_cap   candidates kept per clip, ∅ included (2 .. candidates.POOL_CAP): the first ones in pool order
  readiness  {"n_min", "window", "acc_min"} for the run's per-phrase readiness (readiness.py)

Metrics:
  all clips, from the predict-then-train history: vtt_acc, gemini_acc, muse_acc, pool_recall, and the curve
    (accuracy over the trailing CURVE_WINDOW clips at each clip)
  held-out = the last 20% of clips scored by ONE model trained (with its auto-send threshold) on the first 80%:
    vtt_acc_heldout, autosend_coverage, autosend_precision (None when nothing is sent)
Gate: a run against a baseline run is "pass" only if its held-out VTT accuracy is strictly higher; the baseline must
have run on the same dataset id and clip order, else ValueError.

Config hash: every knob (costs resolved to their values), the dataset id and the clip order (clip ids, or a content
hash of the raw outputs when a record has no id). Run notes (name, hypothesis, author, user, code commit, system
version) are outside it.
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
from readiness import ACC_MIN, N_MIN, WINDOW, readiness
from scorer import STAGES, auto_send, predict, prequential, train

CURVE_WINDOW = 20
AUTHORS = ("agent", "human")
DEFAULTS = {
    "costs": None,
    "drop": [],
    "stages": {name: minimum for minimum, name in STAGES},
    "pool_cap": POOL_CAP,
    "readiness": {"n_min": N_MIN, "window": WINDOW, "acc_min": ACC_MIN},
}
_STAGE_NAMES = {name for _, name in STAGES}
_RAW = ("samples", "closed", "gemini", "bank", "stt")


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
    spec = spec or {}
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


def _cap(clip: Clip, cap: int) -> Clip:
    # Candidate rows don't depend on each other, so keeping the first cap-1 equals featurizing with that cap.
    if len(clip.pool) <= cap:
        return clip
    keep = cap - 1
    label = clip.label if clip.label is None or clip.label < keep else keep  # a cut-off answer is a pool miss
    return replace(clip, pool=clip.pool[:keep] + clip.pool[-1:], label=label,
                   features=np.vstack([clip.features[:keep], clip.features[-1:]]))


def _clips(records: list[dict], knobs: dict) -> list[Clip]:
    distance = partial(visemes.viseme_distance, costs=knobs["costs"])
    dropped = [FEATURES.index(name) for name in knobs["drop"]]
    clips = []
    for record in records:
        if record.get("stt") is None:
            continue
        clip = featurize(record["samples"], record.get("closed"), record.get("gemini"), record["bank"], record["stt"],
                         to_visemes=visemes.to_visemes, viseme_distance=distance)
        clip = _cap(clip, knobs["pool_cap"])
        clip.features[:, dropped] = 0.0
        clips.append(clip)
    return clips


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _vtt_correct(clip: Clip, probs: np.ndarray) -> bool:
    top = int(probs.argmax())
    return top != len(clip.pool) - 1 and match(clip.stt, [clip.pool[top]]) is not None


def _head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(records: list[dict], spec: dict | None = None, *, dataset: str, name: str, baseline: dict | None = None,
        user_id: str | None = None, hypothesis: str = "", author: str = "human", system_version: str | None = None,
        code_commit: str | None = None) -> dict:
    """Replay `records` in order under `spec` and return the run record (`GET /runs` shape + run notes).

    `baseline`: an earlier run record on the same dataset id and clip order; sets `baseline` and the gate.
    """
    if author not in AUTHORS:
        raise ValueError(f"author {author!r} must be one of {AUTHORS}")
    config = config_for(records, spec, dataset)
    if baseline is not None and (baseline["dataset"], baseline["config"]["order"]) != (dataset, config["order"]):
        raise ValueError("baseline ran on a different dataset or clip order; compare runs on the same clips only")
    clips = _clips(records, config)
    if not clips:
        raise ValueError("no clips with STT to score")
    plan = [(minimum, stage) for stage, minimum in config["stages"].items()]

    history = prequential(clips, stages=plan)
    split = len(clips) * 4 // 5
    model = train(clips[:split], stages=plan)  # the held-out 20% never reaches training
    held_out = clips[split:]
    probs = predict(model, held_out)
    correct = [_vtt_correct(c, p) for c, p in zip(held_out, probs)]
    sent = [auto_send(model, p) for p in probs]
    sent_correct = [ok for ok, send in zip(correct, sent) if send]
    metrics = {
        "clips": len(clips),
        "vtt_acc": _mean(h["fused"] for h in history),
        "vtt_acc_heldout": _mean(correct),
        "gemini_acc": _mean(h["gemini"] for h in history),
        "muse_acc": _mean(h["muse"] for h in history),
        "pool_recall": _mean(h["pool_hit"] for h in history),
        "autosend_precision": _mean(sent_correct) if sent_correct else None,
        "autosend_coverage": _mean(sent),
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
        "readiness": readiness(history, [c.stt for c in clips], **config["readiness"]),
        "hypothesis": hypothesis,
        "author": author,
        "code_commit": code_commit if code_commit is not None else _head(),
        "system_version": system_version,
    }


def compare(records: list[dict], variants: dict[str, dict], *, dataset: str, baseline: dict | None = None,
            hypotheses: dict[str, str] | None = None, **notes) -> list[dict]:
    """Run the baseline spec, then each named variant spec against it, all on `records` as given.

    Returns [baseline record, *variant records]. `hypotheses`: {variant name: why this change}. `notes`: user_id,
    author, system_version, code_commit, shared by every run.
    """
    base = run(records, baseline, dataset=dataset, name="baseline", **notes)
    return [base] + [run(records, spec, dataset=dataset, name=name, baseline=base,
                         hypothesis=(hypotheses or {}).get(name, ""), **notes) for name, spec in variants.items()]
