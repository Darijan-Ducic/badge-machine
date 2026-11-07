"""What the iterating agent looks at before it changes anything: where and why lip reading fails, what training
currently does for a user, and how post-processing turned one clip's raw model outputs into one answer.

Usage:
  uv run --env-file .env python tools/diagnose.py failures --user ID [--split train|heldout] [--json]
  uv run --env-file .env python tools/diagnose.py recipe --user ID [--json]
  uv run --env-file .env python tools/diagnose.py clip ID [--json]

Clips come from MongoDB Atlas when MONGODB_URI is set, else data/clips/, or from --store DIR (a file store). A user's
stream is their v1 clips under the phrase list of their latest clip, in store order, featurized as the Clips page does
(clips._featurized: each clip's bank counts only earlier labels). Nothing calls a model; everything is a free replay.

failures  the labeled clips of one split (experiment.splits: train = first 60%, heldout = next 20%; the test split is
          refused) where VTT != STT. A train clip is scored by a model trained on the labeled clips before it; a
          heldout clip by one model trained on the whole train split, as experiment.evaluate scores them.
          Cause, exactly one: pool_miss (the STT text is not in the pool), ranked_wrong (it is, another candidate
          won), abstained (∅ won while it was in the pool). Flags, any: bad_video (∅'s bad_quality or no_speech
          >= 0.5), stt_suspect (the STT text is empty after normalizing, or repeats a word or phrase back to back).
recipe    the user's current model (scorer.train on every labeled clip, as the Stats page): stage and what unlocks
          the next, each feature's weight and whether this stage fits it, the auto-send threshold, the share of the
          user's clips it would auto-send / confirm / abstain on (in-sample, every split), and pool recall.
clip      one clip as the Clips page scores it (a model trained on the user's earlier clips): raw open samples,
          closed call, Gemini, the pool with each candidate's sources and features, each feature's contribution
          (weight x value) to the winner's and runner-up's scores, and the features that separate them most.
"""

import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
# Run as a script, this directory heads sys.path: drop it and put the repo root first so the repo's modules import.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _TOOLS]
sys.path.insert(0, os.path.dirname(_TOOLS))

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import quote  # noqa: E402

import numpy as np  # noqa: E402

import clips  # noqa: E402
import experiment  # noqa: E402
import scorer  # noqa: E402
import visemes  # noqa: E402
from candidates import FEATURES, _observed, _recombinations, _top, match  # noqa: E402
from preheat import _normalize  # noqa: E402

CLIPS_PAGE = "http://localhost:8765/clips.html?clip={}"
SPLITS = ("train", "heldout")  # never "test": the agent must not tune on it
CAUSES = ("pool_miss", "ranked_wrong", "abstained")
FLAGS = ("bad_video", "stt_suspect")
BUCKETS = CAUSES + FLAGS
MEANING = {
    "pool_miss": "STT text not in the pool",
    "ranked_wrong": "in the pool, another candidate won",
    "abstained": "∅ won while the STT text was in the pool",
    "bad_video": "flag: ∅'s bad_quality or no_speech >= 0.5",
    "stt_suspect": "flag: STT text empty or repeating itself",
}
BAD_VIDEO = 0.5
TOP = 3
SOURCES = ("hypothesis", "slots", "closed", "gemini", "bank")
_BAD, _SILENT = FEATURES.index("bad_quality"), FEATURES.index("no_speech")


def _stream(records: list[dict], user: str) -> tuple[str, list[dict], list]:
    """(vocab, the user's v1 records under their latest phrase list, those records as candidates.Clip)."""
    mine = [r for r in records if r.get("record") == clips.V1 and r.get("user_id") == user]
    if not mine:
        raise LookupError(f"no v1 clips for user {user!r}")
    vocab = mine[-1]["vocab"]
    stream = clips._replayed(mine, user, vocab)
    return vocab, stream, clips._featurized(stream)


def _order(probs: np.ndarray) -> list[int]:
    return [int(j) for j in np.argsort(-probs, kind="stable")]


def _link(clip_id: str) -> str:
    return CLIPS_PAGE.format(quote(clip_id))


# ---------------------------------------------------------------- failures


def _stt_suspect(text: str) -> bool:
    if not _normalize(text):
        return True
    words = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()  # preheat._normalize's words, doubling kept
    return any(words[i:i + n] == words[i + n:i + 2 * n]
               for n in range(1, len(words) // 2 + 1) for i in range(len(words) - 2 * n + 1))


def _failure(record: dict, clip, probs: np.ndarray, model: scorer.Model, trained_on: int) -> dict | None:
    none, order = len(clip.pool) - 1, _order(probs)
    top = order[0]
    if top != none and match(clip.stt, [clip.pool[top]]) is not None:
        return None
    in_pool = clip.label != none
    cause = "pool_miss" if not in_pool else "abstained" if top == none else "ranked_wrong"
    bad, silent = float(clip.features[none, _BAD]), float(clip.features[none, _SILENT])
    flags = [name for name, on in (("bad_video", max(bad, silent) >= BAD_VIDEO),
                                   ("stt_suspect", _stt_suspect(clip.stt))) if on]
    return {
        "id": record["id"], "stt": clip.stt, "vtt": clips._shown(clip, top), "cause": cause, "flags": flags,
        "label_rank": order.index(clip.label) + 1 if in_pool else None,
        "label_p": float(probs[clip.label]) if in_pool else None,
        "top": [{"text": clips._shown(clip, j), "p": float(probs[j])} for j in order[:TOP]],
        "bad_quality": bad, "no_speech": silent, "stage": model.stage, "trained_on": trained_on,
        "link": _link(record["id"]),
    }


def failures(records: list[dict], user: str, split: str = "train") -> dict:
    """Every labeled clip of `split` where VTT != STT, by cause, each scored by a model that never saw it."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r}: inspect {' or '.join(SPLITS)} only; the test split is never inspected, "
                         "so no change can be tuned on it")
    vocab, stream, featurized = _stream(records, user)
    labeled = [(r, c) for r, c in zip(stream, featurized) if c.label is not None]
    parts = experiment.splits([{"stt": c.stt, "at": j} for j, (_, c) in enumerate(labeled)])
    scored = [s["at"] for s in parts[split]]
    known = [c for _, c in labeled]
    before = len(parts["train"])
    heldout_model = scorer.train(known[:before], gate=False) if split == "heldout" else None
    rows = []
    for j in scored:
        if heldout_model is None:
            model, trained_on = scorer.train(known[:j], gate=False), j
        else:
            model, trained_on = heldout_model, before
        record, clip = labeled[j]
        row = _failure(record, clip, scorer.predict(model, [clip])[0], model, trained_on)
        if row is not None:
            rows.append(row)
    counts = dict.fromkeys(BUCKETS, 0)
    for row in rows:
        for name in [row["cause"], *row["flags"]]:
            counts[name] += 1
    return {"user": user, "vocab": vocab, "split": split, "scored": len(scored), "failures": len(rows),
            "counts": counts, "rows": rows}


# ---------------------------------------------------------------- recipe


def _status(stage: str, feature: str) -> str:
    """How `scorer.train` sets this feature's weight at `stage` (scorer._fit)."""
    if stage == "defaults":
        return "default"
    if stage == "temperature":
        return "scaled" if scorer.DEFAULT_WEIGHTS[FEATURES.index(feature)] else "off"
    if stage == "core":
        return "fitted" if feature in scorer.CORE else "off"
    return "fitted"


def recipe(records: list[dict], user: str) -> dict:
    """What training currently does for `user`: the model trained on all their labeled clips, and what it decides."""
    vocab, stream, featurized = _stream(records, user)
    labeled = [c for c in featurized if c.label is not None]
    n, model = len(labeled), scorer.train(labeled)
    stages = sorted(scorer.STAGES)
    upcoming = [(at, name) for at, name in stages if at > n]
    actions = Counter(clips._action(model, p) for p in scorer.predict(model, featurized))
    scale = None
    if model.stage == "temperature":
        k = int(np.flatnonzero(scorer.DEFAULT_WEIGHTS)[0])
        scale = float(model.weights[k] / scorer.DEFAULT_WEIGHTS[k])
    hits = sum(c.label != len(c.pool) - 1 for c in labeled)
    return {
        "user": user, "vocab": vocab, "clips": len(stream), "labeled": n, "stage": model.stage,
        "stage_at": next(at for at, name in stages if name == model.stage),
        "next_stage": {"name": upcoming[0][1], "at": upcoming[0][0], "needed": upcoming[0][0] - n} if upcoming
        else None,
        "stages": [{"name": name, "at": at} for at, name in stages],
        "weights": [{"feature": f, "weight": float(w), "default": float(d), "status": _status(model.stage, f)}
                    for f, w, d in zip(FEATURES, model.weights, scorer.DEFAULT_WEIGHTS)],
        "scale": scale,
        "threshold": clips._threshold(model),
        "actions": {name: actions[key] / len(featurized) for name, key in (("auto", "auto"), ("confirm", "confirm"),
                                                                          ("abstain", "none"))},
        "pool_recall": hits / n if n else None,
        "pool_hits": hits,
    }


# ---------------------------------------------------------------- one clip


def _bank_nearest(record: dict, earlier: list[dict], outputs: list[dict]) -> list[str]:
    """The 3 bank entries candidates.featurize put in the pool: nearest in viseme distance to any sample's visemes."""
    distance = partial(visemes.viseme_distance, costs=visemes.load_costs(clips.COSTS))
    said = [t for t in map(clips._label_text, earlier) if t is not None]
    entries = {}
    for text in record["phrases"] + list(dict.fromkeys(said)):
        entries.setdefault(_normalize(text), text)
    entries.pop("", None)
    observed = [_observed(o) for o in outputs]

    def nearness(entry: str) -> float:
        return min((distance(visemes.to_visemes(entry), seen) for seen in observed), default=np.inf)

    return sorted(entries.values(), key=nearness)[:3]


def _sources(record: dict, earlier: list[dict], clip) -> list[list[str]]:
    """Per pool entry, every pool rule that proposed it (∅, last, has none)."""
    outputs = [s["output"] for s in record["samples"] if s.get("output") is not None]
    proposed = {
        "hypothesis": [h["text"] for o in outputs for h in o.get("hypotheses") or []],
        "slots": [text for o in outputs for text in _recombinations(o)],
        "closed": _top(((record["closed"].get("output") or {}).get("scores")) or {}, 3),
        "gemini": _top(record["gemini"].get("output") or {}, 3),
        "bank": _bank_nearest(record, earlier, outputs),
    }
    keys = {name: {_normalize(t) for t in texts} for name, texts in proposed.items()}
    return [[name for name in SOURCES if _normalize(text) in keys[name]] for text in clip.pool[:-1]] + [[]]


def _side(clip, probs: np.ndarray, model: scorer.Model, j: int) -> dict:
    contributions = model.weights * clip.features[j] + 0.0  # + 0.0 turns -0.0 into 0.0
    return {"text": clips._shown(clip, j), "p": float(probs[j]), "score": float(contributions.sum()),
            "contributions": dict(zip(FEATURES, map(float, contributions)))}


def clip_story(records: list[dict], clip_id: str) -> dict:
    """How post-processing turned one clip's raw outputs into its answer, scored as the Clips page scores it."""
    record = next((r for r in records if r.get("id") == clip_id), None)
    if record is None or record.get("record") != clips.V1:
        raise LookupError(f"no v1 clip {clip_id!r}")
    stream = clips._replayed(records, record["user_id"], record["vocab"])
    i = [r["id"] for r in stream].index(clip_id)
    featurized = clips._featurized(stream[:i + 1])  # later clips can't reach this clip's bank or model
    clip, model = featurized[i], clips._trained_before(featurized, i)
    probs = scorer.predict(model, [clip])[0]
    order, none = _order(probs), len(clip.pool) - 1
    sources = _sources(record, stream[:i], clip)
    winner = _side(clip, probs, model, order[0])
    runner = _side(clip, probs, model, order[1]) if len(order) > 1 else None
    why = [] if runner is None else sorted(
        ({"feature": f, "gap": winner["contributions"][f] - runner["contributions"][f],
          "winner": winner["contributions"][f], "runner_up": runner["contributions"][f]} for f in FEATURES),
        key=lambda w: -w["gap"])
    closed, gemini = record["closed"], record["gemini"]
    closed_scores = (closed.get("output") or {}).get("scores") or {}
    gemini_probs = gemini.get("output") or {}
    samples = []
    for s in record["samples"]:
        o = s.get("output")
        quality = (o or {}).get("quality") or {}
        samples.append({"idx": s.get("sample_idx"), "error": s.get("error"),
                        "hypotheses": None if o is None else [{"text": h["text"], "s": h["s"]}
                                                              for h in o.get("hypotheses") or []],
                        "speech": None if o is None else (o.get("speech") or {}).get("present"),
                        "mouth_visible": quality.get("mouth_visible"), "blur": quality.get("blur")})
    return {
        "id": clip_id, "user": record["user_id"], "vocab": record["vocab"], "stt": clip.stt,
        "vtt": clips._shown(clip, order[0]),
        "correct": None if clip.stt is None else (order[0] != none
                                                  and match(clip.stt, [clip.pool[order[0]]]) is not None),
        "stage": model.stage, "trained_on": sum(c.label is not None for c in featurized[:i]),
        "action": clips._action(model, probs), "threshold": clips._threshold(model), "link": _link(clip_id),
        "samples": samples,
        "closed": {"top": [{"phrase": p, "s": closed_scores[p]} for p in _top(closed_scores, 5)],
                   "none": (closed.get("output") or {}).get("none"), "error": closed.get("error")},
        "gemini": {"top": [{"phrase": p, "p": gemini_probs[p]} for p in _top(gemini_probs, 5)],
                   "error": gemini.get("error")},
        "weights": dict(zip(FEATURES, map(float, model.weights))),
        "pool": [{"text": clips._shown(clip, j), "p": float(probs[j]), "is_label": j == clip.label,
                  "sources": sources[j], "features": dict(zip(FEATURES, map(float, clip.features[j])))}
                 for j in order],
        "winner": winner, "runner_up": runner, "why": why,
    }


# ---------------------------------------------------------------- text


def _num(value, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _table(rows: list[list[str]], right: set[int] = frozenset()) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  " + "  ".join(cell.rjust(w) if i in right else cell.ljust(w) for i, (cell, w) in
                             enumerate(zip(row, widths))).rstrip() for row in rows]


def render_failures(report: dict) -> str:
    how = ("each by a model trained on the labeled clips before it" if report["split"] == "train"
           else "each by one model trained on the whole train split")
    lines = [f"failures · user {report['user']} · vocab {report['vocab']} · {report['split']} split: "
             f"{report['scored']} labeled clips scored, {how}",
             f"VTT ≠ STT on {report['failures']} of {report['scored']}"]
    lines += _table([[name, str(report["counts"][name]), MEANING[name]] for name in BUCKETS], right={1})
    for cause in CAUSES:
        rows = [r for r in report["rows"] if r["cause"] == cause]
        if not rows:
            continue
        lines += ["", f"{cause} ({len(rows)})"]
        for r in rows:
            flags = f"  flags {','.join(r['flags'])}" if r["flags"] else ""
            lines += [f"  {r['id']}  STT {json.dumps(r['stt'], ensure_ascii=False)}  VTT "
                      f"{json.dumps(r['vtt'], ensure_ascii=False)}  label rank {_num(r['label_rank'], 'd')} "
                      f"p {_num(r['label_p'])}{flags}  [{r['stage']}, trained on {r['trained_on']}]",
                      "      top " + " · ".join(f"{t['text']} {t['p']:.3f}" for t in r["top"]),
                      f"      {r['link']}"]
    return "\n".join(lines)


def render_recipe(report: dict) -> str:
    nxt = report["next_stage"]
    lines = [f"recipe · user {report['user']} · vocab {report['vocab']} · {report['clips']} clips, "
             f"{report['labeled']} labeled",
             f"stage {report['stage']} (from {report['stage_at']} labeled) · next: "
             + (f"{nxt['name']} at {nxt['at']} labeled ({nxt['needed']} more)" if nxt else "none, the last stage"),
             "stages: " + " · ".join(f"{s['name']} {s['at']}" for s in report["stages"])]
    if report["scale"] is not None:
        lines.append(f"fitted scale on the default weights: {report['scale']:.3f}")
    lines += _table([["feature", "weight", "default", "status"]]
                    + [[w["feature"], f"{w['weight']:.3f}", f"{w['default']:.1f}", w["status"]]
                       for w in report["weights"]], right={1, 2})
    threshold = report["threshold"]
    lines.append("auto-send threshold: " + (f"P(top) >= {threshold:.3f}, and ∅ never auto-sends" if threshold is not
                                             None else "none (nothing auto-sends at this stage or precision)"))
    shares = report["actions"]
    lines.append(f"current model on the user's {report['clips']} clips (in-sample, every split): "
                 + " · ".join(f"{name} {shares[name]:.2f}" for name in ("auto", "confirm", "abstain")))
    lines.append(f"pool recall: {_num(report['pool_recall'])} ({report['pool_hits']} of {report['labeled']} labeled "
                 "clips have the STT text in the pool)")
    return "\n".join(lines)


def render_story(story: dict) -> str:
    verdict = "" if story["correct"] is None else " (correct)" if story["correct"] else " (wrong)"
    lines = [f"clip {story['id']} · user {story['user']} · STT {json.dumps(story['stt'], ensure_ascii=False)} · VTT "
             f"{json.dumps(story['vtt'], ensure_ascii=False)}{verdict}",
             f"scored by a {story['stage']} model trained on {story['trained_on']} earlier labeled clips · action "
             f"{story['action']} · threshold {_num(story['threshold'])}",
             story["link"], "",
             f"raw open samples ({len(story['samples'])}, {sum(s['error'] is None for s in story['samples'])} valid)"]
    for s in story["samples"]:
        if s["hypotheses"] is None:
            lines.append(f"  #{s['idx']}  error: {s['error']}")
        else:
            said = " · ".join(f"{h['text']} {h['s']}" for h in s["hypotheses"]) or "(no hypotheses)"
            lines.append(f"  #{s['idx']}  {said}   speech {s['speech']} · mouth {s['mouth_visible']} · "
                         f"blur {s['blur']}")
    closed, gemini = story["closed"], story["gemini"]
    lines.append("closed call: " + (f"error: {closed['error']}" if closed["error"] else
                                    " · ".join(f"{c['phrase']} {c['s']}" for c in closed["top"])
                                    + f" · none {closed['none']}"))
    lines.append("Gemini: " + (f"error: {gemini['error']}" if gemini["error"] else
                               " · ".join(f"{g['phrase']} {g['p']:.3f}" for g in gemini["top"])))
    lines += ["", f"pool ({len(story['pool'])} candidates, by p)"]
    def mark(c: dict) -> str:  # on a pool miss the training label is ∅
        return "" if not c["is_label"] else "<- label (STT not in the pool)" if not c["sources"] else "<- STT"

    lines += _table([["p", "candidate", "sources", ""]]
                    + [[f"{c['p']:.3f}", c["text"], ",".join(c["sources"]), mark(c)] for c in story["pool"]],
                    right={0})
    lines += ["", "features"]
    lines += _table([["candidate", *FEATURES]] + [[c["text"], *(f"{c['features'][f]:.2f}" for f in FEATURES)]
                                                  for c in story["pool"]], right=set(range(1, len(FEATURES) + 1)))
    winner, runner = story["winner"], story["runner_up"]
    if runner is not None:
        lines += ["", f"contributions (weight x value): winner {json.dumps(winner['text'], ensure_ascii=False)} vs "
                      f"runner-up {json.dumps(runner['text'], ensure_ascii=False)}"]
        lines += _table([["feature", "weight", "winner", "runner-up", "gap"]]
                        + [[w["feature"], f"{story['weights'][w['feature']]:.3f}", f"{w['winner']:.3f}",
                            f"{w['runner_up']:.3f}", f"{w['gap']:+.3f}"] for w in story["why"]]
                        + [["score", "", f"{winner['score']:.3f}", f"{runner['score']:.3f}",
                            f"{winner['score'] - runner['score']:+.3f}"]], right={1, 2, 3, 4})
        lines.append(f"why {json.dumps(winner['text'], ensure_ascii=False)} won: "
                     + " · ".join(f"{w['feature']} {w['gap']:+.3f}" for w in story["why"][:3] if w["gap"] > 0))
    return "\n".join(lines)


RENDER = {"failures": render_failures, "recipe": render_recipe, "clip": render_story}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine output")
    common.add_argument("--store", type=Path, help="a file clip store (default: MongoDB or data/clips/)")
    commands = parser.add_subparsers(dest="command", required=True)
    failing = commands.add_parser("failures", parents=[common], help="VTT != STT clips by cause")
    failing.add_argument("--user", required=True)
    failing.add_argument("--split", choices=SPLITS, default="train")
    commands.add_parser("recipe", parents=[common], help="what training does for a user").add_argument(
        "--user", required=True)
    commands.add_parser("clip", parents=[common], help="one clip's post-processing").add_argument("id")
    args = parser.parse_args(argv)

    store = clips.backend(args.store)
    try:
        if args.command == "clip":
            record = store.get(args.id)
            if record is None:
                raise LookupError(f"no clip {args.id!r}")
            report = clip_story(store.load(record["user_id"]), args.id)
        elif args.command == "failures":
            report = failures(store.load(args.user), args.user, args.split)
        else:
            report = recipe(store.load(args.user), args.user)
    except LookupError as e:
        parser.error(str(e))
    print(json.dumps(report, indent=1, ensure_ascii=False) if args.json else RENDER[args.command](report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
