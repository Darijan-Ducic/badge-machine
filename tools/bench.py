"""Closed-vocabulary benchmarks on public lip-reading data: pick the right phrase from a short known list.

Usage:
  uv run python tools/bench.py manifest crema DIR --out M.jsonl [--per-actor N]   CREMA-D *.flv in DIR
  uv run python tools/bench.py manifest grid DIR --out M.jsonl [--per-speaker N] [--distractors N] [--seed S]
  uv run --env-file .env python tools/bench.py ingest M.jsonl --dataset crema-d [ingest_practice flags]
  uv run --env-file .env python tools/bench.py measure --dataset crema-d [--save] [--json OUT]

Datasets become practice users `practice:<dataset>:<speaker>` (tools/ingest_practice.py; every manifest line
carries its own phrase list): CREMA-D (91 actors x 12 fixed sentences, ODbL; phrase list = the 12 sentences, chance
1/12) and GRID (34 speakers, 6-slot grammar, CC BY 4.0; phrase list per speaker = their sentences in the benchmark
plus seeded distractors that change 2 slots of a present one). `ingest` runs ingest_practice with at most
MAX_CALLS model calls in flight and retries 429/5xx with backoff. `measure` replays the stored raw outputs (no
model calls) per speaker and pooled:
  open exact / word acc   VTT · Muse (the most common top hypothesis across K) vs the label
  closed / Gemini top-k   label among the k best-scored phrases; ties split as a random tie-break would
  fused                   the staged scorer predict-then-train: defaults on every clip; temperature on clips with
                          >= 10 earlier labels; core on clips with >= 30 (each paired with defaults on the same clips)
  auto-send               the staged scorer's decision (temperature from 10, core from 30, 5-fold threshold) on clips
                          with >= 10 earlier labels: coverage = share sent, precision = share of sent that are right
  slots (GRID)            per-slot word hits of the open read and the fused pick, aligned word by word
Personal = one speaker's own stream; pooled = every speaker's clips in store order, so the scorer trains across
speakers (the framework loop). Each clip's features use only its own speaker's earlier labels (as the app does).
"""

import argparse
import contextlib
import dataclasses
import difflib
import json
import random
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import clips  # noqa: E402
from candidates import match  # noqa: E402
from experiment import _edits, clip_cost  # noqa: E402
from preheat import _normalize  # noqa: E402
from scorer import DEFAULT_WEIGHTS, Model, auto_send, predict, train  # noqa: E402

CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock", "TIE": "That is exactly what happened", "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about", "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold", "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment", "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before", "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}
GRID_GRAMMAR = (("bin", "lay", "place", "set"), ("blue", "green", "red", "white"), ("at", "by", "in", "with"),
                tuple("abcdefghijklmnopqrstuvxyz"),
                ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
                ("again", "now", "please", "soon"))
MAX_CALLS = 4
SAM_COST = 0.01  # USD per SAM video call, an estimate: SAM returns no token usage
DEFAULTS_PLAN = [(0, "defaults")]
TEMPERATURE_PLAN = [(0, "defaults"), (10, "temperature")]
CORE_PLAN = [(0, "defaults"), (10, "temperature"), (30, "core")]


# ---------------------------------------------------------------- loaders and labels


def crema_item(path) -> dict:
    """A CREMA-D clip `<actor>_<sentence>_<emotion>_<level>.flv` as a manifest line (phrases added by the caller)."""
    actor, code, emotion, _ = Path(path).stem.split("_")
    if code not in CREMA_SENTENCES:
        raise ValueError(f"{path}: unknown CREMA-D sentence code {code!r}")
    return {"clip": str(path), "text": CREMA_SENTENCES[code], "speaker": actor, "emotion": emotion}


def grid_sentence(align: str) -> str:
    """A GRID .align file (start end word per line) as its sentence, silence (sil) and pauses (sp) dropped."""
    words = [line.split()[2] for line in align.splitlines() if len(line.split()) == 3]
    return " ".join(w for w in words if w not in ("sil", "sp"))


def grid_distractors(present: list[str], n: int, seed: int) -> list[str]:
    """`n` grammatical GRID sentences not in `present`, each `present` sentence with 2 slots changed (seeded)."""
    rng, taken, out = random.Random(seed), set(present), []
    while len(out) < n:
        words = rng.choice(present).split()
        for slot in rng.sample(range(6), 2):
            words[slot] = rng.choice([w for w in GRID_GRAMMAR[slot] if w != words[slot]])
        text = " ".join(words)
        if text not in taken and min(sum(a != b for a, b in zip(words, p.split())) for p in present) == 2:
            taken.add(text)
            out.append(text)
    return out


# ---------------------------------------------------------------- metrics


def topk_credit(scores: dict | None, truth: str, k: int) -> float:
    """Credit for `truth` being among the k best of `scores` {phrase: score}; a tie straddling the cut gets the
    chance a uniform random tie-break puts it inside. A failed call (None) or a truth not scored is a miss."""
    if not scores:
        return 0.0
    at = match(truth, list(scores))
    if at is None:
        return 0.0
    mine = list(scores.values())[at]
    above = sum(v > mine for v in scores.values())
    tied = sum(v == mine for v in scores.values())
    return 0.0 if above >= k else min(1.0, (k - above) / tied)


def slot_hits(read: str | None, truth: str) -> list[int]:
    """Per word of `truth`, 1 if the aligned word of `read` equals it (difflib matching blocks over words)."""
    said, seen = _normalize(truth).split(), _normalize(read or "").split()
    hits = [0] * len(said)
    for block in difflib.SequenceMatcher(a=said, b=seen, autojunk=False).get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            hits[i] = 1
    return hits


def _gemini_scores(record: dict) -> dict | None:
    return record["gemini"].get("output")


@contextlib.contextmanager
def _uncapped_pool():
    """Featurize with every closed / Gemini-scored phrase in the pool and no pool cap (bench replay only)."""
    import candidates
    cap, top = candidates.POOL_CAP, candidates._top
    candidates.POOL_CAP, candidates._top = 10_000, lambda scores, n: top(scores, len(scores))
    try:
        yield
    finally:
        candidates.POOL_CAP, candidates._top = cap, top


def closed_pool(clip, phrases: list[str]):
    """`clip` with its pool cut to the phrase-list entries (pool order) plus ∅: closed-vocabulary mode."""
    keep = [j for j, text in enumerate(clip.pool[:-1]) if match(text, phrases) is not None]
    rows = keep + [len(clip.pool) - 1]
    pool = [clip.pool[j] for j in rows]
    label = None
    if clip.stt is not None:
        hit = match(clip.stt, pool[:-1])
        label = hit if hit is not None else len(pool) - 1
    return dataclasses.replace(clip, pool=pool, features=clip.features[rows], label=label)


def reads(records: list[dict]) -> list[dict]:
    """Per labeled v1 clip, in the given order: each reader against the label, the featurized clip as the app pools
    it (`clip`) and in closed-vocabulary mode (`closed_clip`: phrase list + ∅)."""
    records = [r for r in records if r.get("record") == "v1" and r["label"].get("stt_text")]
    featurized, uncapped = {}, {}
    for user in dict.fromkeys(r["user_id"] for r in records):
        mine = [r for r in records if r["user_id"] == user]
        featurized.update(zip((r["id"] for r in mine), clips._featurized(mine)))  # own earlier labels only
        with _uncapped_pool():
            uncapped.update(zip((r["id"] for r in mine), clips._featurized(mine)))
    rows = []
    for r in records:
        clip, truth = featurized[r["id"]], r["label"]["stt_text"]
        closed = (r["closed"].get("output") or {}).get("scores")
        grid = len(truth.split()) == 6 and all(w in s for w, s in zip(_normalize(truth).split(), GRID_GRAMMAR))
        rows.append({
            "user": r["user_id"], "id": r["id"], "truth": truth, "clip": clip, "open_pick": clip.muse_pick,
            "closed_clip": closed_pool(uncapped[r["id"]], r["phrases"]),
            "open_exact": clip.muse_pick is not None and match(truth, [clip.muse_pick]) is not None,
            "open_words": (_edits(_normalize(clip.muse_pick or "").split(), _normalize(truth).split()),
                           len(_normalize(truth).split())),
            "closed_top1": topk_credit(closed, truth, 1), "closed_top3": topk_credit(closed, truth, 3),
            "gemini_top1": topk_credit(_gemini_scores(r), truth, 1),
            "gemini_top3": topk_credit(_gemini_scores(r), truth, 3),
            "slots": slot_hits(clip.muse_pick, truth) if grid else None,
            "errors": sum(s.get("error") is not None for s in r["samples"]) + (r["closed"].get("error") is not None)
                      + (r["gemini"].get("error") is not None),
            "cost": clip_cost(r),
        })
    return rows


def _pick(clip, probs) -> tuple[bool, str | None]:
    top, none = int(np.argmax(probs)), len(clip.pool) - 1
    text = None if top == none else clip.pool[top]
    return text is not None and match(clip.stt, [text]) is not None, text


def stream(rows: list[dict]) -> list[dict]:
    """Predict-then-train over `rows` in order: each clip scored by models fit on the clips before it only."""
    defaults = Model("defaults", DEFAULT_WEIGHTS.copy(), np.inf)
    out = []
    for i, row in enumerate(rows):
        clip, earlier = row["clip"], [r["clip"] for r in rows[:i]]
        staged = train(earlier, stages=CORE_PLAN)  # gated: carries the auto-send threshold
        temperature = train(earlier, gate=False, stages=TEMPERATURE_PLAN)
        probs = predict(staged, [clip])[0]
        core_ok, core_text = _pick(clip, probs)
        out.append({
            **row, "n_before": i, "stage": staged.stage,
            "fused_defaults": _pick(clip, predict(defaults, [clip])[0])[0],
            "fused_temperature": _pick(clip, predict(temperature, [clip])[0])[0],
            "fused_core": core_ok, "fused_pick": core_text, "sent": bool(auto_send(staged, probs)),
            "fused_slots": slot_hits(core_text, row["truth"]) if row["slots"] is not None else None,
        })
    return out


def _mean(values) -> float | None:
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict]) -> dict:
    """Benchmark numbers over stream rows (one speaker's, or pooled)."""
    after10 = [r for r in rows if r["n_before"] >= 10]
    after30 = [r for r in rows if r["n_before"] >= 30]
    sent = [r for r in after10 if r["sent"]]
    words = sum(r["open_words"][1] for r in rows)
    summary = {
        "n": len(rows),
        "open_exact": _mean(r["open_exact"] for r in rows),
        "open_word_acc": 1 - sum(r["open_words"][0] for r in rows) / words if words else None,
        **{key: _mean(r[key] for r in rows) for key in ("closed_top1", "closed_top3", "gemini_top1", "gemini_top3")},
        "fused_defaults": _mean(r["fused_defaults"] for r in rows),
        "n_10": len(after10), "fused_10": _mean(r["fused_temperature"] for r in after10),
        "fused_10_defaults": _mean(r["fused_defaults"] for r in after10),
        "n_30": len(after30), "fused_30": _mean(r["fused_core"] for r in after30),
        "fused_30_defaults": _mean(r["fused_defaults"] for r in after30),
        "autosend_coverage": _mean(r["sent"] for r in after10),
        "autosend_precision": _mean(r["fused_core"] for r in sent),
        "pool_recall": _mean(r["clip"].label != len(r["clip"].pool) - 1 for r in rows),
        "errors": sum(r.get("errors", 0) for r in rows),
        "cost": sum(r.get("cost") or 0 for r in rows),
    }
    slotted = [r for r in rows if r.get("slots") is not None]
    if slotted:
        summary["open_slots"] = [float(x) for x in np.mean([r["slots"] for r in slotted], axis=0)]
        summary["fused_slots"] = [float(x) for x in np.mean([r["fused_slots"] for r in slotted], axis=0)]
    return summary


# ---------------------------------------------------------------- manifests


def crema_manifest(folder: Path, per_actor: int | None) -> list[dict]:
    phrases = list(CREMA_SENTENCES.values())
    items = [crema_item(p) for p in sorted(Path(folder).glob("*.flv"))]
    by_actor = {}
    for item in items:
        by_actor.setdefault(item["speaker"], []).append(item)
    return [{**item, "phrases": phrases} for group in by_actor.values() for item in group[:per_actor]]


def grid_manifest(folder: Path, per_speaker: int, distractors: int, seed: int) -> list[dict]:
    """GRID videos under folder/<speaker>/*.mpg with alignments under folder/align/<speaker>/<name>.align."""
    out = []
    for speaker_dir in sorted(p for p in Path(folder).iterdir() if re.fullmatch(r"s\d+", p.name)):
        lines = []
        for video in sorted(speaker_dir.glob("*.mpg"))[:per_speaker]:
            align = Path(folder, "align", speaker_dir.name, video.stem + ".align")
            if align.exists():
                lines.append({"clip": str(video), "text": grid_sentence(align.read_text()),
                              "speaker": speaker_dir.name})
        present = list(dict.fromkeys(line["text"] for line in lines))
        phrases = sorted(present + grid_distractors(present, distractors, seed))
        out += [{**line, "phrases": phrases} for line in lines]
    return out


# ---------------------------------------------------------------- ingest with a call limit and retries

_RETRY = re.compile(r"\b(429|500|502|503|504)\b|RateLimit|InternalServer|Timeout|Connection|UNAVAILABLE|"
                    r"RESOURCE_EXHAUSTED|overloaded", re.I)


def limit_calls(max_calls: int = MAX_CALLS, tries: int = 5) -> None:
    """At most `max_calls` Muse / Gemini / SAM calls in flight process-wide; transient failures retried with
    exponential backoff (2, 4, 8, 16 s)."""
    import muse_sampler
    import sam
    import scorers

    gate = threading.Semaphore(max_calls)
    call, gemini, segment = muse_sampler._call, scorers.gemini_scores, sam.segment_video

    def muse(*args, **kwargs):
        for attempt in range(tries):
            with gate:
                item = call(*args, **kwargs)
            if item["output"] is not None or not _RETRY.search(item["error"] or "") or attempt == tries - 1:
                return item
            time.sleep(2 ** (attempt + 1))

    def retried(fn):
        def wrapper(*args, **kwargs):
            for attempt in range(tries):
                try:
                    with gate:
                        return fn(*args, **kwargs)
                except Exception as e:
                    if not _RETRY.search(f"{type(e).__name__}: {e}") or attempt == tries - 1:
                        raise
                time.sleep(2 ** (attempt + 1))
        return wrapper

    muse_sampler._call, scorers.gemini_scores, sam.segment_video = muse, retried(gemini), retried(segment)


# ---------------------------------------------------------------- measure


def _run_record(dataset: str, scope: str, summary: dict, rows: list[dict], user: str | None) -> dict:
    import uuid

    from experiment import _head
    best = summary["fused_30"] if summary["n_30"] else summary["fused_10"]
    curve = [{"i": i, "vtt": _mean(r["fused_core"] for r in rows[max(0, i - 19):i + 1]),
              "gemini": _mean(r["gemini_top1"] for r in rows[max(0, i - 19):i + 1]),
              "muse": _mean(r["open_exact"] for r in rows[max(0, i - 19):i + 1])} for i in range(len(rows))]
    return {
        "id": uuid.uuid4().hex[:12], "t": time.time(), "name": f"bench {dataset} · {scope}", "dataset": dataset,
        "user_id": user, "config": {"stages": dict((n, m) for m, n in CORE_PLAN), "bench": "closed-set v1"},
        "baseline": None, "gate": None, "curve": curve, "author": "agent:lab-bench", "loop": "framework",
        "hypothesis": f"closed-set benchmark {dataset} ({scope}): how well each reader picks the phrase from the list",
        "metrics": {"clips": summary["n"], "vtt_acc": summary["fused_defaults"], "vtt_acc_heldout": best,
                    "gemini_acc": summary["gemini_top1"], "muse_acc": summary["open_exact"],
                    "pool_recall": _mean(r["clip"].label != len(r["clip"].pool) - 1 for r in rows),
                    "autosend_precision": summary["autosend_precision"],
                    "autosend_coverage": summary["autosend_coverage"], **summary},
        "n": summary["n"], "cost": summary["cost"], "code_commit": _head(),
        "system_version": None, "candidate_version": None, "parent_version": None,
    }


def measure(records: list[dict], dataset: str) -> dict:
    prefix = f"practice:{dataset}:"
    records = [r for r in records if str(r.get("user_id", "")).startswith(prefix)]
    rows = reads(records)
    users = list(dict.fromkeys(r["user"] for r in rows))
    closed_rows = [{**r, "clip": r["closed_clip"]} for r in rows]
    personal = {u: stream([r for r in rows if r["user"] == u]) for u in users}
    personal_cp = {u: stream([r for r in closed_rows if r["user"] == u]) for u in users}
    pooled, pooled_cp = stream(rows), stream(closed_rows)
    return {"dataset": dataset, "chance": None,
            "personal": {u: summarize(s) for u, s in personal.items()}, "pooled": summarize(pooled),
            "closed_pool": {**{u: summarize(s) for u, s in personal_cp.items()}, "pooled": summarize(pooled_cp)},
            "_rows": {"pooled": pooled, **personal}, "_rows_cp": {"pooled": pooled_cp, **personal_cp}}


def _fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v}"


def _grid(lines: list[list[str]]) -> str:
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(line, widths)) for line in lines)


def _scopes(block: dict) -> list[tuple[str, dict]]:
    return [(k, v) for k, v in block.items() if k != "pooled"] + [("pooled", block["pooled"])]


def table(result: dict, chance: float | None) -> str:
    """Readers per speaker and pooled, then the fused scorer with the app's pool and with the closed pool."""
    app = {**result["personal"], "pooled": result["pooled"]}
    lines = [["scope", "n", "open=", "word", "cl@1", "cl@3", "gem@1", "gem@3", "err", "$"]]
    for scope, s in _scopes(app):
        lines.append([scope.split(":")[-1], str(s["n"]), _fmt(s["open_exact"]), _fmt(s["open_word_acc"]),
                      _fmt(s["closed_top1"]), _fmt(s["closed_top3"]), _fmt(s["gemini_top1"]),
                      _fmt(s["gemini_top3"]), str(s["errors"]), f"{s['cost']:.2f}"])
    text = _grid(lines)
    for title, block in [("fused, app pool", app), ("fused, closed pool (phrase list + ∅)", result["closed_pool"])]:
        lines = [["scope", "pool", "@def", "@10 (def same clips)", "@30 (def same clips)", "send cov", "send prec"]]
        for scope, s in _scopes(block):
            lines.append([scope.split(":")[-1], _fmt(s["pool_recall"]), _fmt(s["fused_defaults"]),
                          f"{_fmt(s['fused_10'])} ({_fmt(s['fused_10_defaults'])}) n={s['n_10']}",
                          f"{_fmt(s['fused_30'])} ({_fmt(s['fused_30_defaults'])}) n={s['n_30']}",
                          _fmt(s["autosend_coverage"]), _fmt(s["autosend_precision"])])
        text += f"\n\n{title}\n" + _grid(lines)
    if chance is not None:
        text += f"\nchance top-1 {chance * 100:.1f}%"
    for scope, s in _scopes(app):
        if "open_slots" in s:
            cp = (result["closed_pool"].get(scope) or {}).get("fused_slots") or []
            text += (f"\n{scope.split(':')[-1]} slots cmd/color/prep/letter/digit/adv: open "
                     f"{' '.join(_fmt(x) for x in s['open_slots'])} | fused, closed pool "
                     f"{' '.join(_fmt(x) for x in cp)}")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("kind", choices=["crema", "grid"])
    m.add_argument("folder", type=Path)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--per-actor", "--per-speaker", dest="per", type=int)
    m.add_argument("--distractors", type=int, default=12)
    m.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("ingest")
    i.add_argument("manifest")
    i.add_argument("--max-calls", type=int, default=MAX_CALLS)
    q = sub.add_parser("measure")
    q.add_argument("--dataset", required=True)
    q.add_argument("--save", action="store_true", help="save one run per speaker and one pooled (runs.save)")
    q.add_argument("--json", type=Path)
    args, rest = parser.parse_known_args(argv)

    if args.command == "manifest":
        items = (crema_manifest(args.folder, args.per) if args.kind == "crema"
                 else grid_manifest(args.folder, args.per or 12, args.distractors, args.seed))
        args.out.write_text("".join(json.dumps(item) + "\n" for item in items))
        print(f"{len(items)} clips, {len({i['speaker'] for i in items})} speakers -> {args.out}")
        return 0
    if args.command == "ingest":
        from tools import ingest_practice
        limit_calls(args.max_calls)
        return ingest_practice.main([args.manifest, *rest])
    records = [r for r in clips.load() if str(r.get("user_id", "")).startswith(f"practice:{args.dataset}:")]
    result = measure(records, args.dataset)
    sizes = sorted({len(r["phrases"]) for r in records})
    chance = 1 / sizes[0] if len(sizes) == 1 else None
    result["chance"] = chance
    print(table(result, chance))
    if len(sizes) > 1:
        print(f"phrase list sizes {sizes}")
    if args.json:
        args.json.write_text(json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, indent=1))
    if args.save:
        import runs
        app = {**result["personal"], "pooled": result["pooled"]}
        for mode, block, rows in [("app pool", app, result["_rows"]),
                                  ("closed pool", result["closed_pool"], result["_rows_cp"])]:
            for scope, s in _scopes(block):
                readers = {k: app[scope][k] for k in ("open_exact", "open_word_acc", "closed_top1", "closed_top3",
                                                      "gemini_top1", "gemini_top3", "cost")}
                record = _run_record(args.dataset, f"{scope.split(':')[-1]} · {mode}", {**s, **readers}, rows[scope],
                                     scope if scope != "pooled" else None)
                record["metrics"]["chance"] = chance
                runs.save(record)
                print(f"saved run {record['id']} ({scope}, {mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
