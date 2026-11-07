"""Paid lab-frames experiments: how frames reach Muse (grid, tile width, fps, stamps) and turn length, read on the same
stored turns as the original clips. Open samples only (no closed call, no Gemini): the metrics are VTT · Muse.

  uv run --env-file .env python tools/exp_frames.py recut  --user U --n 15 [--min-ms 6000] [--name recut]
  uv run --env-file .env python tools/exp_frames.py layout --user U --n 15 --name grid3x3 --grid 3x3 [--fps 24]
                                                           [--tile-width 320] [--no-stamps]

Turn set: the first N processed turns of the user (oldest session first, turn order), at least --min-ms long; the
same N always picks the same turns, so a larger N extends a run. Re-reads are stored as clips of user
`exp:lab-frames-<name>:<U>` (mode "experiment"); a turn (or piece) already stored there is reused, not paid again.

recut: each turn is cut at its deepest loudness dips into pieces of MIN_MS..MAX_MS; each piece gets its own STT label
(asr.transcribe_turns on the piece's audio) and its own read. Control = the turn's original clip (stored, free).
Per turn, the arm is scored two ways over the same speech: the pieces' reads joined in time order against the turn's
STT (same label as the control), and each piece's read against its own STT (pooled per turn).

Every run saves a run record (runs.save, author "agent:lab-frames") with metrics per arm, paired up/down counts on
per-turn viseme accuracy, an exact two-sided sign-test p, chance baselines and the arm's cost. Evidence only.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from reread import muse_pick  # noqa: E402

MIN_MS, MAX_MS = 2000, 4000
K = 5
CALLS = threading.Semaphore(4)  # at most 4 model calls in flight, across every clip of the run
STT_CALLS = [0]


# ---------------------------------------------------------------- re-cut


def recut(start: int, end: int, levels: list[float], hop_ms: int = 50, min_ms: int = MIN_MS,
          max_ms: int = MAX_MS) -> list[tuple[int, int]]:
    """[start, end) cut at its deepest loudness dips into pieces of min_ms..max_ms. A cut leaves both sides >= min_ms;
    among those, one leaving both sides <= max_ms is preferred (fewest pieces); without any level, cut at the middle."""
    if end - start <= max_ms:
        return [(start, end)]
    middle = lambda k: k * hop_ms + hop_ms // 2  # noqa: E731
    allowed = [k for k in range(len(levels)) if middle(k) - start >= min_ms and end - middle(k) >= min_ms]
    good = [k for k in allowed if middle(k) - start <= max_ms and end - middle(k) <= max_ms]
    pick = good or allowed
    cut = middle(min(pick, key=lambda k: levels[k])) if pick else (start + end) // 2
    return recut(start, cut, levels, hop_ms, min_ms, max_ms) + recut(cut, end, levels, hop_ms, min_ms, max_ms)


# ---------------------------------------------------------------- metrics


def _edits_visemes(read: str, said: str) -> tuple[int, int]:
    import experiment
    import visemes
    a, b = visemes.to_visemes(read or ""), visemes.to_visemes(said or "")
    return experiment._edits(a, b), len(b)


def _edits_words(read: str, said: str) -> tuple[int, int]:
    import experiment
    from preheat import _normalize
    a, b = _normalize(read or "").split(), _normalize(said or "").split()
    return experiment._edits(a, b), len(b)


def best_of_k(samples: list[dict], said: str) -> tuple[int, int]:
    """(fewest viseme edits of any valid sample's top hypothesis vs `said`, viseme count of `said`)."""
    tops = [max(o["hypotheses"], key=lambda h: h["s"])["text"]
            for o in (s.get("output") for s in samples) if o and o.get("hypotheses")]
    total = _edits_visemes("", said)[1]
    return (min(_edits_visemes(t, said)[0] for t in tops) if tops else total), total


def _acc(pairs) -> float | None:
    total = sum(n for _, n in pairs)
    return 1 - sum(e for e, _ in pairs) / total if total else None


def sign_test(up: int, down: int) -> float:
    from scipy.stats import binomtest
    return 1.0 if up + down == 0 else float(binomtest(up, up + down, 0.5).pvalue)


def chance_viseme(stts: list[str]) -> float | None:
    """Viseme acc of reading each turn as another turn's STT: mean over every cyclic shift of the turn set."""
    import experiment
    n = len(stts)
    accs = [experiment.viseme_acc(stts[s:] + stts[:s], stts) for s in range(1, n)]
    accs = [a for a in accs if a is not None]
    return sum(accs) / len(accs) if accs else None


def paired(a: list[float | None], b: list[float | None]) -> dict:
    up = sum(1 for x, y in zip(a, b) if x is not None and y is not None and y > x)
    down = sum(1 for x, y in zip(a, b) if x is not None and y is not None and y < x)
    return {"up": up, "down": down, "ties": len(a) - up - down, "p": sign_test(up, down)}


# ---------------------------------------------------------------- calls


class _Throttled:
    """An OpenAI client whose chat calls wait for a slot in CALLS; the SDK retries 429/5xx with backoff."""

    def __init__(self):
        from openai import OpenAI

        from muse_video import BASE_URL
        self._client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"], timeout=600, max_retries=6)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        with CALLS:
            return self._client.chat.completions.create(**request)


@functools.cache
def _client():
    return _Throttled()


def _stt(wav: bytes) -> str:
    import asr
    for delay in (2, 5, 10, 20, None):
        try:
            with CALLS:
                STT_CALLS[0] += 1
                return asr.transcribe_turns(wav)["text"]
        except Exception:
            if delay is None:
                raise
            time.sleep(delay)


def _skip_closed(clip, phrases):
    return {"call": "closed", "sample_idx": 0, "latency_ms": 0, "output": None, "error": "skipped: open-only experiment"}


def _skip_gemini(clip, phrases, usage=None):
    raise RuntimeError("skipped: open-only experiment")


def _read(segment: bytes, *, user_id: str, stt: str, boxes: list, extra: dict, open_options: dict,
          k: int = K) -> dict:
    import clips
    import muse_sampler
    from scorers import load_phrases
    muse_sampler.make_client = _client  # this process only: every open call goes through the throttle
    sample_open = functools.partial(muse_sampler.sample_open, **open_options) if open_options else None
    record, video = clips.read_clip(segment, phrases=load_phrases(), user_id=user_id, mode="experiment", k=k,
                                    transcript=stt, mouths=boxes, extra=extra, sample_open=sample_open,
                                    score_closed=_skip_closed, gemini=_skip_gemini,
                                    **({"config": open_options} if open_options else {}))
    clips.save(record, video)
    return record


# ---------------------------------------------------------------- turn set


def turn_set(user: str, n: int, min_ms: int) -> list[dict]:
    import clips
    import sessions
    store = sessions._backend(None)
    found = []
    for summary in sorted(sessions.list_sessions(user)["sessions"], key=lambda s: s["t"]):
        record = store.get(summary["id"])
        for turn in record["turns"]:
            if turn["status"] == "processed" and turn["end_ms"] - turn["start_ms"] >= min_ms and turn.get("clip_id"):
                found.append({"session": record["id"], "container": record["container"], **turn})
                if len(found) == n:
                    break
        if len(found) == n:
            break
    store_clips = clips.backend(None)
    for t in found:
        t["old"] = store_clips.get(t["clip_id"])
    return found


def _existing(user_id: str) -> dict:
    import clips
    return {(r["experiment"]["session_id"], r["experiment"]["turn"], r["experiment"].get("piece_start_ms")): r
            for r in clips.backend(None).load(user_id) if r.get("experiment")}


# ---------------------------------------------------------------- arms


def _session_media(sid: str):
    import sessions
    store = sessions._backend(None)
    return store.get(sid), store.recording(sid), store.boxes(sid)


def run_recut(turns: list[dict], user_id: str, name: str, k: int = K) -> list[dict]:
    """Per turn: {"turn", "pieces": [{"start_ms", "end_ms", "stt", "record"}]}, pieces in time order."""
    import sessions
    have = _existing(user_id)
    jobs = []
    by_session = {}
    for t in turns:
        by_session.setdefault(t["session"], []).append(t)
    tmp = tempfile.TemporaryDirectory()
    plans = {}
    for sid, ts in by_session.items():
        record, recording, boxes = _session_media(sid)
        container = record["container"]
        source = Path(tmp.name, f"{sid}.{container}")
        source.write_bytes(recording)
        pcm = sessions._decode(source, Path(tmp.name, f"{sid}.wav"))
        levels = sessions._levels(pcm)
        for t in ts:
            pieces = recut(t["start_ms"], t["end_ms"], levels)
            plans[(sid, t["i"])] = pieces
            for a, b in pieces:
                key = (sid, t["i"], a)
                if key in have:
                    continue
                jobs.append((key, source, container, pcm, boxes, a, b, t))

    def job(item):
        (sid, i, a), source, container, pcm, boxes, a, b, t = item
        per_ms = sessions.SAMPLE_RATE // 1000
        stt = _stt(sessions._wav_bytes(pcm[a * per_ms:b * per_ms]))
        segment = Path(tmp.name, f"{sid}-{i}-{a}.{container}")
        sessions._cut(source, a, b, container, segment)
        extra = {"experiment": {"name": name, "arm": "recut", "session_id": sid, "turn": i, "piece_start_ms": a,
                                "piece_end_ms": b, "source_clip": t["clip_id"], "turn_stt": t["stt"]}}
        return _read(segment.read_bytes(), user_id=user_id, stt=stt, boxes=sessions.slice_boxes(boxes, a, b),
                     extra=extra, open_options={}, k=k)

    with ThreadPoolExecutor(2) as pool:
        for record in pool.map(job, jobs):
            e = record["experiment"]
            have[(e["session_id"], e["turn"], e["piece_start_ms"])] = record
            print(f"  piece {e['session_id'][:8]}#{e['turn']} @{e['piece_start_ms']}: {record['label']['stt_text']!r}",
                  flush=True)
    tmp.cleanup()
    out = []
    for t in turns:
        pieces = [{"start_ms": a, "end_ms": b, "record": have[(t["session"], t["i"], a)]}
                  for a, b in plans[(t["session"], t["i"])]]
        for p in pieces:
            p["stt"] = p["record"]["label"]["stt_text"] or ""
        out.append({"turn": t, "pieces": pieces})
    return out


def tops(samples: list[dict]) -> list[str]:
    """Each valid sample's top hypothesis ("" for a valid sample with none: it read no speech)."""
    return [max(o["hypotheses"], key=lambda h: h["s"])["text"] if o.get("hypotheses") else ""
            for o in (s.get("output") for s in samples) if o]


def _turn_acc(read: str, said: str, kind: str, floor: bool = False) -> float | None:
    e, n = (_edits_visemes if kind == "viseme" else _edits_words)(read, said)
    return None if not n else max(0.0, 1 - e / n) if floor else 1 - e / n


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def lift(reads: list[str], i: int, stts: list[str], kind: str, floor: bool = False
         ) -> tuple[float | None, float | None]:
    """(mean acc of the reads vs their own STT, that minus their mean acc vs every OTHER STT of the set). With
    `floor`, each acc is floored at 0, so a read far longer than a short impostor STT can't inflate the lift."""
    own = _mean(_turn_acc(r, stts[i], kind, floor) for r in reads)
    impostor = _mean(_turn_acc(r, stts[j], kind, floor) for r in reads for j in range(len(stts)) if j != i)
    return own, (None if own is None or impostor is None else own - impostor)


def score_set(reads: list[list[str]], stts: list[str]) -> list[dict]:
    """Per item: own acc and lift, viseme and word, each the mean over the item's sample reads."""
    out = []
    for i, rs in enumerate(reads):
        v_own, v_lift = lift(rs, i, stts, "viseme")
        w_own, w_lift = lift(rs, i, stts, "word")
        out.append({"vis": v_own, "vis_lift": v_lift, "word": w_own, "word_lift": w_lift, "n_reads": len(rs),
                    "vis_lift0": lift(rs, i, stts, "viseme", True)[1], "word_lift0": lift(rs, i, stts, "word", True)[1]})
    return out


def _summary(items: list[dict], best: list[tuple[int, int]] | None = None) -> dict:
    out = {k: _mean(x[k] for x in items) for k in ("vis_lift", "word_lift", "vis_lift0", "word_lift0", "vis", "word")}
    if best is not None:
        out["best_of_k_vis"] = _acc(best)
    return out


def _pairs(ctrl: list[dict], arm: list[dict]) -> dict:
    return {k: paired([c[k] for c in ctrl], [a[k] for a in arm]) for k in ("vis_lift", "word_lift", "vis_lift0", "word_lift0", "vis")}


def _open_cost(record: dict) -> float:
    import experiment
    return sum(experiment._call_cost(s["usage"], "muse") for s in record["samples"] if s.get("usage"))


def _per_sample_cost(records: list[dict]) -> float | None:
    n = sum(sum(1 for s in r["samples"] if s.get("usage")) for r in records)
    return sum(_open_cost(r) for r in records) / n if n else None


def score_recut(rows: list[dict]) -> dict:
    stts = [r["turn"]["stt"] for r in rows]
    ctrl_reads = [tops(r["turn"]["old"]["samples"]) for r in rows]
    piece_rows = [p for r in rows for p in r["pieces"]]
    piece_reads = [tops(p["record"]["samples"]) for p in piece_rows]
    piece_stts = [p["stt"] for p in piece_rows]
    joined = []
    for r in rows:
        per = [tops(p["record"]["samples"]) for p in r["pieces"]]
        k = max((len(x) for x in per), default=0)
        joined.append([" ".join(x[s % len(x)] for x in per if x).strip() for s in range(k)])
    ctrl = score_set(ctrl_reads, stts)
    arm_joined = score_set(joined, stts)
    pieces = score_set(piece_reads, piece_stts)
    at, arm_pieces = 0, []  # per turn: the mean over its pieces
    for r in rows:
        mine = pieces[at:at + len(r["pieces"])]
        at += len(r["pieces"])
        arm_pieces.append({k: _mean(x[k] for x in mine) for k in ("vis", "vis_lift", "word", "word_lift", "vis_lift0", "word_lift0")})
    ctrl_best = [best_of_k(r["turn"]["old"]["samples"], r["turn"]["stt"]) for r in rows]
    piece_best = [best_of_k(p["record"]["samples"], p["stt"]) for p in piece_rows]
    label = _acc([_edits_words(" ".join(p["stt"] for p in r["pieces"]), r["turn"]["stt"]) for r in rows])
    arm_records = [p["record"] for p in piece_rows]
    per_sample = _per_sample_cost(arm_records)
    ctrl_per_sample = _per_sample_cost([r["turn"]["old"] for r in rows])
    metrics = {
        "clips": len(rows), "pieces": len(piece_rows),
        "control": {**_summary(ctrl, ctrl_best), "usd_per_turn_k5": ctrl_per_sample and ctrl_per_sample * 5},
        "arm_joined": _summary(arm_joined),
        "arm_pieces": {**_summary(pieces, piece_best),
                       "usd_per_turn_k5": per_sample and per_sample * 5 * len(piece_rows) / len(rows)},
        "piece_labels_vs_turn_stt_word_acc": label,
        "chance_viseme_acc": {"turns": chance_viseme(stts), "pieces": chance_viseme(piece_stts)},
        "vtt_acc": None, "muse_acc": None,
    }
    per_turn = [{"clip": r["turn"]["clip_id"], "session": r["turn"]["session"], "turn": r["turn"]["i"],
                 "ms": r["turn"]["end_ms"] - r["turn"]["start_ms"],
                 "pieces": [[p["start_ms"], p["end_ms"]] for p in r["pieces"]], "stt": r["turn"]["stt"],
                 "piece_stt": [p["stt"] for p in r["pieces"]], "control": c, "joined": j, "by_piece": bp}
                for r, c, j, bp in zip(rows, ctrl, arm_joined, arm_pieces)]
    return {"metrics": metrics, "flips": {"joined_vs_control": _pairs(ctrl, arm_joined),
                                          "pieces_vs_control": _pairs(ctrl, arm_pieces)},
            "cost": round(sum(_open_cost(r) for r in arm_records), 4), "per_turn": per_turn}


def save_run(*, name: str, lever: str, hypothesis: str, user: str, exp_user: str, turns: list[dict], scored: dict,
             verdict: str, arm: dict, reopen: str | None = None) -> dict:
    import experiment
    import runs
    record = {
        "id": f"labframes-{name}-{uuid.uuid4().hex[:6]}", "t": time.time(), "name": f"lab-frames · {name}",
        "dataset": user, "user_id": exp_user, "author": "agent:lab-frames", "loop": "framework", "lever": lever,
        "hypothesis": hypothesis, "arm": arm, "turn_ids": [t["clip_id"] for t in turns],
        "metrics": scored["metrics"], "flips": scored["flips"], "cost": scored["cost"], "stt_calls": STT_CALLS[0],
        "verdict": verdict, "reopen": reopen, "per_turn": scored["per_turn"], "baseline": None, "gate": None,
        "curve": [], "code_commit": experiment._head(),
    }
    runs.save(record)
    return record


def run_layout(turns: list[dict], user_id: str, name: str, open_options: dict, k: int) -> list[dict]:
    """Per turn: the re-read of the turn's own segment with `open_options`, K=k (stored ones reused)."""
    import sessions
    have = _existing(user_id)
    tmp = tempfile.TemporaryDirectory()
    media, jobs = {}, []
    for t in turns:
        if (t["session"], t["i"], None) in have:
            continue
        if t["session"] not in media:
            record, recording, boxes = _session_media(t["session"])
            source = Path(tmp.name, f"{t['session']}.{record['container']}")
            source.write_bytes(recording)
            media[t["session"]] = (source, record["container"], boxes)
        jobs.append(t)

    def job(t):
        source, container, boxes = media[t["session"]]
        segment = Path(tmp.name, f"{t['session']}-{t['i']}.{container}")
        sessions._cut(source, t["start_ms"], t["end_ms"], container, segment)
        extra = {"experiment": {"name": name, "arm": open_options, "session_id": t["session"], "turn": t["i"],
                                "source_clip": t["clip_id"]}}
        return _read(segment.read_bytes(), user_id=user_id, stt=t["stt"], k=k, extra=extra, open_options=open_options,
                     boxes=sessions.slice_boxes(boxes, t["start_ms"], t["end_ms"]))

    with ThreadPoolExecutor(2) as pool:
        for record in pool.map(job, jobs):
            e = record["experiment"]
            have[(e["session_id"], e["turn"], None)] = record
            print(f"  turn {e['session_id'][:8]}#{e['turn']}: {tops(record['samples'])[:1]}", flush=True)
    tmp.cleanup()
    return [{"turn": t, "record": have[(t["session"], t["i"], None)]} for t in turns]


def score_layout(rows: list[dict]) -> dict:
    stts = [r["turn"]["stt"] for r in rows]
    ctrl = score_set([tops(r["turn"]["old"]["samples"]) for r in rows], stts)
    arm = score_set([tops(r["record"]["samples"]) for r in rows], stts)
    ctrl_best = [best_of_k(r["turn"]["old"]["samples"][:len(r["record"]["samples"])], r["turn"]["stt"]) for r in rows]
    arm_best = [best_of_k(r["record"]["samples"], r["turn"]["stt"]) for r in rows]
    arm_records = [r["record"] for r in rows]
    per_sample, ctrl_per_sample = _per_sample_cost(arm_records), _per_sample_cost([r["turn"]["old"] for r in rows])
    metrics = {
        "clips": len(rows),
        "control": {**_summary(ctrl, ctrl_best), "usd_per_turn_k5": ctrl_per_sample and ctrl_per_sample * 5},
        "arm": {**_summary(arm, arm_best), "usd_per_turn_k5": per_sample and per_sample * 5},
        "chance_viseme_acc": chance_viseme(stts), "vtt_acc": None, "muse_acc": None,
    }
    per_turn = [{"clip": r["turn"]["clip_id"], "session": r["turn"]["session"], "turn": r["turn"]["i"],
                 "ms": r["turn"]["end_ms"] - r["turn"]["start_ms"], "stt": r["turn"]["stt"], "control": c, "arm": a}
                for r, c, a in zip(rows, ctrl, arm)]
    return {"metrics": metrics, "flips": {"arm_vs_control": _pairs(ctrl, arm)},
            "cost": round(sum(_open_cost(r) for r in arm_records), 4), "per_turn": per_turn}


def _fmt(d: dict) -> str:
    return "  ".join(f"{k} {v:+.3f}" if "lift" in k else f"{k} {v:.3f}" for k, v in d.items()
                     if isinstance(v, float))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["recut", "layout"])
    p.add_argument("--user", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--min-ms", type=int, default=None, help="shortest turn kept (default: recut 6000, layout 0)")
    p.add_argument("--name", required=True)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--grid")
    p.add_argument("--tile-width", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--no-stamps", action="store_true")
    p.add_argument("--verdict", default="inconclusive")
    p.add_argument("--reopen")
    p.add_argument("--hypothesis")
    p.add_argument("--dry", action="store_true", help="print the turn set, read nothing")
    args = p.parse_args(argv)
    min_ms = args.min_ms if args.min_ms is not None else (6000 if args.command == "recut" else 0)
    turns = turn_set(args.user, args.n, min_ms)
    print(f"{len(turns)} turns >= {min_ms} ms: {[t['clip_id'] for t in turns]}", flush=True)
    exp_user = f"exp:lab-frames-{args.name}:{args.user}"
    if args.dry:
        return 0
    if args.command == "recut":
        scored = score_recut(run_recut(turns, exp_user, args.name, args.k))
        arm, lever = {"recut_ms": [MIN_MS, MAX_MS], "k": args.k}, "turn length"
        hypothesis = args.hypothesis or ("Re-cutting 6-8 s turns into 2-4 s pieces at loudness dips raises VTT · Muse "
                                         "lift (read vs own STT minus vs impostor STTs) over the same speech.")
    else:
        options = {k: v for k, v in (("grid", args.grid), ("tile_width", args.tile_width), ("fps", args.fps)) if v}
        if args.no_stamps:
            options["stamps"] = False
        scored = score_layout(run_layout(turns, exp_user, args.name, options, args.k))
        arm, lever = {**options, "k": args.k}, "frame layout"
        hypothesis = args.hypothesis or f"Reading turns with {options} changes VTT · Muse lift."
    for r in scored["per_turn"]:
        print(f"{r['session'][:8]}#{r['turn']:<3} {r['ms']:>5} ms  STT {r['stt']}")
    m = scored["metrics"]
    for name in ("control", "arm", "arm_joined", "arm_pieces"):
        if name in m:
            print(f"{name:<11} {_fmt(m[name])}")
    for name, f in scored["flips"].items():
        print(f"{name}: " + "  ".join(f"{k} {v['up']}up/{v['down']}down p={v['p']:.3f}" for k, v in f.items()))
    record = save_run(name=args.name, lever=lever, user=args.user, exp_user=exp_user, turns=turns, scored=scored,
                      verdict=args.verdict, arm=arm, hypothesis=hypothesis, reopen=args.reopen)
    print(json.dumps({"run": record["id"], "cost": record["cost"], "stt_calls": record["stt_calls"],
                      "chance": m["chance_viseme_acc"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
