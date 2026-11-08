"""Paid open-call schema arms on a FIXED set of stored turns (their stored mouth crops), paired against the turns'
stored v1 samples. Each re-read is stored as a clip of user `exp:lab-schema-<schema>:<user>` (mode "experiment"; the
closed and Gemini calls are copied from the source clip, same crop), so the Clips page shows them.

  uv run --env-file .env python tools/exp_schema.py turns --user U --n 40 --out turns.json   # fix the turn set
  uv run --env-file .env python tools/exp_schema.py arm --turns turns.json --schema h1 [--k 5]  # one paid arm
  uv run --env-file .env python tools/exp_schema.py report --turns turns.json --arms h1 nb1 [--save]

Metrics per arm (reads = each valid sample's top hypothesis, "sample", and VTT · Muse, the most common top hypothesis,
"pick"): LIFT = acc(read, own STT) - mean acc(read, the other turns' STTs), for word and viseme acc (primary); raw
word / viseme acc; agreement = mean pairwise 1 - word edits / longer length over the K top hypotheses; valid % (no
validation error); tokens per call; $ per turn (all K calls, input + output, experiment.PRICES); s per turn (slowest
of the K calls). Paired vs v1 per turn on sample viseme lift: up / down counts and the exact sign test.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import statistics as st  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

CACHE = Path("/tmp/lab-schema")
CONCURRENCY = 4  # budget rule: at most 4 Muse calls in flight
_RETRYABLE = ("429", "500", "502", "503", "504", "RateLimit", "InternalServer", "APIConnection", "Timeout")


# ---------------------------------------------------------------- metrics (pure)


def top_text(output: dict | None) -> str | None:
    hypotheses = (output or {}).get("hypotheses") or []
    return max(hypotheses, key=lambda h: h["s"])["text"] if hypotheses else None


def lift(read: str, own: int, stts: list[str], acc) -> float:
    """acc(read, stts[own]) - mean over the other turns u of acc(read, stts[u])."""
    others = [acc([read], [s]) for u, s in enumerate(stts) if u != own]
    return acc([read], [stts[own]]) - sum(others) / len(others)


def matched_lift(read: str, own: int, stts: list[str], acc) -> float:
    """Length-matched lift: acc(read, own STT) - mean acc(read, every window of the other turns' STT words as long as
    the own STT), so a read earns nothing for tracking the turn's length alone."""
    from preheat import _normalize
    n = len(_normalize(stts[own]).split())
    others = [w for u, s in enumerate(stts) if u != own for w in _normalize(s).split()]
    windows = [" ".join(others[j:j + n]) for j in range(0, max(1, len(others) - n + 1), max(1, n // 2))]
    return acc([read], [stts[own]]) - sum(acc([read], [w]) for w in windows) / len(windows)


def agreement(texts: list[str]) -> float | None:
    import experiment
    from preheat import _normalize
    words = [_normalize(t).split() for t in texts]
    pairs = [1 - experiment._edits(a, b) / max(len(a), len(b), 1) for a, b in combinations(words, 2)]
    return sum(pairs) / len(pairs) if pairs else None


def turn_metrics(samples: list[dict], i: int, stts: list[str]) -> dict:
    import experiment
    from tools.reread import muse_pick
    tops = [t for t in map(top_text, (s.get("output") for s in samples)) if t is not None]
    pick = muse_pick(samples) or ""
    usage = [s.get("usage") or {} for s in samples]
    row = {"valid": sum(s.get("output") is not None and not s.get("error") for s in samples) / len(samples),
           "agreement": agreement(tops),
           "completion": st.mean(u.get("completion_tokens", 0) for u in usage),
           "reasoning": st.mean((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage),
           "prompt": st.mean(u.get("prompt_tokens", 0) for u in usage),
           "usd": sum(experiment._call_cost(u, "muse") for u in usage),
           "s": max(s["latency_ms"] for s in samples) / 1000}
    from preheat import _normalize
    row["len_ratio"] = (st.mean(len(_normalize(t).split()) for t in tops) / len(_normalize(stts[i]).split())
                        if tops else 0.0)
    for name, acc in (("word", experiment.word_acc), ("viseme", experiment.viseme_acc)):
        row[f"{name}_lift"] = st.mean(lift(t, i, stts, acc) for t in tops) if tops else 0.0
        row[f"{name}_mlift"] = st.mean(matched_lift(t, i, stts, acc) for t in tops) if tops else 0.0
        row[f"{name}_acc"] = st.mean(acc([t], [stts[i]]) for t in tops) if tops else 0.0
        row[f"pick_{name}_lift"] = lift(pick, i, stts, acc)
        row[f"pick_{name}_acc"] = acc([pick], [stts[i]])
    return row


def summary(rows: list[dict]) -> dict:
    return {key: st.mean(r[key] for r in rows if r[key] is not None) for key in rows[0]} | {"n": len(rows)}


def paired(base: list[dict], cand: list[dict], key: str) -> dict:
    from tools.compare import sign_test
    up = sum(c[key] > b[key] for b, c in zip(base, cand))
    down = sum(c[key] < b[key] for b, c in zip(base, cand))
    return {"up": up, "down": down, "tie": len(base) - up - down, "p": sign_test(down, up)}


# ---------------------------------------------------------------- turns, crops, calls


def fix_turns(user: str, n: int, out: Path) -> list[dict]:
    import clips
    stored = [r for r in clips.backend(None).load(user) if r.get("record") == clips.V1 and r["label"]["stt_text"]]
    turns = [{"id": r["id"], "stt": r["label"]["stt_text"], "user": user} for r in stored[:n]]
    out.write_text(json.dumps({"user": user, "fixed_at": time.time(), "turns": turns}, indent=1))
    return turns


def crop(clip_id: str) -> Path:
    path = CACHE / "crops" / f"{clip_id}.mp4"
    if not path.exists():
        import clips
        video = clips.backend(None).video(clip_id)
        if video is None:
            raise SystemExit(f"no stored crop for clip {clip_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(video)
    return path


def _with_retry(client, i, request, finish, tries: int = 5) -> dict:
    from muse_sampler import _call
    for attempt in range(tries):
        item = _call(client, "open", i, request, finish)
        if not item["error"] or item["output"] is not None or not any(c in item["error"] for c in _RETRYABLE):
            return item
        time.sleep(2 ** attempt + random.random())
    return item


def run_arm(turns_file: Path, schema: str, k: int) -> Path:
    import clips
    import muse_sampler
    from muse_video import make_client
    spec = json.loads(turns_file.read_text())
    user, turns = spec["user"], spec["turns"]
    out = CACHE / f"{turns_file.stem}-{schema}.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    todo = [t for t in turns if t["id"] not in done]
    client, lock = make_client(), threading.Lock()
    prepared = {t["id"]: muse_sampler.variant_request(crop(t["id"]), schema) for t in todo}
    jobs = [(t["id"], i) for t in todo for i in range(k)]
    results: dict[str, list] = {t["id"]: [None] * k for t in todo}

    def one(job):
        clip_id, i = job
        request, finish, _ = prepared[clip_id]
        item = _with_retry(client, i, request, finish)
        with lock:
            results[clip_id][i] = item
            if all(results[clip_id]):
                _store(clip_id, results[clip_id], schema, user, k)
                done[clip_id] = results[clip_id]
                out.write_text(json.dumps(done))
                print(f"  {schema} {clip_id}: {sum(not s['error'] for s in results[clip_id])}/{k} valid, "
                      f"{max(s['latency_ms'] for s in results[clip_id]) / 1000:.1f}s", flush=True)

    with ThreadPoolExecutor(CONCURRENCY) as pool:
        list(pool.map(one, jobs))
    return out


def _store(clip_id: str, samples: list[dict], schema: str, user: str, k: int) -> None:
    import clips
    import muse_sampler
    source = clips.backend(None).get(clip_id)
    record = {**source, "id": str(time.time_ns()), "t": time.time(), "user_id": f"exp:lab-schema-{schema}:{user}",
              "mode": "experiment", "samples": samples,
              "config": {**source["config"], **muse_sampler.CONFIG, "schema": schema, "k": k,
                         "crop": source["config"]["crop"]},
              "experiment": {"name": f"lab-schema-{schema}", "schema": schema, "source_clip": clip_id,
                             "closed_gemini": "copied from source clip (same crop)"}}
    clips.save(record, crop(clip_id).read_bytes())


# ---------------------------------------------------------------- report


def arm_samples(turns_file: Path, schema: str) -> dict[str, list]:
    if schema == "v1-stored":
        import clips
        spec = json.loads(turns_file.read_text())
        path = CACHE / f"{turns_file.stem}-v1-stored.json"
        if not path.exists():
            store = clips.backend(None)
            path.write_text(json.dumps({t["id"]: store.get(t["id"])["samples"] for t in spec["turns"]}))
        return json.loads(path.read_text())
    return json.loads((CACHE / f"{turns_file.stem}-{schema}.json").read_text())


def report(turns_file: Path, arms: list[str]) -> dict:
    spec = json.loads(turns_file.read_text())
    arms = ["v1-stored", *[a for a in arms if a != "v1-stored"]]
    data = {a: arm_samples(turns_file, a) for a in arms}
    ids = [t["id"] for t in spec["turns"] if all(t["id"] in data[a] for a in arms)]
    stts = [t["stt"] for t in spec["turns"] if t["id"] in ids]
    rows = {a: [turn_metrics(data[a][cid], i, stts) for i, cid in enumerate(ids)] for a in arms}
    result = {"turns": ids, "arms": {}}
    print(f"{len(ids)} turns (all arms present) of {len(spec['turns'])} fixed; user {spec['user']}")
    print(f"{'arm':<10} {'vis lift':>8} {'word lift':>9} {'pick vis':>8} {'vis acc':>7} {'word acc':>8} {'agree':>6} "
          f"{'valid':>5} {'out tok':>7} {'reas':>5} {'in tok':>6} {'$/turn':>6} {'s/turn':>6}  paired vis lift vs v1")
    for a in arms:
        s = summary(rows[a])
        pv = paired(rows["v1-stored"], rows[a], "viseme_lift")
        pw = paired(rows["v1-stored"], rows[a], "word_lift")
        pm = paired(rows["v1-stored"], rows[a], "viseme_mlift")
        result["arms"][a] = {"summary": s, "paired_viseme_lift": pv, "paired_word_lift": pw,
                             "paired_viseme_mlift": pm, "rows": rows[a]}
        print(f"{a:<10} {s['viseme_lift']:>+8.3f} {s['word_lift']:>+9.3f} {s['pick_viseme_lift']:>+8.3f} "
              f"{s['viseme_acc']:>7.3f} {s['word_acc']:>8.3f} {s['agreement']:>6.3f} {s['valid']:>5.0%} "
              f"{s['completion']:>7.0f} {s['reasoning']:>5.0f} {s['prompt']:>6.0f} {s['usd']:>6.3f} {s['s']:>6.1f}"
              f"  up {pv['up']} down {pv['down']} p={pv['p']:.3g} | word up {pw['up']} down {pw['down']} "
              f"p={pw['p']:.3g}")
        print(f"{'':<10} len ratio {s['len_ratio']:.2f}; length-matched lift vis {s['viseme_mlift']:+.3f} word "
              f"{s['word_mlift']:+.3f}; paired vis mlift up {pm['up']} down {pm['down']} p={pm['p']:.3g}")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("turns")
    t.add_argument("--user", required=True)
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("arm")
    a.add_argument("--turns", type=Path, required=True)
    a.add_argument("--schema", required=True)
    a.add_argument("--k", type=int, default=5)
    r = sub.add_parser("report")
    r.add_argument("--turns", type=Path, required=True)
    r.add_argument("--arms", nargs="+", required=True)
    r.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "turns":
        turns = fix_turns(args.user, args.n, args.out)
        print(f"fixed {len(turns)} turns -> {args.out}")
    elif args.cmd == "arm":
        print(f"arm written to {run_arm(args.turns, args.schema, args.k)}")
    else:
        result = report(args.turns, args.arms)
        if args.json:
            args.json.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
