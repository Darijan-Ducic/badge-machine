"""Lab-prompt experiments: paired re-reads of stored dictation turns with ONE change to the Muse open call (sampling,
earlier-STT context, or the prior-only static-frame control), scored against the turn's stored baseline samples.

  uv run --env-file .env python tools/exp_prompt.py freeze --user U --set NAME [--max N]
  uv run --env-file .env python tools/exp_prompt.py free --set NAME            # K=3 vs K=5, chance, noise: no calls
  uv run --env-file .env python tools/exp_prompt.py run --set NAME --arm ARM [--k 3] [--temperature T] [--effort E]
      [--context history|session] [--context-n 8] [--static] --lever L --hypothesis H

`freeze` fixes a turn set (processed turns' clip ids, in time order) in data/exp/lab-prompt/<NAME>.json. `run` reads
each turn's stored mouth crop (the exact video the readers saw), calls Muse K times with the arm's change (≤ 4 calls
in flight, 429/5xx retried with backoff), stores each re-read as a clip of user `exp:lab-prompt-<ARM>:<U>`, and
saves a run record (runs.save) comparing the arm with the stored samples[:K] of the same turns: VTT · Muse word and
viseme accuracy, best-of-K viseme accuracy, sample agreement, per-turn up/down/tie and an exact sign-test p, cost.
A re-run of `run` reuses re-reads already stored for that arm. Context = STT of EARLIER turns of the set only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402

from preheat import _normalize  # noqa: E402

FPS = 24
SETS = Path("data/exp/lab-prompt")
AUTHOR = "agent:lab-prompt"
GENERIC = "I want to talk with you about what we can do together to make this country stronger for everyone."
_TRANSIENT = re.compile(r"RateLimit|InternalServer|APIConnection|APITimeout|ServiceUnavailable|Error code: (429|5\d\d)")


# ---------------------------------------------------------------- turns and context


def ordered_turns(records: list[dict]) -> list[dict]:
    """Processed turns of the session records, in time order: session start, then turn start."""
    out = []
    for r in sorted(records, key=lambda r: r["t"]):
        for t in sorted(r["turns"], key=lambda t: t["start_ms"]):
            if t["status"] == "processed" and t.get("clip_id"):
                out.append({"session_id": r["id"], "turn": t["i"], "t": r["t"], "start_ms": t["start_ms"],
                            "end_ms": t["end_ms"], "stt": t["stt"], "clip_id": t["clip_id"],
                            "user_id": r.get("user_id")})
    return out


def context_for(turns: list[dict], index: int, *, scope: str = "history", n: int = 8) -> list[str]:
    """STT of the (at most n most recent) turns BEFORE turns[index]; "session" keeps only its own session's."""
    earlier = turns[:index]
    if scope == "session":
        earlier = [t for t in earlier if t["session_id"] == turns[index]["session_id"]]
    elif scope != "history":
        raise ValueError(f"context scope {scope!r} must be history or session")
    texts = [t["stt"] for t in earlier if t.get("stt")]
    return texts[-n:] if n > 0 else []


# ---------------------------------------------------------------- paid calls


def transient(error: str | None) -> bool:
    return bool(error) and bool(_TRANSIENT.search(error))


def with_retry(call, *, sleep=time.sleep, delays=(2, 8, 30, 60)) -> dict:
    """One sample item from `call()` (a one-item list), retried after each delay while its error is transient."""
    for delay in (*delays, None):
        [item] = call()
        if not transient(item.get("error")) or delay is None:
            return item
        sleep(delay)


def static_video(source: Path, out: Path) -> None:
    """The prior-only control's clip: the middle frame of `source` held for as many frames as `source` has at FPS."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", source, "-vf", f"fps={FPS}", f"{tmp}/%05d.png"],
                       check=True)
        frames = sorted(Path(tmp).glob("*.png"))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i",
                        frames[len(frames) // 2], "-frames:v", str(len(frames)), "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                       check=True)


def _params(arm: dict, context) -> dict:
    params = {key: arm[key] for key in ("temperature", "effort") if arm.get(key) is not None}
    return params if context is None else {**params, "context": context}


def run_arm(turns: list[dict], arm: dict, *, video_for, sample=None, workers: int = 4, sleep=time.sleep,
            indices: list[int] | None = None) -> list[dict]:
    """Read turns[i] (every i, or `indices`) K times with the arm's change; contexts come from the full `turns`."""
    if sample is None:
        import muse_sampler
        sample = muse_sampler.sample_open
    indices = list(range(len(turns))) if indices is None else indices
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(workers) as pool:
        plans = []
        for i in indices:
            clip = Path(tmp, f"{i}.mp4")
            clip.write_bytes(video_for(turns[i]))
            if arm.get("static"):
                static_video(clip, Path(tmp, f"{i}-static.mp4"))
                clip = Path(tmp, f"{i}-static.mp4")
            context = context_for(turns, i, scope=arm["context"], n=arm.get("context_n", 8)) if arm.get(
                "context") else None
            params = _params(arm, context)
            futures = [pool.submit(with_retry, lambda c=clip, p=params: sample(c, 1, **p), sleep=sleep)
                       for _ in range(arm["k"])]
            plans.append((i, clip, context, futures))
        results = []
        for i, clip, context, futures in plans:
            samples = [{**f.result(), "sample_idx": j} for j, f in enumerate(futures)]
            results.append({"index": i, "turn": turns[i], "context": context, "samples": samples,
                            "video": clip.read_bytes()})
    return results


# ---------------------------------------------------------------- metrics


def _edits(read: list, said: list) -> int:
    from experiment import _edits as edits
    return edits(read, said)


def _top(output: dict) -> str | None:
    return _normalize(max(output["hypotheses"], key=lambda h: h["s"])["text"]) if output.get("hypotheses") else None


def turn_metrics(samples: list[dict], stt: str, k: int | None = None) -> dict:
    """One turn's read by the first k samples (all without k): VTT · Muse pick = the most common top hypothesis."""
    from visemes import to_visemes
    valid = [s["output"] for s in (samples[:k] if k else samples) if s.get("output") is not None]
    tops = [t for t in map(_top, valid) if t is not None]
    pick = Counter(tops).most_common(1)[0][0] if tops else None
    said_words, said_visemes = _normalize(stt).split(), to_visemes(stt)

    def vis(text):
        return _edits(to_visemes(text or ""), said_visemes), len(said_visemes)

    best = min((vis(t) for t in tops), default=vis(None))
    sample_accs = [1 - vis(t)[0] / len(said_visemes) for t in tops] if said_visemes else []
    return {"stt": stt, "pick": pick, "pick_words": (_edits(_normalize(pick or "").split(), said_words),
                                                      len(said_words)),
            "pick_visemes": vis(pick), "best_visemes": best,
            "sample_viseme_acc": sum(sample_accs) / len(sample_accs) if sample_accs else 0.0,
            "agreement": sum(t == pick for t in tops) / len(valid) if valid else 0.0, "valid": len(valid),
            "empty": len(valid) - len(tops), "exact": pick == _normalize(stt)}


def _acc(pair) -> float:
    edits, total = pair
    return 1 - edits / total if total else 0.0


def summarize(rows: list[dict]) -> dict:
    def pooled(key):
        total = sum(r[key][1] for r in rows)
        return 1 - sum(r[key][0] for r in rows) / total if total else None

    def mean(key):
        return sum(r[key] for r in rows) / len(rows) if rows else None

    valid = sum(r["valid"] for r in rows)
    return {"clips": len(rows), "word_acc": pooled("pick_words"), "viseme_acc": pooled("pick_visemes"),
            "best_of_k_viseme_acc": pooled("best_visemes"), "sample_viseme_acc": mean("sample_viseme_acc"),
            "agreement": mean("agreement"), "muse_acc": mean("exact"),
            "empty_read_share": sum(r["empty"] for r in rows) / valid if valid else None, "valid_samples": valid}


def sign_test(ups: int, downs: int) -> float:
    """Exact two-sided sign test (ties dropped)."""
    n = ups + downs
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(ups, downs) + 1)) / 2 ** n)


def flips(a: list[dict], b: list[dict], key: str = "pick_visemes") -> dict:
    """Per turn, does arm b read more (up) or fewer (down) visemes right than arm a."""
    out = {"up": 0, "down": 0, "tie": 0}
    for x, y in zip(a, b):
        dx, dy = (_acc(x[key]), _acc(y[key])) if key != "sample_viseme_acc" else (x[key], y[key])
        out["up" if dy > dx + 1e-9 else "down" if dy < dx - 1e-9 else "tie"] += 1
    return out


def samples_cost(samples: list[dict]) -> float:
    from experiment import _call_cost
    return sum(_call_cost(s["usage"], "muse") for s in samples if s.get("usage"))


# ---------------------------------------------------------------- storage


def exp_user(arm_name: str, user: str) -> str:
    return f"exp:lab-prompt-{arm_name}:{user}"


def store_reread(turn: dict, arm: dict, samples: list[dict], video: bytes, context, *, source: dict,
                 store=None) -> dict:
    """The re-read as a v1 clip of the experiment user (never the source user); closed and Gemini are not re-run."""
    import clips
    import muse_sampler
    not_run = "not run: lab-prompt re-reads the Muse open call only"
    crop = (source.get("config") or {}).get("crop")
    record = {
        "record": clips.V1, "id": str(time.time_ns()), "t": time.time(), "user_id": exp_user(arm["name"],
                                                                                             turn["user_id"]),
        "mode": "experiment", "fps": None, "prompted": None, "phrases": list(source["phrases"]),
        "vocab": clips.vocab_id(source["phrases"]),
        "config": {**muse_sampler.CONFIG, "crop": f"static-middle({crop})" if arm.get("static") else crop,
                   "k": arm["k"], **{key: arm[key] for key in ("temperature", "effort") if arm.get(key) is not None},
                   "prompt_id": "p3" + (f"+ctx-{arm['context']}{arm.get('context_n', 8)}" if arm.get("context")
                                        else "")},
        "tracker_flags": [], "frames": source.get("frames"), "label": {"stt_text": turn["stt"], "user_edit": None},
        "samples": samples,
        "closed": {"call": "closed", "sample_idx": 0, "latency_ms": 0, "output": None, "error": not_run},
        "gemini": {"call": "gemini", "latency_ms": 0, "output": None, "error": not_run},
        "experiment": {"lab": "lab-prompt", "name": arm["name"], "arm": arm, "session_id": turn["session_id"],
                       "turn": turn["turn"], "source_clip": turn["clip_id"], "context": context},
    }
    clips.save(record, video, store)
    return record


def run_record(*, name: str, lever: str, hypothesis: str, arms: dict, baseline: str, variant: str,
               clip_ids: list[str], cost: float, verdict: str, user_id: str, dataset: str, reopen: str | None = None,
               config: dict | None = None, extra: dict | None = None) -> dict:
    """A run record for the Training page's Runs tab: per-arm metrics, flips, sign-test p, cost, verdict."""
    from experiment import _head
    base, var = arms[baseline], arms[variant]
    f = flips(base, var)
    f_sample = flips(base, var, key="sample_viseme_acc")
    curve, total = [], [0, 0]
    for i, row in enumerate(var):
        total = [total[0] + row["pick_visemes"][0], total[1] + row["pick_visemes"][1]]
        curve.append({"i": i, "vtt": _acc(total)})
    per_turn = [{"clip_id": c, "stt": base[i]["stt"],
                 **{arm: {"pick": rows[i]["pick"], "viseme_acc": _acc(rows[i]["pick_visemes"])}
                    for arm, rows in arms.items()}} for i, c in enumerate(clip_ids)]
    return {
        "id": uuid.uuid4().hex[:12], "t": time.time(), "name": name, "dataset": dataset, "user_id": user_id,
        "author": AUTHOR, "loop": "framework", "lab": "lab-prompt", "lever": lever, "hypothesis": hypothesis,
        "baseline_arm": baseline, "variant_arm": variant, "turn_clip_ids": list(clip_ids),
        "metrics": {"clips": len(var), "muse_acc": summarize(var)["muse_acc"],
                    "arms": {arm: summarize(rows) for arm, rows in arms.items()}},
        "flips": f, "p": sign_test(f["up"], f["down"]), "flips_sample_mean": f_sample,
        "p_sample_mean": sign_test(f_sample["up"], f_sample["down"]), "cost_usd": cost, "verdict": verdict,
        "reopen": reopen, "config": config or {}, "baseline": None, "gate": None, "curve": curve,
        "per_turn": per_turn, "code_commit": _head(), **(extra or {}),
    }


# ---------------------------------------------------------------- CLI


def _set_path(name: str) -> Path:
    return SETS / f"{name}.json"


def load_set(name: str) -> tuple[str, list[dict]]:
    """(user, the set's turns in time order) rebuilt from the session store, so contexts see the same turns."""
    import sessions
    frozen = json.loads(_set_path(name).read_text())
    store = sessions._backend(None)
    records = [store.get(s["id"]) for s in sessions.list_sessions(frozen["user"])["sessions"]]
    keep = set(frozen["clip_ids"])
    turns = [t for t in ordered_turns(records) if t["clip_id"] in keep]
    missing = keep - {t["clip_id"] for t in turns}
    if missing:
        raise SystemExit(f"turn set {name}: {len(missing)} clip ids no longer in the session store")
    return frozen["user"], turns


def _freeze(args) -> int:
    import sessions
    store = sessions._backend(None)
    records = [store.get(s["id"]) for s in sessions.list_sessions(args.user)["sessions"]]
    turns = ordered_turns(records)[:args.max] if args.max else ordered_turns(records)
    SETS.mkdir(parents=True, exist_ok=True)
    _set_path(args.set).write_text(json.dumps({"user": args.user, "frozen_at": time.time(),
                                               "clip_ids": [t["clip_id"] for t in turns]}, indent=1))
    print(f"turn set {args.set}: {len(turns)} processed turns of {args.user} -> {_set_path(args.set)}")
    return 0


def _print(record: dict) -> None:
    print(f"\n{record['name']} · n={record['metrics']['clips']} · run {record['id']}")
    print(f"{'arm':<14} {'word':>6} {'viseme':>6} {'best-K':>6} {'sample':>6} {'agree':>6} {'empty':>6}")
    for arm, m in record["metrics"]["arms"].items():
        print(f"{arm:<14} {m['word_acc']:>6.3f} {m['viseme_acc']:>6.3f} {m['best_of_k_viseme_acc']:>6.3f} "
              f"{m['sample_viseme_acc']:>6.3f} {m['agreement']:>6.2f} {m['empty_read_share'] or 0:>6.2f}")
    f, g = record["flips"], record["flips_sample_mean"]
    print(f"pick viseme flips up/down/tie {f['up']}/{f['down']}/{f['tie']} p={record['p']:.3f}; sample-mean "
          f"{g['up']}/{g['down']}/{g['tie']} p={record['p_sample_mean']:.3f}; cost ${record['cost_usd']:.2f}; "
          f"verdict {record['verdict']}")


def _free(args) -> int:
    import clips
    import runs
    user, turns = load_set(args.set)
    store = clips.backend(None)
    sources = [store.get(t["clip_id"]) for t in turns]
    stts = [t["stt"] for t in turns]
    k5 = [turn_metrics(s["samples"], stt) for s, stt in zip(sources, stts)]
    k3 = [turn_metrics(s["samples"], stt, k=3) for s, stt in zip(sources, stts)]
    one = lambda text, stt: turn_metrics([{"output": {"hypotheses": [{"text": text, "s": 1}]}}], stt)  # noqa: E731
    chance = {"other_turn_stt": summarize([one(stts[(i + 1) % len(stts)], stt) for i, stt in enumerate(stts)]),
              "generic_sentence": summarize([one(GENERIC, stt) for stt in stts])}
    aa_a = [turn_metrics(s["samples"][:2], stt) for s, stt in zip(sources, stts)]
    aa_b = [turn_metrics(s["samples"][3:5], stt) for s, stt in zip(sources, stts)]
    aa = flips(aa_a, aa_b)
    rng, boot = random.Random(0), []
    for _ in range(args.reps):
        rows = [turn_metrics([rng.choice(s["samples"]) for _ in range(3)], stt) for s, stt in zip(sources, stts)]
        boot.append(summarize(rows)["viseme_acc"])
    mean = sum(boot) / len(boot)
    sd = (sum((b - mean) ** 2 for b in boot) / (len(boot) - 1)) ** 0.5
    noise = {"k3_viseme_acc_bootstrap_mean": mean, "k3_viseme_acc_bootstrap_sd": sd, "reps": args.reps,
             "aa_k2_disjoint_flips": aa, "aa_k2_p": sign_test(aa["up"], aa["down"])}
    record = run_record(name=f"K=3 vs K=5 (stored samples) · {args.set}", lever="K",
                        hypothesis="Dropping the open call from K=5 to K=3 samples costs no VTT · Muse accuracy.",
                        arms={"base-k5": k5, "k3": k3}, baseline="base-k5", variant="k3",
                        clip_ids=[t["clip_id"] for t in turns], cost=0.0, verdict=args.verdict or "inconclusive",
                        user_id=user, dataset=user, extra={"chance": chance, "noise": noise})
    if not args.dry:
        runs.save(record)
    _print(record)
    for name, m in chance.items():
        print(f"chance {name}: word {m['word_acc']:.3f} viseme {m['viseme_acc']:.3f}")
    print(f"noise: K=3 viseme acc bootstrap {mean:.3f} ± {sd:.3f} (sd); A/A K=2 disjoint flips "
          f"{aa['up']}/{aa['down']}/{aa['tie']} p={noise['aa_k2_p']:.3f}")
    return 0


def _run(args) -> int:
    import clips
    import runs
    user, turns = load_set(args.set)
    arm = {"name": args.arm, "k": args.k, "temperature": args.temperature, "effort": args.effort,
           "context": args.context, "context_n": args.context_n, "static": args.static}
    arm = {key: value for key, value in arm.items() if value not in (None, False)}
    store = clips.backend(None)
    sources = {t["clip_id"]: store.get(t["clip_id"]) for t in turns}
    done = {r["experiment"]["source_clip"]: r for r in store.load(exp_user(args.arm, user))
            if r.get("experiment", {}).get("arm") == arm}
    todo = [i for i, t in enumerate(turns) if t["clip_id"] not in done]
    print(f"{args.arm}: {len(turns)} turns, {len(turns) - len(todo)} already re-read, {len(todo)} to read, arm {arm}")
    for result in run_arm(turns, arm, video_for=lambda t: store.video(t["clip_id"]), indices=todo):
        turn = result["turn"]
        done[turn["clip_id"]] = store_reread(turn, arm, result["samples"], result["video"], result["context"],
                                             source=sources[turn["clip_id"]])
    reread = [done[t["clip_id"]] for t in turns]
    stts = [t["stt"] for t in turns]
    base = [turn_metrics(sources[t["clip_id"]]["samples"], stt, k=args.k) for t, stt in zip(turns, stts)]
    var = [turn_metrics(r["samples"], stt) for r, stt in zip(reread, stts)]
    errors = sum(s.get("output") is None for r in reread for s in r["samples"])
    cost = sum(samples_cost(r["samples"]) for r in reread)
    record = run_record(name=f"{args.arm} vs base-k{args.k} · {args.set}", lever=args.lever,
                        hypothesis=args.hypothesis, arms={f"base-k{args.k}": base, args.arm: var},
                        baseline=f"base-k{args.k}", variant=args.arm, clip_ids=[t["clip_id"] for t in turns],
                        cost=cost, verdict=args.verdict or "inconclusive", reopen=args.reopen,
                        user_id=exp_user(args.arm, user), dataset=user, config=arm,
                        extra={"reread_clip_ids": [r["id"] for r in reread], "failed_samples": errors,
                               "contexts": [r["experiment"]["context"] for r in reread]})
    runs.save(record)
    _print(record)
    print(f"failed samples {errors}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--user", required=True)
    f.add_argument("--set", required=True)
    f.add_argument("--max", type=int)
    g = sub.add_parser("free")
    g.add_argument("--set", required=True)
    g.add_argument("--reps", type=int, default=500)
    g.add_argument("--verdict")
    g.add_argument("--dry", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--temperature", type=float)
    r.add_argument("--effort")
    r.add_argument("--context", choices=["history", "session"])
    r.add_argument("--context-n", type=int)
    r.add_argument("--static", action="store_true")
    r.add_argument("--lever", required=True)
    r.add_argument("--hypothesis", required=True)
    r.add_argument("--verdict")
    r.add_argument("--reopen")
    args = p.parse_args(argv)
    return {"freeze": _freeze, "free": _free, "run": _run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
