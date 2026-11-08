"""Paid few-shot re-read (lab-personal): read a user's stored dictation turns again with that user's own EARLIER labeled
clips (frames + STT) as examples in the Muse open prompt, paired against each turn's original stored read.

  uv run --env-file .env python tools/exp_personal_fewshot.py --user U [--examples 2] [--k 5] [--targets 15]
      [--from 2] [--name lab-personal-fewshot] [--save]

Targets: the user's labeled clips (store order) from index --from on, at most --targets new ones per run; a target
already re-read under --name is reused, never paid twice. Examples for target i: personal.earlier(records, i,
last=--examples), the most recent earlier labeled clips of the same user, never i itself or a later clip. Each re-read
is stored as user exp:<name>:<U> (mode "experiment"; closed and Gemini copied from the source clip, not re-called).
Control: the source clip's own stored K samples (same crop, same images, default prompt). Metric: VTT · Muse (most
common top hypothesis) word and viseme accuracy vs STT; paired per-clip word-acc up/down and an exact sign test.
At most 4 calls in flight; 429/5xx/connection errors retry with backoff. --save writes control + few-shot run records.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import personal  # noqa: E402

RETRYABLE = ("429", "500", "502", "503", "504", "rate", "overloaded", "Connection", "Timeout", "timed out")
_slots = threading.Semaphore(4)  # <= 4 model calls in flight, whatever the thread count


def _one_call(clip: Path, examples: list[dict], idx: int, tries: int = 5) -> dict:
    import muse_sampler
    for attempt in range(tries):
        with _slots:
            [item] = muse_sampler.sample_open(clip, 1, examples=examples)
        if not item["error"] or not any(m in item["error"] for m in RETRYABLE) or attempt == tries - 1:
            return {**item, "sample_idx": idx}
        time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def reread(records: list[dict], i: int, *, n_examples: int, k: int, name: str, pool: ThreadPoolExecutor) -> dict:
    """Re-read records[i] with its user's n most recent earlier labeled clips as examples; store and return it."""
    import clips
    import muse_sampler
    store = clips.backend(None)
    target, shots = records[i], personal.earlier(records, i, last=n_examples)
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp, "clip.mp4")
        clip.write_bytes(store.video(target["id"]))
        examples = []
        for n, ex in enumerate(shots):
            path = Path(tmp, f"example{n}.mp4")
            path.write_bytes(store.video(ex["id"]))
            examples.append({"clip": path, "text": personal.label(ex)})
        samples = list(pool.map(lambda idx: _one_call(clip, examples, idx), range(k)))
        video = clip.read_bytes()
    record = {**target, "id": str(time.time_ns()), "t": time.time(), "user_id": f"exp:{name}:{target['user_id']}",
              "mode": "experiment", "samples": samples,
              "config": {**target["config"], "k": k,
                         "prompt_id": f"{muse_sampler.CONFIG['prompt_id']}+{muse_sampler.FEWSHOT_ID}"},
              "experiment": {"name": name, "lever": "few-shot", "source_clip": target["id"],
                             "examples": [ex["id"] for ex in shots], "copied": ["closed", "gemini"]}}
    clips.save(record, video)
    return record


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _open_cost(samples: list[dict]) -> float:
    import experiment
    return sum(experiment._call_cost(s["usage"], "muse") for s in samples if s.get("usage"))


def paired(rows: list[dict], a: str, b: str, key: str = "word") -> dict:
    from tools.compare import sign_test
    up = [r["id"] for r in rows if r[b][key] > r[a][key] + 1e-12]
    down = [r["id"] for r in rows if r[b][key] < r[a][key] - 1e-12]
    return {"up": len(up), "down": len(down), "ties": len(rows) - len(up) - len(down),
            "p": sign_test(len(down), len(up)), "up_ids": up, "down_ids": down}


def run_record(rows, arm, *, name, user, hypothesis, lever, cost, baseline=None, flips=None, verdict=None,
               extra=None) -> dict:
    import uuid

    import experiment
    stts = [r["stt"] for r in rows]
    texts = [r[arm]["text"] for r in rows]
    curve = [{"i": j, "vtt": experiment.word_acc(texts[:j + 1], stts[:j + 1]),
              "viseme": experiment.viseme_acc(texts[:j + 1], stts[:j + 1])} for j in range(len(rows))]
    word, vis = experiment.word_acc(texts, stts), experiment.viseme_acc(texts, stts)
    return {"id": uuid.uuid4().hex[:12], "t": time.time(), "name": name, "dataset": user, "user_id": user,
            "author": "agent:lab-personal", "loop": "personal", "lever": lever, "hypothesis": hypothesis,
            "arm": arm, "clips": [r["id"] for r in rows], "baseline": baseline, "gate": None,
            "metrics": {"clips": len(rows), "vtt_acc": word, "word_acc": word, "viseme_acc": vis,
                        "word_lift": _mean(r[arm].get("word_lift", 0.0) for r in rows),
                        "viseme_lift": _mean(r[arm].get("viseme_lift", 0.0) for r in rows),
                        "metric_note": "PRIMARY = lift (score vs own STT - mean vs the set's other STTs); "
                                       "vtt_acc here = raw word acc (1 - WER) of the arm's VTT text"},
            "curve": curve, "flips": flips, "p": None if flips is None else flips["p"], "cost": cost,
            "verdict": verdict, "code_commit": experiment._head(), **(extra or {})}


def main(argv=None) -> int:
    import clips
    import experiment
    from tools.reread import muse_pick
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user", required=True)
    p.add_argument("--examples", type=int, default=2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--targets", type=int, default=15, help="new (paid) targets this run at most")
    p.add_argument("--from", dest="start", type=int, default=2, help="first target index among labeled clips")
    p.add_argument("--name", default="lab-personal-fewshot")
    p.add_argument("--save", action="store_true", help="save control + few-shot run records (runs.save)")
    p.add_argument("--verdict", default="inconclusive")
    args = p.parse_args(argv)

    records = [r for r in clips.load(None, args.user) if r.get("record") == clips.V1 and personal.label(r)]
    done = {r["experiment"]["source_clip"]: r for r in clips.load(None, f"exp:{args.name}:{args.user}")
            if r.get("experiment", {}).get("examples") is not None
            and len(r["experiment"]["examples"]) == args.examples}
    todo = [i for i in range(args.start, len(records)) if records[i]["id"] not in done][:args.targets]
    print(f"{len(records)} labeled clips; {len(done)} already re-read; {len(todo)} new targets", file=sys.stderr)
    def one(i):  # targets run side by side so 4 calls stay in flight; _slots caps the calls themselves
        new = reread(records, i, n_examples=args.examples, k=args.k, name=args.name, pool=pool)
        print(f"  re-read #{i} {records[i]['id']} (${_open_cost(new['samples']):.3f})", file=sys.stderr, flush=True)
        return records[i]["id"], new

    with ThreadPoolExecutor(4 * args.k) as pool, ThreadPoolExecutor(3) as targets:
        done.update(targets.map(one, todo))

    rows, cost = [], 0.0
    for i, r in enumerate(records):
        if r["id"] not in done:
            continue
        stt, new = personal.label(r), done[r["id"]]
        row = {"id": r["id"], "i": i, "stt": stt, "exp_clip": new["id"]}
        for arm, samples in (("control", r["samples"]), ("fewshot", new["samples"])):
            text = muse_pick(samples) or ""
            row[arm] = {"text": text, "word": experiment.word_acc([text], [stt]),
                        "viseme": experiment.viseme_acc([text], [stt])}
        cost += _open_cost(new["samples"])
        rows.append(row)
    stts = [r["stt"] for r in rows]
    for arm in ("control", "fewshot"):
        for metric in ("word", "viseme"):
            for r, v in zip(rows, personal.lift([r[arm]["text"] for r in rows], stts, metric,
                                                nearest=personal.LIFT_NEAREST)):
                r[arm][f"{metric}_lift"] = v
    for r in rows:
        print(f"#{r['i']:<3} word {r['control']['word']:+.3f} -> {r['fewshot']['word']:+.3f}   viseme "
              f"{r['control']['viseme']:.3f} -> {r['fewshot']['viseme']:.3f}\n  STT {r['stt']}\n"
              f"  old {r['control']['text']}\n  new {r['fewshot']['text']}")
    summary = {"n": len(rows), "cost": round(cost, 4)}
    for arm in ("control", "fewshot"):
        texts = [r[arm]["text"] for r in rows]
        summary[arm] = {"word_lift": _mean(r[arm]["word_lift"] for r in rows),
                        "viseme_lift": _mean(r[arm]["viseme_lift"] for r in rows),
                        "word": experiment.word_acc(texts, stts), "viseme": experiment.viseme_acc(texts, stts),
                        "words_per_read": sum(len(t.split()) for t in texts) / max(len(texts), 1)}
    for key in ("word_lift", "viseme_lift", "word", "viseme"):
        summary[f"flips_{key}"] = {k: v for k, v in paired(rows, "control", "fewshot", key).items()
                                   if not k.endswith("ids")}
    print(json.dumps(summary, indent=1))
    if args.save and rows:
        import runs
        hyp = (f"Few-shot: the user's {args.examples} most recent earlier labeled clips (one 4x4 frame image + STT "
               f"each) in the Muse open prompt raise VTT · Muse word/viseme acc on the same turns")
        base = run_record(rows, "control", name="fewshot-control", user=args.user, lever="few-shot",
                          hypothesis="Control: the turns' original stored K=5 reads, default prompt", cost=0.0)
        treat = run_record(rows, "fewshot", name=f"fewshot-{args.examples}", user=args.user, lever="few-shot",
                           hypothesis=hyp, cost=round(cost, 4), baseline=base["id"],
                           flips=paired(rows, "control", "fewshot", "word_lift"), verdict=args.verdict,
                           extra={"flips_viseme_lift": paired(rows, "control", "fewshot", "viseme_lift"),
                                  "flips_word_raw": paired(rows, "control", "fewshot", "word"),
                                  "exp_user": f"exp:{args.name}:{args.user}"})
        runs.save(base)
        runs.save(treat)
        print(json.dumps({"saved": [base["id"], treat["id"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
