"""Paid multi-pass reading experiment (lab-decode): run decode.py's second passes on a fixed set of a user's stored
turns and compare each arm, paired per turn, against the same turns' stored single-pass VTT · Muse.

  uv run --env-file .env python tools/exp_decode.py --user U [--arms refine,decode] [--skip 0] [--first 15]
                                                     [--name NAME] [--save] [--out /tmp/x.json]

Turns: the user's v1 clips with an STT label, a stored crop and at least one usable open sample, in store order,
then [skip : skip + first]. Arms: `refine` (re-watch the crop with the K readings), `decode` (text-only decoder),
`decode_vis` ($0 on top of decode: its candidate closest in visemes to the readers' observed shapes). Per arm: word
and viseme accuracy vs control, oracle best-of-K (the best sample's top hypothesis per turn), paired up/down turns on
per-turn word and viseme accuracy with an exact two-sided sign test, extra $ and seconds per turn. --save writes one
run record per arm (runs.save; Training page, Runs tab), with every output; nothing goes into a user's clip stream.
≤ 4 concurrent calls; the client retries 429 / 5xx with backoff.
"""

import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.append(TOOLS)  # after the root: tools/experiment.py must not shadow experiment.py

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import decode  # noqa: E402
import experiment  # noqa: E402
import visemes  # noqa: E402
from compare import sign_test  # noqa: E402
from muse_sampler import CONFIG as OPEN_CONFIG, labeled_grids  # noqa: E402
from reread import muse_pick  # noqa: E402

ARMS = ("refine", "decode")
DERIVED = {"decode_vis": "decode"}  # $0 arms computed from a paid arm's output
WORKERS = 4


def run_turn(record: dict, video: bytes, arms: list[str], client) -> dict:
    """The paid arms on one turn. Only the record's samples and its crop reach a call: never its label."""
    samples, items = record["samples"], {}
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp, "crop.mp4")
        clip.write_bytes(video)
        if "refine" in arms:
            grids = labeled_grids(clip, OPEN_CONFIG["grid"], OPEN_CONFIG["tile_width"])
            items["refine"] = decode.refine(samples, grids, client)
            n_frames = grids.n_frames
        else:
            n_frames = record.get("frames") or 0
        if "decode" in arms:
            items["decode"] = decode.decode(samples, n_frames, client)
    return items


def _observed(output: dict) -> list[str]:
    codes = [v["v"] for v in output.get("visemes") or [] if v["v"] != "REST"]
    return [c for i, c in enumerate(codes) if i == 0 or codes[i - 1] != c]


def vis_rerank(item: dict, samples: list[dict]) -> str | None:
    """decode_vis: the decoder candidate with the smallest mean viseme distance to the readers' observed shapes."""
    output, seqs = item.get("output"), [_observed(o) for o in decode.readers(samples)]
    texts = [c["text"] for c in (output or {}).get("candidates") or [] if c.get("text", "").strip()]
    seqs = [s for s in seqs if s]
    if not texts or not seqs:
        return decode.text_of(item)
    return min(texts, key=lambda t: sum(visemes.viseme_distance(visemes.to_visemes(t), s) for s in seqs))


def oracle(samples: list[dict], stt: str, metric) -> str:
    tops = [max(o["hypotheses"], key=lambda h: h["s"])["text"] for o in decode.readers(samples) if o.get("hypotheses")]
    return max(tops, key=lambda t: metric([t], [stt]) or 0) if tops else ""


def _turns(user: str, skip: int, first: int | None):
    import clips
    store = clips.backend(None)
    picked = [r for r in store.load(user) if r.get("record") == "v1" and (r.get("label") or {}).get("stt_text")
              and decode.readers(r["samples"])]
    picked = picked[skip:None if first is None else skip + first]
    return [(r, store.video(r["id"])) for r in picked]


def paired(control: list[str], arm: list[str], stts: list[str], metric) -> dict:
    up = [i for i, (c, a, s) in enumerate(zip(control, arm, stts)) if metric([a], [s]) > metric([c], [s])]
    down = [i for i, (c, a, s) in enumerate(zip(control, arm, stts)) if metric([a], [s]) < metric([c], [s])]
    return {"up": len(up), "down": len(down), "ties": len(stts) - len(up) - len(down), "p": sign_test(len(down), len(up)),
            "up_idx": up, "down_idx": down}


def lifts(texts: list[str], stts: list[str], metric) -> list[float]:
    """Per turn: score vs its own STT minus the mean score vs the other turns' STTs (lab-metrics' LIFT). A generic
    sentence scores the same against every STT, so its lift is ~0 however high its raw accuracy."""
    score = lambda t, s: metric([t], [s]) or 0.0  # noqa: E731
    return [score(t, stts[i]) - sum(score(t, s) for j, s in enumerate(stts) if j != i) / (len(stts) - 1)
            for i, t in enumerate(texts)]


def paired_lift(control: list[float], arm: list[float]) -> dict:
    up, down = sum(a > c for c, a in zip(control, arm)), sum(a < c for c, a in zip(control, arm))
    return {"up": up, "down": down, "ties": len(arm) - up - down, "p": sign_test(down, up)}


def summarize(rows: list[dict], arm: str) -> dict:
    stts = [r["stt"] for r in rows]
    control = [r["control"] or "" for r in rows]
    texts = [r["texts"].get(arm) or "" for r in rows]
    paid = DERIVED.get(arm, arm)
    items = [r["items"].get(paid) or {} for r in rows]
    costs = [experiment._call_cost(i["usage"], "muse") for i in items if i.get("usage")] if arm == paid else [0.0]
    lat = [i["latency_ms"] / 1000 for i in items if i.get("latency_ms")] if arm == paid else [0.0]
    w, v = paired(control, texts, stts, lambda a, b: experiment.word_acc(a, b)), \
        paired(control, texts, stts, lambda a, b: experiment.viseme_acc(a, b))
    lw = [lifts(x, stts, experiment.word_acc) for x in (control, texts)]
    lv = [lifts(x, stts, experiment.viseme_acc) for x in (control, texts)]
    exact = lambda xs: sum(experiment._matches(s, x or None) for x, s in zip(xs, stts)) / len(stts)  # noqa: E731
    return {
        "clips": len(rows), "arm": arm,
        "lift_word": sum(lw[1]) / len(rows), "lift_word_control": sum(lw[0]) / len(rows),
        "lift_viseme": sum(lv[1]) / len(rows), "lift_viseme_control": sum(lv[0]) / len(rows),
        "lift_w": paired_lift(*lw), "lift_v": paired_lift(*lv),
        "words_per_stt": sum(len(t.split()) for t in texts) / max(1, sum(len(t.split()) for t in stts)),
        "word_acc": experiment.word_acc(texts, stts), "word_acc_control": experiment.word_acc(control, stts),
        "viseme_acc": experiment.viseme_acc(texts, stts), "viseme_acc_control": experiment.viseme_acc(control, stts),
        "oracle_word_acc": experiment.word_acc([r["oracle_w"] for r in rows], stts),
        "oracle_viseme_acc": experiment.viseme_acc([r["oracle_v"] for r in rows], stts),
        "vtt_acc": exact(texts), "muse_acc": exact(control),
        "word": {k: w[k] for k in ("up", "down", "ties", "p")}, "viseme": {k: v[k] for k in ("up", "down", "ties", "p")},
        "errors": sum(1 for i in items if i.get("error")),
        "extra_cost_per_turn": sum(costs) / len(rows), "extra_latency_s": sum(lat) / max(len(lat), 1),
        "cost_total": sum(costs),
        "flips": {"word_up": [rows[i]["clip"] for i in w["up_idx"]], "word_down": [rows[i]["clip"] for i in w["down_idx"]]},
    }


def run_record(summary: dict, rows: list[dict], *, user: str, name: str, skip: int, hypothesis: str) -> dict:
    s, arm = summary, summary["arm"]
    verdict = (f"LIFT word {s['lift_word_control']:+.3f} -> {s['lift_word']:+.3f} ({s['lift_w']['up']} up / "
               f"{s['lift_w']['down']} down, p={s['lift_w']['p']:.3g}); LIFT viseme {s['lift_viseme_control']:+.3f} "
               f"-> {s['lift_viseme']:+.3f} ({s['lift_v']['up']}/{s['lift_v']['down']}, p={s['lift_v']['p']:.3g}); "
               f"raw word {s['word_acc_control']:.3f} -> {s['word_acc']:.3f} ({s['word']['up']} up / {s['word']['down']} "
               f"down, p={s['word']['p']:.3g}); viseme {s['viseme_acc_control']:.3f} -> {s['viseme_acc']:.3f} "
               f"({s['viseme']['up']}/{s['viseme']['down']}, p={s['viseme']['p']:.3g})")
    return {
        "id": uuid.uuid4().hex[:12], "t": time.time(), "name": f"{name}:{arm}", "author": "agent:lab-decode",
        "loop": "framework", "lever": f"multi-pass reading / {arm}", "hypothesis": hypothesis,
        "dataset": f"{user} turns[{skip}:{skip + len(rows)}]", "user_id": user, "speaker": user,
        "turns": [r["clip"] for r in rows], "baseline": None, "gate": None, "curve": [],
        "config": {**decode.CONFIG, "arm": arm, "open_config": OPEN_CONFIG},
        "code_commit": experiment._head(),
        "metrics": {k: s[k] for k in ("clips", "lift_word", "lift_word_control", "lift_viseme", "lift_viseme_control",
                                      "words_per_stt", "vtt_acc", "muse_acc", "word_acc", "word_acc_control", "viseme_acc",
                                      "viseme_acc_control", "oracle_word_acc", "oracle_viseme_acc",
                                      "extra_cost_per_turn", "extra_latency_s", "errors")},
        "flips": {**s["flips"], "word": s["word"], "viseme": s["viseme"], "lift_word": s["lift_w"],
                  "lift_viseme": s["lift_v"]}, "p": s["lift_w"]["p"],
        "cost": s["cost_total"], "verdict": verdict,
        "outputs": [{"clip": r["clip"], "stt": r["stt"], "control": r["control"], "text": r["texts"].get(arm),
                     "output": (r["items"].get(DERIVED.get(arm, arm)) or {}).get("output")} for r in rows],
    }


HYPOTHESES = {
    "refine": "A second Muse call that re-watches the crop with the K readings as hints, checking each word's frames "
              "against its mouth shape, reads more words right than the single-pass VTT · Muse.",
    "decode": "A text-only decoder over the K readings' mouth shapes and word slots finds a sentence closer to what "
              "was said than the single-pass VTT · Muse.",
    "decode_vis": "Re-ranking the decoder's candidates by viseme distance to the observed shapes beats its own pick.",
}


def main(argv=None) -> int:
    from muse_video import make_client
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user", required=True)
    p.add_argument("--arms", default="refine,decode")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--first", type=int)
    p.add_argument("--name", default="lab-decode")
    p.add_argument("--save", action="store_true")
    p.add_argument("--out")
    p.add_argument("--cache", default="/tmp/lab-decode-cache.jsonl", help="paid items, reused across runs")
    args = p.parse_args(argv)
    arms = [a for a in args.arms.split(",") if a in ARMS]
    turns = _turns(args.user, args.skip, args.first)
    turns = [(r, v) for r, v in turns if v]
    if not turns:
        print(f"no usable turns for {args.user!r}", file=sys.stderr)
        return 2
    client = make_client().with_options(max_retries=6)
    cache_path, cache = Path(args.cache), {}  # (clip, arm, config) -> item, appended the moment a call returns
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            entry = json.loads(line)
            cache[(entry["clip"], entry["arm"], entry["config"])] = entry["item"]
    key, lock = json.dumps(decode.CONFIG, sort_keys=True), threading.Lock()

    def task(job):
        (record, video), arm = job
        if (record["id"], arm, key) in cache:
            return cache[(record["id"], arm, key)]
        item = run_turn(record, video, [arm], client)[arm]
        if not item.get("error"):
            with lock, cache_path.open("a") as f:
                f.write(json.dumps({"clip": record["id"], "arm": arm, "config": key, "item": item}) + "\n")
        return item

    jobs = [(t, arm) for t in turns for arm in arms]
    with ThreadPoolExecutor(WORKERS) as pool:  # WORKERS concurrent calls in all
        flat = list(pool.map(task, jobs))
    all_items = [{arm: flat[i * len(arms) + j] for j, arm in enumerate(arms)} for i in range(len(turns))]
    raw = Path(args.out or f"/tmp/{args.name}-items.json")
    rows = []
    for (record, _), items in zip(turns, all_items):
        stt = record["label"]["stt_text"]
        texts = {arm: decode.text_of(items[arm]) for arm in arms}
        if "decode" in arms:
            texts["decode_vis"] = vis_rerank(items["decode"], record["samples"])
        rows.append({"clip": record["id"], "stt": stt, "control": muse_pick(record["samples"]), "texts": texts,
                     "items": items, "oracle_w": oracle(record["samples"], stt, experiment.word_acc),
                     "oracle_v": oracle(record["samples"], stt, experiment.viseme_acc)})
    shown = arms + (["decode_vis"] if "decode" in arms else [])
    print(f"{args.name}: {len(rows)} turns of {args.user} [{args.skip}:{args.skip + len(rows)}], arms {shown}\n")
    for r in rows:
        print(f"{r['clip']}\n  STT     {r['stt']}\n  control {r['control']}")
        for arm in shown:
            print(f"  {arm:<7} {r['texts'].get(arm)}")
    print(f"\n{'arm':<11}{'n':>3} {'LIFT word':>15} {'up/dn p':>12} {'LIFT viseme':>15} {'up/dn p':>12} {'len':>5}")
    summaries = [summarize(rows, arm) for arm in shown]
    for s in summaries:
        print(f"{s['arm']:<11}{s['clips']:>3} {s['lift_word_control']:>+7.3f}>{s['lift_word']:<+7.3f} "
              f"{s['lift_w']['up']:>3}/{s['lift_w']['down']:<3}{s['lift_w']['p']:>6.3f} "
              f"{s['lift_viseme_control']:>+7.3f}>{s['lift_viseme']:<+7.3f} "
              f"{s['lift_v']['up']:>3}/{s['lift_v']['down']:<3}{s['lift_v']['p']:>6.3f} {s['words_per_stt']:>5.2f}")
    print(f"\n{'arm':<11}{'n':>3} {'word':>13} {'viseme':>13} {'oracle w/v':>12} {'w up/dn p':>14} "
          f"{'v up/dn p':>14} {'$/turn':>7} {'s/turn':>7} {'err':>4}")
    for s in summaries:
        print(f"{s['arm']:<11}{s['clips']:>3} {s['word_acc_control']:>6.3f}>{s['word_acc']:<6.3f} "
              f"{s['viseme_acc_control']:>6.3f}>{s['viseme_acc']:<6.3f} "
              f"{s['oracle_word_acc']:>5.3f}/{s['oracle_viseme_acc']:<6.3f} "
              f"{s['word']['up']:>3}/{s['word']['down']:<3}{s['word']['p']:>7.3f} "
              f"{s['viseme']['up']:>3}/{s['viseme']['down']:<3}{s['viseme']['p']:>7.3f} "
              f"{s['extra_cost_per_turn']:>7.4f} {s['extra_latency_s']:>7.1f} {s['errors']:>4}")
    print(f"\ntotal spend ${sum(s['cost_total'] for s in summaries):.3f}")
    raw.write_text(json.dumps({"rows": rows, "summaries": summaries}, default=str))
    if args.save:
        import runs
        for s in summaries:
            record = run_record(s, rows, user=args.user, name=args.name, skip=args.skip,
                                hypothesis=HYPOTHESES[s["arm"]])
            runs.save(record)
            print(f"saved run {record['id']} {record['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
