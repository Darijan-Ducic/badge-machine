"""Free personal-loop replay (lab-personal): every personal.py arm on one user's stored clips, predict-then-train, $0.

  uv run --env-file .env python tools/exp_personal.py --user U [--from 1] [--curve 5,10,20,40] [--save] [--json]
      [--source exp:lab-personal-fewshot:U]   # read the samples of these re-reads instead (paired by source_clip)

Each clip i is read by layers built from personal.earlier(records, i) only. PRIMARY metric: lift (score vs the clip's
own STT minus mean score vs the 10 other STTs in the set closest in word count; personal.lift with nearest=
personal.LIFT_NEAREST, because plain lift pays a content-free read for matching length), word and viseme; raw is
secondary. Paired vs VTT · Muse per clip on lift, with an exact two-sided sign test. --curve re-runs every arm with
the personal layers limited to the last N earlier clips, on the fixed set of clips that have >= max(N) earlier ones.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402

import personal  # noqa: E402


def score(rows: list[dict], arms) -> dict:
    """Per arm: mean word/viseme lift over the rows, raw word/viseme acc, and the paired lift flips vs muse."""
    import experiment
    from tools.compare import sign_test
    stts = [r["stt"] for r in rows]
    out = {}
    for arm in arms:
        texts = [r["reads"][arm] for r in rows]
        lifts = {m: personal.lift(texts, stts, m, nearest=personal.LIFT_NEAREST) for m in ("word", "viseme")}
        for r, w, v in zip(rows, lifts["word"], lifts["viseme"]):
            r.setdefault("lift", {})[arm] = {"word": w, "viseme": v}
        out[arm] = {"n": len(rows),
                    "word_lift": sum(lifts["word"]) / len(rows), "viseme_lift": sum(lifts["viseme"]) / len(rows),
                    "word": experiment.word_acc(texts, stts), "viseme": experiment.viseme_acc(texts, stts),
                    "words_per_read": sum(len(t.split()) for t in texts) / len(rows)}
    for arm in arms:
        for m in ("word", "viseme"):
            up = [r["id"] for r in rows if r["lift"][arm][m] > r["lift"]["muse"][m] + 1e-12]
            down = [r["id"] for r in rows if r["lift"][arm][m] < r["lift"]["muse"][m] - 1e-12]
            out[arm][f"{m}_flips"] = {"up": len(up), "down": len(down), "p": sign_test(len(down), len(up)),
                                      "up_ids": up, "down_ids": down}
    return out


def replay(records: list[dict], targets: list[int], last: int | None = None) -> list[dict]:
    p = personal.Personal(records, last=last)
    return [{"id": records[i]["id"], "i": i, "stt": personal.label(records[i]), "reads": p.reads(i)}
            for i in targets]


def load(user: str, source: str | None) -> list[dict]:
    """The user's v1 records in store order; with `source`, each record whose re-read exists there gets the
    re-read's samples (the rest keep their own), so personal layers still see the user's real history."""
    import clips
    records = [r for r in clips.load(None, user) if r.get("record") == clips.V1]
    if source:
        reread = {r["experiment"]["source_clip"]: r for r in clips.load(None, source) if r.get("experiment")}
        records = [{**r, "samples": reread[r["id"]]["samples"], "reread": reread[r["id"]]["id"]}
                   if r["id"] in reread else r for r in records]
    return records


def table(scores: dict) -> str:
    lines = [f"{'arm':<16} {'n':>3} {'w.lift':>7} {'v.lift':>7} {'word':>7} {'viseme':>7} {'wpr':>5}  "
             f"{'w up/down p':>14}  {'v up/down p':>14}"]
    for arm, s in scores.items():
        wf, vf = s["word_flips"], s["viseme_flips"]
        lines.append(f"{arm:<16} {s['n']:>3} {s['word_lift']:>+7.3f} {s['viseme_lift']:>+7.3f} {s['word']:>+7.3f} "
                     f"{s['viseme']:>7.3f} {s['words_per_read']:>5.1f}  {wf['up']:>3}/{wf['down']:<3} {wf['p']:.3f}"
                     f"  {vf['up']:>3}/{vf['down']:<3} {vf['p']:.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--user", required=True)
    ap.add_argument("--from", dest="start", type=int, default=1, help="first evaluated index (store order)")
    ap.add_argument("--curve", default="", help="comma-separated history sizes, e.g. 5,10,20,40")
    ap.add_argument("--source", help="user id of re-reads whose samples replace the source clips' samples")
    ap.add_argument("--only-source", action="store_true", help="evaluate only clips that have a re-read")
    ap.add_argument("--arms", default=",".join(personal.ARMS))
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    arms = args.arms.split(",")
    if "muse" not in arms:
        arms = ["muse"] + arms
    records = load(args.user, args.source)
    labeled = [i for i, r in enumerate(records) if personal.label(r) is not None]
    targets = [i for i in labeled if i >= args.start and (not args.only_source or "reread" in records[i])]
    rows = replay(records, targets)
    for r in rows:
        r["reads"] = {a: r["reads"][a] for a in arms}
    report = {"user": args.user, "source": args.source, "all": score(rows, arms), "curve": {}}
    print(f"{args.user}{' via ' + args.source if args.source else ''}: {len(rows)} clips, all earlier history")
    print(table(report["all"]))
    sizes = [int(n) for n in args.curve.split(",") if n]
    if sizes:
        fixed = [i for i in targets if len(personal.earlier(records, i)) >= max(sizes)]
        print(f"\nlearning curve on the {len(fixed)} clips with >= {max(sizes)} earlier labeled clips")
        for n in sizes + [None]:
            curve_rows = replay(records, fixed, last=n)
            for r in curve_rows:
                r["reads"] = {a: r["reads"][a] for a in arms}
            report["curve"][str(n or "all")] = score(curve_rows, arms)
            print(f"\nlast {n or 'all'}:\n" + table(report["curve"][str(n or "all")]))
    if args.json:
        print(json.dumps({k: v for k, v in report.items()}, default=str))
    if args.save:
        save(report, rows, args, arms)
    return 0


def save(report: dict, rows: list[dict], args, arms) -> None:
    import time
    import uuid

    import experiment
    import runs

    def curve(arm):
        stts = [r["stt"] for r in rows]
        texts = [r["reads"][arm] for r in rows]
        lifts = [r["lift"][arm]["word"] for r in rows]
        return [{"i": j, "vtt": sum(lifts[:j + 1]) / (j + 1),
                 "word": experiment.word_acc(texts[:j + 1], stts[:j + 1])} for j in range(len(rows))]

    tag = f"{args.user}{'@' + args.source if args.source else ''}"
    ids = {}
    for arm in arms:
        s = report["all"][arm]
        record = {"id": uuid.uuid4().hex[:12], "t": time.time(), "name": f"personal-{arm}", "dataset": tag,
                  "user_id": args.user, "author": "agent:lab-personal", "loop": "personal",
                  "lever": {"muse": "baseline", "prior_only": "control (no video)", "lm_rescore": "language prior",
                            "slot_lm": "language prior", "scorer_staged": "personal scorer",
                            "scorer_personal": "personal scorer", "viseme_default": "viseme costs",
                            "viseme_personal": "viseme costs"}[arm],
                  "hypothesis": HYPOTHESES[arm], "arm": arm, "clips": [r["id"] for r in rows],
                  "baseline": ids.get("muse"), "gate": None,
                  "metrics": {"clips": s["n"], "vtt_acc": s["word_lift"], "word_lift": s["word_lift"],
                              "viseme_lift": s["viseme_lift"], "word_acc": s["word"], "viseme_acc": s["viseme"],
                              "metric_note": "vtt_acc here = mean word LIFT (score vs own STT - mean vs the set's "
                                             "other STTs); raw word/viseme acc secondary"},
                  "flips": {k: v for k, v in s["word_flips"].items()}, "viseme_flips": s["viseme_flips"],
                  "p": s["word_flips"]["p"], "cost": 0.0, "curve": curve(arm),
                  "learning_curve": {n: {"word_lift": c[arm]["word_lift"], "viseme_lift": c[arm]["viseme_lift"]}
                                     for n, c in report["curve"].items()},
                  "verdict": VERDICTS.get(arm, "inconclusive"), "code_commit": experiment._head()}
        runs.save(record)
        ids[arm] = record["id"]
    print(json.dumps({"saved": ids}))


HYPOTHESES = {
    "muse": "Baseline: VTT · Muse, the most common top hypothesis across K samples (no personal data)",
    "prior_only": "Control: the user's medoid earlier STT, no video; any personal arm must beat it on lift",
    "lm_rescore": "A personal trigram LM over earlier STT, weighted per clip on earlier clips, picks a hypothesis "
                  "that matches the turn better than Muse's vote",
    "slot_lm": "Decoding each sample's slot look-alikes with the personal trigram LM recovers the user's words",
    "scorer_staged": "The staged scorer fit on this user's earlier exact-match labels picks better than Muse",
    "scorer_personal": "A conditional logit fit to each earlier clip's best-lift candidate, with personal LM and "
                       "speaking-rate features, picks better than Muse",
    "viseme_default": "Picking the candidate nearest the observed visemes (default costs) beats Muse",
    "viseme_personal": "Viseme costs learned from this user's confusions pick better than default costs",
}
VERDICTS: dict[str, str] = {}  # set by the session after reading the numbers (see the book chapter)


if __name__ == "__main__":
    sys.exit(main())
