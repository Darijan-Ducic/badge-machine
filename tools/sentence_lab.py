"""Sentence post-processing lab: score free word-level combiners of the K stored open samples on real turns ($0).

Every arm reads only the stored samples, so none fits anything and every labeled clip is scored (no split needed).
Baselines: `vtt_main` is the app's pick before the sentence path (the scorer's pool pick, each clip read by a model
trained on the clips before it), `muse` the most common top hypothesis (VTT · Muse). `rover_top` is what the
sentence path ships (sentence.read).

Primary metric is LIFT (gameable raw accuracies are reported beside it): per clip, accuracy of the read against its
own STT minus its mean accuracy against every other turn's STT in the set (the impostors); per-clip accuracy is
1 - edits / STT length, for words and for visemes (visemes.to_visemes, unit costs). Paired per clip against the
baseline: up / down counts of lift, exact two-sided sign test on the non-ties, 95% bootstrap CI of the mean lift
delta (2000 resamples, seed 0). Oracles: per clip, the hypothesis with the fewest word edits among the top
hypotheses (`oracle_top`) or all hypotheses (`oracle_all`).

Usage:
  uv run --env-file .env python tools/sentence_lab.py --user ID [--export snap.json] [--json out.json]
  uv run python tools/sentence_lab.py --records snap.json [--baseline muse] [--json out.json]
  add --save NAME --arm ARM --hypothesis TEXT --verdict active|parked|inconclusive [--reopen TEXT] (needs .env for
  the Runs tab) to store a run record (author agent:lab-fusion, loop framework)
"""

import argparse
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiment  # noqa: E402
import sentence  # noqa: E402
import visemes  # noqa: E402
from scorer import predict, train  # noqa: E402

_spec = importlib.util.spec_from_file_location("compare_tool", ROOT / "tools" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

TAUS = (0.2, 0.3, 0.4)
RESAMPLES, SEED = 2000, 0


# ---------------------------------------------------------------- arms


def _hyps(output: dict) -> list[dict]:
    return [h for h in output.get("hypotheses") or [] if sentence.words(h["text"])]


def _top(output: dict) -> dict | None:
    hyps = _hyps(output)
    return max(hyps, key=lambda h: h["s"]) if hyps else None


def _top_n(outputs: list[dict], n: int) -> tuple[list[list[str]], list[float]]:
    ranked = [h for o in outputs for h in sorted(_hyps(o), key=lambda h: -h["s"])[:n]]
    return [sentence.words(h["text"]) for h in ranked], [h["s"] for h in ranked]


def _decode(columns: list[dict], tau: float) -> list[str]:
    """Per network column, its best word if that word holds at least tau of the column's weight."""
    out = []
    for column in columns:
        named = [(w, x) for w, x in column.items() if w is not None]
        if named:
            word, weight = max(named, key=lambda item: item[1])
            if weight >= tau * sum(column.values()):
                out.append(word)
    return out


def arms(outputs: list[dict]) -> dict[str, list[str]]:
    """Every arm's word list for one clip's valid sample outputs."""
    tops = sorted([t for t in map(_top, outputs) if t is not None], key=lambda h: -h["s"])
    top_words = [sentence.words(t["text"]) for t in tops]
    every = [h for o in outputs for h in _hyps(o)]
    every_words, every_s = [sentence.words(h["text"]) for h in every], [h["s"] for h in every]
    slots = [w for w in (sentence.slot_words(o) for o in outputs) if w]
    out = {
        "rover_top": sentence.rover(top_words),
        "rover_top_nogap": sentence.rover(top_words, gaps=False),
        "rover_top_s": sentence.rover(top_words, [t["s"] for t in tops]),
        "rover_top2_s": sentence.rover(*_top_n(outputs, 2)),
        "rover_all_s": sentence.rover(every_words, every_s),
        "rover_slots": sentence.rover(slots),
        "rover_tops_slots": sentence.rover(top_words + slots),
        "mbr_top": [], "mbr_all_s": [], "rover_medoid_first": [], "longest_top": [], "viseme_rerank": [],
    }
    columns = sentence.network(top_words)
    out.update({f"rover_tau{tau}": _decode(columns, tau) for tau in TAUS})
    if top_words:
        medoid = sentence.mbr(top_words)
        out["mbr_top"] = list(top_words[medoid])
        out["rover_medoid_first"] = sentence.rover([top_words[medoid]] + top_words[:medoid] + top_words[medoid + 1:])
        out["longest_top"] = max(top_words, key=len)
    if every_words:
        out["mbr_all_s"] = list(every_words[sentence.mbr(every_words, every_s)])
        observed = [c for c in (_observed(o) for o in outputs) if c]
        if observed:
            fit = [np.mean([visemes.viseme_distance(visemes.to_visemes(" ".join(h)), seen) for seen in observed])
                   for h in every_words]
            out["viseme_rerank"] = list(every_words[int(np.argmin(fit))])
    return out


def _observed(output: dict) -> list[str]:
    codes = [v["v"] for v in output.get("visemes") or [] if v["v"] != "REST"]
    return [c for i, c in enumerate(codes) if i == 0 or c != codes[i - 1]]


# ---------------------------------------------------------------- lift


def _batch_edits(a: list[int], bs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Unit-cost Levenshtein from sequence a to each row of bs (padded with -1), at each row's own length."""
    width = bs.shape[1]
    previous = np.tile(np.arange(width + 1), (len(bs), 1))
    for i, x in enumerate(a, 1):
        current = np.empty_like(previous)
        current[:, 0] = i
        substitute = previous[:, :-1] + (bs != x)
        for j in range(1, width + 1):
            current[:, j] = np.minimum(np.minimum(previous[:, j] + 1, current[:, j - 1] + 1), substitute[:, j - 1])
        previous = current
    return previous[np.arange(len(bs)), lengths]


def accuracy_matrix(reads: list[list], saids: list[list]) -> np.ndarray:
    """acc[i, j] = 1 - edits(read i, said j) / len(said j); 0 where said j is empty."""
    vocab = {token: k for k, token in enumerate({t for seq in reads + saids for t in seq})}
    lengths = np.array([len(s) for s in saids])
    bs = np.full((len(saids), max(lengths.max(initial=0), 1)), -1)
    for j, s in enumerate(saids):
        bs[j, :len(s)] = [vocab[t] for t in s]
    out = np.zeros((len(reads), len(saids)))
    for i, read in enumerate(reads):
        edits = _batch_edits([vocab[t] for t in read], bs, lengths)
        out[i] = np.where(lengths > 0, 1 - edits / np.maximum(lengths, 1), 0.0)
    return out


def lift(acc: np.ndarray, impostors: np.ndarray | None = None) -> np.ndarray:
    """Per clip: own accuracy minus the mean accuracy against the impostor STTs (default every other clip's;
    `impostors[i, j]` True where clip j's STT is an impostor for clip i; a clip without impostors gets 0)."""
    n = len(acc)
    impostors = ~np.eye(n, dtype=bool) if impostors is None else impostors
    counts = impostors.sum(1)
    mean = np.where(counts > 0, (acc * impostors).sum(1) / np.maximum(counts, 1), np.diag(acc))
    return np.diag(acc) - mean


def length_matched(stts: list[str], tolerance: float = 0.25) -> np.ndarray:
    """Impostors whose STT word count is within ±tolerance of the clip's own: lift then can't reward length."""
    lens = np.array([len(sentence.words(s)) for s in stts])
    return (np.abs(lens[None, :] - lens[:, None]) <= tolerance * lens[:, None]) & ~np.eye(len(stts), dtype=bool)


def paired(mine: np.ndarray, base: np.ndarray) -> dict:
    up, down = int((mine > base + 1e-12).sum()), int((mine < base - 1e-12).sum())
    draws = np.random.default_rng(SEED).integers(0, len(mine), size=(RESAMPLES, len(mine)))
    spread = (mine - base)[draws].mean(1)
    return {"up": up, "down": down, "p": compare.sign_test(up, down), "delta": float((mine - base).mean()),
            "ci95": [float(np.percentile(spread, 2.5)), float(np.percentile(spread, 97.5))]}


# ---------------------------------------------------------------- the lab


def _app_vtt(clips: list) -> list[str]:
    """The scorer's pool pick (∅ = ""), each clip read by a model trained on the clips before it."""
    texts = []
    for i, clip in enumerate(clips):
        probs = predict(train(clips[:i], gate=False), [clip])[0]
        top = int(probs.argmax())
        texts.append("" if top == len(clip.pool) - 1 else clip.pool[top])
    return texts


def lab(records: list[dict], baseline: str = "vtt_main") -> dict:
    labeled, clips = experiment._featurized(records, experiment.resolve(None))
    stts = [r["stt"] for r in labeled]
    texts = {"vtt_main": _app_vtt(clips), "muse": [c.muse_pick or "" for c in clips]}
    for record in labeled:
        outputs = [s["output"] for s in record["samples"] if s.get("output") is not None]
        said = sentence.words(record["stt"])
        for name, words in arms(outputs).items():
            texts.setdefault(name, []).append(" ".join(words))
        best = lambda options: min(options, key=lambda t: sentence.edits(sentence.words(t), said), default="")  # noqa: E731
        texts.setdefault("oracle_top", []).append(best([t["text"] for t in map(_top, outputs) if t]))
        texts.setdefault("oracle_all", []).append(best([h["text"] for o in outputs for h in _hyps(o)]))
    said_words = [sentence.words(s) for s in stts]
    said_visemes = [visemes.to_visemes(s) for s in stts]
    lifts, matched = {}, length_matched(stts)
    for name, arm in texts.items():
        word = accuracy_matrix([sentence.words(t) for t in arm], said_words)
        vis = accuracy_matrix([visemes.to_visemes(t) for t in arm], said_visemes)
        lifts[name] = {"word": lift(word), "viseme": lift(vis), "word_matched": lift(word, matched),
                       "viseme_matched": lift(vis, matched)}
    rows = {}
    for name, arm in texts.items():
        rows[name] = {
            "word": experiment.word_acc(arm, stts), "viseme": experiment.viseme_acc(arm, stts),
            "word_lift": float(lifts[name]["word"].mean()), "viseme_lift": float(lifts[name]["viseme"].mean()),
            "word_lift_matched": float(lifts[name]["word_matched"].mean()),
            "viseme_lift_matched": float(lifts[name]["viseme_matched"].mean()),
            "mean_words": float(np.mean([len(sentence.words(t)) for t in arm])),
            "vs_baseline": {m: paired(lifts[name][m], lifts[baseline][m]) for m in lifts[name]},
            "per_clip_word_lift": lifts[name]["word"].tolist(),
        }
    return {"baseline": baseline, "clips": len(stts), "ids": [str(r.get("id")) for r in labeled], "stts": stts,
            "stt_mean_words": float(np.mean([len(w) for w in said_words])), "arms": rows, "texts": texts}


def table(report: dict) -> str:
    lines = [f"{report['clips']} labeled clips, STT mean {report['stt_mean_words']:.1f} words; paired vs "
             f"{report['baseline']} on per-clip lift",
             f"{'arm':19} {'word':>6} {'vis':>6} {'w.lift':>7} {'v.lift':>7} {'len':>5}  {'word lift up/dn p':>18} "
             f"{'Δ [95% CI]':>24}  {'vis lift up/dn p':>17}  {'w.lift≈len up/dn p':>24}"]
    for name, r in report["arms"].items():
        w, v, m = (r["vs_baseline"][k] for k in ("word", "viseme", "word_matched"))
        lines.append(f"{name:19} {r['word']:6.3f} {r['viseme']:6.3f} {r['word_lift']:+7.3f} {r['viseme_lift']:+7.3f} "
                     f"{r['mean_words']:5.1f}  {w['up']:4d}/{w['down']:<4d} {w['p']:8.4f} "
                     f"{w['delta']:+.3f} [{w['ci95'][0]:+.3f},{w['ci95'][1]:+.3f}]  "
                     f"{v['up']:4d}/{v['down']:<4d} {v['p']:7.4f}  "
                     f"{r['word_lift_matched']:+.3f} {m['up']:4d}/{m['down']:<4d} {m['p']:7.4f}")
    return "\n".join(lines)


def run_record(report: dict, name: str, arm: str, hypothesis: str, verdict: str | None, reopen: str,
               user: str | None) -> dict:
    """A run record for the Training page Runs tab (the `GET /runs` keys) plus this lab's evidence."""
    r, base = report["arms"][arm], report["arms"][report["baseline"]]
    per = r["per_clip_word_lift"]
    window = experiment.CURVE_WINDOW
    keep = ("word", "viseme", "word_lift", "viseme_lift", "mean_words", "vs_baseline")
    return {
        "id": uuid.uuid4().hex[:12], "t": time.time(), "name": name, "dataset": f"clips:{user}", "user_id": user,
        "config": {"lab": "tools/sentence_lab.py", "arm": arm, "baseline_arm": report["baseline"]}, "baseline": None,
        "metrics": {"clips": report["clips"], "vtt_acc": None, "vtt_acc_heldout": None, "gemini_acc": None,
                    "muse_acc": None, "pool_recall": None, "autosend_precision": None, "autosend_coverage": None,
                    **{f"{k}": r[k] for k in keep[:5]}, **{f"baseline_{k}": base[k] for k in keep[:4]},
                    "oracle_word_acc": report["arms"]["oracle_all"]["word"],
                    "oracle_top_word_acc": report["arms"]["oracle_top"]["word"]},
        "arms": {k: {key: v[key] for key in keep} for k, v in report["arms"].items()},
        "gate": None, "curve": [{"i": i, "vtt": float(np.mean(per[max(0, i - window + 1):i + 1]))}
                                for i in range(len(per))],
        "hypothesis": hypothesis, "author": "agent:lab-fusion", "lever": "free post-processing (sentence path)",
        "loop": "framework", "clip_ids": report["ids"], "flips": r["vs_baseline"],
        "p": r["vs_baseline"]["word"]["p"], "verdict": verdict, "reopen": reopen or None,
        "code_commit": experiment._head(), "system_version": None, "candidate_version": None, "parent_version": None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--user")
    source.add_argument("--records", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--baseline", default="vtt_main")
    parser.add_argument("--save", metavar="NAME")
    parser.add_argument("--arm", default="rover_top")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--verdict", choices=("active", "parked", "inconclusive"))
    parser.add_argument("--reopen", default="")
    parser.add_argument("--user-id", help="user id to tag the run with (default: --user)")
    args = parser.parse_args(argv)
    records = compare.store_records(None, args.user)[0] if args.user else compare.file_records(args.records)[0]
    if args.export:
        args.export.write_text(json.dumps(records, default=str))
    report = lab(records, args.baseline)
    print(table(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=float))
    if args.save:
        import runs
        record = run_record(report, args.save, args.arm, args.hypothesis, args.verdict, args.reopen,
                            args.user_id or args.user)
        runs.save(record)
        print(f"saved run {record['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
