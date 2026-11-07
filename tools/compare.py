"""Replay two CODE versions on the exact same stored clips and report paired evidence. No verdict: you judge.

A side is a git ref, materialized with `git archive <ref> | tar -x` into a temp dir, or `worktree` (this checkout as
it is, uncommitted edits included). Each side replays in its own Python process with its tree first on sys.path and
as its working directory, so its own candidates.py, scorer.py, visemes.py, experiment.py and data/viseme_costs.json
decide its result. No model is called: features and scores are recomputed from the stored raw outputs.

Usage:
  uv run --env-file .env python tools/compare.py --user ID [--base main] [--cand worktree] [--json out.json]
  uv run python tools/compare.py --records snapshot.json [--base REF] [--cand REF|worktree] [--json out.json]
  add --export snapshot.json to write the records both sides read (replay them later with --records)
  --split train|heldout (default heldout); --split test only with --final, once, at the end, never while tuning

Records: --user loads that user's v1 clips once from the clip store (MongoDB when MONGODB_URI is set, else
data/clips/, or --store DIR), under the phrase list of their latest clip, each clip's bank counting only the labels
before it (as clips.py replays them). --records is a JSON list (or JSON lines) of raw clip records
{id, samples, closed, gemini, bank, stt} in replay order, e.g. an --export. Both sides read one file.

Evidence, on experiment.py's split (chronological 60/20/20 of the clips with STT; heldout and test are each scored by
one model trained on every clip before them, train predict-then-train):
  per side     VTT (= STT exactly), word, viseme and top-3 accuracy and cost per clip, each side by its own code
  paired       per clip, VTT = STT on each side: base right/cand wrong, base wrong/cand right, both, neither; the ids
               of every flip; an exact two-sided sign test on the flips
  word delta   cand - base word accuracy, 95% percentile bootstrap CI over clips resampled in pairs (fixed seed)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORKTREE = "worktree"
SPLITS = ("train", "heldout", "test")
RESAMPLES, SEED = 2000, 0
COLUMNS = [("VTT", "sentence_acc"), ("word", "word_acc"), ("viseme", "viseme_acc"), ("top-3", "top3_acc"),
           ("$/clip", "cost_per_clip")]
TEST_WARNING = ("WARNING: scoring the TEST split. The test split must not be used for tuning: read it once, at the "
                "end, and never change code, prompts or features because of it.")


# ---------------------------------------------------------------- one side, in its own process


def _side(tree: str, records_path: str, split: str, out: str) -> None:
    """Replay the records on `split` with the code in `tree`. Uses only experiment APIs that exist at bfc4434."""
    tree = Path(tree).resolve()
    sys.path.insert(0, str(tree))
    os.chdir(tree)  # visemes.load_costs reads data/viseme_costs.json relative to the working directory
    data = Path(records_path).read_bytes()
    records = json.loads(data)
    import experiment
    for name in ("experiment", "candidates", "scorer", "visemes"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(tree):
            raise SystemExit(f"{name} was imported from {sys.modules[name].__file__}, not from {tree}")

    calls, word_acc = [], experiment.word_acc

    def spy(vtts, stts):  # evaluate scores the split's VTT texts with word_acc: read them on the way through
        calls.append((list(vtts), list(stts)))
        return word_acc(vtts, stts)

    experiment.word_acc = spy
    metrics = experiment.evaluate(records, None, split)
    experiment.word_acc = word_acc
    scored = experiment.splits(records)[split]
    if len(calls) != 1 or len(calls[0][0]) != len(scored):
        raise SystemExit("this tree's evaluate no longer calls experiment.word_acc(vtts, stts) once per split; "
                         "compare.py cannot read its per-clip VTT")

    position = {id(r): i for i, r in enumerate(records)}
    clips = []
    for record, vtt, stt in zip(scored, *calls[0]):
        said = experiment._normalize(stt).split()
        clips.append({"id": str(record["id"]) if record.get("id") is not None else f"#{position[id(record)]}",
                      "stt": stt, "vtt": vtt, "correct": bool(vtt) and experiment.match(stt, [vtt]) is not None,
                      "word_edits": experiment._edits(experiment._normalize(vtt).split(), said),
                      "stt_words": len(said)})
    words = sum(c["stt_words"] for c in clips)
    if clips and (abs(np.mean([c["correct"] for c in clips]) - metrics["sentence_acc"]) > 1e-9
                  or words and abs(1 - sum(c["word_edits"] for c in clips) / words - metrics["word_acc"]) > 1e-9):
        raise SystemExit("per-clip VTT disagrees with this tree's own sentence/word accuracy; its scoring changed in "
                         "a way compare.py does not read")
    Path(out).write_text(json.dumps({"records_sha256": hashlib.sha256(data).hexdigest(), "metrics": metrics,
                                     "clips": clips}, default=float))


# ---------------------------------------------------------------- trees


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def materialize(ref: str, dest: Path) -> dict:
    """The side's tree: `worktree` is this checkout as it is; a ref is `git archive <commit> | tar -x` into dest."""
    if ref == WORKTREE:
        return {"ref": ref, "commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
                "tree": ROOT}
    try:
        commit = _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"{ref!r} is not a commit in {ROOT}") from None
    dest.mkdir(parents=True)
    archive = subprocess.Popen(["git", "-C", str(ROOT), "archive", commit], stdout=subprocess.PIPE)
    untar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() or untar.returncode:
        raise SystemExit(f"could not materialize {ref} ({commit}) into {dest}")
    return {"ref": ref, "commit": commit, "dirty": False, "tree": dest}


def _replay(tree: Path, records_file: Path, split: str, out: Path) -> subprocess.Popen:
    # -P: this script's directory (tools/, which has its own experiment.py) is not on the side's sys.path
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONHASHSEED": "0"}
    return subprocess.Popen([sys.executable, "-P", str(Path(__file__).resolve()), "_side", str(tree),
                             str(records_file), split, str(out)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


# ---------------------------------------------------------------- records


def store_records(store: Path | None, user: str) -> tuple[list[dict], str]:
    """The user's v1 clips as raw clip records, under their latest clip's phrase list, banks of earlier labels only."""
    sys.path.insert(0, str(ROOT))
    import clips
    stored = clips.load(store, user)
    v1 = [r for r in stored if r.get("record") == clips.V1 and r.get("user_id") == user]
    if not v1:
        raise SystemExit(f"no v1 clips for user {user!r} in the clip store")
    vocab = v1[-1]["vocab"]
    counts, records = Counter(), []
    for r in clips._replayed(stored, user, vocab):
        text = clips._label_text(r)
        records.append({"id": r["id"], "t": r.get("t"), "samples": r["samples"], "closed": r["closed"],
                        "gemini": r["gemini"], "bank": {"phrases": r["phrases"], "counts": dict(counts)},
                        "stt": text})
        if text is not None:
            counts[text] += 1
    return records, (f"user {user}: {len(records)} v1 clips under phrase list {vocab} "
                     f"({len(v1) - len(records)} under other lists skipped)")


def file_records(path: Path) -> tuple[list[dict], str]:
    text = path.read_text()
    records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines()
                                                                       if x.strip()]
    return records, str(path)


# ---------------------------------------------------------------- evidence


def sign_test(base_only: int, cand_only: int) -> float:
    """Exact two-sided sign test: P(a split of the flips at least this lopsided | each flip is a fair coin)."""
    n, k = base_only + cand_only, min(base_only, cand_only)
    if n == 0:
        return 1.0
    return min(1.0, float(Fraction(2 * sum(comb(n, i) for i in range(k + 1)), 2 ** n)))


def paired(base: list[dict], cand: list[dict]) -> dict:
    """Per clip, VTT = STT on each side; the flips by direction, in clip order; the sign test on them."""
    if [(c["id"], c["stt"]) for c in base] != [(c["id"], c["stt"]) for c in cand]:
        raise ValueError("the sides scored different clips or a different order; only the same clips can be paired")
    lost = [b["id"] for b, c in zip(base, cand) if b["correct"] and not c["correct"]]
    gained = [b["id"] for b, c in zip(base, cand) if c["correct"] and not b["correct"]]
    return {"clips": len(base), "base_right_cand_wrong": len(lost), "base_wrong_cand_right": len(gained),
            "both_right": sum(b["correct"] and c["correct"] for b, c in zip(base, cand)),
            "neither_right": sum(not b["correct"] and not c["correct"] for b, c in zip(base, cand)),
            "flips": {"base_right_cand_wrong": lost, "base_wrong_cand_right": gained},
            "sign_test_p": sign_test(len(lost), len(gained))}


def word_delta(base: list[dict], cand: list[dict], resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """cand - base word accuracy (1 - word edits / STT words, summed over clips) with a 95% percentile bootstrap CI;
    each resample draws clips with replacement and scores both sides on the same draw."""
    edits = np.array([[c["word_edits"] for c in side] for side in (base, cand)], float).reshape(2, -1)
    words = np.array([[c["stt_words"] for c in side] for side in (base, cand)], float).reshape(2, -1)
    result = {"delta": None, "ci95": None, "resamples": resamples, "seed": seed}
    if not words.sum(1).all():
        return result
    draws = np.random.default_rng(seed).integers(0, edits.shape[1], size=(resamples, edits.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = edits[0][draws].sum(1) / words[0][draws].sum(1) - edits[1][draws].sum(1) / words[1][draws].sum(1)
    low, high = np.nanpercentile(spread, [2.5, 97.5])
    point = edits[0].sum() / words[0].sum() - edits[1].sum() / words[1].sum()
    return {**result, "delta": float(point), "ci95": [float(low), float(high)]}


def report_of(split: str, source: str, records: list[dict], digest: str, sides: dict) -> dict:
    base, cand = sides["base"]["clips"], sides["cand"]["clips"]
    return {
        "split": split,
        "records": {"source": source, "count": len(records), "sha256": digest},
        **{name: {key: side[key] for key in ("ref", "commit", "dirty", "records_sha256", "metrics")}
           for name, side in sides.items()},
        "paired": paired(base, cand),
        "word_acc_delta": word_delta(base, cand),
        "clips": [{"id": b["id"], "stt": b["stt"], "base_vtt": b["vtt"], "cand_vtt": c["vtt"],
                   "base_correct": b["correct"], "cand_correct": c["correct"]} for b, c in zip(base, cand)],
    }


def _number(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def table(report: dict) -> str:
    sides = {name: report[name] for name in ("base", "cand")}
    lines = [f"{name} {s['ref']} @ {s['commit'][:7]}{' + uncommitted edits' if s['dirty'] else ''}"
             for name, s in sides.items()]
    p = report["paired"]
    lines.append(f"records: {report['records']['count']} from {report['records']['source']} "
                 f"(sha256 {report['records']['sha256'][:12]}); split {report['split']}: {p['clips']} clips with STT")
    rows = [["", *(header for header, _ in COLUMNS)]]
    rows += [[name, *(_number(s["metrics"].get(key), ".3f") for _, key in COLUMNS)] for name, s in sides.items()]
    deltas = [None if None in (b := sides["base"]["metrics"].get(key), c := sides["cand"]["metrics"].get(key))
              else c - b for _, key in COLUMNS]
    rows.append(["delta", *(_number(d, "+.3f") for d in deltas)])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines += ["", *("  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
                    for row in rows)]
    lost, gained = p["flips"]["base_right_cand_wrong"], p["flips"]["base_wrong_cand_right"]
    flips = len(lost) + len(gained)
    lines += ["", f"paired by clip (VTT = STT on each side), {p['clips']} clips:",
              f"  base right, cand wrong  {len(lost):4d}  {' '.join(lost)}",
              f"  base wrong, cand right  {len(gained):4d}  {' '.join(gained)}",
              f"  both right              {p['both_right']:4d}",
              f"  neither right           {p['neither_right']:4d}",
              f"  sign test on {flips} flips (exact, two-sided): p = {p['sign_test_p']:.4g}"]
    w = report["word_acc_delta"]
    lines.append("word accuracy, cand - base: -" if w["delta"] is None else
                 f"word accuracy, cand - base: {w['delta']:+.3f}, 95% bootstrap CI [{w['ci95'][0]:+.3f}, "
                 f"{w['ci95'][1]:+.3f}] ({w['resamples']} paired resamples of the clips)")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_side"]:  # the child process of one side
        _side(*argv[1:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--user", help="replay this user's v1 clips from the clip store")
    source.add_argument("--records", type=Path, help="JSON list (or JSON lines) of raw clip records, in order")
    parser.add_argument("--store", type=Path, help="clip store directory for --user (default: MongoDB or data/clips)")
    parser.add_argument("--base", default="main", help="git ref of the base side (default: main)")
    parser.add_argument("--cand", default=WORKTREE, help="git ref of the candidate side, or 'worktree' (default)")
    parser.add_argument("--split", choices=SPLITS, default="heldout")
    parser.add_argument("--final", action="store_true", help="allow --split test: once, at the end, never to tune")
    parser.add_argument("--export", type=Path, help="write the records both sides read, as JSON")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final:
        parser.error("--split test needs --final: the test split is read once, at the end, and never used for tuning")
    if args.final and args.split != "test":
        parser.error("--final only goes with --split test")
    if args.store and not args.user:
        parser.error("--store goes with --user")
    if args.final:
        print(TEST_WARNING, file=sys.stderr)

    records, source = store_records(args.store, args.user) if args.user else file_records(args.records)
    data = json.dumps(records, default=str).encode()
    digest = hashlib.sha256(data).hexdigest()
    if args.export:
        args.export.write_bytes(data)
    with tempfile.TemporaryDirectory(prefix="compare-") as tmp:
        tmp = Path(tmp)
        (tmp / "records.json").write_bytes(data)  # one file: both sides read the same bytes
        sides = {name: materialize(ref, tmp / name) for name, ref in (("base", args.base), ("cand", args.cand))}
        running = {name: _replay(side["tree"], tmp / "records.json", args.split, tmp / f"{name}.out.json")
                   for name, side in sides.items()}
        for name, proc in running.items():
            _, err = proc.communicate()
            if proc.returncode:
                raise SystemExit(f"{name} side ({sides[name]['ref']}) failed:\n{err[-4000:]}")
            sides[name].update(json.loads((tmp / f"{name}.out.json").read_text()))
    for name, side in sides.items():
        if side["records_sha256"] != digest:
            raise SystemExit(f"{name} side read different records than were written")
    report = report_of(args.split, source, records, digest, sides)
    print(table(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
