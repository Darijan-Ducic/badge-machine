"""lab-features paired arms: lip measurements (lipfeatures.py) as Muse's input, vs today's photo frames, same turns.

  uv run --env-file .env python tools/exp_features.py run <dir> --arms base,sil,table,text [--n 40] [--workers 4]
  uv run --env-file .env python tools/exp_features.py report <dir> --arms sil,table,text [--save]

<dir> holds turns.jsonl + <clip>.mp4 (tools/exp_features_fetch.py) and sam/<clip>.json (tools/exp_features_sam.py).
Arms (one schema-v1 open call each, K=1, prompt p3 otherwise unchanged; reads cached in <dir>/arms/<arm>/<clip>.json):
  base   today's 2x2 photo grids (muse_sampler.labeled_grids), run here so every arm is read at the same time
  sil    (a) the same grid layout, each tile replaced by the frame's lip silhouette (lipfeatures.silhouette)
  table  (b) today's photo grids plus the per-frame measurement table in the prompt
  text   (c) the measurement table only, no images
Metric: lift = acc(read vs own STT) - mean acc(read vs the other turns' STTs), per turn, viseme and word; paired
per-turn wins/losses of each arm's lift vs base, two-sided sign test. Evidence only; defaults untouched.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

AUTHOR = "agent:lab-features"
TABLE_COLS = ("open", "width", "aspect", "dark", "teeth")

_TABLE_NOTE = """\
Per-frame lip measurements (columns, all measured by a program from the mouth region of the same video):
  frame   frame number (24 per second; with images, the number printed on each tile)
  open    height of the open gap between the lips / lip width (0 = lips together; about 0.15+ = clearly open)
  width   mouth width / this speaker's usual mouth width (above 1 = spread, below 1 = narrowed or rounded)
  aspect  outer lip height / outer lip width
  dark    share of the mouth centre that is dark open cavity (0 to 1)
  teeth   share of the mouth centre that is visible teeth (0 to 1)
How to use them: a run of frames with open and teeth near 0 is the lips together (P, B, M, or rest); a rise in open
is a vowel or an open consonant; width below 1 with low teeth is rounding (W, O, R); teeth with small open is S, F,
TH or E. The measurements are noisy at the clip edges and on blurred frames."""


def _sil_intro(prompt: str) -> str:
    old = ("The images show a muted video clip of a person; there is no audio. Read their lips and report what you "
           "see, with your uncertainty.")
    new = ("The images show a person's mouth while they speak, drawn as a diagram, one drawing per video frame; there "
           "is no audio and no face. In each drawing the gray ellipse is the outer edge of the lips (its width and "
           "height follow the lips), black inside it is the open mouth cavity or the dark gap between the lips, and "
           "white inside it is visible teeth. Read the speech from these mouth shapes and their changes over time, "
           "and report it with your uncertainty.")
    assert old in prompt, "muse_sampler open prompt changed; update _sil_intro"
    return prompt.replace(old, new)


def _text_prompt(prompt: str, n_frames: int, tbl: str) -> str:
    old_intro = ("The images show a muted video clip of a person; there is no audio. Read their lips and report what "
                 "you see, with your uncertainty.")
    new_intro = ("There are no images and no audio. Below is a per-frame measurement table of a person's lips while "
                 "they speak (a muted video, 24 frames per second, frames 0 to %d). Read the speech from these lip "
                 "measurements and report it with your uncertainty." % (n_frames - 1))
    head, sep, rest = prompt.partition("\n\nFill in the JSON fields")
    assert old_intro in head and sep, "muse_sampler open prompt changed; update _text_prompt"
    rest = rest.replace("frame numbers printed on the tiles", "frame numbers from the table")
    rest = rest.replace('quality: mouth_visible is "full"', 'quality (judge from the measurements; "full" and "low" '
                        'when they look consistent): mouth_visible is "full"')
    return (f"{new_intro}\n\n{_TABLE_NOTE}\n\n{tbl}{sep}{rest}")


def _table_prompt(prompt: str, tbl: str) -> str:
    return (prompt + "\n\nAlso use this table, measured by a program from the same frames; when the table and the "
            "images disagree, trust the images.\n" + _TABLE_NOTE + "\n\n" + tbl)


def _grids_from_images(images, tile_width=320):
    """muse_sampler.Grids from per-frame PIL images (silhouettes), same 2x2 layout, stamps and frame skipping."""
    import muse_sampler as ms
    from PIL import Image
    cols, rows = 2, 2
    frames = [im.resize((tile_width, round(im.height * tile_width / im.width))) for im in images]
    capacity = ms.MAX_IMAGES * cols * rows
    keep = list(range(len(frames)))
    if len(frames) > capacity:
        keep = [round(i * (len(frames) - 1) / (capacity - 1)) for i in range(capacity)]
    out, labels = [], []
    for start in range(0, len(keep), cols * rows):
        chunk = keep[start:start + cols * rows]
        width, height = frames[0].size
        sheet = Image.new("RGB", (cols * width, rows * height))
        for slot, frame in enumerate(chunk):
            sheet.paste(ms.stamp(frames[frame].copy(), frame), (slot % cols * width, slot // cols * height))
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=90)
        out.append(buf.getvalue())
        labels.append(chunk)
    return ms.Grids(out, labels, len(frames))


def features(row, root: Path):
    """(frames, boxes, series, ref) for a turn, from the cached SAM boxes; None without a SAM cache."""
    import lipfeatures as lf
    from scorers import FPS
    video = Path(row["video"])
    cache = root / "sam" / f"{video.stem}.json"
    if not cache.exists():
        return None
    frames = lf.decode(video, FPS)
    boxes = lf.track(json.loads(cache.read_text())["frames"], len(frames))
    return frames, boxes, lf.series(frames, boxes), lf.Reference(frames, boxes)


def request_for(arm: str, row, root: Path) -> dict:
    import lipfeatures as lf
    import muse_sampler as ms
    video = Path(row["video"])
    grids = ms.labeled_grids(video, ms.CONFIG["grid"], ms.CONFIG["tile_width"])
    if arm == "base":
        return ms._request(ms._open_prompt(grids), grids, "lipread_v1", ms.WIRE_SCHEMA), grids.n_frames
    frames, boxes, s, ref = features(row, root)
    n = min(len(frames), grids.n_frames)
    tbl = lf.table({k: v[:n] for k, v in s.items()}, TABLE_COLS)
    if arm == "sil":
        sil = _grids_from_images([lf.silhouette(f, b, ref) for f, b in zip(frames[:n], boxes[:n])])
        return ms._request(_sil_intro(ms._open_prompt(sil)), sil, "lipread_v1", ms.WIRE_SCHEMA), sil.n_frames
    if arm == "table":
        return ms._request(_table_prompt(ms._open_prompt(grids), tbl), grids, "lipread_v1", ms.WIRE_SCHEMA), n
    if arm == "text":
        empty = ms.Grids([], [], n)
        return ms._request(_text_prompt(ms._open_prompt(grids), n, tbl), empty, "lipread_v1", ms.WIRE_SCHEMA), n
    raise ValueError(arm)


def read(arm: str, row, root: Path, client, sem) -> dict:
    import muse_sampler as ms
    out = root / "arms" / arm / f"{Path(row['video']).stem}.json"
    if out.exists():
        return json.loads(out.read_text())
    request, n = request_for(arm, row, root)

    def finish(raw):
        output = ms.from_wire(raw)
        problems = ms.validate_open(output, n)
        return output, ("; ".join(problems) or None)

    for attempt in range(4):
        with sem:
            item = ms._call(client, "open", 0, request, finish)
        err = item.get("error") or ""
        if item.get("output") is None and any(c in err for c in ("429", "500", "502", "503", "504", "Timeout", "timed out")):
            time.sleep(5 * 2 ** attempt)
            continue
        break
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(item))
    return item


def top(item) -> str:
    from preheat import _normalize
    o = (item or {}).get("output")
    if not o or not o.get("hypotheses"):
        return ""
    return _normalize(max(o["hypotheses"], key=lambda h: h["s"])["text"])


def rows_of(root: Path, n: int | None):
    rows = [json.loads(line) for line in (root / "turns.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("video") and (root / "sam" / f"{Path(r['video']).stem}.json").exists()]
    return rows[:n] if n else rows


def cmd_run(args) -> int:
    from muse_video import make_client
    root = Path(args.dir)
    rows = rows_of(root, args.n)
    client, sem = make_client(), threading.Semaphore(args.workers)
    jobs = [(arm, row) for row in rows for arm in args.arms.split(",")]

    def one(job):
        arm, row = job
        try:
            item = read(arm, row, root, client, sem)
            return f"{arm:5} {Path(row['video']).stem}: {top(item)[:70]!r} {('ERR ' + item['error'][:80]) if item.get('error') else ''}"
        except Exception as e:
            return f"{arm:5} {Path(row['video']).stem}: FAIL {type(e).__name__}: {e}"

    with ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(one, jobs):
            print(line, flush=True)
    return 0


def lifts(reads: list[str], stts: list[str]) -> dict:
    import experiment
    out = {"viseme": [], "word": [], "lift_viseme": [], "lift_word": []}
    for i, (r, s) in enumerate(zip(reads, stts)):
        for key, fn in (("viseme", experiment.viseme_acc), ("word", experiment.word_acc)):
            own = fn([r], [s])
            imp = np.mean([fn([r], [o]) for j, o in enumerate(stts) if j != i])
            out[key].append(own)
            out["lift_" + key].append(own - imp)
    return out


def sign_test(a, b) -> dict:
    from scipy.stats import binomtest
    wins = int(sum(x > y + 1e-12 for x, y in zip(a, b)))
    losses = int(sum(y > x + 1e-12 for x, y in zip(a, b)))
    p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    return {"wins": wins, "losses": losses, "ties": len(a) - wins - losses, "p": float(p)}


def cmd_report(args) -> int:
    import experiment
    root = Path(args.dir)
    user = json.loads((root / "turns.jsonl").read_text().splitlines()[0]).get("user") or root.name
    rows = rows_of(root, args.n)
    arms = ["base"] + [a for a in args.arms.split(",") if a != "base"]
    items = {a: [] for a in arms}
    keep = []
    for row in rows:
        stem = Path(row["video"]).stem
        got = {a: root / "arms" / a / f"{stem}.json" for a in arms}
        if all(p.exists() for p in got.values()):
            keep.append(row)
            for a in arms:
                items[a].append(json.loads(got[a].read_text()))
    stts = [r["stt"] for r in keep]
    sam_usd = np.mean([json.loads((root / "sam" / f"{Path(r['video']).stem}.json").read_text())["usd"] for r in keep])
    print(f"{root.name}: n={len(keep)} turns with every arm read; SAM ${sam_usd:.4f}/turn")
    results = {}
    for a in arms:
        reads = [top(i) for i in items[a]]
        m = lifts(reads, stts)
        muse_usd = np.mean([experiment._call_cost(i["usage"], "muse") if i.get("usage") else 0 for i in items[a]])
        valid = sum(1 for i in items[a] if i.get("output"))
        results[a] = {"m": m, "usd": float(muse_usd + (0 if a == "base" else sam_usd)), "valid": valid, "reads": reads}
        print(f"  {a:5} valid {valid:>3}  viseme {np.mean(m['viseme']):.3f} lift {np.mean(m['lift_viseme']):+.3f} | "
              f"word {np.mean(m['word']):.3f} lift {np.mean(m['lift_word']):+.3f} | ${results[a]['usd']:.4f}/turn")
    base = results["base"]["m"]
    for a in arms[1:]:
        m = results[a]["m"]
        sv, sw = sign_test(m["lift_viseme"], base["lift_viseme"]), sign_test(m["lift_word"], base["lift_word"])
        results[a]["sign"] = {"viseme": sv, "word": sw}
        print(f"  {a} vs base: lift viseme W/L {sv['wins']}/{sv['losses']} p={sv['p']:.3f}; "
              f"lift word W/L {sw['wins']}/{sw['losses']} p={sw['p']:.3f}")
    if args.show:
        for k, r in enumerate(keep[: args.show]):
            print(f"\nSTT  {r['stt']}")
            for a in arms:
                print(f"{a:5}{results[a]['reads'][k]}")
    if args.save:
        import runs
        for a in arms[1:]:
            save_run(runs, a, results, keep, user, args)
    return 0


VERDICTS = {}  # arm -> (verdict, reopen), filled from the command line


def save_run(runs, arm, results, keep, user, args):
    m, b = results[arm]["m"], results["base"]["m"]
    lv, lw = float(np.mean(m["lift_viseme"])), float(np.mean(m["lift_word"]))
    blv, blw = float(np.mean(b["lift_viseme"])), float(np.mean(b["lift_word"]))
    verdict, reopen = args.verdict or "inconclusive", args.reopen or ""
    record = {
        "id": f"labfeatures-{arm}-{uuid.uuid4().hex[:6]}", "t": time.time(),
        "name": (f"features {arm} · lift viseme {lv:+.3f} vs base {blv:+.3f} (p {results[arm]['sign']['viseme']['p']:.2f})"
                 f" · lift word {lw:+.3f} vs {blw:+.3f} · n {len(keep)} · {verdict}"),
        "dataset": user, "user_id": f"exp:lab-features-{arm}:{user}", "author": AUTHOR, "loop": "framework",
        "lever": "input representation (lip features)", "experiment": "lab-features", "arm": arm,
        "hypothesis": args.hypothesis or "", "turns": [Path(r["video"]).stem for r in keep],
        "metrics": {"clips": len(keep), "lift_viseme": lv, "lift_word": lw,
                    "vtt_muse_viseme_acc": float(np.mean(m["viseme"])), "vtt_muse_word_acc": float(np.mean(m["word"])),
                    "valid": results[arm]["valid"], "usd_per_turn": results[arm]["usd"]},
        "baseline_metrics": {"clips": len(keep), "lift_viseme": blv, "lift_word": blw,
                             "vtt_muse_viseme_acc": float(np.mean(b["viseme"])),
                             "vtt_muse_word_acc": float(np.mean(b["word"])), "usd_per_turn": results["base"]["usd"]},
        "flips": results[arm]["sign"], "p": results[arm]["sign"]["viseme"]["p"], "cost": results[arm]["usd"],
        "verdict": verdict, "reopen": reopen, "baseline": None, "gate": None,
        "config": {"k": 1, "calls": "open only", "prompt": "p3 + " + arm, "sam": "video lips boxes"},
        "curve": [], "code_commit": __import__("experiment")._head(),
    }
    runs.save(record)
    print(f"  saved run {record['id']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        s = sub.add_parser(name)
        s.add_argument("dir")
        s.add_argument("--arms", default="base,sil,table,text")
        s.add_argument("--n", type=int)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--save", action="store_true")
        s.add_argument("--show", type=int, default=0)
        s.add_argument("--verdict")
        s.add_argument("--reopen")
        s.add_argument("--hypothesis")
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
