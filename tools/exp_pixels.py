"""lab-pixels: re-read a FIXED set of stored dictation turns with pixel variants of the mouth crop (pixels.py), paired
against the original clips (today's pipeline = control), and score them with the lift metric.

  uv run --env-file .env python tools/exp_pixels.py sheet --user U --arms stab,sharp [--turns 3]   # $0 contact sheets
  uv run --env-file .env python tools/exp_pixels.py run --user U --name pixels-1 --arms stab,sharp --n 15 [--save]

`run` freezes the first N processed turns (oldest first) into data/exp/<name>-turns.json on first use and reuses that
file afterwards (a larger --n appends newer turns). Each re-read is a clip of user `exp:lab-pixels-<arm>:<U>` read by
Muse only (K open samples; closed and Gemini skipped), at most 4 model calls in flight, retried on 429/5xx. A turn
already re-read for an arm is reused, so reruns only pay for what is missing. Metrics, per arm, over the same turns:
VTT · Muse (most common top hypothesis) word/viseme acc, LIFT (viseme acc vs its own STT minus the mean vs the other
turns' STTs), best-of-K viseme acc, paired lift up/down vs control with an exact sign test, and $.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import pixels  # noqa: E402
from pixels import Spec  # noqa: E402
from tools.reread import muse_pick  # noqa: E402

ARMS = {
    "stab": Spec(stabilize=True),
    "sharp": Spec(contrast=True, sharpen=1.0, crf=14),
    "stab-sharp": Spec(stabilize=True, contrast=True, sharpen=1.0, crf=14),
    "stab-sharp-w320": Spec(stabilize=True, contrast=True, sharpen=1.0, crf=14, width=320),
    "stab-gray": Spec(stabilize=True, gray=True, contrast=True, sharpen=1.0, crf=14),
    "stab-sam": Spec(stabilize=True, sam_lips=True),
    "stab-sam-sharp": Spec(stabilize=True, sam_lips=True, contrast=True, sharpen=1.0, crf=14),
}
EXP = Path("data/exp")
SHEETS = Path("/tmp/lab-pixels")
CALLS = threading.Semaphore(4)  # model calls in flight, across all clips


# ---------------------------------------------------------------- turns


def _all_turns(user: str) -> list[dict]:
    import sessions
    store = sessions._backend(None)
    out = []
    for summary in sorted(sessions.list_sessions(user)["sessions"], key=lambda s: s.get("t") or 0):
        record = store.get(summary["id"])
        for turn in record["turns"]:
            if turn["status"] == "processed" and turn.get("clip_id") and turn.get("stt"):
                out.append({"session": summary["id"], "turn": turn["i"], "clip": turn["clip_id"],
                            "start_ms": turn["start_ms"], "end_ms": turn["end_ms"], "stt": turn["stt"]})
    return out


def turn_set(user: str, name: str, n: int) -> list[dict]:
    path = EXP / f"{name}-turns.json"
    frozen = json.loads(path.read_text()) if path.exists() else []
    if len(frozen) < n:
        known = {t["clip"] for t in frozen}
        frozen += [t for t in _all_turns(user) if t["clip"] not in known][:n - len(frozen)]
        EXP.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(frozen, indent=1))
    return frozen[:n]


@functools.lru_cache(maxsize=None)
def _session(session_id: str) -> tuple[bytes, list]:
    import sessions
    store = sessions._backend(None)
    return store.recording(session_id), store.boxes(session_id)


def _segment(turn: dict, tmp: str) -> tuple[Path, list]:
    """The turn's segment, cut exactly as the dictation path cuts it (sessions._cut), cached under data/exp/segments."""
    import sessions
    cache = EXP / "segments"
    for container in ("webm", "mp4"):
        if (cache / f"{turn['clip']}.{container}").exists():
            return cache / f"{turn['clip']}.{container}", json.loads((cache / f"{turn['clip']}.boxes").read_text())
    recording, boxes = _session(turn["session"])
    container = sessions._container(recording)
    source, segment = Path(tmp, f"rec.{container}"), Path(tmp, f"turn.{container}")
    source.write_bytes(recording)
    sessions._cut(source, turn["start_ms"], turn["end_ms"], container, segment)
    sliced = sessions.slice_boxes(boxes, turn["start_ms"], turn["end_ms"])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{turn['clip']}.boxes").write_text(json.dumps(sliced))
    (cache / f"{turn['clip']}.{container}").write_bytes(segment.read_bytes())
    return segment, sliced


# ---------------------------------------------------------------- Muse only, bounded and retried


def _transient(error: str | None) -> bool:
    return bool(error) and any(s in error for s in ("429", "RateLimit", "InternalServerError", "APIConnection",
                                                    "Timeout", " 500", " 502", " 503", " 504"))


def sample_open(clip: Path, k: int) -> list[dict]:
    import muse_sampler as ms
    from muse_video import make_client
    grids = ms.labeled_grids(clip, ms.CONFIG["grid"], ms.CONFIG["tile_width"])
    request = ms._request(ms._open_prompt(grids), grids, "lipread_v1", ms.WIRE_SCHEMA)

    def finish(raw):
        output = ms.from_wire(raw)
        return output, ("; ".join(ms.validate_open(output, grids.n_frames)) or None)

    client = make_client()

    def one(i):
        for attempt in range(5):
            with CALLS:
                item = ms._call(client, "open", i, request, finish)
            if not (item["output"] is None and _transient(item["error"])):
                return item
            time.sleep(2 ** attempt * 3)
        return item

    with ThreadPoolExecutor(k) as pool:
        return list(pool.map(one, range(k)))


def sam_lips(clip_id: str):
    """segment_video for one turn: SAM 3.1 "lips" on the segment, cached in data/exp/sam-lips/<clip>.json."""
    def segment_video(source, prompt):
        import sam
        path = EXP / "sam-lips" / f"{clip_id}.json"
        if path.exists():
            out = json.loads(path.read_text())
        else:
            for attempt in range(4):
                try:
                    with CALLS:
                        out = sam.segment_video(source, prompt)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3 or not _transient(str(e)):
                        raise
                    time.sleep(2 ** attempt * 5)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
        return {**out, "frames": {int(i): v for i, v in out["frames"].items()}}
    return segment_video


def _skip_closed(clip, phrases):
    return {"call": "closed", "sample_idx": 0, "latency_ms": 0, "output": None, "error": "skipped (lab-pixels)"}


def _skip_gemini(clip, phrases, usage=None):
    raise RuntimeError("skipped (lab-pixels)")


def _reread(turn: dict, arm: str, user: str, name: str, k: int, existing: dict) -> dict:
    import clips
    from scorers import load_phrases
    if turn["clip"] in existing:
        return existing[turn["clip"]]
    with tempfile.TemporaryDirectory() as tmp:
        segment, boxes = _segment(turn, tmp)
        return clips.record_clip(
            segment.read_bytes(), phrases=load_phrases(), user_id=f"exp:lab-pixels-{arm}:{user}", mode="experiment",
            k=k, transcript=turn["stt"], mouths=boxes, crop_fn=pixels.cropper(ARMS[arm], sam_lips(turn["clip"])),
            sample_open=sample_open,
            score_closed=_skip_closed, gemini=_skip_gemini,
            extra={"experiment": {"name": name, "lab": "lab-pixels", "arm": arm, "spec": ARMS[arm].__dict__,
                                  "session_id": turn["session"], "turn": turn["turn"], "source_clip": turn["clip"]}})


# ---------------------------------------------------------------- metrics


def sign_p(up: int, down: int) -> float:
    """Exact two-sided sign test (ties dropped)."""
    n = up + down
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(up, down) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _top(output) -> str:
    from preheat import _normalize
    return _normalize(max(output["hypotheses"], key=lambda h: h["s"])["text"]) if output and output.get(
        "hypotheses") else ""


def score(picks: list[str], stts: list[str], samples: list[list[dict]]) -> dict:
    import experiment as ex
    own = [ex.viseme_acc([p], [s]) or 0.0 for p, s in zip(picks, stts)]
    others = [np.mean([ex.viseme_acc([p], [s]) or 0.0 for j, s in enumerate(stts) if j != i])
              for i, p in enumerate(picks)]
    word_own = [ex.word_acc([p], [s]) or 0.0 for p, s in zip(picks, stts)]
    word_others = [np.mean([ex.word_acc([p], [s]) or 0.0 for j, s in enumerate(stts) if j != i])
                   for i, p in enumerate(picks)]
    best = [max([ex.viseme_acc([_top(x.get("output"))], [s]) or 0.0 for x in smp] or [0.0])
            for smp, s in zip(samples, stts)]
    return {"word": ex.word_acc(picks, stts), "viseme": ex.viseme_acc(picks, stts),
            "lift": float(np.mean(np.subtract(own, others))), "word_lift": float(np.mean(np.subtract(word_own,
                                                                                                     word_others))),
            "best_of_k_viseme": float(np.mean(best)), "per_turn_lift": list(np.subtract(own, others)),
            "per_turn_viseme": own}


def run(args) -> int:
    import clips
    import experiment as ex
    turns = turn_set(args.user, args.name, args.n)
    store = clips.backend(None)
    stored = {arm: {r["experiment"]["source_clip"]: r for r in clips.load(None, f"exp:lab-pixels-{arm}:{args.user}")
                    if r.get("experiment")} for arm in args.arms}
    if args.score_only:  # $0: only the turns every arm already has
        turns = [t for t in turns if all(t["clip"] in stored[arm] for arm in args.arms)]
    print(f"{args.name}: {len(turns)} turns of {args.user}; arms {args.arms}", file=sys.stderr)
    control = [store.get(t["clip"]) for t in turns]
    stts = [t["stt"] for t in turns]
    rows = {"control": {"records": control}}
    for arm in args.arms:
        existing = stored[arm]
        with ThreadPoolExecutor(6) as pool:  # clips in flight; CALLS caps the model calls
            records = list(pool.map(lambda t: _reread(t, arm, args.user, args.name, args.k, existing), turns))
        rows[arm] = {"records": records}
    out = {}
    for arm, row in rows.items():
        picks = [muse_pick(r["samples"]) or "" for r in row["records"]]
        m = score(picks, stts, [r["samples"] for r in row["records"]])
        m["cost"] = sum(ex.clip_cost(r) or 0 for r in row["records"]) if arm != "control" else 0.0
        m["picks"] = picks
        out[arm] = m
    base = out["control"]
    print(f"\n{'arm':<16} {'n':>3} {'word':>6} {'vis':>6} {'lift':>7} {'wlift':>7} {'bestK':>6} "
          f"{'up':>3} {'dn':>3} {'p':>6} {'$':>6}")
    for arm, m in out.items():
        diff = np.subtract(m["per_turn_lift"], base["per_turn_lift"])
        up, down = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
        m.update(up=up, down=down, p=sign_p(up, down))
        print(f"{arm:<16} {len(turns):>3} {m['word']:>6.3f} {m['viseme']:>6.3f} {m['lift']:>+7.3f} "
              f"{m['word_lift']:>+7.3f} {m['best_of_k_viseme']:>6.3f} {up:>3} {down:>3} {m['p']:>6.3f} "
              f"{m['cost']:>6.2f}")
    result = {"name": args.name, "user": args.user, "turns": [t["clip"] for t in turns], "arms": {
        arm: {k: v for k, v in m.items() if k not in ("picks",)} | {"spec": ARMS[arm].__dict__ if arm in ARMS else
                                                                   "today's pipeline (stored clip)"}
        for arm, m in out.items()}}
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / f"{args.name}-result.json").write_text(json.dumps(result, indent=1, default=float))
    if args.verbose:
        for i, t in enumerate(turns):
            print(f"\nSTT {t['stt']}")
            for arm, m in out.items():
                print(f"  {arm:<14} {m['per_turn_lift'][i]:+.2f} {m['picks'][i]}")
    return 0


# ---------------------------------------------------------------- contact sheets ($0)


def sheet(args) -> int:
    import crop
    from muse_sampler import _decode, stamp
    SHEETS.mkdir(parents=True, exist_ok=True)
    turns = _all_turns(args.user)[:args.turns]
    for n, turn in enumerate(turns):
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            segment, boxes = _segment(turn, tmp)
            for arm in ["control", *args.arms]:
                clip = Path(tmp, f"{arm}.mp4")
                if arm == "control":
                    crop.crop(segment, clip, mouths=boxes)
                else:
                    pixels.cropper(ARMS[arm], sam_lips(turn["clip"]))(segment, clip, mouths=boxes)
                frames = _decode(clip, 320)
                pick = [round(i * (len(frames) - 1) / 7) for i in range(8)]
                rows.append((arm, [stamp(frames[i].copy(), i) for i in pick]))
        tw, th = 320, max(f.height for _, fs in rows for f in fs)
        image = Image.new("RGB", (120 + 8 * tw, len(rows) * th), "white")
        draw = ImageDraw.Draw(image)
        for r, (arm, frames) in enumerate(rows):
            draw.text((4, r * th + th // 2), arm, fill="black")
            for c, f in enumerate(frames):
                image.paste(f, (120 + c * tw, r * th))
        path = SHEETS / f"sheet-{n}-{turn['clip']}.png"
        image.save(path)
        print(path, "|", turn["stt"])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--user", required=True)
    s.add_argument("--arms", type=lambda v: v.split(","), default=list(ARMS))
    s.add_argument("--turns", type=int, default=3)
    r = sub.add_parser("run")
    r.add_argument("--user", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--arms", type=lambda v: v.split(","), required=True)
    r.add_argument("--n", type=int, default=15)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--score-only", action="store_true", help="no model calls: score the turns every arm has")
    args = p.parse_args(argv)
    return sheet(args) if args.cmd == "sheet" else run(args)


if __name__ == "__main__":
    sys.exit(main())
